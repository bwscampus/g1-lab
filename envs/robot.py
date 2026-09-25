"""Stage 3: live deployment through unitree_sdk2py.

High-level control with LocoClient, then upper-body targets published on
``rt/arm_sdk`` at 50 Hz with the blend weight in ``motor_cmd[29].q``. Handover
protocol: release any stale takeover first, read a fresh LowState, and always
release the arms in teardown (normal end, Ctrl-C, or exception).

Two modes, ``--mode`` (required, no default):

  gantry    full bring-up and shutdown, for a robot hanging in a gantry:
            Damp -> FSM 4 (locked stand) -> FSM 200 (main operation) -> run
            -> release arms -> Damp
  standing  robot already standing under its own controller (any FSM):
            remember the current FSM -> FSM 200 -> run -> release arms
            -> back to the remembered FSM. Never damps.

Walking (``--walk``): a policy's ``Action.base`` velocity is sent as
``LocoClient.Move(vx, vy, vyaw)`` from a 10 Hz commander thread (the RPC blocks,
so it never runs on the tick thread), clamped to ``config.BASE_VEL_MAX``. Move
is a 1 s dead-man command on the robot; the commander re-sends while a command
is set, sends ``StopMove`` when it clears, and stops before the arms are
released. Move works in FSM 200, so no extra FSM transition is needed.
Pre-flight for walking: standing on the floor or hoisted with feet touching,
~2 m clear in every direction, no tether to snag, spotter on the remote (L2+B).

Camera: for policies that use it, the head camera is streamed over WebRTC
(``--camera-ip``, ``$UNITREE_AES_128_KEY``) and connected *before* any FSM
transition, so a bad camera link fails before the robot is touched. The robot
accepts one WebRTC client: disconnect the Unitree app first.

Pre-flight:
  * robot NOT in debug mode
  * clear space around the arms
  * someone on the remote with L2+B ready

    python run.py --env robot --policy tpose --iface eth0 --mode standing
"""
from __future__ import annotations

import argparse
import os
import threading
import time

import numpy as np

from camera import WebRTCCamera
from config import ARM_SDK_WEIGHT_IDX, BASE_VEL_MAX, CONTROL_DT, NUM_JOINTS, UPPER_BODY
from policy import Action
from envs.base import Env
from envs.base import shield_sigint
from envs.monitor import JointMonitor
from skills import LOCO_METHODS


FSM_LOCKED_STAND = 4
FSM_MAIN = 200
MODES = ("gantry", "standing")


class ArmSdk:
    """Thin publisher for rt/arm_sdk plus a LowState subscriber."""

    def __init__(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.state = None
        self.crc = CRC()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        # Zero gains everywhere so a stale buffer can never command the legs.
        for j in range(NUM_JOINTS):
            mc = self.cmd.motor_cmd[j]
            mc.q = mc.dq = mc.tau = mc.kp = mc.kd = 0.0

    def _on_state(self, msg) -> None:
        self.state = msg

    def fresh_state(self, settle: float = 0.5, timeout: float = 5.0) -> np.ndarray:
        t0 = time.time()
        while self.state is None:
            if time.time() - t0 > timeout:
                raise TimeoutError("no LowState received; check the network interface")
            time.sleep(0.05)
        time.sleep(settle)
        return np.array([m.q for m in self.state.motor_state[:NUM_JOINTS]])

    def release(self, duration: float = 1.0) -> None:
        """Publish weight=0 with zero gains so ai_sport drops any arm_sdk takeover."""
        for j in UPPER_BODY:
            mc = self.cmd.motor_cmd[j]
            mc.q = mc.dq = mc.tau = mc.kp = mc.kd = 0.0
        self.cmd.motor_cmd[ARM_SDK_WEIGHT_IDX].q = 0.0
        self.cmd.crc = self.crc.Crc(self.cmd)
        for _ in range(int(duration / CONTROL_DT)):
            self.pub.Write(self.cmd)
            time.sleep(CONTROL_DT)

    def send(self, action: Action) -> None:
        for j in action.joints:
            mc = self.cmd.motor_cmd[j]
            mc.q = float(action.q[j])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = action.kp
            mc.kd = action.kd
        self.cmd.motor_cmd[ARM_SDK_WEIGHT_IDX].q = float(action.weight)
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)


class BaseCommander:
    """Owns LocoClient.Move on its own thread with a latest-only command slot.
    ``command(base)`` never blocks; the worker re-sends every ``period`` seconds
    while a command is set and sends StopMove once when it clears or on stop."""

    def __init__(self, loco, period: float = 0.1, limits=BASE_VEL_MAX) -> None:
        self.loco = loco
        self.period = period
        self.limits = np.asarray(limits, dtype=float)
        self._cond = threading.Condition()
        self._cmd = None
        self._changed = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._last_sent = None
        self.calls = 0
        self.failures = 0
        self.latency_max = 0.0
        self._latency_sum = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="base-commander", daemon=True)
        self._thread.start()

    def command(self, base) -> None:
        if base is not None:
            base = tuple(float(v) for v in np.clip(base, -self.limits, self.limits))
        with self._cond:
            self._cmd = base
            self._changed = True
            self._cond.notify()

    def stop(self, join: float = 3.0) -> None:
        with self._cond:
            self._cmd = None
            self._stop = True
            self._changed = True
            self._cond.notify()
        t = self._thread
        if t is not None:
            t.join(timeout=join)     # the robot's own 1 s dead-man is the backstop

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._changed or self._stop, timeout=self.period)
                cmd, stop = self._cmd, self._stop
                self._changed = False
            if stop:
                break
            if cmd is not None:
                self._send(lambda: self.loco.Move(*cmd))
                self._last_sent = cmd
            elif self._last_sent is not None:
                self._send(self.loco.StopMove)
                self._last_sent = None
        if self._last_sent is not None:
            self._send(self.loco.StopMove)
            self._last_sent = None

    def _send(self, fn) -> None:
        t0 = time.monotonic()
        try:
            code = fn()
        except Exception as e:           # a failed RPC must not kill the commander
            code = e
        dt = time.monotonic() - t0
        self.calls += 1
        self._latency_sum += dt
        self.latency_max = max(self.latency_max, dt)
        if code not in (0, None):
            self.failures += 1
            if self.failures <= 3:
                print(f"base: command failed ({code})")

    def summary(self) -> str:
        if not self.calls:
            return "base: no commands sent"
        return (f"base: {self.calls} command(s), {self.failures} failure(s), RPC latency mean "
                f"{self._latency_sum / self.calls * 1e3:.0f} ms max {self.latency_max * 1e3:.0f} ms")


class RobotEnv(Env):
    name = "robot"
    monitor = None        # created in setup(); report-only

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("robot")
        g.add_argument("--iface", default=None,
                       help="network interface or IP of the robot (required for --env robot)")
        g.add_argument("--mode", choices=sorted(MODES), default=None,
                       help="required for --env robot. gantry: Damp->FSM4->FSM200, run, release, Damp. "
                            "standing: remember FSM, ->FSM200, run, release, ->remembered FSM, no Damp")
        g.add_argument("--countdown", type=int, default=3,
                       help="seconds to count down before taking over the arms")
        g.add_argument("--walk", action="store_true",
                       help="let the policy drive the base with LocoClient.Move (FSM 200), clamped to "
                            "BASE_VEL_MAX. PRE-FLIGHT: on the floor or hoisted with feet touching, ~2 m "
                            "clear all round, no tether to snag, spotter on the remote with L2+B")
        g.add_argument("--camera-ip", default=os.environ.get("UNITREE_ROBOT_IP"),
                       help="robot IP for the head camera stream (default: $UNITREE_ROBOT_IP)")
        g.add_argument("--camera-timeout", type=float, default=15.0,
                       help="seconds to wait for the first camera frame (default 15)")

    def setup(self) -> None:
        if not self.args.iface:
            raise SystemExit("--env robot requires --iface <network_interface_or_ip>")
        if self.args.mode not in MODES:
            raise SystemExit("--env robot requires --mode gantry|standing (no default: "
                             "gantry damps the robot at the end, standing does not)")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        self.camera = None
        if self.use_camera:
            if not self.args.camera_ip:
                raise SystemExit("this policy uses the camera: pass --camera-ip or set "
                                 "$UNITREE_ROBOT_IP")
            key = os.environ.get("UNITREE_AES_128_KEY")
            if not key:
                print("warning: $UNITREE_AES_128_KEY not set; firmware >= 1.5.1 needs it "
                      "(see unitree-fetch-aes-key)")
            print(f"Connecting to the head camera at {self.args.camera_ip}")
            self.camera = WebRTCCamera(self.args.camera_ip, key, timeout=self.args.camera_timeout)
            self.camera.start()          # before any FSM change
            print(f"camera: first frame {self.camera.latest().image.shape}")

        ChannelFactoryInitialize(0, self.args.iface)
        self.loco = LocoClient()
        self.loco.SetTimeout(10.0)
        self.loco.Init()

        if self.args.mode == "gantry":
            print("Damp");                     self.loco.Damp();        time.sleep(1.0)
            print("FSM 4 (locked stand)");     self.loco.SetFsmId(FSM_LOCKED_STAND); time.sleep(7.0)
        else:  # standing
            self.initial_fsm = self._fsm()
            print(f"fsm: {self.initial_fsm} (will return here after the run)")
        print("FSM 200 (main operation)");     self.loco.SetFsmId(FSM_MAIN); time.sleep(3.0)
        print("fsm:", self._fsm(), "(robot should be balancing on its own now)")
        for s in range(self.args.countdown, 0, -1):
            print(f"Taking over arms in {s}...")
            time.sleep(1.0)
        self.arm = ArmSdk()
        # report-only: flag any joint that left its bounds, but never stop a live run
        self.monitor = JointMonitor(strict=False, gate_targets=False, gate_velocity=False,
                                    gate_base=False)
        self.base = None
        if self.args.walk:
            print("WALKING ENABLED: the policy may drive the base (Move, <= "
                  f"{BASE_VEL_MAX[0]} m/s). Clear floor, spotter ready."
                  + (" Gantry mode: mind the tether." if self.args.mode == "gantry" else ""))
            self.base = BaseCommander(self.loco)
            self.base.start()

    def _fsm(self):
        code, fsm = self.loco.GetFsmId()
        if code != 0:
            raise RuntimeError(f"GetFsmId failed with code {code}")
        return fsm

    def teardown(self) -> None:
        # Whatever happened (normal end, Ctrl-C, exception): release the arms,
        # then Damp (gantry) or return to the FSM the robot was in (standing).
        # The camera is closed last; it never gates the arm release.
        with shield_sigint("interrupt ignored: finishing the hand-over to the onboard controller"):
            try:
                self._teardown_robot()
            finally:
                camera = getattr(self, "camera", None)
                if camera is not None:
                    camera.stop()

    def _teardown_robot(self) -> None:
        base = getattr(self, "base", None)
        if base is not None:
            print("Stopping the base")
            base.stop()                  # StopMove before the arms are released
        arm = getattr(self, "arm", None)
        if arm is not None:
            print("Releasing arms")
            arm.release(1.0)
        loco = getattr(self, "loco", None)
        if loco is None:
            return
        if self.args.mode == "gantry":
            print("Damp")
            loco.Damp()
            time.sleep(1.0)
        else:  # standing
            initial = getattr(self, "initial_fsm", None)
            if initial is not None and initial != FSM_MAIN:
                print(f"FSM {initial} (restoring initial state)")
                loco.SetFsmId(initial)
                time.sleep(3.0)
        print("fsm:", self._fsm(), " Done.")

    def reset(self) -> np.ndarray:
        print("Releasing any stale arm_sdk state")
        self.arm.release(1.0)
        q0 = self.arm.fresh_state()
        print("Got fresh LowState. Taking over arms.")
        self.overruns = 0
        self._wall = time.time()
        return q0

    def frame(self):
        return None if self.camera is None else self.camera.latest()

    @property
    def can_walk(self) -> bool:
        return bool(self.args.walk)

    @property
    def has_loco(self) -> bool:
        return True                    # LocoClient is always up on the robot

    def _motor(self, attr: str):
        arm = getattr(self, "arm", None)
        if arm is None or arm.state is None:
            return None
        return np.array([getattr(m, attr) for m in arm.state.motor_state[:NUM_JOINTS]], dtype=float)

    def joint_vel(self):
        return self._motor("dq")

    def joint_torque(self):
        return self._motor("tau_est")

    def step(self, action: Action) -> np.ndarray:
        bad = [j for j in action.joints if j not in UPPER_BODY]
        if bad:
            raise RuntimeError(f"arm_sdk can only command waist+arms, policy tried {bad}")
        base = getattr(self, "base", None)
        if action.base is not None and base is None:
            raise RuntimeError("the policy commands the base; pass --walk (read its pre-flight first)")
        if base is not None:
            base.command(action.base)
        if action.command is not None:
            name, kw = action.command
            if name not in LOCO_METHODS:
                raise RuntimeError(f"onboard call {name!r} is not allowed (allowed: {sorted(LOCO_METHODS)})")
            print(f"LocoClient.{name}({kw})")
            code = getattr(self.loco, name)(**kw)
            self.last_command = {"name": name, "args": dict(kw), "code": code}
            if code not in (0, None):
                print(f"warning: LocoClient.{name} returned {code}")
        self.arm.send(action)
        self._wall += CONTROL_DT
        lag = self._wall - time.time()
        if lag > 0:
            time.sleep(lag)
        elif lag < -CONTROL_DT:
            self.overruns += 1          # the policy step took longer than a tick
        q = np.array([m.q for m in self.arm.state.motor_state[:NUM_JOINTS]])
        if self.monitor is not None:
            self.monitor.observe(q)
        return q

    def report(self) -> bool:
        monitor = getattr(self, "monitor", None)
        if monitor is not None and monitor.ticks:
            monitor.report("robot")
        base = getattr(self, "base", None)
        if base is not None:
            print(base.summary())
        overruns = getattr(self, "overruns", 0)
        if overruns:
            print(f"robot: {overruns} tick overrun(s) > {CONTROL_DT * 1e3:.0f} ms; "
                  f"the policy step is too slow for 50 Hz")
        else:
            print("robot: no tick overruns")
        return True
