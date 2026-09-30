"""Stage 3: live deployment through unitree_sdk2py.

High-level control with LocoClient, then upper-body targets published on
``rt/arm_sdk`` at 50 Hz with the blend weight in ``motor_cmd[29].q``. Handover
protocol: release any stale takeover first, read a fresh LowState, and always
release the arms in teardown (normal end, Ctrl-C, or exception).

Two modes, ``--mode`` (required, no default). FSM 200 is Unitree's *main
operation* mode; entering it is what gives this process the arms (arm_sdk) and
the base (LocoClient.Move) — "control mode" in the messages below.

  gantry    full bring-up and shutdown, for a robot hanging in a gantry:
            Damp -> FSM 4 (locked stand) -> FSM 200 (control mode) -> run
            -> release arms -> Damp
  standing  robot already standing under its own controller (any FSM):
            remember the current FSM -> FSM 200 (skipped when already there)
            -> run -> release arms -> back to the remembered FSM. Never damps.

The arms are taken where they are (a 2 s weight ramp, nothing moves) and
released where the last tool left them; no move to a neutral pose, no
countdown.

Walking (``--walk``): a tool's ``Action.base`` velocity is sent as
``LocoClient.Move(vx, vy, vyaw)`` from a 10 Hz commander thread (the RPC blocks,
so it never runs on the tick thread), clamped to ``config.BASE_VEL_MAX``. Move
is a 1 s dead-man command on the robot; the commander re-sends while a command
is set, sends ``StopMove`` when it clears, and stops before the arms are
released. Move works in FSM 200, so no extra FSM transition is needed.
Pre-flight for walking: standing on the floor or hoisted with feet touching,
~2 m clear in every direction, no tether to snag, spotter on the remote (L2+B).

Camera: for the agent (which decides on it), the head camera is streamed over WebRTC
(``--camera-ip``, ``$UNITREE_AES_128_KEY``) and connected *before* any FSM
transition, so a bad camera link fails before the robot is touched. The robot
accepts one WebRTC client: disconnect the Unitree app first.

Pre-flight:
  * robot NOT in debug mode
  * clear space around the arms
  * someone on the remote with L2+B ready

    g1 run --env robot --tools tpose --iface eth0 --mode standing
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from typing import Optional

import numpy as np

from g1.camera import WebRTCCamera
from g1.core.action import Action
from g1.core.config import (ARM_SDK_WEIGHT_IDX, AUDIO_METHODS, BASE_VEL_MAX, CONTROL_DT, LOCO_METHODS, NUM_JOINTS,
                            UPPER_BODY)
from g1.envs import sdk
from g1.envs.base import Env, shield_sigint
from g1.envs.monitor import JointMonitor


FSM_LOCKED_STAND = 4
FSM_MAIN = 200
MODES = ("gantry", "standing")


class Health:
    """LowState arrival times, written on the DDS thread, read on the tick thread."""

    def __init__(self) -> None:
        self.count = 0
        self.last = 0.0            # time.monotonic() of the newest LowState
        self._window: list[float] = []

    def tick(self) -> None:
        now = time.monotonic()
        self.count += 1
        self.last = now
        self._window.append(now)
        if len(self._window) > 1500:
            del self._window[:500]

    def rate(self, seconds: float = 1.0) -> float:
        now = time.monotonic()
        return sum(1 for t in self._window if t > now - seconds) / seconds

    def age_ms(self) -> float:
        return math.inf if not self.last else (time.monotonic() - self.last) * 1e3


class ArmSdk:
    """Thin publisher for rt/arm_sdk plus a LowState subscriber."""

    def __init__(self, health: Optional[Health] = None) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.state = None
        self.health = health or Health()
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
        self.health.tick()

    def fresh_state(self, settle: float = 0.5, timeout: float = 5.0, iface: str = "") -> np.ndarray:
        t0 = time.time()
        while self.state is None:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"no LowState received in {timeout:.0f}s on {iface or 'the interface'}: "
                                   "the DDS link (interface, subnet, the robot's services); run g1 status")
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
        self.last = None                  # (name, code, latency_s) of the newest RPC

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
                self._send(lambda: self.loco.Move(*cmd), "Move")
                self._last_sent = cmd
            elif self._last_sent is not None:
                self._send(self.loco.StopMove, "StopMove")
                self._last_sent = None
        if self._last_sent is not None:
            self._send(self.loco.StopMove, "StopMove")
            self._last_sent = None

    def _send(self, fn, name: str = "Move") -> None:
        t0 = time.monotonic()
        try:
            code = fn()
        except Exception as e:           # a failed RPC must not kill the commander
            code = e
        dt = time.monotonic() - t0
        self.calls += 1
        self._latency_sum += dt
        self.latency_max = max(self.latency_max, dt)
        self.last = (name, code, dt)
        if code not in (0, None):
            self.failures += 1
            if self.failures <= 3:
                print(f"base: command failed ({sdk.explain(code)})")

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
        g.add_argument("--iface", default=os.environ.get("UNITREE_IFACE"),
                       help="network interface (e.g. en7) or IP of the robot for DDS (default: $UNITREE_IFACE, "
                            "e.g. from .env; required for --env robot)")
        g.add_argument("--mode", choices=sorted(MODES), default=None,
                       help="required for --env robot. gantry: Damp->FSM4->FSM200, run, release, Damp. "
                            "standing: remember FSM, ->FSM200, run, release, ->remembered FSM, no Damp")
        g.add_argument("--walk", action="store_true",
                       help="let a tool drive the base with LocoClient.Move (FSM 200), clamped to "
                            "BASE_VEL_MAX. PRE-FLIGHT: on the floor or hoisted with feet touching, ~2 m "
                            "clear all round, no tether to snag, spotter on the remote with L2+B")
        g.add_argument("--camera-ip", default=os.environ.get("UNITREE_ROBOT_IP"),
                       help="robot IP for the head camera stream (default: $UNITREE_ROBOT_IP)")
        g.add_argument("--camera-timeout", type=float, default=15.0,
                       help="seconds to wait for the first camera frame (default 15)")
        g.add_argument("--volume", type=int, default=None, metavar="0-100",
                       help="set the speaker volume before the run (the say tool uses the onboard TTS)")

    def setup(self) -> None:
        if not self.args.iface:
            raise SystemExit("--env robot requires --iface <network_interface_or_ip> (or UNITREE_IFACE in .env)")
        if self.args.mode not in MODES:
            raise SystemExit("--env robot requires --mode gantry|standing (no default: "
                             "gantry damps the robot at the end, standing does not)")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        self.camera = None
        if self.use_camera:
            if not self.args.camera_ip:
                raise SystemExit("this run uses the camera: pass --camera-ip or set "
                                 "$UNITREE_ROBOT_IP")
            key = os.environ.get("UNITREE_AES_128_KEY")
            if not key:
                print("warning: $UNITREE_AES_128_KEY not set; firmware >= 1.5.1 needs it "
                      "(see unitree-fetch-aes-key)")
            print(f"camera: connecting to {self.args.camera_ip}")
            t0 = time.monotonic()
            self.camera = WebRTCCamera(self.args.camera_ip, key, timeout=self.args.camera_timeout)
            self.camera.start()          # before any FSM change
            print(f"camera: first frame {self.camera.latest().image.shape} after {time.monotonic() - t0:.1f}s")

        self._say(f"dds: ChannelFactoryInitialize(0, {self.args.iface!r})")
        ChannelFactoryInitialize(0, self.args.iface)
        self.health = Health()
        self.loco = LocoClient()
        self.loco.SetTimeout(10.0)
        self.loco.Init()
        self.audio = None
        self.audio_calls = 0
        self._audio_warned = False
        try:
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            self.audio = AudioClient()
            self.audio.SetTimeout(10.0)
            self.audio.Init()
            if self.args.volume is not None:
                self._call(self.audio, "SetVolume", max(0, min(100, int(self.args.volume))))
            code, volume = self._call(self.audio, "GetVolume")
            if code == 0:
                print(f"audio: volume {volume}")
            else:
                raise RuntimeError(sdk.explain(code))
        except Exception as e:                  # a robot without the voice service still runs
            print(f"warning: audio service unavailable ({e}); say will be skipped")
            self.audio = None

        if self.args.mode == "gantry":
            self._require(self._call(self.loco, "Damp")[0], "Damp")
            time.sleep(1.0)
            self._require(self._call(self.loco, "SetFsmId", FSM_LOCKED_STAND)[0], "SetFsmId(4)")
            print(f"fsm: {sdk.fsm_name(FSM_LOCKED_STAND)}")
            time.sleep(7.0)
            self.initial_fsm = None
        else:  # standing
            self.initial_fsm = self._fsm()
            self._say(f"fsm at start: {sdk.fsm_name(self.initial_fsm)}")
            if self.initial_fsm not in sdk.STANDING_FSMS:
                raise SystemExit(f"--mode standing needs a robot standing under its own controller, but it "
                                 f"reports FSM {sdk.fsm_name(self.initial_fsm)}; stand it up first (or use --mode gantry)")
        if self.initial_fsm == FSM_MAIN:
            print(f"already in control mode (FSM {FSM_MAIN}, main operation)")
        else:
            print(f"entering control mode (FSM {FSM_MAIN}, main operation)"
                  + (f"; will return to FSM {self.initial_fsm} after the run" if self.initial_fsm is not None else ""))
            self._require(self._call(self.loco, "SetFsmId", FSM_MAIN)[0], f"SetFsmId({FSM_MAIN})")
            t0 = time.monotonic()
            while True:
                fsm = self._fsm()
                self._say(f"fsm: {sdk.fsm_name(fsm)} after {time.monotonic() - t0:.1f}s")
                if fsm == FSM_MAIN:
                    break
                if time.monotonic() - t0 > 3.0:
                    raise SystemExit(f"the robot did not enter FSM {FSM_MAIN} within 3 s (reports "
                                     f"{sdk.fsm_name(fsm)}): another controller may own the FSM, or it is in "
                                     "a protection state; check the app and the remote, run g1 status")
                time.sleep(0.2)
        self.arm = ArmSdk(self.health)
        # report-only: flag any joint that left its bounds, but never stop a live run
        self.monitor = JointMonitor(strict=False, gate_targets=False, gate_velocity=False,
                                    gate_base=False)
        self.base = None
        if self.args.walk:
            print("WALKING ENABLED: the program may drive the base (Move, <= "
                  f"{BASE_VEL_MAX[0]} m/s). Clear floor, spotter ready."
                  + (" Gantry mode: mind the tether." if self.args.mode == "gantry" else ""))
            self.base = BaseCommander(self.loco)
            self.base.start()

    # -- the SDK, checked ---------------------------------------------------------------------
    def _call(self, client, name: str, *args, label: Optional[str] = None, **kw):
        if label is None:
            label = ("LocoClient" if client is getattr(self, "loco", None) else
                     "AudioClient" if client is getattr(self, "audio", None) else type(client).__name__)
        return sdk.call(client, name, *args, log=self.sdk_log, verbose=self.verbose, label=label, **kw)

    def _require(self, code: int, what: str) -> None:
        """A failed call during bring-up stops before the arms are touched."""
        if code == 0:
            return
        hint = ("no reply from the robot: check the interface and the DDS link (g1 status --iface ...)"
                if sdk.is_transport(code) else
                "the robot refused: the app or the remote may own the FSM, or it is in a protection state "
                "(g1 status --iface ... shows what it reports)")
        raise SystemExit(f"{what} failed with {sdk.explain(code)}; {hint}")

    def _fsm(self):
        code, fsm = self._call(self.loco, "GetFsmId")
        if code != 0:
            self._require(code, "GetFsmId")
        return fsm

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text)

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
        """Nothing here raises: the release must always finish."""
        base = getattr(self, "base", None)
        if base is not None:
            print("Stopping the base")
            base.stop()                  # StopMove before the arms are released
        arm = getattr(self, "arm", None)
        if arm is not None:
            print("leaving control mode: releasing the arms")
            arm.release(1.0)
        loco = getattr(self, "loco", None)
        if loco is None:
            return
        if self.args.mode == "gantry":
            code, _ = self._call(loco, "Damp")
            print("Damp" if code == 0 else f"warning: Damp failed with {sdk.explain(code)}")
            time.sleep(1.0)
        else:  # standing
            initial = getattr(self, "initial_fsm", None)
            if initial is not None and initial != FSM_MAIN:
                if initial in sdk.STANDING_FSMS:
                    code, _ = self._call(loco, "SetFsmId", initial)
                    time.sleep(3.0)
                    print(f"FSM {sdk.fsm_name(initial)} restored" if code == 0 else
                          f"warning: could not restore FSM {initial}: {sdk.explain(code)}")
                else:
                    print(f"staying in control mode (FSM {FSM_MAIN}); the run started in FSM "
                          f"{sdk.fsm_name(initial)}, which is not a standing state")
        code, fsm = self._call(loco, "GetFsmId")
        print(f"fsm: {sdk.fsm_name(fsm) if code == 0 else sdk.explain(code)}; done")

    def reset(self) -> np.ndarray:
        self.arm.release(1.0)                 # drop any stale arm_sdk takeover
        q0 = self.arm.fresh_state(iface=self.args.iface)
        self._say(f"lowstate: first message, age {self.health.age_ms():.0f} ms, {self.health.rate():.0f}/s")
        self.overruns = 0
        self.ticks = 0
        self._wall = time.time()
        self._health_at = time.monotonic()
        return q0

    def frame(self):
        return None if self.camera is None else self.camera.latest()

    def source(self):
        return self.camera

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
            raise RuntimeError(f"arm_sdk can only command waist+arms, the program tried {bad}")
        base = getattr(self, "base", None)
        if action.base is not None and base is None:
            raise RuntimeError("the program commands the base; pass --walk (read its pre-flight first)")
        if base is not None:
            base.command(action.base)
        if action.command is not None:
            self._command(*action.command)
        self.arm.send(action)
        self._wall += CONTROL_DT
        lag = self._wall - time.time()
        if lag > 0:
            time.sleep(lag)
        elif lag < -CONTROL_DT:
            self.overruns += 1          # the program step took longer than a tick
        q = np.array([m.q for m in self.arm.state.motor_state[:NUM_JOINTS]])
        if self.monitor is not None:
            self.monitor.observe(q)
        self.ticks += 1
        if self.verbose and time.monotonic() - self._health_at >= 1.0:
            self._health_at = time.monotonic()
            print(self.health_line())
        return q

    def health_line(self) -> str:
        """One line a second in --verbose: the link, the tick, the base, the camera."""
        h = getattr(self, "health", None)
        parts = [f"t={self.ticks * CONTROL_DT:6.1f}"]
        if h is not None:
            parts.append(f"lowstate {h.rate():.0f}/s (age {h.age_ms():.0f} ms)")
        parts.append(f"ticks {self.ticks} overruns {getattr(self, 'overruns', 0)}")
        base = getattr(self, "base", None)
        if base is not None and base.last is not None:
            name, code, dt = base.last
            parts.append(f"base {name} {'ok' if code in (0, None) else sdk.explain(code)} {dt * 1e3:.0f} ms")
        cam = getattr(self, "camera", None)
        if cam is not None:
            f = cam.latest()
            parts.append(f"camera {cam.count} frames" + (f", age {(time.monotonic() - f.stamp) * 1e3:.0f} ms" if f else ""))
        return "  ".join(parts)

    def _command(self, name: str, kw: dict) -> None:
        """One onboard call, routed by its allow-list: LocoClient for gestures,
        AudioClient for speech. A failed speech never stops a live run."""
        if name in LOCO_METHODS:
            client, label = self.loco, "LocoClient"
        elif name in AUDIO_METHODS:
            client, label = getattr(self, "audio", None), "AudioClient"
            if client is None:
                if not self._audio_warned:
                    print("warning: audio unavailable; say skipped")
                    self._audio_warned = True
                return
            self.audio_calls += 1
        else:
            raise RuntimeError(f"onboard call {name!r} is not allowed (allowed: {sorted(LOCO_METHODS | AUDIO_METHODS)})")
        print(f"{label}.{name}({kw})")
        code, data = self._call(client, name, label=label, **kw)
        if code == -1 and label != "AudioClient":
            raise RuntimeError(f"{label}.{name} raised: {data}")
        self.last_command = {"name": name, "args": dict(kw), "code": code}
        if code != 0:
            print(f"warning: {label}.{name} returned {sdk.explain(code) if code != -1 else data}")

    @property
    def has_audio(self) -> bool:
        return getattr(self, "audio", None) is not None

    def report(self) -> bool:
        monitor = getattr(self, "monitor", None)
        if monitor is not None and monitor.ticks:
            monitor.report("robot")
        base = getattr(self, "base", None)
        if base is not None:
            print(base.summary())
        if getattr(self, "audio_calls", 0):
            print(f"audio: {self.audio_calls} call(s)")
        failed = [e for e in self.sdk_log if e["code"] != 0]
        if self.sdk_log:
            print(f"sdk: {len(self.sdk_log)} call(s), {len(failed)} failed"
                  + (f" (codes {sorted({e['code'] for e in failed})})" if failed else ""))
        overruns = getattr(self, "overruns", 0)
        if overruns:
            print(f"robot: {overruns} tick overrun(s) > {CONTROL_DT * 1e3:.0f} ms; "
                  f"the program step is too slow for 50 Hz")
        else:
            print("robot: no tick overruns")
        return True


# --------------------------------------------------------------------------
# g1 status: is it the link, or is the robot refusing?
# --------------------------------------------------------------------------

def probe(iface: str, seconds: float = 5.0, verbose: bool = False, factory=None, loco_cls=None,
          subscribe=None) -> dict:
    """Read-only: count LowState for ``seconds``, ask the FSM, and say which of
    the two failures it is. Never publishes to arm_sdk, never changes the FSM.
    The SDK pieces are injectable for tests."""
    if factory is None or loco_cls is None or subscribe is None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        factory = factory or (lambda: ChannelFactoryInitialize(0, iface))
        loco_cls = loco_cls or LocoClient

        def subscribe(on_state):
            sub = ChannelSubscriber("rt/lowstate", LowState_)
            sub.Init(on_state, 10)
            return sub
    log: list = []
    out: dict = {"iface": iface, "seconds": seconds}
    factory()
    health = Health()
    latest: dict = {}

    def on_state(msg):
        latest["msg"] = msg
        health.tick()

    subscribe(on_state)
    loco = loco_cls()
    loco.SetTimeout(min(seconds, 5.0))
    loco.Init()
    t0 = time.monotonic()
    code, fsm = sdk.call(loco, "GetFsmId", log=log, verbose=verbose)
    out["rpc"] = {"code": code, "fsm": fsm, "explain": sdk.explain(code), "elapsed_s": log[-1]["elapsed_s"]}
    while time.monotonic() - t0 < seconds:
        time.sleep(0.05)
    out["lowstate"] = {"count": health.count, "rate": health.count / seconds, "age_ms": health.age_ms()}
    msg = latest.get("msg")
    if msg is not None:
        try:
            out["lowstate"]["joint0_q"] = float(msg.motor_state[0].q)
            out["lowstate"]["battery_soc"] = int(msg.bms_state.soc)
        except Exception:
            pass
    link = health.count > 0
    if link and code == 0:
        verdict = f"link ok, FSM {sdk.fsm_name(fsm)}" + ("" if fsm in sdk.STANDING_FSMS else
                                                          " — not a standing state; --mode standing will refuse")
    elif not link and (code != 0):
        verdict = (f"no link: LowState absent in {seconds:.0f}s and RPC failed with {sdk.explain(code)} — "
                   f"check the interface ({iface}), the subnet (192.168.123.x) and that the robot's services are up")
    elif link and code != 0:
        verdict = (f"link ok ({health.count / seconds:.0f} LowState/s) but requests fail with {sdk.explain(code)}: "
                   "the service refuses — check the app, the remote's mode, and the robot's error state")
    else:
        verdict = f"RPC answers (FSM {sdk.fsm_name(fsm)}) but no LowState: the state topic is not reaching this host"
    out["verdict"] = verdict
    out["calls"] = log
    return out


def status_main(argv: list[str]) -> int:
    from g1.vlm import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(prog="g1 status", description="is the robot reachable, and what state is it in? "
                                "Read-only: no arm_sdk, no FSM change.")
    p.add_argument("--iface", default=os.environ.get("UNITREE_IFACE"),
                   help="network interface or IP of the robot for DDS (default: $UNITREE_IFACE, e.g. from .env)")
    p.add_argument("--seconds", type=float, default=5.0, help="how long to count LowState (default 5)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    if not args.iface:
        p.error("--iface is required (or UNITREE_IFACE in .env)")
    try:
        out = probe(args.iface, args.seconds, args.verbose)
    except ImportError as e:
        p.error(f"unitree_sdk2py is not installed ({e})")
    if args.json:
        print(json.dumps(out, indent=1, default=str))
        return 0
    ls = out["lowstate"]
    print(f"lowstate: {ls['count']} messages in {args.seconds:.0f}s = {ls['rate']:.0f}/s"
          + (f", newest {ls['age_ms']:.0f} ms old" if ls["count"] else "")
          + (f", joint0 q {ls['joint0_q']:+.3f}" if "joint0_q" in ls else "")
          + (f", battery {ls['battery_soc']}%" if "battery_soc" in ls else ""))
    r = out["rpc"]
    print(f"rpc:      GetFsmId -> {r['explain']} in {r['elapsed_s']:.3f}s"
          + (f", FSM {sdk.fsm_name(r['fsm'])}" if r["code"] == 0 else ""))
    print(f"verdict:  {out['verdict']}")
    return 0 if out["rpc"]["code"] == 0 and out["lowstate"]["count"] else 1
