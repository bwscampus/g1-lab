"""Stage 3: live deployment through unitree_sdk2py.

High-level bring-up with LocoClient (Damp -> FSM 4 locked stand -> FSM 200 main
operation, ai_sport balancing), then upper-body targets are published on
``rt/arm_sdk`` at 50 Hz with the blend weight in ``motor_cmd[29].q``. Same
handover protocol as arm_movements/tpose.py: release any stale takeover first,
read a fresh LowState, and always release + damp in teardown.

Pre-flight:
  * robot standing on the floor, NOT in debug mode
  * clear space around the arms
  * someone on the remote with L2+B ready

    python run.py --env robot --policy tpose --iface eth0
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from config import ARM_SDK_WEIGHT_IDX, CONTROL_DT, NUM_JOINTS, UPPER_BODY
from policy import Action
from envs.base import Env


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


class RobotEnv(Env):
    name = "robot"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("robot")
        g.add_argument("--iface", default=None,
                       help="network interface or IP of the robot (required for --env robot)")
        g.add_argument("--skip-bringup", action="store_true",
                       help="assume the robot is already in FSM 200; skip Damp/FSM4/FSM200")
        g.add_argument("--countdown", type=int, default=3,
                       help="seconds to count down before taking over the arms")

    def setup(self) -> None:
        if not self.args.iface:
            raise SystemExit("--env robot requires --iface <network_interface_or_ip>")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        ChannelFactoryInitialize(0, self.args.iface)
        self.loco = LocoClient()
        self.loco.SetTimeout(10.0)
        self.loco.Init()

        if not self.args.skip_bringup:
            print("Damp");                self.loco.Damp();        time.sleep(1.0)
            print("FSM 4 (locked stand)"); self.loco.SetFsmId(4);   time.sleep(7.0)
            print("FSM 200 (main operation)"); self.loco.SetFsmId(200); time.sleep(3.0)
        print("fsm:", self.loco.GetFsmId(), "(robot should be balancing on its own now)")
        for s in range(self.args.countdown, 0, -1):
            print(f"Taking over arms in {s}...")
            time.sleep(1.0)
        self.arm = ArmSdk()

    def teardown(self) -> None:
        # Whatever happened (normal end, Ctrl-C, exception): release, then damp.
        arm = getattr(self, "arm", None)
        if arm is not None:
            print("Releasing arms")
            arm.release(1.0)
        loco = getattr(self, "loco", None)
        if loco is not None:
            print("Damp")
            loco.Damp()
            time.sleep(1.0)
            print("fsm:", loco.GetFsmId(), " Done.")

    def reset(self) -> np.ndarray:
        print("Releasing any stale arm_sdk state")
        self.arm.release(1.0)
        q0 = self.arm.fresh_state()
        print("Got fresh LowState. Taking over arms.")
        self._wall = time.time()
        return q0

    def step(self, action: Action) -> np.ndarray:
        bad = [j for j in action.joints if j not in UPPER_BODY]
        if bad:
            raise RuntimeError(f"arm_sdk can only command waist+arms, policy tried {bad}")
        self.arm.send(action)
        self._wall += CONTROL_DT
        lag = self._wall - time.time()
        if lag > 0:
            time.sleep(lag)
        return np.array([m.q for m in self.arm.state.motor_state[:NUM_JOINTS]])
