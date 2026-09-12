import time
import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
# from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


# Joint indices (29-DoF G1)
WAIST = [12, 13, 14]
L_ARM = [15, 16, 17, 18, 19, 20, 21]   # shoulder P/R/Y, elbow, wrist R/P/Y
R_ARM = [22, 23, 24, 25, 26, 27, 28]
ARM_JOINTS = L_ARM + R_ARM
CTRL_JOINTS = WAIST + ARM_JOINTS
WEIGHT_IDX = 29                        # motor_cmd[29].q carries the arm-sdk weight

KP, KD = 60.0, 1.5
DT = 0.02 

class ArmSdk:
    def __init__(self):
        self.state = None
        self.crc = CRC()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        # Make sure every joint we don't control has zero gains, so a stale buffer
        # can never command the legs or anything else.
        for j in range(29):
            mc = self.cmd.motor_cmd[j]
            mc.q = 0.0; mc.dq = 0.0; mc.tau = 0.0; mc.kp = 0.0; mc.kd = 0.0

    def _on_state(self, msg):
        self.state = msg

    def fresh_state(self, settle=0.5):
        """Wait for a LowState, then keep reading for `settle` seconds and return the latest."""
        while self.state is None:
            time.sleep(0.05)
        time.sleep(settle)
        return np.array([m.q for m in self.state.motor_state[:29]])

    def release(self, duration=1.0):
        """Publish weight=0 with zero gains so ai_sport drops any stale arm_sdk takeover."""
        for j in CTRL_JOINTS:
            mc = self.cmd.motor_cmd[j]
            mc.q = 0.0; mc.dq = 0.0; mc.tau = 0.0; mc.kp = 0.0; mc.kd = 0.0
        self.cmd.motor_cmd[WEIGHT_IDX].q = 0.0
        self.cmd.crc = self.crc.Crc(self.cmd)
        for _ in range(int(duration / DT)):
            self.pub.Write(self.cmd)
            time.sleep(DT)

    def send(self, targets, weight):
        """targets: dict joint_idx -> q. weight: 0..1 arm-sdk blend."""
        for j in CTRL_JOINTS:
            mc = self.cmd.motor_cmd[j]
            mc.q = float(targets[j])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = KP
            mc.kd = KD
        self.cmd.motor_cmd[WEIGHT_IDX].q = float(weight)
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)