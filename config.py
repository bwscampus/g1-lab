"""Static facts about the 29-DoF Unitree G1.

Joint indices follow the DDS order used by ``LowCmd_.motor_cmd`` /
``LowState_.motor_state`` (see unitree_mujoco's g1_joint_index_dds.md), which
is also the joint/actuator order of the MuJoCo Menagerie ``unitree_g1`` model.
The limits below were dumped from that model; ``tests/test_config.py`` asserts
they still match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_JOINTS = 29
CONTROL_DT = 0.02          # 50 Hz policy tick
ARM_SDK_WEIGHT_IDX = 29    # motor_cmd[29].q carries the arm-sdk blend weight on the robot


@dataclass(frozen=True)
class Joint:
    index: int
    name: str
    lo: float
    hi: float


# fmt: off
JOINTS: tuple[Joint, ...] = (
    Joint(0,  "left_hip_pitch",       -2.5307,   2.8798),
    Joint(1,  "left_hip_roll",        -0.5236,   2.9671),
    Joint(2,  "left_hip_yaw",         -2.7576,   2.7576),
    Joint(3,  "left_knee",            -0.087267, 2.8798),
    Joint(4,  "left_ankle_pitch",     -0.87267,  0.5236),
    Joint(5,  "left_ankle_roll",      -0.2618,   0.2618),
    Joint(6,  "right_hip_pitch",      -2.5307,   2.8798),
    Joint(7,  "right_hip_roll",       -2.9671,   0.5236),
    Joint(8,  "right_hip_yaw",        -2.7576,   2.7576),
    Joint(9,  "right_knee",           -0.087267, 2.8798),
    Joint(10, "right_ankle_pitch",    -0.87267,  0.5236),
    Joint(11, "right_ankle_roll",     -0.2618,   0.2618),
    Joint(12, "waist_yaw",            -2.618,    2.618),
    Joint(13, "waist_roll",           -0.52,     0.52),
    Joint(14, "waist_pitch",          -0.52,     0.52),
    Joint(15, "left_shoulder_pitch",  -3.0892,   2.6704),
    Joint(16, "left_shoulder_roll",   -1.5882,   2.2515),
    Joint(17, "left_shoulder_yaw",    -2.618,    2.618),
    Joint(18, "left_elbow",           -1.0472,   2.0944),
    Joint(19, "left_wrist_roll",      -1.97222,  1.97222),
    Joint(20, "left_wrist_pitch",     -1.61443,  1.61443),
    Joint(21, "left_wrist_yaw",       -1.61443,  1.61443),
    Joint(22, "right_shoulder_pitch", -3.0892,   2.6704),
    Joint(23, "right_shoulder_roll",  -2.2515,   1.5882),
    Joint(24, "right_shoulder_yaw",   -2.618,    2.618),
    Joint(25, "right_elbow",          -1.0472,   2.0944),
    Joint(26, "right_wrist_roll",     -1.97222,  1.97222),
    Joint(27, "right_wrist_pitch",    -1.61443,  1.61443),
    Joint(28, "right_wrist_yaw",      -1.61443,  1.61443),
)
# fmt: on
assert len(JOINTS) == NUM_JOINTS and all(j.index == i for i, j in enumerate(JOINTS))

JOINT_NAMES: tuple[str, ...] = tuple(j.name for j in JOINTS)
JOINT_LO = np.array([j.lo for j in JOINTS])
JOINT_HI = np.array([j.hi for j in JOINTS])

# Joint groups.
LEFT_LEG = list(range(0, 6))
RIGHT_LEG = list(range(6, 12))
LEGS = LEFT_LEG + RIGHT_LEG
WAIST = [12, 13, 14]
LEFT_ARM = list(range(15, 22))   # shoulder P/R/Y, elbow, wrist R/P/Y
RIGHT_ARM = list(range(22, 29))
ARMS = LEFT_ARM + RIGHT_ARM
UPPER_BODY = WAIST + ARMS        # everything arm_sdk may command

# Default "stand" pose (Menagerie keyframe): arms hanging, elbows bent.
STAND_Q = np.zeros(NUM_JOINTS)
STAND_Q[[15, 16, 18]] = [0.2, 0.2, 1.28]
STAND_Q[[22, 23, 25]] = [0.2, -0.2, 1.28]


def joint_index(name: str) -> int:
    return JOINT_NAMES.index(name)


# Head camera (Intel RealSense D435 colour stream). Mount pose is the URDF's
# d435_joint origin in the torso_link frame: forward/up offset, pitched down.
HEAD_CAMERA_POS = (0.0576235, 0.01753, 0.42987)   # m, in torso_link
HEAD_CAMERA_PITCH = 0.8307767                     # rad, looking down
HEAD_CAMERA_FOVY = 58.0                           # degrees, vertical
HEAD_CAMERA_SIZE = (480, 640)                     # sim render height, width
