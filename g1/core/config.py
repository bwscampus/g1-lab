"""Static facts about the 29-DoF Unitree G1.

Joint indices follow the DDS order used by ``LowCmd_.motor_cmd`` /
``LowState_.motor_state`` (see unitree_mujoco's g1_joint_index_dds.md), which
is also the motor order of the twin's model (unitree_mujoco's ``g1_29dof``).
The limits below were dumped from the Menagerie ``unitree_g1`` model and match
the twin's; ``tests/core/test_config.py`` asserts they still do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from g1.core import limits

NUM_JOINTS = 29
CONTROL_DT = limits.get("control_dt_s")          # the control tick (configs/limits.json)
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

# Default "stand" pose: arms hanging, elbows bent (the twin's onboard controller holds it).
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
# Where the right arm pivots: the URDF's right_shoulder_pitch_joint origin in the torso_link frame.
# A pointing arm's line starts here, 12 cm right of and 19 cm below the camera.
R_SHOULDER_POS = (0.0039563, -0.10021, 0.23778)   # m, in torso_link
R_SHOULDER_FROM_CAMERA = tuple(s - c for s, c in zip(R_SHOULDER_POS, HEAD_CAMERA_POS))   # x fwd, y left, z up
HEAD_CAMERA_SIZE = (int(limits.get("twin_camera_height_px")), int(limits.get("twin_camera_width_px")))   # sim render H, W
# horizontal field of view of the sim render, from the vertical one and the aspect
HEAD_CAMERA_FOV_H = 2 * math.degrees(math.atan(math.tan(math.radians(HEAD_CAMERA_FOVY / 2))
                                               * HEAD_CAMERA_SIZE[1] / HEAD_CAMERA_SIZE[0]))

# The only onboard LocoClient calls a tool may make (Action.command). Everything else
# (FSM switching, damp, torque, squat, stand height) is unreachable from a model reply by construction.
# Sit is the one FSM a tool may enter, and only as this name: RobotEnv sends it as SetFsmId(3)
# (LocoClient.Sit drops the return code), and the teardown stands the robot up (SetFsmId(801)).
LOCO_METHODS = frozenset({"WaveHand", "ShakeHand", "Sit"})
LOCO_FSM = {"Sit": 3}                     # allow-listed names that are an FSM switch
FSM_SIT, FSM_SIT_TO_STAND = 3, 801
# The only call a tool may make on the audio service ("voice"): text-to-speech. PlayStream
# (arbitrary bytes), SetVolume and LedControl are not reachable from a reply; volume is a flag.
AUDIO_METHODS = frozenset({"TtsMaker"})
# What AudioClient.PlayStream plays: raw 16 kHz mono 16-bit little-endian PCM, no header and no
# rate field (measured 2026-10-30; docs/running-on-the-robot.md). g1/core/sounds.py converts to it.
SPEAKER_RATE_HZ = 16000
SPEAKER_SAMPLE_BYTES = 2

# Base velocity limits for Action.base: |vx| m/s forward, |vy| m/s left, |vyaw| rad/s.
# Values and their provenance: configs/limits.json.
BASE_VEL_MAX = np.array([limits.get("base_vx_max"), limits.get("base_vy_max"), limits.get("base_vyaw_max")])
