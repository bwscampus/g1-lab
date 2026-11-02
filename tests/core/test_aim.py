"""g1/core/aim.py against the twin's own kinematics: the chains in config are the URDF's, so the
arm's line, the camera and the solve agree with MuJoCo to a fraction of a degree."""
import math

import numpy as np
import pytest

from g1.core import aim
from g1.core.config import WAIST
from g1.tools.arms import ELBOW_STRAIGHT, POINT_ROLL, R_ARM

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def twin():
    from g1.twin.core import Twin
    return Twin()


def pose(twin, waist, arm):
    """Set the joints, run the kinematics: world points into the IMU's frame, and the IMU (the pelvis quat)."""
    d = twin.data
    q = d.qpos[7:7 + 29]
    q[WAIST], q[R_ARM] = waist, arm
    mujoco.mj_kinematics(twin.model, d)
    mujoco.mj_camlight(twin.model, d)                     # cameras are placed here, not in mj_kinematics
    to_feet = lambda p: np.asarray(p) - d.xpos[1]            # the IMU's frame: the world's axes at the pelvis
    return to_feet, np.eye(3), d.qpos[3:7].copy()


def body(twin, name):
    return twin.data.xpos[mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_BODY, name)]


CASES = [((0.0, 0.0, 0.0), (-1.2, POINT_ROLL, 0, ELBOW_STRAIGHT, 0, 0, 0)),
         ((-0.6, 0.05, 0.1), (-0.8, POINT_ROLL, 0, ELBOW_STRAIGHT, 0, 0, 0)),
         ((0.9, -0.05, -0.1), (-1.5, -0.3, 0.2, ELBOW_STRAIGHT, 0, 0, 0))]


@pytest.mark.parametrize("waist, arm", CASES)
def test_the_arm_line_matches_the_twin(twin, waist, arm):
    to_feet, _, imu = pose(twin, waist, arm)
    shoulder, d = aim.arm_ray(arm, waist, imu)
    s, w = to_feet(body(twin, "right_shoulder_pitch_link")), to_feet(body(twin, "right_wrist_yaw_link"))
    assert shoulder == pytest.approx(s, abs=2e-3)
    assert math.degrees(math.acos(np.clip(np.dot(d, (w - s) / np.linalg.norm(w - s)), -1, 1))) < 0.2


@pytest.mark.parametrize("waist, arm", CASES)
def test_the_camera_matches_the_twin(twin, waist, arm):
    from g1.twin.model import HEAD_CAMERA
    to_feet, unhead, imu = pose(twin, waist, arm)
    cid = mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA)
    R, c = aim.camera(waist, imu)
    assert c == pytest.approx(to_feet(twin.data.cam_xpos[cid]), abs=2e-3)
    view = unhead @ -twin.data.cam_xmat[cid].reshape(3, 3)[:, 2]        # MuJoCo cameras look along -z
    assert np.dot(R[:, 0], view) > math.cos(math.radians(0.2))
    p = aim.pixel_point(400, 100, 640, 480, 1.2, waist, imu)
    assert aim.project(p, 640, 480, waist, imu) == pytest.approx((400, 100), abs=1e-6)


def test_the_solve_puts_the_arm_line_through_the_target(twin):
    target = np.array([0.6, -0.5, 0.0])                       # ahead, right, about table height
    rest = (-1.2, POINT_ROLL, 0, ELBOW_STRAIGHT, 0, 0, 0)
    _, _, imu = pose(twin, (0.0, 0.0, 0.0), rest)               # the pelvis tilt the solve allows for
    yaw, pitch = aim.solve_point(target, rest, 0.0, 0.0, imu)
    arm = (pitch, *rest[1:])
    to_feet, _, imu = pose(twin, (yaw, 0.0, 0.0), arm)
    s, w = to_feet(body(twin, "right_shoulder_pitch_link")), to_feet(body(twin, "right_wrist_yaw_link"))
    to_t = target - s
    err = math.degrees(math.acos(np.dot(w - s, to_t) / np.linalg.norm(w - s) / np.linalg.norm(to_t)))
    assert err < 0.2 and aim.miss(target, arm, (yaw, 0, 0), imu) == pytest.approx((0, 0), abs=0.01)
