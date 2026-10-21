"""locate_point: a pixel's direction from the head camera, checked against the twin's own camera."""
import math

import numpy as np
import pytest

from g1.core import limits
from g1.core.config import HEAD_CAMERA_PITCH, joint_index
from g1.tools import TOOLS
from g1.tools.control import bearing


def test_the_centre_the_edges_and_the_waist():
    c = bearing(320, 240, 640, 480)
    assert c["bearing_from_torso_deg"] == 0 and c["below_horizontal_deg"] == pytest.approx(math.degrees(HEAD_CAMERA_PITCH), abs=0.1)
    right, left = bearing(640, 240, 640, 480), bearing(0, 240, 640, 480)
    assert right["bearing_from_torso_deg"] < -30 and left["bearing_from_torso_deg"] == -right["bearing_from_torso_deg"]
    assert bearing(320, 0, 640, 480)["below_horizontal_deg"] < c["below_horizontal_deg"]     # higher in the image
    turned = bearing(320, 240, 640, 480, (math.radians(-40), 0.0, 0.0))
    assert turned["bearing_from_torso_deg"] == 0 and turned["look_yaw_deg"] == -40                # the waist adds on
    far = bearing(0, 240, 640, 480, (math.radians(130), 0.0, 0.0))
    assert far["look_yaw_deg"] == limits.get("look_yaw_max_deg") and "beyond the waist" in far["beyond_waist"]


def test_the_query_needs_an_image_and_a_pixel_inside_it():
    tool = TOOLS["locate_point"](x_px=100, y_px=50, note="the laptop")
    cmd = np.zeros(29)
    with pytest.raises(ValueError, match="no camera image"):
        tool.query(cmd, [], view={"frame_size": None, "q": cmd})
    with pytest.raises(ValueError, match="outside the 64x48 image"):
        tool.query(cmd, [], view={"frame_size": (64, 48), "q": cmd})
    out = tool.query(cmd, [], view={"frame_size": (640, 360), "q": cmd, "imu_quat": None})
    assert out["status"] == "ok" and out["image_size"] == [640, 360] and "no distance" in out["note"]


def test_it_agrees_with_the_twin_camera():
    mujoco = pytest.importorskip("mujoco")
    from g1.twin.core import Twin
    from g1.twin.model import HEAD_CAMERA
    twin = Twin(start=(0.5, -0.3, 30))
    m, d = twin.model, twin.data
    waist = math.radians(20)
    d.qpos[7 + joint_index("waist_yaw")] = waist
    mujoco.mj_forward(m, d)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA)
    pos, rot = d.cam_xpos[cam], d.cam_xmat[cam].reshape(3, 3)       # columns: camera x right, y up, -z forward
    W, H = 640, 480
    f = (H / 2) / math.tan(math.radians(m.cam_fovy[cam]) / 2)
    _, _, base_yaw = twin.truth()
    heading = base_yaw + waist
    # points placed by (distance, degrees from the torso's heading, degrees below horizontal), all in view
    for r, side, below in ((1.5, 10, 30), (1.0, -20, 50), (2.0, 15, 25), (0.8, 0, 65)):
        a = heading + math.radians(side)
        target = pos + np.array([r * math.cos(a), r * math.sin(a), -r * math.tan(math.radians(below))])
        v = target - pos
        xc, yc, zc = rot.T @ v
        assert zc < 0                                                  # in front of the camera
        u, w = W / 2 + f * xc / -zc, H / 2 - f * yc / -zc
        assert 0 <= u <= W and 0 <= w <= H, (r, side, below)
        # what the robot measures: its waist joints (the walker holds the pitch) and the IMU's pelvis tilt
        waist_q = tuple(d.qpos[7 + joint_index(n)] for n in ("waist_yaw", "waist_roll", "waist_pitch"))
        got = bearing(u, w, W, H, waist_q, d.qpos[3:7])
        want_yaw = math.degrees(math.remainder(math.atan2(v[1], v[0]) - base_yaw, 2 * math.pi))
        want_below = math.degrees(math.atan2(-v[2], math.hypot(v[0], v[1])))
        assert got["look_yaw_deg"] == pytest.approx(want_yaw, abs=2.0), (r, side, below)     # from the feet
        assert got["below_horizontal_deg"] == pytest.approx(want_below, abs=2.0), (r, side, below)
