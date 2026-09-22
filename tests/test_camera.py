import math

import numpy as np
import pytest

from camera import Camera, DirCamera, Frame, NoiseCamera
from config import CONTROL_DT, STAND_Q, UPPER_BODY, joint_index
from envs import CheckEnv
from envs.base import Env
from motions import SixSeven
from policy import Obs, ReactivePolicy
from routines import Selector
from run import build_parser, run
from vision import Look, red_blob

WAIST_YAW = joint_index("waist_yaw")


def solid(color, h=48, w=64):
    return np.full((h, w, 3), color, dtype=np.uint8)


def check_env(*extra):
    return CheckEnv(build_parser().parse_args(["--env", "check", "--policy", "x", *extra]))


def write_frames(path, colors):
    cv2 = pytest.importorskip("cv2")
    for i, c in enumerate(colors):
        bgr = cv2.cvtColor(solid(c), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path / f"{i:04d}.png"), bgr)


# -- frame slot -------------------------------------------------------------

def test_slot_keeps_latest_only():
    cam = Camera()
    for i in range(3):
        cam.publish(solid((i, 0, 0)), 0.1 * i)
    f = cam.latest()
    assert f.seq == 3 and f.stamp == pytest.approx(0.2) and f.image[0, 0, 0] == 2
    assert cam.count == 3
    with pytest.raises(ValueError):
        cam.publish(np.zeros((4, 4), np.uint8), 0.0)


def test_observe_without_camera():
    obs = Env(None).observe(STAND_Q)
    assert obs.frame is None and obs.frame_age == math.inf


def test_dir_camera_replays_on_clock(tmp_path):
    write_frames(tmp_path, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    cam = DirCamera(tmp_path, fps=10.0)
    cam.start()
    f = cam.poll(0.0)
    assert f.seq == 1 and tuple(f.image[0, 0]) == (255, 0, 0)      # RGB, not cv2's BGR
    assert cam.poll(0.05).seq == 1
    f = cam.poll(0.1)
    assert f.seq == 2 and tuple(f.image[0, 0]) == (0, 255, 0) and f.stamp == pytest.approx(0.1)
    assert cam.poll(0.35).seq == 3      # skipped straight to the newest due frame
    assert cam.poll(1.0).seq == 3       # ran out: the last frame stays, no re-publish


def test_noise_camera_frames_differ():
    cam = NoiseCamera(fps=10.0, size=(32, 32))
    a = cam.poll(0.0).image.copy()
    b = cam.poll(0.1).image
    assert a.shape == (32, 32, 3) and not np.array_equal(a, b)


# -- vision helper ----------------------------------------------------------

def test_red_blob():
    img = solid((0, 0, 0))
    assert red_blob(img) is None
    img[:, 48:] = (220, 20, 20)                     # right quarter red
    u, v, frac = red_blob(img)
    assert u > 0.5 and abs(v) < 1e-6 and frac == pytest.approx(0.25)


# -- ReactivePolicy envelope --------------------------------------------------

class Nod(ReactivePolicy):
    name = "nod"

    def __init__(self):
        super().__init__(1.0, ramp=0.1, to_stand=0.1)
        self.calls = 0

    def track(self, t, obs):
        self.calls += 1
        return {WAIST_YAW: 0.3}


def test_reactive_tracks_per_frame_and_holds_when_stale():
    p = Nod()
    p.reset(Obs(STAND_Q))
    t = 0.0
    while t < 0.2 - 1e-9:                           # through both bookends
        p.step(t, Obs(STAND_Q))
        t += CONTROL_DT
    img = solid((0, 0, 0))
    a = p.step(0.2, Obs(STAND_Q, Frame(img, 0.2, 1), 0.0))
    assert p.calls == 1
    assert a.q[WAIST_YAW] == pytest.approx(3.0 * CONTROL_DT)   # rate-limited toward 0.3
    p.step(0.22, Obs(STAND_Q, Frame(img, 0.2, 1), 0.02))      # same seq: no new call
    assert p.calls == 1
    p.step(0.24, Obs(STAND_Q, Frame(img, 0.2, 2), 1.0))       # new seq but stale: no call
    assert p.calls == 1
    p.step(0.26, Obs(STAND_Q, Frame(img, 0.26, 3), 0.0))
    assert p.calls == 2
    assert p.duration == pytest.approx(1.4)


def test_reactive_rejects_joint_outside_policy():
    class Bad(ReactivePolicy):
        joints = [WAIST_YAW]

        def track(self, t, obs):
            return {18: 0.0}

    p = Bad(1.0, ramp=0.0, to_stand=0.0)
    p.reset(Obs(STAND_Q))
    with pytest.raises(ValueError):
        p.step(0.0, Obs(STAND_Q, Frame(solid((0, 0, 0)), 0.0, 1), 0.0))


# -- through the check env ----------------------------------------------------

def test_look_passes_check_under_noise():
    env = check_env("--camera-noise")
    p = Look(duration=3.0)
    assert run(p, env) is True
    assert env.violations == []
    assert env.ticks == round(p.duration / CONTROL_DT)
    assert env.camera.count > 30                      # frames were actually fed


def test_look_holds_without_frames():
    env = check_env()
    assert run(Look(duration=2.0), env) is True
    assert env.q_min[WAIST_YAW] == env.q_max[WAIST_YAW] == 0.0


def test_selector_triggers_motion_then_hands_back(tmp_path):
    write_frames(tmp_path, [(0, 0, 0)] * 30 + [(230, 10, 10)] * 5)   # red appears at 3 s
    env = check_env("--camera-dir", str(tmp_path), "--camera-fps", "10")
    p = Selector([(lambda f: red_blob(f.image) is not None, "sixseven")])
    assert run(p, env) is True
    assert env.violations == []
    # takeover 5 s -> triggers as soon as idle begins -> sixseven -> handback 5 s
    assert env.ticks == round((10.0 + SixSeven().duration) / CONTROL_DT)
    assert env.q_min[19] < -1.0 and env.q_max[26] > 1.0   # sixseven turned the wrists


def test_selector_times_out_without_trigger():
    env = check_env()
    p = Selector([(lambda f: True, "tpose")], timeout=1.0)
    assert run(p, env) is True
    assert env.ticks == round(11.0 / CONTROL_DT)
    assert env.q_max[16] == pytest.approx(0.2)        # never left STAND
    assert p.joints == sorted(UPPER_BODY)
