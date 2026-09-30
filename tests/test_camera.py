import math

import numpy as np
import pytest

from g1.camera import Camera, DirCamera
from g1.cli import run
from g1.core.action import Obs, Runnable
from g1.core.config import CONTROL_DT, STAND_Q, UPPER_BODY
from g1.envs.base import Env
from tests.doubles import red_blob, sim_env


def solid(color, h=48, w=64):
    return np.full((h, w, 3), color, dtype=np.uint8)


def write_frames(path, colors):
    from g1.core import images
    for i, c in enumerate(colors):
        images.write_png(path / f"{i:04d}.png", solid(c))


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
    assert f.seq == 1 and tuple(f.image[0, 0]) == (255, 0, 0)      # RGB as written
    assert cam.poll(0.05).seq == 1
    f = cam.poll(0.1)
    assert f.seq == 2 and tuple(f.image[0, 0]) == (0, 255, 0) and f.stamp == pytest.approx(0.1)
    assert cam.poll(0.35).seq == 3      # skipped straight to the newest due frame
    assert cam.poll(1.0).seq == 3       # ran out: the last frame stays, no re-publish


def test_red_blob_double():
    img = solid((0, 0, 0))
    assert red_blob(img) is None
    img[:, 48:] = (220, 20, 20)                     # right quarter red
    u, v, frac = red_blob(img)
    assert u > 0.5 and abs(v) < 1e-6 and frac == pytest.approx(0.25)


# -- frames reach the program through the env, on the env's clock ----------------------

class Watch(Runnable):
    """Holds STAND for ``seconds`` and notes every distinct frame it is shown."""

    name = "watch"
    joints = UPPER_BODY
    uses_camera = True

    def __init__(self, seconds):
        self.seconds = seconds
        self.seen = []

    def step(self, t, obs):
        if obs.frame is not None and (not self.seen or self.seen[-1][0] != obs.frame.seq):
            self.seen.append((obs.frame.seq, obs.frame.stamp, obs.frame_age, tuple(obs.frame.image[0, 0])))
        return None if t >= self.seconds - 1e-9 else self.action(obs.q.copy())


def test_camera_dir_feeds_the_program_on_sim_time(tmp_path):
    write_frames(tmp_path, [(0, 0, 0)] * 10 + [(230, 10, 10)] * 5)   # red appears at 1 s
    env = sim_env("--camera-dir", str(tmp_path), "--camera-fps", "10")
    p = Watch(2.0)
    assert run(p, env) is True and env.violations == []
    assert len(p.seen) == 15 and env.replay.count == 15               # every file, once, at 10 fps
    seqs, stamps, ages, pixels = zip(*p.seen)
    assert list(seqs) == list(range(1, 16))
    assert stamps[10] == pytest.approx(1.0, abs=CONTROL_DT) and all(a <= CONTROL_DT + 1e-9 for a in ages)
    assert pixels[9] == (0, 0, 0) and pixels[10] == (230, 10, 10)
    env = sim_env()
    assert run(Watch(0.5), env) is True and env.renderer is not None    # no --camera-dir: the render feeds it
