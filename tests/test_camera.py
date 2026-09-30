import math
import time

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


# -- taps: subscribe, the live view, the recording ---------------------------------------

def test_subscribers_see_every_publish_and_a_bad_one_is_dropped(capsys):
    cam = Camera()
    seen = []
    cam.subscribe(lambda f: seen.append(f.seq))

    def bad(f):
        raise RuntimeError("boom")
    cam.subscribe(bad)
    for i in range(3):
        cam.publish(solid((i, 0, 0)), 0.1 * i)                    # the producer never sees the failure
    assert seen == [1, 2, 3] and "dropped" in capsys.readouterr().out
    cam.unsubscribe(cam._subscribers[0])
    cam.publish(solid((9, 0, 0)), 0.3)
    assert seen == [1, 2, 3] and cam._subscribers == []


def test_viewer_serves_a_frame_a_page_and_a_stream():
    import urllib.request
    from g1.camera import Viewer
    from g1.core import images
    cam = Camera()
    v = Viewer(cam, 0, width=32, title="t")
    v.start()
    try:
        assert v.port > 0 and v.url == f"http://127.0.0.1:{v.port}/"
        with pytest.raises(urllib.error.HTTPError):               # no frame yet
            urllib.request.urlopen(v.url + "frame", timeout=2)
        cam.publish(solid((200, 30, 30)), 0.0)
        data = urllib.request.urlopen(v.url + "frame", timeout=2).read()
        img = images.decode(data)
        assert img.shape == (24, 32, 3) and img[12, 16, 0] > 150 and img[12, 16, 1] < 80   # scaled, RGB
        page = urllib.request.urlopen(v.url, timeout=2).read().decode()
        assert "/stream" in page and "<title>t</title>" in page
        resp = urllib.request.urlopen(v.url + "stream", timeout=2)
        assert resp.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        head = resp.read(40)
        assert head.startswith(b"--frame\r\nContent-Type: image/jpeg")
        resp.close()
    finally:
        t0 = time.monotonic()
        v.stop()
        assert time.monotonic() - t0 < 2.0 and v.server is None


def test_recorder_writes_frames_at_their_own_stamps(tmp_path):
    av = pytest.importorskip("av")
    from g1.camera import Recorder
    cam = Camera()
    rec = Recorder(cam, tmp_path / "out" / "cam.mp4")
    rec.start()
    for i in range(30):
        cam.publish(solid((i * 8, 0, 0), h=49, w=65), 5.0 + i / 15)     # odd size: cropped to even
        time.sleep(0.005)
    rec.stop()
    assert rec.frames == 30 and rec.error is None and "30 frames" in rec.summary()
    assert rec.duration_s == pytest.approx(29 / 15, abs=0.002)
    c = av.open(str(rec.path))
    s = c.streams.video[0]
    frames = list(c.decode(s))
    assert len(frames) == 30 and (s.width, s.height) == (64, 48)
    assert float(frames[-1].pts * frames[-1].time_base) == pytest.approx(29 / 15, abs=0.002)
    assert float(frames[0].pts * frames[0].time_base) == 0.0
    c.close()


def test_recorder_drops_the_oldest_when_behind(tmp_path):
    av = pytest.importorskip("av")
    from g1.camera import Recorder
    cam = Camera()
    rec = Recorder(cam, tmp_path / "cam.mp4", queue_size=2)
    rec.start()
    for i in range(200):                              # far faster than the encoder
        cam.publish(solid((i % 255, 0, 0), h=240, w=320), i / 15)
    rec.stop()
    assert rec.error is None and rec.dropped > 0 and rec.frames + rec.dropped == 200
    c = av.open(str(rec.path))
    assert sum(1 for _ in c.decode(c.streams.video[0])) == rec.frames
    c.close()


def test_recorder_needs_pyav(monkeypatch):
    import builtins
    from g1.camera import Recorder
    real = builtins.__import__

    def no_av(name, *a, **k):
        if name == "av":
            raise ImportError
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_av)
    with pytest.raises(RuntimeError, match="pip install av"):
        Recorder(Camera(), "x.mp4")
