import math
import time

import numpy as np
import pytest

from g1.camera import Camera, DirCamera
from g1.cli import run
from g1.core.action import Runnable
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
        return None if t >= self.seconds - 1e-9 else self.action(STAND_Q.copy())


def test_camera_dir_feeds_the_program_on_sim_time(tmp_path):
    write_frames(tmp_path, [(0, 0, 0)] * 10 + [(230, 10, 10)] * 5)   # red appears at 1 s
    env = sim_env(camera_dir=tmp_path, camera_fps=10.0)
    p = Watch(2.0)
    assert run(p, env) is True and env.violations == []
    assert len(p.seen) == 15 and env.replay.count == 15               # every file, once, at 10 fps
    seqs, stamps, ages, pixels = zip(*p.seen)
    assert list(seqs) == list(range(1, 16))
    assert stamps[10] == pytest.approx(1.0, abs=CONTROL_DT) and all(a <= CONTROL_DT + 1e-9 for a in ages)
    assert pixels[9] == (0, 0, 0) and pixels[10] == (230, 10, 10)
    env = sim_env()
    assert run(Watch(0.5), env) is True and env.renderer is not None    # no --camera-dir: the render feeds it


# -- the robot's stream: aiortc's jitter buffer -----------------------------------------

def assembled(packets_per_frame: int) -> list:
    """Frames that come out of the jitter buffer an aiortc receiver makes for video, fed
    one frame of ``packets_per_frame`` packets and the first packet of the next."""
    from aiortc import rtcrtpreceiver
    from aiortc.rtp import RtpPacket
    buffer = rtcrtpreceiver.JitterBuffer(capacity=128, is_video=True)   # as RTCRtpReceiver asks
    out = []
    for seq in range(packets_per_frame + 1):
        p = RtpPacket(sequence_number=seq, timestamp=0 if seq < packets_per_frame else 3000)
        p._data = b"x"
        out.append(buffer.add(p)[1])
    return [f for f in out if f is not None]


def test_a_keyframe_larger_than_aiortcs_buffer_still_arrives(monkeypatch):
    """The G1's keyframes run ~150 packets; aiortc's 128-packet video buffer passed on
    only each one's tail (no SPS/PPS, no start), so no frame ever decoded.
    WebRTCCamera.start enlarges it."""
    pytest.importorskip("aiortc")
    from aiortc import jitterbuffer, rtcrtpreceiver
    from g1.camera import roomy_video_jitter_buffer
    monkeypatch.setattr(rtcrtpreceiver, "JitterBuffer", jitterbuffer.JitterBuffer)
    assert [len(f.data) for f in assembled(150)] == [22]   # aiortc as shipped: the tail only
    roomy_video_jitter_buffer()
    roomy_video_jitter_buffer()                         # again, as a second start would
    assert [len(f.data) for f in assembled(150)] == [150]
    assert rtcrtpreceiver.JitterBuffer(capacity=16, prefetch=4).capacity == 16   # audio untouched


# -- subscribers ----------------------------------------------------------------------

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
