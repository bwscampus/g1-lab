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


# -- DirectCamera (the Jetson: the video hub's multicast, decoded by a GStreamer child) --------

def rgba_writer(frames, h=6, w=8, hang=False, say=""):
    """A child that stands in for gst-launch-1.0: ``frames`` RGBA frames (pixel value i+1, alpha 255)
    written to the fd it is given, then exits (or sleeps, as a live stream would)."""
    import sys

    def command(fd):
        script = (f"import os, sys, time\nsys.stderr.write({say!r})\n"
                  f"for i in range({frames}):\n"
                  f"    px = bytes([i + 1, i + 1, i + 1, 255])\n"
                  f"    os.write({fd}, px * ({h} * {w})); time.sleep(0.02)\n"
                  f"time.sleep(60 if {hang} else 0)\n")
        return [sys.executable, "-c", script]
    return command


def test_direct_camera_publishes_rgb_frames_from_the_decoder():
    from g1.camera import DirectCamera
    cam = DirectCamera("eth0", timeout=5.0, size=(6, 8), command=rgba_writer(3, hang=True))
    cam.start()
    try:
        t0 = time.monotonic()
        while cam.count < 3 and time.monotonic() - t0 < 5.0:
            time.sleep(0.01)
        frame = cam.latest()
        assert frame.image.shape == (6, 8, 3) and (frame.image == 3).all()     # alpha dropped, RGB kept
        assert cam.count == 3 and cam.error is None
        proc = cam.proc
    finally:
        cam.stop()
    assert proc.poll() is not None and cam.proc is None                        # the child is stopped with it


def test_direct_camera_explains_a_decoder_that_stops_before_a_frame():
    from g1.camera import DirectCamera
    cam = DirectCamera("wlan0", timeout=5.0, size=(6, 8), command=rgba_writer(0, say="no element nvv4l2decoder"))
    with pytest.raises(ConnectionError, match=r"no element nvv4l2decoder.*eth0, not 'wlan0'"):
        cam.start()
    assert cam.proc is None


def test_direct_camera_says_when_the_stream_ends_mid_run(capsys):
    from g1.camera import DirectCamera
    cam = DirectCamera("eth0", timeout=5.0, size=(6, 8), command=rgba_writer(2))
    cam.start()
    try:
        t0 = time.monotonic()
        while cam.error is None and time.monotonic() - t0 < 5.0:
            time.sleep(0.01)
        assert isinstance(cam.error, EOFError) and "camera: stream ended" in capsys.readouterr().out
    finally:
        cam.stop()


def test_the_video_hub_pipeline_reads_its_multicast_on_the_robot_interface():
    from g1.camera import VIDEO_HUB_GROUP, VIDEO_HUB_PORT, video_hub_pipeline
    argv = video_hub_pipeline("eth0", 7)
    assert argv[0] == "gst-launch-1.0"
    assert {f"address={VIDEO_HUB_GROUP}", f"port={VIDEO_HUB_PORT}", "multicast-iface=eth0",
            "nvv4l2decoder", "fdsink", "fd=7"} <= set(argv)
    assert "video/x-raw,format=RGBA,width=1280,height=720" in argv
    # a slow reader drops decoded frames, never packets (a lost packet smears the picture)
    assert argv.index("queue") == argv.index("rtph264depay") - 2
    leaky = argv.index("leaky=downstream")
    assert argv[leaky - 1] == "queue" and argv[leaky + 1] == "max-size-buffers=1" and leaky < argv.index("fdsink")


@pytest.mark.parametrize("jetson, source, iface, expected", [
    (True, "auto", "eth0", "DirectCamera"),       # on the Jetson: straight from the video hub
    (False, "auto", "en7", "WebRTCCamera"),       # a laptop: the robot's WebRTC stream, as before
    (True, "webrtc", "eth0", "WebRTCCamera"),     # asked for by name
    (False, "direct", "eth0", "DirectCamera"),
])
def test_the_robot_env_takes_the_camera_direct_only_on_the_jetson(monkeypatch, jetson, source, iface, expected):
    from types import SimpleNamespace

    import g1.camera
    import g1.envs.robot as robot

    made = []

    class Fake(Camera):
        def __init__(self, name):
            super().__init__()
            made.append(name)

        def start(self):
            self.publish(solid((1, 2, 3)), 0.0)
    monkeypatch.setattr(g1.camera, "on_the_jetson", lambda: jetson)
    monkeypatch.setattr(g1.camera, "DirectCamera", lambda *a, **k: Fake("DirectCamera"))
    monkeypatch.setattr(robot, "WebRTCCamera", lambda *a, **k: Fake("WebRTCCamera"))
    args = SimpleNamespace(camera_source=source, iface=iface, camera_ip="192.168.123.161", camera_timeout=1.0)
    robot.RobotEnv._open_camera(SimpleNamespace(args=args))
    assert made == [expected]
