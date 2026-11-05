"""g1 lidar: the PointCloud2 decoding, the map format, the child-process feed, and the command's
SLAM calls (allow-listed, always ended) against a fake link."""
import math
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from g1.core import config, limits
from g1.lidar import (MapAccumulator, Scan, SlamLidar, lidar_main, points_from_cloud2, pose_from_odom,
                      pose_matrix, read_pcd, write_pcd, yaw_deg)


def header(t=12.5, frame="map"):
    return NS(stamp=NS(sec=int(t), nanosec=int(round((t % 1) * 1e9))), frame_id=frame)


def cloud(points, fields=("x", "y", "z", "intensity"), step=None, frame="map"):
    """A sensor_msgs PointCloud2 of float32 fields, ``step`` bytes a point (padding after the fields)."""
    step = step or 4 * len(fields)
    rows = np.zeros((len(points), step), np.uint8)
    for i, _ in enumerate(fields):
        rows[:, 4 * i: 4 * i + 4] = np.asarray(points, "<f4")[:, i: i + 1].view(np.uint8)
    return NS(header=header(frame=frame), height=1, width=len(points), point_step=step, row_step=step * len(points),
              is_bigendian=False, is_dense=True, data=list(rows.tobytes()),
              fields=[NS(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate(fields)])


def odom(x=1.0, y=2.0, z=0.5, yaw_deg_=90.0, t=12.5, frame="map"):
    h = math.radians(yaw_deg_) / 2
    return NS(header=header(t, frame), child_frame_id="body",
              pose=NS(pose=NS(position=NS(x=x, y=y, z=z), orientation=NS(x=0.0, y=0.0, z=math.sin(h), w=math.cos(h)))))


# -- the format ------------------------------------------------------------------------

def test_a_cloud_decodes_to_xyz_and_intensity_in_metres():
    pts = [[1.0, 2.0, 3.0, 40.0], [-1.5, 0.25, -0.5, 7.0]]
    out = points_from_cloud2(cloud(pts))
    assert out.dtype == np.float32 and out.shape == (2, 4)
    assert np.allclose(out, pts)


def test_padding_field_order_and_a_missing_intensity_are_handled():
    pts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    out = points_from_cloud2(cloud(pts, fields=("x", "y", "z"), step=32))     # 20 bytes of padding a point
    assert np.allclose(out[:, :3], pts) and (out[:, 3] == 0).all()
    msg = cloud([[9.0, 1.0, 2.0, 3.0]], fields=("reflectivity", "x", "y", "z"))
    assert np.allclose(points_from_cloud2(msg), [[1.0, 2.0, 3.0, 9.0]])


def test_nan_points_are_dropped_and_a_cloud_without_xyz_is_refused():
    out = points_from_cloud2(cloud([[1, 2, 3, 4], [np.nan, 0, 0, 1]]))
    assert len(out) == 1
    with pytest.raises(ValueError, match="no \\['z'\\]"):
        points_from_cloud2(cloud([[1, 2, 3]], fields=("x", "y", "i")))


def test_odometry_is_a_4x4_pose():
    stamp, frame, pose = pose_from_odom(odom(yaw_deg_=90.0))
    assert stamp == pytest.approx(12.5) and frame == "map"
    assert np.allclose(pose[:3, 3], [1, 2, 0.5]) and yaw_deg(pose) == pytest.approx(90.0)
    assert np.allclose(pose[:3, :3] @ [1, 0, 0], [0, 1, 0])          # body x is map y after a 90 deg yaw


def test_a_scan_in_the_body_frame_is_moved_into_the_map_and_one_in_the_map_is_not():
    pose = pose_matrix(1, 2, 0, 0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    pts = np.array([[1, 0, 0, 5]], np.float32)
    in_body = Scan(pts, 0.0, "body", pose, "map")
    assert np.allclose(in_body.map_points(), [[1, 3, 0, 5]], atol=1e-6)
    assert np.allclose(Scan(pts, 0.0, "map", pose, "map").map_points(), pts)


def test_the_accumulator_keeps_one_point_per_voxel():
    acc = MapAccumulator(voxel_m=0.1)
    pts = np.array([[0.01, 0.01, 0.01, 1], [0.02, 0.02, 0.02, 2], [0.5, 0, 0, 3]], np.float32)
    acc.add(Scan(pts, 0.0, "map", np.eye(4), "map"))
    acc.add(Scan(pts + [0.001, 0, 0, 0], 0.1, "map", np.eye(4), "map"))
    assert acc.scans == 2 and len(acc.points()) == 2
    assert acc.points()[0, 3] == 1                                   # the first point of a voxel is kept


def test_the_accumulator_grows_across_scans_and_keeps_its_keys_sorted():
    acc = MapAccumulator(voxel_m=0.05)
    rng = np.random.default_rng(1)
    for i in range(20):
        pts = rng.uniform(-5, 5, size=(1000, 4)).astype(np.float32)
        acc.add(Scan(pts, float(i), "map", np.eye(4), "map"))
    keys = acc.keys(acc.points())
    assert acc.scans == 20 and len(np.unique(keys)) == len(acc.points())   # one point a voxel
    assert np.all(np.diff(acc._keys) > 0) and len(acc._keys) == len(acc.points())
    acc.add(Scan(acc.points()[:500].copy(), 21.0, "map", np.eye(4), "map"))
    assert len(acc.points()) == len(keys)                            # the same voxels again: nothing new


def test_slam_mapping_points_in_map_are_not_moved_again():
    pose = pose_matrix(5, 0, 0, 0, 0, 0, 1)
    pts = np.array([[1, 2, 3, 0]], np.float32)
    assert np.allclose(Scan(pts, 0.0, "map", pose, "odom").map_points(), pts)


def test_a_pcd_round_trips_with_a_standard_header(tmp_path):
    pts = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    path = write_pcd(tmp_path / "map.pcd", pts)
    head = path.read_bytes().split(b"DATA binary\n")[0].decode()
    assert "FIELDS x y z intensity" in head and "POINTS 50" in head and "TYPE F F F F" in head
    assert np.array_equal(read_pcd(path), pts)


# -- the live feed ----------------------------------------------------------------------

def fake_subscriber(iface, domain, conn):
    """Stands in for the DDS child: two odometry messages and three scans, then quiet."""
    pose_a = pose_matrix(0, 0, 0, 0, 0, 0, 1)
    pose_b = pose_matrix(1, 0, 0, 0, 0, 0, 1)
    conn.send(("odom", 10.0, "map", pose_a.tobytes()))
    conn.send(("odom", 11.0, "map", pose_b.tobytes()))
    for i, t in enumerate((10.1, 10.9, 11.2)):
        pts = np.full((3 + i, 4), i, np.float32)
        conn.send(("scan", t, "body", len(pts), pts.tobytes(), [("x", 0, 7)], 16))
        time.sleep(0.05)
    time.sleep(60)


def dying_subscriber(iface, domain, conn):
    raise SystemExit(0)


def test_the_feed_pairs_each_scan_with_the_nearest_odometry():
    lidar = SlamLidar("eth0", 0, timeout=10.0, target=fake_subscriber)
    lidar.start()
    try:
        t0 = time.monotonic()
        while lidar.count < 3 and time.monotonic() - t0 < 10:
            time.sleep(0.01)
        scan = lidar.latest()
        assert lidar.count == 3 and lidar.odom_count == 2 and scan.seq == 3
        assert scan.points.shape == (5, 4) and scan.frame_id == "body" and scan.map_frame == "map"
        assert np.allclose(scan.pose[:3, 3], [1, 0, 0])                  # 11.2 is nearest 11.0
        assert lidar.layout == ([("x", 0, 7)], 16)
        proc = lidar.proc
    finally:
        lidar.stop()
    assert lidar.proc is None and not proc.is_alive()


def test_a_subscriber_that_dies_before_a_scan_is_an_error():
    lidar = SlamLidar("eth0", 0, timeout=10.0, target=dying_subscriber)
    with pytest.raises(ConnectionError, match="before a scan"):
        lidar.start()


# -- g1 lidar ---------------------------------------------------------------------------

class FakeState:
    def __init__(self, statuses, code=0):
        self.statuses, self.code = statuses, code

    def SetTimeout(self, s): pass
    def Init(self): pass

    def ServiceList(self):
        return self.code, [NS(name=n, status=s, protect=0) for n, s in self.statuses.items()]

    def ServiceSwitch(self, name, on):            # must never be called
        raise AssertionError("g1 lidar switched a service")


class FakeSlam:
    def __init__(self, calls, refuse=()):
        self.calls, self.refuse = calls, refuse

    def SetTimeout(self, s): pass
    def Init(self): pass

    def _call(self, name, *args):
        self.calls.append((name, *args))
        return (3202 if name in self.refuse else 0), "{}"

    def StartMapping(self, slam_type): return self._call("StartMapping", slam_type)
    def EndMapping(self, address): return self._call("EndMapping", address)
    def CloseSlam(self): return self._call("CloseSlam")


class FakeLink:
    def __init__(self, statuses=None, refuse=(), list_code=0):
        self.calls = []
        self.state = FakeState(statuses or {"lidar_driver": 0, "unitree_slam": 0, "video_hub_pc4": 0}, list_code)
        self.slam_client = FakeSlam(self.calls, refuse)

    def init(self, domain, iface): self.calls.append(("init", domain, iface))
    def robot_state(self): return self.state
    def slam(self): return self.slam_client


class FakeLidar:
    """A feed of a growing map, or one that fails or is interrupted."""
    fail = None

    def __init__(self, iface, domain):
        self.count = self.odom_count = 0
        self.layout = ([("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7)], 16)
        self.stopped = False
        self.t0 = None

    def start(self):
        if self.fail:
            raise self.fail
        self.t0 = time.monotonic()

    def latest(self):
        n = int((time.monotonic() - self.t0) / 0.02) + 1
        if self.count != n:
            self.count = self.odom_count = n
        pts = np.random.default_rng(n).uniform(-3, 3, size=(100 * n, 4)).astype(np.float32)
        return Scan(pts, float(n), "map", pose_matrix(0.1 * n, 0, 0, 0, 0, 0, 1), "map", n)

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def healthy_fake_lidar():
    FakeLidar.fail = None
    yield
    FakeLidar.fail = None


@pytest.fixture
def robot(monkeypatch):
    monkeypatch.setenv("UNITREE_IFACE", "eth0")
    monkeypatch.delenv("UNITREE_DOMAIN", raising=False)
    monkeypatch.setitem(limits.LIMITS["lidar_max_s"], "value", 0.3)
    monkeypatch.setitem(limits.LIMITS["lidar_view_port"], "value", 0)
    FakeLidar.fail = None


def test_g1_lidar_checks_starts_serves_prints_ends_and_saves_the_map(robot, capsys, tmp_path):
    link = FakeLink()
    assert lidar_main([], link=link, lidar_cls=FakeLidar, runs=tmp_path) == 0
    out = capsys.readouterr().out
    assert link.calls == [("init", 0, "eth0"), ("StartMapping", "indoor"),
                          ("EndMapping", limits.get("slam_map_path")), ("CloseSlam",)]
    assert "fields [('x', 0, 7)" in out and "pts" in out and "pose (" in out and "map " in out
    assert "live view http://127.0.0.1:" in out and "ssh -L" in out
    assert "growing: a map" in out and "voxel map" in out and "4 call(s), 0 failed" in out
    saved = list(tmp_path.glob("*/map.pcd"))
    assert len(saved) == 1 and len(read_pcd(saved[0])) > 0 and f"map saved: {saved[0]}" in out
    assert {c[0] for c in link.calls} - {"init"} <= config.SLAM_METHODS


@pytest.mark.parametrize("statuses, named", [
    ({"lidar_driver": 1, "unitree_slam": 0}, "lidar_driver is off"),
    ({"lidar_driver": 0, "unitree_slam": 1}, "unitree_slam is off"),
    ({"lidar_driver": 1}, "lidar_driver and unitree_slam are off"),
])
def test_a_service_that_is_off_stops_before_any_slam_call(robot, capsys, statuses, named):
    link = FakeLink(statuses)
    assert lidar_main([], link=link, lidar_cls=FakeLidar) == 1
    out = capsys.readouterr().out
    assert named in out and "Unitree app" in out
    assert link.calls == [("init", 0, "eth0")]


def test_a_refused_start_closes_slam_and_fails(robot, capsys):
    link = FakeLink(refuse=("StartMapping",))
    assert lidar_main([], link=link, lidar_cls=FakeLidar) == 1
    assert [c[0] for c in link.calls[1:]] == ["StartMapping", "CloseSlam"]
    assert "refused to start mapping" in capsys.readouterr().out


@pytest.mark.parametrize("fail, rc", [(TimeoutError("lidar: no scan"), 1), (KeyboardInterrupt(), 0)])
def test_mapping_is_always_ended_when_the_feed_fails_or_is_interrupted(robot, capsys, tmp_path, fail, rc):
    FakeLidar.fail = fail
    link = FakeLink()
    assert lidar_main([], link=link, lidar_cls=FakeLidar, runs=tmp_path) == rc
    assert not list(tmp_path.glob("*/map.pcd"))                      # nothing mapped, nothing saved
    assert [c[0] for c in link.calls[1:]] == ["StartMapping", "EndMapping", "CloseSlam"]


def test_g1_lidar_refuses_the_twin(monkeypatch):
    monkeypatch.setenv("UNITREE_IFACE", "lo0")
    with pytest.raises(SystemExit):
        lidar_main([], link=FakeLink(), lidar_cls=FakeLidar)


def test_the_live_view_serves_the_page_the_state_the_scan_and_the_map():
    import json
    import urllib.request
    from g1.lidar_view import LidarViewer
    lidar = FakeLidar("eth0", 0)
    lidar.start()
    acc = MapAccumulator(voxel_m=0.05)
    for _ in range(3):
        acc.add(lidar.latest())
        time.sleep(0.03)
    viewer = LidarViewer(lidar, acc, port=0, max_points=100)
    viewer.start()
    try:
        get = lambda path: urllib.request.urlopen(viewer.url + path, timeout=5).read()
        page = get("").decode()
        assert "<canvas" in page and "state.json" in page and "map.bin" in page and "http" not in page.split("<script>")[1]
        state = json.loads(get("state.json"))
        assert state["seq"] > 0 and len(state["pose"]) == 16 and state["map_scans"] == 3
        assert state["n_map"] == len(acc.points()) and state["n_map_sent"] == 100
        scan = np.frombuffer(get("scan.bin"), "<f4").reshape(-1, 3)
        assert len(scan) == state["n_scan"] or len(scan) >= 100
        m = np.frombuffer(get("map.bin"), "<i2").reshape(-1, 3)
        assert 50 <= len(m) <= 100 and np.abs(m).max() <= 300                  # cm, within the fake's +-3 m
        with pytest.raises(Exception):
            get("nope")
    finally:
        viewer.stop()


def test_the_live_view_takes_a_free_port_when_its_own_is_taken():
    import socket
    from g1.lidar_view import LidarViewer
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen()
    try:
        viewer = LidarViewer(FakeLidar("eth0", 0), MapAccumulator(), port=held.getsockname()[1])
        viewer.start()
        assert viewer.port != held.getsockname()[1]
        viewer.stop()
    finally:
        held.close()


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP"])
def test_a_kill_or_a_dropped_ssh_ends_mapping_and_saves_the_map_like_ctrl_c(robot, capsys, tmp_path, monkeypatch, sig):
    import os
    import signal
    import threading
    monkeypatch.setitem(limits.LIMITS["lidar_max_s"], "value", 30.0)
    before = signal.getsignal(getattr(signal, sig))
    threading.Timer(0.5, lambda: os.kill(os.getpid(), getattr(signal, sig))).start()
    link = FakeLink()
    t0 = time.monotonic()
    assert lidar_main([], link=link, lidar_cls=FakeLidar, runs=tmp_path) == 0
    assert time.monotonic() - t0 < 10                                   # it stopped on the signal, not at 30 s
    assert [c[0] for c in link.calls[1:]] == ["StartMapping", "EndMapping", "CloseSlam"]
    assert len(list(tmp_path.glob("*/map.pcd"))) == 1 and "stopping" in capsys.readouterr().out
    assert signal.getsignal(getattr(signal, sig)) == before             # the old handler is back


def test_a_second_signal_during_the_teardown_does_not_cut_it_short(robot, capsys, tmp_path, monkeypatch):
    import os
    import signal

    class SignalledSlam(FakeSlam):
        def EndMapping(self, address):
            os.kill(os.getpid(), signal.SIGTERM)                       # a kill while mapping ends
            return super().EndMapping(address)
    link = FakeLink()
    link.slam_client = SignalledSlam(link.calls)
    assert lidar_main([], link=link, lidar_cls=FakeLidar, runs=tmp_path) == 0
    assert [c[0] for c in link.calls[1:]] == ["StartMapping", "EndMapping", "CloseSlam"]
    assert len(list(tmp_path.glob("*/map.pcd"))) == 1 and "this finishes on its own" in capsys.readouterr().out
