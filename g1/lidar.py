"""The head lidar (a Livox MID-360), live, through Unitree's SLAM.

The lidar sends its points only to the main board (.161), where Unitree's ``lidar_driver`` and
``unitree_slam`` services turn them into DDS topics. A mapping session (``slam_operate`` 1801)
publishes ``rt/unitree/slam_mapping/points`` (sensor_msgs PointCloud2) and
``rt/unitree/slam_mapping/odom`` (nav_msgs Odometry); 1802 ends it and saves Unitree's own map as
a .pcd on the robot, 1901 closes SLAM. Both services are switched on in the Unitree app; this code
never switches a service (it only reads the service list), and the SLAM calls it may make are
``config.SLAM_METHODS``.

The format for a 3D map is ``Scan``: ``points`` (N, 4) float32 ``x, y, z, intensity`` in metres,
``pose`` (4, 4) map<-body at the scan's time, and ``map_points()`` in the map frame either way.
``MapAccumulator`` voxel-downsamples scans into one cloud; ``write_pcd`` saves it for Open3D, PCL or
CloudCompare. Every ``g1 lidar`` session saves its map as ``runs/lidar/<time>/map.pcd`` and serves a
live 3D view in a browser (``g1/lidar_view.py``), as ``--view`` does for the camera.

PointCloud2's ``data`` reaches Python as a sequence of ~0.5 MB a scan; decoded next to rt/lowstate it
would starve the run for the GIL (the WebRTC camera on the Jetson did), so ``SlamLidar`` subscribes in
a child process that sends numpy bytes back, and the parent keeps a latest-only slot.

  g1 lidar      check the services, start mapping, serve the live view and print a line a second until
                Ctrl-C (or lidar_max_s), then end mapping and save runs/lidar/<time>/map.pcd
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from g1.core import config, limits

POINTS_TOPIC = "rt/unitree/slam_mapping/points"
ODOM_TOPIC = "rt/unitree/slam_mapping/odom"
SERVICES = ("lidar_driver", "unitree_slam")     # the onboard services the feed needs, switched on in the app
SERVICE_RUNNING = 0                             # RobotStateClient.ServiceList status of a running service

# sensor_msgs/PointField datatypes
FIELD_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
INTENSITY_NAMES = ("intensity", "reflectivity", "i")


# -- the format ------------------------------------------------------------------------

@dataclass
class Scan:
    points: np.ndarray          # (N, 4) float32: x, y, z (m), intensity, in ``frame_id``
    stamp: float                # the cloud's header time (s)
    frame_id: str               # the cloud's frame
    pose: np.ndarray            # (4, 4) float64: map <- body at ``stamp`` (the nearest odometry)
    map_frame: str = ""         # the odometry's frame (the map's); "" before any odometry
    seq: int = 0

    def map_points(self) -> np.ndarray:
        """The points in the map frame: as they are when the cloud is already in it (Unitree SLAM's
        mapping points come in ``map``), else moved by the pose."""
        if not self.map_frame or self.frame_id in (self.map_frame, "map"):
            return self.points
        xyz = self.points[:, :3].astype(np.float64) @ self.pose[:3, :3].T + self.pose[:3, 3]
        return np.column_stack([xyz, self.points[:, 3]]).astype(np.float32)


def stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def points_from_cloud2(msg) -> np.ndarray:
    """A PointCloud2 as (N, 4) float32 ``x, y, z, intensity``, whatever its layout (padding,
    field order, datatypes); intensity is 0 when the cloud has none. NaN points are dropped."""
    fields = {f.name: f for f in msg.fields}
    missing = [a for a in "xyz" if a not in fields]
    if missing:
        raise ValueError(f"point cloud has no {missing} field (fields: {list(fields)})")
    n = int(msg.width) * int(msg.height)
    raw = np.frombuffer(bytes(msg.data), np.uint8)
    if n == 0 or raw.size == 0:
        return np.zeros((0, 4), np.float32)
    rows = raw[: n * msg.point_step].reshape(n, msg.point_step)
    endian = ">" if msg.is_bigendian else "<"

    def column(f) -> np.ndarray:
        dt = np.dtype(endian + FIELD_DTYPES[f.datatype])
        return rows[:, f.offset: f.offset + dt.itemsize].copy().view(dt).reshape(n).astype(np.float32)

    out = np.zeros((n, 4), np.float32)
    for i, a in enumerate("xyz"):
        out[:, i] = column(fields[a])
    name = next((k for k in INTENSITY_NAMES if k in fields), None)
    if name is not None:
        out[:, 3] = column(fields[name])
    return out[np.isfinite(out[:, :3]).all(axis=1)]


def pose_matrix(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """A 4x4 transform from a position and a unit quaternion."""
    q = np.array([qw, qx, qy, qz], float)
    q /= np.linalg.norm(q) or 1.0
    w, a, b, c = q
    t = np.eye(4)
    t[:3, :3] = [[1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w)],
                 [2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w)],
                 [2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)]]
    t[:3, 3] = (x, y, z)
    return t


def pose_from_odom(msg) -> tuple[float, str, np.ndarray]:
    """``(stamp, frame_id, 4x4 map <- body)`` from a nav_msgs Odometry."""
    p, o = msg.pose.pose.position, msg.pose.pose.orientation
    return stamp_s(msg.header), msg.header.frame_id, pose_matrix(p.x, p.y, p.z, o.x, o.y, o.z, o.w)


def yaw_deg(pose: np.ndarray) -> float:
    return math.degrees(math.atan2(pose[1, 0], pose[0, 0]))


class MapAccumulator:
    """Scans into one cloud in the map frame, one point per ``voxel_m`` cube (the first one kept).
    Each voxel is one int64 key in a sorted array, so a scan costs a search and an insert, not a
    re-sort of the whole map (it runs 10 times a second while the map grows). ``points()`` is
    replaced, never changed in place, so the live view can read it from its own threads."""
    OFFSET = 1 << 20               # voxel indices fit 21 bits an axis: +-2^20 voxels (52 km at 5 cm)

    def __init__(self, voxel_m: float = limits.get("lidar_voxel_m")) -> None:
        self.voxel_m = voxel_m
        self._points = np.zeros((0, 4), np.float32)
        self._keys = np.zeros(0, np.int64)            # sorted
        self.scans = 0

    def keys(self, points: np.ndarray) -> np.ndarray:
        k = np.clip(np.floor(points[:, :3] / self.voxel_m).astype(np.int64) + self.OFFSET, 0, 2 * self.OFFSET - 1)
        return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]

    def add(self, scan: Scan) -> None:
        pts = scan.map_points()
        self.scans += 1
        if not len(pts):
            return
        keys, first = np.unique(self.keys(pts), return_index=True)     # one point a voxel within the scan
        if len(self._keys):
            at = np.searchsorted(self._keys, keys)
            new = self._keys[np.minimum(at, len(self._keys) - 1)] != keys
        else:
            at, new = np.zeros(len(keys), np.int64), np.ones(len(keys), bool)
        if new.any():
            self._keys = np.insert(self._keys, at[new], keys[new])
            self._points = np.concatenate([self._points, pts[first[new]]])

    def points(self) -> np.ndarray:
        return self._points


def write_pcd(path: str | Path, points: np.ndarray) -> Path:
    """Binary PCD v0.7, fields ``x y z intensity`` float32: opens in Open3D, PCL, CloudCompare."""
    pts = np.ascontiguousarray(points[:, :4], dtype="<f4")
    n = len(pts)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n"
              "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    path = Path(path)
    path.write_bytes(header.encode("ascii") + pts.tobytes())
    return path


def read_pcd(path: str | Path) -> np.ndarray:
    """What ``write_pcd`` wrote, back as (N, 4) float32."""
    data = Path(path).read_bytes()
    end = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    return np.frombuffer(data[end:], "<f4").reshape(-1, 4).copy()


# -- the live feed ----------------------------------------------------------------------

def subscribe(iface: str, domain: int, conn) -> None:
    """The child process: subscribe to SLAM's points and odometry and send each, decoded, to ``conn``
    as ``("odom", stamp, frame, pose bytes)`` or ``("scan", stamp, frame, n, points bytes)``."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    ChannelFactoryInitialize(domain, iface)
    lock = threading.Lock()

    def send(item) -> None:
        with lock:
            try:
                conn.send(item)
            except (BrokenPipeError, OSError):
                os._exit(0)                     # the parent is gone

    def on_odom(msg) -> None:
        stamp, frame, pose = pose_from_odom(msg)
        send(("odom", stamp, frame, pose.tobytes()))

    def on_points(msg) -> None:
        pts = points_from_cloud2(msg)
        send(("scan", stamp_s(msg.header), msg.header.frame_id, len(pts), pts.tobytes(),
              [(f.name, f.offset, f.datatype) for f in msg.fields], int(msg.point_step)))

    subs = [ChannelSubscriber(ODOM_TOPIC, Odometry_), ChannelSubscriber(POINTS_TOPIC, PointCloud2_)]
    subs[0].Init(on_odom, 10)
    subs[1].Init(on_points, 2)
    while True:
        time.sleep(1.0)


class SlamLidar:
    """The live feed: SLAM's scans and odometry from a child process (``target``, ``subscribe`` by
    default; tests swap it), each scan paired with the odometry nearest its stamp. A latest-only slot,
    like the camera's. ``start`` blocks until the first scan or raises."""

    def __init__(self, iface: str, domain: int, timeout: float = limits.get("lidar_timeout_s"),
                 target: Optional[Callable] = None) -> None:
        self.iface, self.domain, self.timeout = iface, domain, timeout
        self.target = target or subscribe
        self._lock = threading.Lock()
        self._scan: Optional[Scan] = None
        self._odom: deque = deque(maxlen=200)
        self.count = 0
        self.odom_count = 0
        self.layout: Optional[tuple] = None       # the first cloud's (fields, point_step)
        self.proc = None
        self.conn = None
        self.thread: Optional[threading.Thread] = None
        self._stopping = False

    def start(self) -> None:
        ctx = mp.get_context("spawn")             # never fork a process that holds DDS
        self.conn, child = ctx.Pipe(duplex=False)
        self.proc = ctx.Process(target=self.target, args=(self.iface, self.domain, child), daemon=True,
                                name="slam-lidar")
        self.proc.start()
        child.close()
        self._stopping = False
        self.thread = threading.Thread(target=self._run, name="slam-lidar", daemon=True)
        self.thread.start()
        t0 = time.monotonic()
        while self.latest() is None:
            if not self.proc.is_alive():
                self.stop()
                raise ConnectionError("lidar: the subscriber exited before a scan")
            if time.monotonic() - t0 > self.timeout:
                odom = f"{self.odom_count} odometry messages" if self.odom_count else "no odometry either"
                self.stop()
                raise TimeoutError(f"lidar: no scan on {POINTS_TOPIC} within {self.timeout:.0f}s ({odom})")
            time.sleep(0.05)

    def _run(self) -> None:
        while not self._stopping:
            try:
                item = self.conn.recv()
            except (EOFError, OSError):
                return
            if item[0] == "odom":
                _, stamp, frame, pose = item
                with self._lock:
                    self._odom.append((stamp, frame, np.frombuffer(pose, np.float64).reshape(4, 4)))
                    self.odom_count += 1
                continue
            _, stamp, frame, n, data, fields, step = item
            points = np.frombuffer(data, np.float32).reshape(n, 4)
            with self._lock:
                if self._odom:
                    ostamp, oframe, pose = min(self._odom, key=lambda o: abs(o[0] - stamp))
                else:
                    oframe, pose = "", np.eye(4)
                self.count += 1
                self.layout = self.layout or (fields, step)
                self._scan = Scan(points, stamp, frame, pose, oframe, self.count)

    def latest(self) -> Optional[Scan]:
        with self._lock:
            return self._scan

    def stop(self) -> None:
        self._stopping = True
        if self.proc is not None:
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2.0)
                if self.proc.is_alive():
                    self.proc.kill()
                    self.proc.join()
            self.proc = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None


# -- g1 lidar ---------------------------------------------------------------------------

def slam_call(slam, name: str, *args, log: list, verbose: bool) -> tuple[int, object]:
    """One SLAM call, allow-listed (``config.SLAM_METHODS``) and through ``sdk.call``."""
    from g1.envs import sdk
    if name not in config.SLAM_METHODS:
        raise ValueError(f"SLAM call {name!r} is not allow-listed (config.SLAM_METHODS)")
    return sdk.call(slam, name, *args, log=log, verbose=verbose, label="SlamClient")


def services_off(state, *, log: list, verbose: bool) -> tuple[int, list[str]]:
    """Read-only: ``(code, the needed services that are not running)``."""
    from g1.envs import sdk
    code, services = sdk.call(state, "ServiceList", log=log, verbose=verbose, label="RobotStateClient")
    if code != 0:
        return code, []
    status = {s.name: s.status for s in services or []}
    return 0, [n for n in SERVICES if status.get(n) != SERVICE_RUNNING]


def scan_line(scan: Scan, rate: float, now: float) -> str:
    p = scan.points
    lo, hi = (p[:, :3].min(axis=0), p[:, :3].max(axis=0)) if len(p) else (np.zeros(3), np.zeros(3))
    x, y, z = scan.pose[:3, 3]
    return (f"scan {scan.seq:4d}  {scan.frame_id or '?'}  {len(p):6d} pts  {rate:4.1f}/s  "
            f"x [{lo[0]:+6.2f}, {hi[0]:+6.2f}] y [{lo[1]:+6.2f}, {hi[1]:+6.2f}] z [{lo[2]:+6.2f}, {hi[2]:+6.2f}] m  "
            f"pose ({x:+.2f}, {y:+.2f}, {z:+.2f}) yaw {yaw_deg(scan.pose):+6.1f} deg"
            + ("" if scan.map_frame else "  (no odometry yet)"))


def stop_like_ctrl_c() -> dict:
    """SIGINT, SIGTERM and SIGHUP all end the session as Ctrl-C does (end mapping, close SLAM, save the
    map): a dropped ssh (SIGHUP) or a kill (SIGTERM) must not leave SLAM mapping and the map unsaved,
    and a shell that started us in the background had set SIGINT to ignored. Returns the old handlers.
    Main thread only (a no-op elsewhere)."""
    def handler(signum, frame):
        raise KeyboardInterrupt
    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            old[sig] = signal.signal(sig, handler)
        except ValueError:                       # not the main thread
            pass
    return old


def save_map(acc: MapAccumulator, runs: Path) -> Optional[Path]:
    """``runs/<time>/map.pcd``: the session's map in the map frame (nothing when it is empty)."""
    if not len(acc.points()):
        return None
    folder = Path(runs) / time.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    return write_pcd(folder / "map.pcd", acc.points())


def lidar_main(argv: list[str], link=None, lidar_cls=SlamLidar, runs: Optional[Path] = None) -> int:
    from g1.envs.base import shield_sigint
    from g1.envs.link import DdsLink, dds_domain, is_loopback
    from g1.lidar_view import LidarViewer
    from g1.vlm import load_dotenv
    load_dotenv()
    runs = Path(runs) if runs is not None else limits.ROOT / "runs" / "lidar"
    p = argparse.ArgumentParser(prog="g1 lidar", description=(
        "the head lidar through Unitree's SLAM: checks lidar_driver and unitree_slam are on (read-only), "
        "starts mapping, serves the live 3D view at http://127.0.0.1:<lidar_view_port> and prints a line a "
        f"second until Ctrl-C (at most lidar_max_s, {limits.get('lidar_max_s'):g} s), then ends mapping, closes "
        "SLAM and saves the map as runs/lidar/<time>/map.pcd. Robot only."))
    p.parse_args(argv)
    iface = os.environ.get("UNITREE_IFACE")
    if not iface:
        p.error("set UNITREE_IFACE=<iface or IP> in .env")
    if is_loopback(iface):
        p.error("g1 lidar is the robot's lidar; the twin has none (UNITREE_IFACE is loopback)")
    verbose = os.environ.get("G1_VERBOSE", "") not in ("", "0")
    log: list = []
    link = link or DdsLink()
    domain = dds_domain(iface)
    try:
        link.init(domain, iface)
    except ImportError as e:
        p.error(f"unitree_sdk2py is not installed ({e})")
    rpc = limits.get("rpc_timeout_s")

    state = link.robot_state()
    state.SetTimeout(rpc)
    state.Init()
    code, off = services_off(state, log=log, verbose=verbose)
    if code != 0:
        print("lidar: cannot read the robot's service list; is the robot reachable? (g1 status)")
        return 1
    if off:
        print(f"lidar: {' and '.join(off)} {'is' if len(off) == 1 else 'are'} off; switch "
              f"{'it' if len(off) == 1 else 'them'} on in the Unitree app (services), then run g1 lidar again")
        return 1

    slam = link.slam()
    slam.SetTimeout(rpc)
    slam.Init()
    print(f"lidar: {', '.join(SERVICES)} running; starting mapping ({limits.get('slam_type')})")
    code, _ = slam_call(slam, "StartMapping", limits.get("slam_type"), log=log, verbose=verbose)
    if code != 0:
        print("lidar: SLAM refused to start mapping (is the app's mapping or navigation in use?)")
        slam_call(slam, "CloseSlam", log=log, verbose=verbose)
        return 1

    previous = stop_like_ctrl_c()
    lidar = lidar_cls(iface, domain)
    acc = MapAccumulator()
    viewer = None
    sizes: list[int] = []
    rc = 0
    saved = None
    t0 = time.monotonic()
    try:
        lidar.start()
        print(f"lidar: first scan after {time.monotonic() - t0:.1f}s")
        if lidar.layout:
            fields, step = lidar.layout
            print(f"lidar: {POINTS_TOPIC} fields {fields}, point_step {step} bytes")
        viewer = LidarViewer(lidar, acc)
        viewer.start()
        print(f"lidar: live view {viewer.url}  (from a laptop: ssh -L {viewer.port}:127.0.0.1:{viewer.port} "
              f"unitree@<the Jetson>, then open {viewer.url})")
        print(f"lidar: mapping until Ctrl-C (at most {limits.get('lidar_max_s'):g} s)")
        seen, t_start, first, printed = 0, time.monotonic(), lidar.count, 0.0
        while time.monotonic() - t_start < limits.get("lidar_max_s"):
            scan = lidar.latest()
            now = time.monotonic()
            if scan is not None and scan.seq != seen:
                seen = scan.seq
                sizes.append(len(scan.points))
                acc.add(scan)
                if now - printed >= limits.get("lidar_print_period_s"):
                    printed = now
                    print(scan_line(scan, (lidar.count - first) / max(now - t_start, 1e-6), now)
                          + f"  map {len(acc.points())} pts")
            time.sleep(0.02)
    except (TimeoutError, ConnectionError) as e:
        print(e)
        rc = 1
    except KeyboardInterrupt:
        print("\nlidar: stopping (Ctrl-C, SIGTERM or SIGHUP); ending mapping and saving the map")
    finally:
        note = "ending mapping, closing SLAM and saving the map; this finishes on its own"
        for sig in previous:                     # SIGTERM/SIGHUP too: the teardown is never cut short
            if sig != signal.SIGINT:
                signal.signal(sig, lambda signum, frame: print(f"\n{note}", flush=True))
        with shield_sigint(note):
            if viewer is not None:
                viewer.stop()
            lidar.stop()
            slam_call(slam, "EndMapping", limits.get("slam_map_path"), log=log, verbose=verbose)
            slam_call(slam, "CloseSlam", log=log, verbose=verbose)
            saved = save_map(acc, runs)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if sizes:
        elapsed = time.monotonic() - t0
        growing = len(sizes) > 2 and sizes[-1] > 1.5 * sizes[0]
        print(f"lidar: {len(sizes)} scans ({lidar.count} received, {lidar.odom_count} odometry) in {elapsed:.1f}s, "
              f"{np.mean(sizes):.0f} points each on average ({'growing: a map' if growing else 'steady: one scan each'}); "
              f"{len(acc.points())} points in the {acc.voxel_m:g} m voxel map")
    if saved is not None:
        shown = saved.relative_to(limits.ROOT) if saved.is_relative_to(limits.ROOT) else saved
        print(f"lidar: map saved: {shown} ({len(acc.points())} points; open it in CloudCompare or Open3D)")
    failed = [e for e in log if e["code"] != 0]
    print(f"sdk: {len(log)} call(s), {len(failed)} failed")
    return rc or (1 if failed else 0)
