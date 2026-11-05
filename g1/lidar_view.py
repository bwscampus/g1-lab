"""``g1 lidar``'s live view: the map and the latest scan in 3D in a browser, as ``--view`` is the camera's.

An HTTP server on ``127.0.0.1:lidar_view_port`` (a free port when that one is taken; from a laptop,
``ssh -L``), on its own threads, reading the lidar's latest-only slot and the map accumulator's array
(replaced, never changed in place), so it never slows the feed.

  /            the page: a canvas drawn with plain 2D (no WebGL, no library, nothing fetched from the
               internet), drag to orbit, shift- or right-drag to pan, wheel to zoom
  /state.json  seq, counts, rate, the 4x4 pose (row-major), the floor's height, elapsed
  /scan.bin    the latest scan in the map frame: float32 x, y, z (little-endian)
  /map.bin     the map: int16 x, y, z in centimetres, at most lidar_view_points_max points
"""
from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

from g1.core import limits


class LidarViewer:
    def __init__(self, lidar, acc, port: int = limits.get("lidar_view_port"), *, host: str = "127.0.0.1",
                 max_points: int = limits.get("lidar_view_points_max")) -> None:
        self.lidar, self.acc = lidar, acc
        self.port, self.host, self.max_points = port, host, max_points
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.t0 = time.monotonic()
        self.count0 = 0
        self._map_cache: tuple[int, bytes] = (-1, b"")
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def floor_z(self, pts: np.ndarray) -> float:
        """The floor's height in the map frame: the 2nd percentile of z (a sample of the map)."""
        if not len(pts):
            return 0.0
        sample = pts[:: max(1, len(pts) // 20000), 2]
        return float(np.percentile(sample, 2))

    def state(self) -> dict:
        scan = self.lidar.latest()
        pts = self.acc.points()
        elapsed = time.monotonic() - self.t0
        out = {"seq": 0, "n_scan": 0, "frame": "", "map_frame": "", "pose": np.eye(4).ravel().tolist(), "yaw_deg": 0.0,
               "rate": (self.lidar.count - self.count0) / elapsed if elapsed > 0 else 0.0,
               "map_scans": self.acc.scans, "n_map": len(pts), "n_map_sent": min(len(pts), self.max_points),
               "voxel_m": self.acc.voxel_m, "floor_z": self.floor_z(pts), "elapsed": elapsed}
        if scan is not None:
            out.update(seq=scan.seq, n_scan=len(scan.points), frame=scan.frame_id, map_frame=scan.map_frame,
                       pose=[float(v) for v in scan.pose.ravel()],
                       yaw_deg=math.degrees(math.atan2(scan.pose[1, 0], scan.pose[0, 0])))
        return out

    def scan_bytes(self) -> bytes:
        scan = self.lidar.latest()
        if scan is None:
            return b""
        return np.ascontiguousarray(scan.map_points()[:, :3], dtype="<f4").tobytes()

    def map_bytes(self) -> bytes:
        with self._lock:
            scans, data = self._map_cache
            if scans == self.acc.scans:
                return data
            pts = self.acc.points()
            step = max(1, math.ceil(len(pts) / self.max_points))
            cm = np.clip(np.round(pts[::step, :3] * 100), -32767, 32767).astype("<i2")
            self._map_cache = (self.acc.scans, cm.tobytes())
            return self._map_cache[1]

    def start(self) -> None:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:      # quiet
                pass

            def send(self, body: bytes, kind: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                try:
                    if path == "/state.json":
                        self.send(json.dumps(viewer.state()).encode(), "application/json")
                    elif path == "/scan.bin":
                        self.send(viewer.scan_bytes(), "application/octet-stream")
                    elif path == "/map.bin":
                        self.send(viewer.map_bytes(), "application/octet-stream")
                    elif path in ("/", "/index.html"):
                        self.send(PAGE.encode(), "text/html; charset=utf-8")
                    else:
                        self.send_error(404)
                except (BrokenPipeError, ConnectionResetError):
                    return

        try:
            self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:                                  # the port is taken: any free one
            self.server = ThreadingHTTPServer((self.host, 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.t0, self.count0 = time.monotonic(), self.lidar.count
        self.thread = threading.Thread(target=self.server.serve_forever, name="lidar-view", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        server, self.server = self.server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>g1 lidar</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;height:100%;background:#0d0f12;color:#cfd6df;font:13px/1.4 -apple-system,system-ui,sans-serif;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;cursor:grab;touch-action:none}
#hud{position:fixed;left:12px;top:10px;background:rgba(13,15,18,.8);padding:8px 10px;border-radius:6px;
     white-space:pre;font:12px/1.5 ui-monospace,Menlo,monospace;pointer-events:none}
#bar{position:fixed;right:12px;top:10px;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
button{background:#1d232b;color:#cfd6df;border:1px solid #333c47;border-radius:5px;padding:5px 10px;cursor:pointer;font:inherit}
button.on{background:#25557c;border-color:#3d7db3}
#help{position:fixed;left:12px;bottom:10px;opacity:.55;font-size:12px;pointer-events:none}
#legend{position:fixed;right:12px;bottom:10px;font-size:12px;display:flex;align-items:center;gap:6px;opacity:.8}
#legend i{display:inline-block;width:120px;height:8px;border-radius:2px;
  background:linear-gradient(90deg,#3040ff,#00d0ff,#30ff60,#ffe000,#ff3000)}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud">g1 lidar: connecting…</div>
<div id="bar">
  <button id="top">Top</button><button id="tilt">3D</button>
  <button id="follow" class="on">Follow robot</button>
  <button id="scanb" class="on">Scan</button><button id="mapb" class="on">Map</button>
</div>
<div id="legend">floor <i></i> +2.5 m</div>
<div id="help">drag: orbit · shift- or right-drag: pan · wheel: zoom · Ctrl-C in the terminal ends mapping and saves map.pcd</div>
<script>
"use strict";
const cv = document.getElementById("c"), ctx = cv.getContext("2d"), hud = document.getElementById("hud");
let W = 0, H = 0, F = 1, img, buf, depth, dpr = 1;
let yaw = -2.4, pitch = 0.75, dist = 9, target = [0, 0, 0];
let follow = true, showScan = true, showMap = true, dirty = true;
let scan = new Float32Array(0), map = new Float32Array(0), st = null, lastSeq = -1, lastMap = -1, lastMapFetch = 0;
const BG = 0xff120f0d;                                   // #0d0f12 as little-endian ABGR

function resize() {
  dpr = window.devicePixelRatio || 1;
  W = cv.width = Math.floor(innerWidth * dpr); H = cv.height = Math.floor(innerHeight * dpr);
  F = 0.5 * H / Math.tan(30 * Math.PI / 180);           // a 60 degree vertical field of view
  img = ctx.createImageData(W, H); buf = new Uint32Array(img.data.buffer); depth = new Float32Array(W * H);
  dirty = true;
}
addEventListener("resize", resize); resize();

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

function camera() {                                      // orbit around the target, z up
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const eye = [target[0] + dist * cp * Math.cos(yaw), target[1] + dist * cp * Math.sin(yaw), target[2] + dist * sp];
  const fwd = norm(sub(target, eye)), right = norm(cross(fwd, [0, 0, 1])), up = cross(right, fwd);
  return {eye, fwd, right, up};
}
function project(p, c) {
  const d = sub(p, c.eye), z = d[0] * c.fwd[0] + d[1] * c.fwd[1] + d[2] * c.fwd[2];
  if (z < 0.05) return null;
  return [W / 2 + F * (d[0] * c.right[0] + d[1] * c.right[1] + d[2] * c.right[2]) / z,
          H / 2 - F * (d[0] * c.up[0] + d[1] * c.up[1] + d[2] * c.up[2]) / z];
}
function colour(t, dim) {                                // height 0..1 -> blue, cyan, green, yellow, red
  t = Math.max(0, Math.min(1, t));
  const ch = x => Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - x)));
  const k = dim ? 0.5 : 1;
  return ((255 << 24) | ((ch(1) * 255 * k) << 16) | ((ch(2) * 255 * k) << 8) | (ch(3) * 255 * k)) >>> 0;
}
function points(arr, c, floor, dim, size) {
  const e = c.eye, f = c.fwd, r = c.right, u = c.up, n = arr.length / 3;
  for (let i = 0; i < n; i++) {
    const dx = arr[3 * i] - e[0], dy = arr[3 * i + 1] - e[1], dz = arr[3 * i + 2] - e[2];
    const z = dx * f[0] + dy * f[1] + dz * f[2];
    if (z < 0.05) continue;
    const sx = (W / 2 + F * (dx * r[0] + dy * r[1] + dz * r[2]) / z) | 0;
    const sy = (H / 2 - F * (dx * u[0] + dy * u[1] + dz * u[2]) / z) | 0;
    if (sx < -size || sy < -size || sx >= W || sy >= H) continue;
    const col = colour((arr[3 * i + 2] - floor) / 2.5, dim);
    for (let oy = 0; oy < size; oy++) for (let ox = 0; ox < size; ox++) {
      const x = sx + ox, y = sy + oy;
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      const k = y * W + x;
      if (z < depth[k]) { depth[k] = z; buf[k] = col; }
    }
  }
}
function line(c, a, b, style, width) {
  const pa = project(a, c), pb = project(b, c);
  if (!pa || !pb) return;
  ctx.strokeStyle = style; ctx.lineWidth = width * dpr;
  ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
}
function grid(c, floor) {                                 // 1 m squares on the floor, 12 m around the target
  const cx = Math.round(target[0]), cy = Math.round(target[1]), R = 12;
  for (let i = -R; i <= R; i++) {
    const style = i === 0 ? "rgba(140,160,185,.35)" : "rgba(140,160,185,.13)";
    for (let j = -R; j < R; j++) {
      line(c, [cx + i, cy + j, floor], [cx + i, cy + j + 1, floor], style, 1);
      line(c, [cx + j, cy + i, floor], [cx + j + 1, cy + i, floor], style, 1);
    }
  }
}
function robot(c, floor) {
  if (!st) return;
  const m = st.pose, p = [m[3], m[7], m[11]], x = [m[0], m[4], m[8]];
  line(c, [p[0], p[1], floor], p, "rgba(255,255,255,.5)", 1);
  line(c, p, [p[0] + 0.7 * x[0], p[1] + 0.7 * x[1], p[2] + 0.7 * x[2]], "#ff4d4d", 3);
  const q = project(p, c);
  if (q) { ctx.fillStyle = "#ff4d4d"; ctx.beginPath(); ctx.arc(q[0], q[1], 5 * dpr, 0, 2 * Math.PI); ctx.fill(); }
}
function render() {
  if (dirty && W) {
    dirty = false;
    const c = camera(), floor = st ? st.floor_z : 0;
    buf.fill(BG); depth.fill(Infinity);
    if (showMap) points(map, c, floor, true, Math.max(1, Math.round(dpr)));
    if (showScan) points(scan, c, floor, false, Math.max(2, Math.round(2 * dpr)));
    ctx.putImageData(img, 0, 0);
    grid(c, floor); robot(c, floor);
  }
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

async function get(path) {
  const r = await fetch(path, {cache: "no-store"});
  if (!r.ok) throw new Error(path + " " + r.status);
  return r;
}
async function poll() {
  try {
    const s = await (await get("state.json")).json();
    st = s;
    if (s.seq !== lastSeq) { lastSeq = s.seq; scan = new Float32Array(await (await get("scan.bin")).arrayBuffer()); dirty = true; }
    const now = performance.now();
    if (s.map_scans !== lastMap && now - lastMapFetch > 1000) {
      lastMapFetch = now; lastMap = s.map_scans;
      const a = new Int16Array(await (await get("map.bin")).arrayBuffer()), m = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) m[i] = a[i] / 100;
      map = m; dirty = true;
    }
    if (follow) { target = [s.pose[3], s.pose[7], s.pose[11]]; dirty = true; }
    const p = s.pose, mins = Math.floor(s.elapsed / 60), secs = Math.floor(s.elapsed % 60);
    hud.textContent =
      `scan ${s.seq}  ${s.n_scan} pts  ${s.rate.toFixed(1)}/s  (${s.frame || "?"})\n` +
      `map  ${s.n_map} pts from ${s.map_scans} scans, ${s.voxel_m * 100} cm voxels` +
      (s.n_map > s.n_map_sent ? `  (showing ${s.n_map_sent})` : "") + "\n" +
      `pose x ${p[3].toFixed(2)}  y ${p[7].toFixed(2)}  z ${p[11].toFixed(2)} m  yaw ${s.yaw_deg.toFixed(1)}°\n` +
      `floor z ${s.floor_z.toFixed(2)} m   ${mins}:${String(secs).padStart(2, "0")} mapping`;
  } catch (e) {
    hud.textContent = "g1 lidar: not connected (" + e.message + ")\nis g1 lidar still running? (its map is saved when it ends)";
  }
  setTimeout(poll, 200);
}
poll();

function setFollow(on) { follow = on; document.getElementById("follow").classList.toggle("on", on); dirty = true; }
function toggle(id, get, set) {
  const b = document.getElementById(id);
  b.onclick = () => { set(!get()); b.classList.toggle("on", get()); dirty = true; };
}
document.getElementById("top").onclick = () => { pitch = 1.553; dirty = true; };
document.getElementById("tilt").onclick = () => { pitch = 0.75; dirty = true; };
document.getElementById("follow").onclick = () => setFollow(!follow);
toggle("scanb", () => showScan, v => showScan = v);
toggle("mapb", () => showMap, v => showMap = v);

let drag = null;
cv.addEventListener("pointerdown", e => {
  drag = {x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2};
  cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing";
});
cv.addEventListener("pointerup", () => { drag = null; cv.style.cursor = "grab"; });
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    setFollow(false);
    const c = camera(), s = dist * 0.0018, g = norm([c.fwd[0] + c.up[0], c.fwd[1] + c.up[1], 0]);
    target = [target[0] - c.right[0] * dx * s + g[0] * dy * s, target[1] - c.right[1] * dx * s + g[1] * dy * s, target[2]];
  } else {
    yaw -= dx * 0.006; pitch = Math.max(-0.3, Math.min(1.553, pitch + dy * 0.006));
  }
  dirty = true;
});
cv.addEventListener("contextmenu", e => e.preventDefault());
cv.addEventListener("wheel", e => {
  e.preventDefault(); dist = Math.max(0.5, Math.min(80, dist * Math.pow(1.0015, e.deltaY))); dirty = true;
}, {passive: false});
</script></body></html>
"""
