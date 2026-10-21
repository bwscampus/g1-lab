"""The twin as its own process, on DDS, in real time: what ``g1 run TASK --sim`` talks to.

    g1 twin fetch      # once: the model, the walker and the room's assets (assets/)
    g1 twin status

``g1 run TASK --sim`` starts one (``TwinProcess``) for the task's scene, on the
loopback interface and a private DDS domain, runs the robot env's code against it
exactly as against the G1, and stops it at the end. ``--mode gantry`` hangs it from
a rope. It always opens the MuJoCo window (under mjpython on macOS); only the tests
run it headless (``G1_TWIN_HEADLESS=1``). The process
itself is ``python -m g1.twin.main serve ...``, not a user command.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from g1.core import limits
from g1.core.config import CONTROL_DT, HEAD_CAMERA_SIZE


def _parser() -> argparse.ArgumentParser:
    from g1.envs.scene import parse_object
    from g1.envs.sim import _size, _xyz
    from g1.twin.onboard import FSMS
    p = argparse.ArgumentParser(prog="g1.twin.main serve", description="the twin's DDS server (started by g1 run --sim)")
    p.add_argument("--iface", default="lo0" if sys.platform == "darwin" else "lo",
                   help="loopback interface to serve on (default lo0 on macOS, lo elsewhere)")
    p.add_argument("--domain", type=int, default=int(limits.get("twin_domain")),
                   help="DDS domain (default: limit twin_domain; the robot uses 0, which is refused)")
    p.add_argument("--headless", action="store_true", help="no viewer")
    p.add_argument("--gantry", action="store_true", help="hang the torso from a rope (for --mode gantry); starts in FSM 1")
    p.add_argument("--fsm", type=int, choices=sorted(FSMS), default=None,
                   help="the FSM it starts in (default 200 standing, 1 with --gantry)")
    p.add_argument("--scene", choices=("none", "room"), default="none")
    p.add_argument("--sim-objects", type=parse_object, nargs="*", default=[], metavar="NAME@X,Y[,Z]")
    p.add_argument("--sim-obstacle", type=_xyz, default=None, metavar="X,Y,Z")
    p.add_argument("--camera-size", type=_size, default=None, metavar="HxW",
                   help=f"head camera size (default {HEAD_CAMERA_SIZE[0]}x{HEAD_CAMERA_SIZE[1]}; the robot streams 720x1280)")
    p.add_argument("--camera-fps", type=float, default=limits.get("twin_camera_fps"))
    p.add_argument("--no-camera", action="store_true", help="do not render the head camera")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until Ctrl-C)")
    return p


def run(args: argparse.Namespace) -> int:
    import mujoco
    import mujoco.viewer
    from g1.twin.bridge import DdsBridge
    from g1.twin.core import Twin
    from g1.twin.model import HEAD_CAMERA
    from g1.twin.onboard import FSM_DAMP, FSM_MAIN
    from g1.twin.shm import FrameWriter

    if args.domain == 0:
        raise SystemExit("g1 twin refuses DDS domain 0: that is the robot's")
    fsm = args.fsm if args.fsm is not None else (FSM_DAMP if args.gantry else FSM_MAIN)
    twin = Twin(scene=args.scene, objects=args.sim_objects, obstacle=args.sim_obstacle,
                camera_size=args.camera_size, fsm=fsm, band=args.gantry, voice="twin")
    viewer = None
    if not args.headless:
        if sys.platform == "darwin" and not getattr(mujoco.viewer, "_MJPYTHON", None):
            raise SystemExit("On macOS the twin's window needs mjpython (it comes with mujoco: pip install mujoco)")
        viewer = mujoco.viewer.launch_passive(twin.model, twin.data)
    bridge = DdsBridge(twin, args.domain, args.iface)
    renderer = writer = None
    if not args.no_camera:
        h, w = args.camera_size or HEAD_CAMERA_SIZE
        renderer = mujoco.Renderer(twin.model, h, w)
        writer = FrameWriter(args.domain, h, w)
    stopping = []
    signal.signal(signal.SIGTERM, lambda *a: stopping.append(1))
    print(f"twin: serving on {args.iface}, DDS domain {args.domain}, FSM {fsm}"
          + (", gantry rope" if args.gantry else "") + (f", camera {renderer.height}x{renderer.width} "
                                                        f"@ {args.camera_fps:g} fps" if renderer else ""))
    bridge.start()
    frame_every = max(1, round(1.0 / (args.camera_fps * CONTROL_DT))) if renderer else 0
    import os
    verbose = os.environ.get("G1_VERBOSE", "") not in ("", "0")
    wall0, ticks, status_at = time.perf_counter(), 0, 0.0
    try:
        while not stopping and (args.seconds is None or twin.time < args.seconds):
            twin.step()
            ticks += 1
            if renderer is not None and ticks % frame_every == 0:
                with twin.lock:
                    renderer.update_scene(twin.data, camera=HEAD_CAMERA)
                writer.write(renderer.render())
            if viewer is not None:
                if not viewer.is_running():
                    break
                with twin.lock:
                    viewer.sync()
            lag = wall0 + twin.time - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            if verbose and twin.time - status_at >= limits.get("twin_status_period_s"):
                status_at = twin.time
                rt = twin.time / max(1e-9, time.perf_counter() - wall0)
                print(f"twin: t={twin.time:7.1f}  {twin.summary()}  lowstate {bridge.published} sent, "
                      f"arm_sdk {bridge.arm_messages} received, real-time factor {rt:.2f}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()
        print(f"twin: stopped at t={twin.time:.1f}; {twin.summary()}")
    return 0


class TwinProcess:
    """``with TwinProcess(task, mode=..., viewer=...) as twin:`` a twin serving the task's
    scene on ``twin.iface`` / ``twin.domain`` until the block ends (normally, on an error
    or on Ctrl-C). Its output is echoed, prefixed ``twin:``, so TTS lines and its summary
    reach the terminal."""

    READY = "twin: serving"

    def __init__(self, task: dict, *, mode: str = "standing", viewer: bool = False, camera: bool = True,
                 timeout: float = limits.get("twin_start_timeout_s")) -> None:
        import os
        self.iface = "lo0" if sys.platform == "darwin" else "lo"
        self.domain = 100 + os.getpid() % 100         # private: never another twin's, never the robot's 0
        exe = sys.executable
        if viewer and sys.platform == "darwin":
            from pathlib import Path
            exe = str(Path(sys.executable).with_name("mjpython"))
            if not Path(exe).exists():
                raise SystemExit(f"the twin's window needs mjpython next to {sys.executable} (pip install mujoco)")
        self.cmd = [exe, "-m", "g1.twin.main", "serve", "--iface", self.iface, "--domain", str(self.domain),
                    "--scene", task["scene"]]
        if task["objects"]:
            self.cmd += ["--sim-objects", *task["objects"]]
        if task.get("camera_size"):
            self.cmd += ["--camera-size", "x".join(str(v) for v in task["camera_size"])]
        if mode == "gantry":
            self.cmd.append("--gantry")
        if not viewer:
            self.cmd.append("--headless")
        if not camera:
            self.cmd.append("--no-camera")
        self.timeout = timeout
        self.proc = None
        self._echo = None

    def __enter__(self) -> "TwinProcess":
        import subprocess
        import threading
        from g1.core import limits
        # its own session: a Ctrl-C at the terminal reaches g1 run, whose safe return still needs the twin
        import os
        self.proc = subprocess.Popen(self.cmd, cwd=limits.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})     # its lines as they happen
        ready = threading.Event()
        lines: list[str] = []

        def echo():
            for line in self.proc.stdout:
                line = line.rstrip()
                if line.startswith(self.READY):
                    ready.set()
                if line.startswith("twin:"):
                    print(line, flush=True)
                else:
                    lines.append(line)            # its startup chatter, shown only if it fails
        self._echo = threading.Thread(target=echo, name="twin-echo", daemon=True)
        self._echo.start()
        t0 = time.monotonic()
        while not ready.wait(0.1):
            if self.proc.poll() is not None or time.monotonic() - t0 > self.timeout:
                self.__exit__()
                raise SystemExit("the twin did not start:\n" + "\n".join(lines[-20:]))
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()                 # SIGTERM: it stops its loop and prints its summary
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                self.proc.wait()
        if self._echo is not None:
            self._echo.join(timeout=2)
        self.proc = None


def main(argv: list[str] | None = None) -> int:
    from g1.envs import scene
    from g1.twin import assets
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["serve"]:                     # the process TwinProcess starts
        return run(_parser().parse_args(argv[1:]))
    p = argparse.ArgumentParser(prog="g1 twin", description="the twin's downloads")
    p.add_argument("command", choices=("fetch", "status"))
    p.add_argument("--force", action="store_true", help="fetch: again, even if present")
    args = p.parse_args(argv)
    if args.command == "fetch":
        assets.fetch(force=args.force)
        scene.fetch(force=args.force)
        print("room: " + ("complete" if not scene.missing(tuple(scene.YCB)) else f"missing {scene.missing(tuple(scene.YCB))}"))
        return 0
    code = assets.status()
    need = scene.missing(tuple(scene.YCB))
    print(f"room:   {scene.ASSETS}" + (f"  MISSING {', '.join(need)}" if need else "  ok"))
    return code or (1 if need else 0)


if __name__ == "__main__":
    raise SystemExit(main())
