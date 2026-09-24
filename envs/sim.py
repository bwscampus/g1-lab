"""MuJoCo: the stage that both runs a policy and checks it.

Every run is watched by a ``JointMonitor`` (``envs/monitor.py``): the measured
angle of every joint must stay inside its limits, as must every commanded
target, the blended command's speed, the arm_sdk weight and any base velocity.
``report()`` prints the table and fails the run on any violation, so
``--env sim --headless`` is the fast pre-flight for a new policy and the viewer
run is the same check with a window.

Loads the Menagerie ``unitree_g1`` scene (position actuators, kp=500), starts
at the ``stand`` keyframe and drives the actuators with the policy's targets,
blended with the hold pose by the arm_sdk weight exactly as the check env does.
Joints the policy does not command hold the keyframe pose.

By default the pelvis is welded to the world so an arm-only policy can be
viewed without the robot needing to balance. Pass ``--free-base`` to drop the
weld (the stiff PD on the legs keeps it standing for a while, but it will not
balance).

Walking: an ``Action.base`` velocity slides the pinned pelvis kinematically
(legs stay in the stand pose, feet drag). That rehearses a behaviour's loop and
its limits, not gait physics. With ``--free-base`` a base command aborts.

Camera: the model gets a ``head`` camera on ``torso_link`` at the D435 mount.
For policies that use the camera it is rendered offscreen every
``--camera-every`` ticks and stamped with sim time. ``--sim-target x,y,z`` adds
a red sphere for the example camera policies to look at.

Scene: ``--scene room`` builds a textured room with furniture (``scene.py``)
and ``--sim-objects mug@1.5,1.2 pencil@1.0,0.3`` places real objects in it, so
the real vision model has something real to look at. ``--camera-size 720x1280``
renders at the robot's resolution.

macOS: the viewer must run under ``mjpython``:
    mjpython run.py --env sim --policy tpose
Pass ``--headless`` to run the physics without a window (e.g. in CI).
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from camera import Camera, DirCamera, NoiseCamera
from config import (CONTROL_DT, HEAD_CAMERA_FOVY, HEAD_CAMERA_PITCH, HEAD_CAMERA_POS,
                    HEAD_CAMERA_SIZE, NUM_JOINTS)
from policy import Action
from envs.base import Env, EnvAbort
from envs.monitor import JointMonitor, TooManyViolations

_LOCAL_MENAGERIE = Path.home() / "Robotics" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
HEAD_CAMERA = "head"


def find_mjcf() -> Path:
    """Path of the G1 scene. Priority: $G1_MJCF, pip menagerie, ~/Robotics clone."""
    env_path = os.environ.get("G1_MJCF")
    if env_path:
        return Path(env_path)
    try:
        import mujoco_menagerie
        p = Path(mujoco_menagerie.__file__).parent / "unitree_g1" / "scene.xml"
        if p.exists():
            return p
    except ImportError:
        pass
    if _LOCAL_MENAGERIE.exists():
        return _LOCAL_MENAGERIE
    raise FileNotFoundError(
        "No G1 MJCF found. Set G1_MJCF=/path/to/unitree_g1/scene.xml or pip install mujoco-menagerie")


def _size(text: str) -> tuple[int, int]:
    try:
        h, w = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected HxW, e.g. 720x1280") from None
    return h, w


def load_model(target: tuple[float, float, float] | None = None,
               obstacle: tuple[float, float, float] | None = None, *,
               scene: str = "none", objects=(), camera_size: tuple[int, int] | None = None):
    """Compiled models are cached: a test suite that builds dozens of envs pays
    the MJCF compile once per distinct scene. Nothing mutates the model."""
    return _load_model(target, obstacle, scene, tuple(tuple(o) for o in objects), camera_size)


@functools.lru_cache(maxsize=8)
def _load_model(target, obstacle, scene, objects, camera_size):
    """Compile the G1 scene with the head camera added on torso_link (pose from
    the URDF's d435_joint; MuJoCo cameras look along -z with y up, hence the
    xyaxes) and, optionally, a red sphere at ``target`` for camera policies and
    a chair-sized box at ``obstacle`` for a vision model to describe."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(find_mjcf()))
    p = HEAD_CAMERA_PITCH
    spec.body("torso_link").add_camera(name=HEAD_CAMERA, pos=list(HEAD_CAMERA_POS),
                                      xyaxes=[0, -1, 0, math.sin(p), 0, math.cos(p)],
                                      fovy=HEAD_CAMERA_FOVY)
    if target is not None:
        body = spec.worldbody.add_body(name="target", pos=list(target))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], rgba=[1, 0, 0, 1],
                      contype=0, conaffinity=0)
    if obstacle is not None:
        body = spec.worldbody.add_body(name="obstacle", pos=list(obstacle))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.225],
                      rgba=[0.5, 0.5, 0.5, 1])           # neutral grey (no red bias); collides
    if camera_size is not None:
        spec.visual.global_.offheight = max(spec.visual.global_.offheight, camera_size[0])
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, camera_size[1])
    if scene == "room":
        from scene import build_room
        build_room(spec, objects, camera_size)
    elif objects:
        raise SystemExit("--sim-objects needs --scene room")
    return spec.compile()


def _xyz(text: str) -> tuple[float, float, float]:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return parts[0], parts[1], parts[2]


class SimEnv(Env):
    name = "sim"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("sim")
        g.add_argument("--headless", action="store_true", help="run physics without the viewer")
        g.add_argument("--free-base", action="store_true",
                       help="do not weld the pelvis to the world")
        g.add_argument("--realtime", type=float, default=None,
                       help="playback speed multiplier; default 1.0 with the viewer, unpaced when "
                            "--headless (give it explicitly to pace a headless run, e.g. for --vision api)")
        g.add_argument("--hold-end", type=float, default=2.0,
                       help="seconds to keep the viewer open after the policy finishes")
        g.add_argument("--camera-every", type=int, default=3,
                       help="render the head camera every N ticks (default 3, ~16 Hz)")
        g.add_argument("--sim-target", type=_xyz, default=None, metavar="X,Y,Z",
                       help="add a red 10 cm sphere at this world position, e.g. 1.0,0.5,0.6")
        g.add_argument("--sim-obstacle", type=_xyz, default=None, metavar="X,Y,Z",
                       help="add a chair-sized box (0.4x0.4x0.45 m, centre) for a vision model to "
                            "describe, e.g. 1.2,0,0.225")
        g.add_argument("--scene", choices=("none", "room"), default="none",
                       help="room: textured walls/floor, table and chairs (assets via python -m scene fetch)")
        from scene import parse_object
        g.add_argument("--sim-objects", type=parse_object, nargs="*", default=[], metavar="NAME@X,Y[,Z]",
                       help="objects to place in the room, e.g. mug@1.5,1.2 pencil@1.0,0.3 (z: on the floor)")
        g.add_argument("--camera-size", type=_size, default=None, metavar="HxW",
                       help="head camera render size (default 480x640; the robot streams 720x1280)")
        g.add_argument("--camera-dir", type=Path, default=None, metavar="DIR",
                       help="feed the image files in DIR (sorted by name) instead of rendering")
        g.add_argument("--camera-noise", action="store_true",
                       help="feed random frames instead of rendering, to fuzz a camera policy")
        g.add_argument("--camera-fps", type=float, default=15.0,
                       help="frame rate of --camera-dir / --camera-noise (default 15)")
        c = parser.add_argument_group("checks (sim)")
        c.add_argument("--margin", type=float, default=0.05,
                       help="safety margin inside the joint limits, rad (default 0.05)")
        c.add_argument("--max-vel", type=float, default=4.0,
                       help="max allowed joint speed of the effective command, rad/s (default 4.0)")
        c.add_argument("--max-violations", type=int, default=20,
                       help="stop the run after this many violations (default 20)")

    def setup(self) -> None:
        import mujoco
        import mujoco.viewer

        self.mujoco = mujoco
        self.camera_size = self.args.camera_size or HEAD_CAMERA_SIZE
        self.model = load_model(self.args.sim_target, self.args.sim_obstacle, scene=self.args.scene,
                                objects=self.args.sim_objects, camera_size=self.args.camera_size)
        self.data = mujoco.MjData(self.model)
        if self.model.nu != NUM_JOINTS:
            raise RuntimeError(f"expected {NUM_JOINTS} actuators, model has {self.model.nu}")
        # Menagerie puts the free joint first, so joint i sits at qpos[7 + i].
        self.qpos_idx = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
                                  for i in range(self.model.nu)])
        self.qvel_idx = np.array([self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
                                  for i in range(self.model.nu)])
        self.substeps = max(1, int(round(CONTROL_DT / self.model.opt.timestep)))

        self._pin_base = not self.args.free_base
        self.viewer = None
        if not self.args.headless:
            if sys.platform == "darwin" and not getattr(mujoco.viewer, "_MJPYTHON", None):
                raise SystemExit("On macOS the viewer needs mjpython:\n"
                                 "    mjpython run.py --env sim ...\n"
                                 "or pass --headless.")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.monitor = JointMonitor(margin=self.args.margin, max_vel=self.args.max_vel,
                                    max_violations=self.args.max_violations)
        self.camera = Camera()
        self.replay = None
        self.renderer = None
        self._tick = 0
        if self.use_camera:
            if self.args.camera_dir is not None:
                self.replay = DirCamera(self.args.camera_dir, fps=self.args.camera_fps)
            elif self.args.camera_noise:
                self.replay = NoiseCamera(fps=self.args.camera_fps)
            if self.replay is not None:
                self.replay.start()          # a recorded or fuzzed feed replaces the render
            else:
                self.renderer = mujoco.Renderer(self.model, *self.camera_size)

    def teardown(self) -> None:
        if self.viewer is not None:
            t_end = time.time() + self.args.hold_end
            while self.viewer.is_running() and time.time() < t_end:
                self._physics_tick()
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()

    def reset(self) -> np.ndarray:
        m, d = self.model, self.data
        key = self.mujoco.mj_name2id(m, self.mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.mujoco.mj_resetDataKeyframe(m, d, key)
        self._base_qpos = d.qpos[:7].copy()
        self._base_qpos0 = self._base_qpos.copy()
        self._base_pose = np.zeros(3)      # x, y, yaw of the slid base
        self._base_moved = False
        self.hold = d.qpos[self.qpos_idx].copy()
        self.monitor.reset(hold=self.hold)
        d.ctrl[:] = self.hold
        self.mujoco.mj_forward(m, d)
        if self.viewer is not None:
            self.viewer.sync()
        self._render()
        self._wall = time.time()
        return self._q()

    def _q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def _physics_tick(self) -> None:
        d = self.data
        if self._pin_base:
            # Pin the pelvis by overriding the free-joint state each substep
            # (no model edit needed, so the same MJCF serves both modes).
            d.qpos[:7] = self._base_qpos
            d.qvel[:6] = 0.0
        self.mujoco.mj_step(self.model, d)

    def _slide_base(self, base) -> None:
        """Integrate a base velocity into the pinned pelvis pose."""
        vx, vy, vyaw = base
        x, y, yaw = self._base_pose
        yaw += vyaw * CONTROL_DT
        x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * CONTROL_DT
        y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * CONTROL_DT
        self._base_pose = np.array([x, y, yaw])
        q0 = self._base_qpos0
        self._base_qpos[0] = q0[0] + x
        self._base_qpos[1] = q0[1] + y
        rot = np.zeros(4)
        self.mujoco.mju_axisAngle2Quat(rot, np.array([0.0, 0.0, 1.0]), yaw)
        quat = np.zeros(4)
        self.mujoco.mju_mulQuat(quat, rot, q0[3:7])
        self._base_qpos[3:7] = quat
        self._base_moved = True

    def _render(self) -> None:
        if self.renderer is None:
            return
        self.renderer.update_scene(self.data, camera=HEAD_CAMERA)
        self.camera.publish(self.renderer.render(), self.data.time)

    def frame(self):
        if self.replay is not None:
            return self.replay.poll(self.clock())
        return self.camera.latest()

    def clock(self) -> float:
        return float(self.data.time)

    @property
    def can_walk(self) -> bool:
        return not self.args.free_base      # known before setup(): the runner gates on it early

    def base_pose(self):
        return tuple(float(v) for v in self._base_pose)

    def step(self, action: Action) -> np.ndarray:
        if self.viewer is not None and not self.viewer.is_running():
            raise KeyboardInterrupt("viewer closed")
        j = list(action.joints)
        ctrl = self.hold.copy()
        ctrl[j] = (1.0 - action.weight) * self.hold[j] + action.weight * action.q[j]
        self.data.ctrl[:] = ctrl
        if action.base is not None:
            if not self._pin_base:
                raise EnvAbort("sim cannot walk: base commands need the pinned base (drop --free-base)")
            self._slide_base(action.base)
        for _ in range(self.substeps):
            self._physics_tick()
        self._tick += 1
        if self._tick % max(1, self.args.camera_every) == 0:
            self._render()
        if self.viewer is not None:
            self.viewer.sync()
        realtime = self.args.realtime
        if realtime is None and self.viewer is not None:
            realtime = 1.0
        if realtime is not None:
            # pace to wall-clock
            self._wall += CONTROL_DT / max(realtime, 1e-6)
            lag = self._wall - time.time()
            if lag > 0:
                time.sleep(lag)
        q = self._q()
        try:
            self.monitor.observe(q, action)
        except TooManyViolations as e:
            raise EnvAbort(str(e)) from None
        return q

    # the monitor is the run's verdict; forward what callers and tests read
    @property
    def violations(self):
        return self.monitor.violations

    @property
    def ticks(self) -> int:
        return self.monitor.ticks

    @property
    def q_min(self):
        return self.monitor.q_min

    @property
    def q_max(self):
        return self.monitor.q_max

    @property
    def cmd_min(self):
        return self.monitor.cmd_min

    @property
    def cmd_max(self):
        return self.monitor.cmd_max

    @property
    def peak_vel(self):
        return self.monitor.peak_vel

    @property
    def base_path(self) -> float:
        return self.monitor.base_path

    @property
    def base_ticks(self) -> int:
        return self.monitor.base_ticks

    @property
    def base_peak(self):
        return self.monitor.base_peak

    def report(self) -> bool:
        q = self._q()
        x, y, yaw = self._base_pose
        frames = self.camera.count if self.renderer is not None else (
            self.replay.count if self.replay is not None else 0)
        print(f"\nsim: finished at sim time {self.data.time:.2f} s; "
              f"pelvis z = {self.data.qpos[2]:.3f} m"
              + (f"; {frames} camera frames" if frames else "")
              + (f"; base slid to ({x:+.2f}, {y:+.2f}) m yaw {math.degrees(yaw):+.0f} deg"
                 if self._base_moved else ""))
        return self.monitor.report("sim") and bool(np.all(np.isfinite(q)))
