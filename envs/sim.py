"""Stage 2: MuJoCo replay in the interactive viewer.

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

macOS: the viewer must run under ``mjpython``:
    mjpython run.py --env sim --policy tpose
Pass ``--headless`` to run the physics without a window (e.g. in CI).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from camera import Camera
from config import (CONTROL_DT, HEAD_CAMERA_FOVY, HEAD_CAMERA_PITCH, HEAD_CAMERA_POS,
                    HEAD_CAMERA_SIZE, NUM_JOINTS)
from policy import Action
from envs.base import Env, EnvAbort

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


def load_model(target: tuple[float, float, float] | None = None,
               obstacle: tuple[float, float, float] | None = None):
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

    def setup(self) -> None:
        import mujoco
        import mujoco.viewer

        self.mujoco = mujoco
        self.model = load_model(self.args.sim_target, self.args.sim_obstacle)
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
        self.camera = Camera()
        self.renderer = None
        self._tick = 0
        if self.use_camera:
            self.renderer = mujoco.Renderer(self.model, *HEAD_CAMERA_SIZE)

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
        return self.camera.latest()

    def clock(self) -> float:
        return float(self.data.time)

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
        return self._q()

    def report(self) -> bool:
        q = self._q()
        x, y, yaw = self._base_pose
        print(f"sim: finished at sim time {self.data.time:.2f} s; "
              f"pelvis z = {self.data.qpos[2]:.3f} m"
              + (f"; {self.camera.count} camera frames" if self.renderer is not None else "")
              + (f"; base slid to ({x:+.2f}, {y:+.2f}) m yaw {math.degrees(yaw):+.0f} deg"
                 if self._base_moved else ""))
        return bool(np.all(np.isfinite(q)))
