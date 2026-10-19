"""The twin's MuJoCo model: unitree_mujoco's 29-DoF G1 on its flat scene, with
the head camera added on ``torso_link`` at the D435 mount and, optionally, an
obstacle or the textured room (``envs/scene.py``).

The model has torque motors (``ctrlrange`` = the real torque limits) in DDS
order, so ``ctrl`` is the motor torque the onboard controller computes; nothing
here is a position servo. Compiled models are cached per scene: a test suite
pays the compile once per distinct scene. Nothing mutates a compiled model.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from g1.core import limits
from g1.core.config import HEAD_CAMERA_FOVY, HEAD_CAMERA_PITCH, HEAD_CAMERA_POS, JOINT_NAMES, NUM_JOINTS
from g1.twin import assets

HEAD_CAMERA = "head"
PHYSICS_DT = limits.get("twin_physics_dt_s")


def load_model(obstacle: tuple[float, float, float] | None = None, *,
               scene: str = "none", objects=(), camera_size: tuple[int, int] | None = None):
    return _load_model(obstacle, scene, tuple(tuple(o) for o in objects), camera_size)


@functools.lru_cache(maxsize=8)
def _load_model(obstacle, scene, objects, camera_size):
    """Compile the scene. MuJoCo cameras look along -z with y up, hence the xyaxes."""
    import mujoco

    assets.require()
    spec = mujoco.MjSpec.from_file(str(assets.SCENE_XML))
    spec.option.timestep = PHYSICS_DT
    p = HEAD_CAMERA_PITCH
    spec.body("torso_link").add_camera(name=HEAD_CAMERA, pos=list(HEAD_CAMERA_POS),
                                      xyaxes=[0, -1, 0, math.sin(p), 0, math.cos(p)],
                                      fovy=HEAD_CAMERA_FOVY)
    if obstacle is not None:
        body = spec.worldbody.add_body(name="obstacle", pos=list(obstacle))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.225],
                      rgba=[0.5, 0.5, 0.5, 1])           # neutral grey; collides
    if camera_size is not None:
        spec.visual.global_.offheight = max(spec.visual.global_.offheight, camera_size[0])
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, camera_size[1])
    if scene == "room":
        from g1.envs.scene import build_room
        build_room(spec, objects, camera_size)
    elif objects:
        raise SystemExit("a task's objects need \"scene\": \"room\"")
    model = spec.compile()
    check(model)
    return model


def check(model) -> None:
    """The twin relies on these facts of the model; a different file fails here, not mid-run."""
    if model.nu != NUM_JOINTS:
        raise RuntimeError(f"expected {NUM_JOINTS} motors, the model has {model.nu}")
    names = tuple(model.actuator(i).name for i in range(model.nu))
    if names != JOINT_NAMES:
        raise RuntimeError(f"motor order differs from the DDS order: {names}")
    qpos = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
    if not np.array_equal(qpos, 7 + np.arange(NUM_JOINTS)):
        raise RuntimeError("expected the free joint first and joint i at qpos[7 + i]")
