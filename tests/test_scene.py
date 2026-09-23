import numpy as np
import pytest

from scene import OBJECTS, YCB, build_room, missing, parse_object


def test_parse_object():
    assert parse_object("mug@1.5,1.2") == ("mug", 1.5, 1.2, None)
    assert parse_object("pencil@-0.5,0.25,0.79") == ("pencil", -0.5, 0.25, 0.79)
    for bad in ("mug", "mug@1", "chair@1,2", "mug@a,b"):
        with pytest.raises(Exception):
            parse_object(bad)
    assert set(YCB) < set(OBJECTS) and "pencil" in OBJECTS


def test_room_needs_assets_message(tmp_path, monkeypatch):
    import scene
    monkeypatch.setattr(scene, "ASSETS", tmp_path)
    assert missing() and missing(["mug"])[-1] == "ycb:mug"
    mujoco = pytest.importorskip("mujoco")
    with pytest.raises(FileNotFoundError, match="python -m scene fetch"):
        build_room(mujoco.MjSpec(), [("mug", 1.0, 0.0, None)])


@pytest.mark.skipif(bool(missing(["mug"])), reason="room assets not fetched (python -m scene fetch mug)")
def test_room_renders_from_the_head_camera():
    pytest.importorskip("mujoco")
    from envs import SimEnv
    from run import build_parser
    args = build_parser().parse_args(["--env", "sim", "--policy", "look", "--headless", "--scene", "room",
                                      "--sim-objects", "mug@0.8,0.25", "pencil@1.0,-0.3", "--camera-size", "720x1280"])
    env = SimEnv(args)
    env.use_camera = True
    with env:
        env.reset()
        f = env.frame()
        assert f.image.shape == (720, 1280, 3) and f.image.std() > 20
        m = env.model
        import mujoco
        mug = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mug1")
        assert mug >= 0 and abs(m.body_pos[mug][2]) < 0.01              # YCB meshes are authored base-down at z = 0
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall_+x_lintel") >= 0
        # the red mug is in the lower half of the image, on the wood-coloured floor
        from vision import red_blob
        blob = red_blob(f.image, min_fraction=0.0002)                     # the mug is small and dark red
        assert blob is not None
        u, v, frac = blob
        assert u < 0 and v > 0 and frac < 0.02                            # left of centre, on the floor, nothing else red


def test_objects_need_the_room():
    pytest.importorskip("mujoco")
    from envs import SimEnv
    from run import build_parser
    args = build_parser().parse_args(["--env", "sim", "--policy", "look", "--headless", "--sim-objects", "pencil@1,0"])
    with pytest.raises(SystemExit, match="--scene room"):
        SimEnv(args).setup()
