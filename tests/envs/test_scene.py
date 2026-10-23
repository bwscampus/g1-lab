import numpy as np
import pytest

from g1.envs.scene import OBJECTS, TABLE_TOP_Z, YCB, build_room, missing, parse_object


def test_parse_object():
    assert parse_object("mug@1.5,1.2") == ("mug", 1.5, 1.2, None)
    assert parse_object("pencil@-0.5,0.25,0.79") == ("pencil", -0.5, 0.25, 0.79)
    assert parse_object("chair@2.6,1.6") == ("chair", 2.6, 1.6, None)
    assert parse_object("laptop@1.75,-1.5,0.77") == ("laptop", 1.75, -1.5, 0.77)
    for bad in ("mug", "mug@1", "sofa@1,2", "mug@a,b"):
        with pytest.raises(Exception):
            parse_object(bad)
    assert set(YCB) < set(OBJECTS) and {"pencil", "chair", "laptop"} <= set(OBJECTS)


def test_room_needs_assets_message(tmp_path, monkeypatch):
    from g1.envs import scene
    monkeypatch.setattr(scene, "ASSETS", tmp_path)
    assert missing() and missing(["mug"])[-1] == "ycb:mug"
    mujoco = pytest.importorskip("mujoco")
    with pytest.raises(FileNotFoundError, match="g1 twin fetch"):
        build_room(mujoco.MjSpec(), [("mug", 1.0, 0.0, None)])


@pytest.mark.skipif(bool(missing(["mug"])), reason="room assets not fetched (g1 scene fetch mug)")
def test_room_renders_from_the_head_camera():
    pytest.importorskip("mujoco")
    from g1.cli import run_settings
    from g1.envs import SimEnv
    args = run_settings({"scene": "room", "objects": ["mug@0.8,0.25", "pencil@1.0,-0.3"], "camera_size": (720, 1280)})
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
        from tests.doubles import red_blob
        blob = red_blob(f.image, min_fraction=0.0002)                     # the mug is small and dark red
        assert blob is not None
        u, v, frac = blob
        assert u < 0 and v > 0 and frac < 0.02                            # left of centre, on the floor, nothing else red


def test_objects_need_the_room():
    pytest.importorskip("mujoco")
    from g1.cli import run_settings
    from g1.envs import SimEnv
    args = run_settings({"objects": ["pencil@1,0"]})
    with pytest.raises(SystemExit, match='"scene": "room"'):
        SimEnv(args).setup()


@pytest.mark.skipif(bool(missing()), reason="room assets not fetched (g1 twin fetch)")
def test_a_laptop_stands_on_the_table_and_the_twin_starts_where_the_task_says():
    mujoco = pytest.importorskip("mujoco")
    from g1.twin.core import Twin
    twin = Twin(scene="room", objects=[parse_object(f"laptop@1.75,-1.5,{TABLE_TOP_Z}")], start=(1.2, -0.95, 30))
    m, d = twin.model, twin.data
    g = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    base, screen, top = g("laptop1_base"), g("laptop1_screen"), g("table_top")
    assert min(base, screen, g("laptop1_display"), top) >= 0
    assert abs(d.geom_xpos[base][2] - m.geom_size[base][2] - TABLE_TOP_Z) < 1e-6          # base-down on the top
    assert abs(d.geom_xpos[top][2] + m.geom_size[top][2] - TABLE_TOP_Z) < 1e-6            # the top's surface
    assert d.geom_xpos[screen][0] > d.geom_xpos[base][0] and d.geom_xpos[screen][2] > TABLE_TOP_Z + 0.08   # hinge away, open
    x, y, yaw = twin.truth()
    assert abs(x - 1.2) < 0.15 and abs(y + 0.95) < 0.15 and abs(np.degrees(yaw) - 30) < 3   # settled where it started
    assert twin.base_pose() == pytest.approx((0, 0, 0), abs=1e-9)                            # measured from there
    assert twin.pelvis_z > 0.7                                                              # standing
    from tests.doubles import red_blob
    r = mujoco.Renderer(m, 240, 320)
    r.update_scene(d, camera="head")
    assert red_blob(r.render()) is None                                                     # nothing red for the double
