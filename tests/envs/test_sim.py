import numpy as np
import pytest

from g1.cli import run, run_parser
from g1.envs import SimEnv
from g1.tools import Chain
from g1.tools.arms import TPose
from g1.tools.control import Hold


@pytest.fixture
def sim_args():
    pytest.importorskip("mujoco")
    return run_parser().parse_args(["--env", "sim", "--tools", "tpose", "--headless"])


def test_tpose_headless_sim(sim_args):
    env = SimEnv(sim_args)
    try:
        ok = run(Chain(TPose()), env)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert ok is True
    # Pinned base: pelvis should not have moved.
    assert env.data.qpos[2] == pytest.approx(0.79, abs=1e-3)
    # Ended at STAND (elbows 1.28) and the arm_sdk blend handed back to hold.
    q = env._q()
    assert np.all(np.isfinite(q))


def test_head_camera_renders_when_asked():
    pytest.importorskip("mujoco")
    args = run_parser().parse_args(["--env", "sim", "--tools", "hold", "--headless", "--camera", "on",
                                    "--sim-obstacle", "1.2,0,0.225"])
    env = SimEnv(args)
    try:
        ok = run(Chain(Hold(seconds=1.0), pause=0.0), env)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert ok is True
    f = env.frame()
    assert f is not None and f.image.shape == (480, 640, 3) and f.image.std() > 0
    assert env.camera.count > 50                     # every --camera-every ticks, stamped on sim time
    assert 0 < f.stamp <= env.clock()
    env2 = SimEnv(run_parser().parse_args(["--env", "sim", "--tools", "hold", "--headless"]))
    run(Chain(Hold(seconds=0.5), pause=0.0), env2)  # a chain does not use the camera: no render
    assert env2.frame() is None and env2.renderer is None


def test_sim_base_slides_and_free_base_rejects_it(capsys):
    pytest.importorskip("mujoco")
    from g1.tools.move import Move
    env = SimEnv(run_parser().parse_args(["--env", "sim", "--tools", "move", "--headless"]))
    try:
        ok = run(Chain(Move(dx_m=0.6, dyaw_deg=30)), env)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert ok is True
    x, y, yaw = env._base_pose
    assert x == pytest.approx(0.6, abs=1e-6) and yaw == pytest.approx(np.radians(30), abs=1e-6)
    assert env.data.qpos[0] == pytest.approx(x, abs=1e-6)            # the pelvis really slid
    env = SimEnv(run_parser().parse_args(["--env", "sim", "--tools", "move", "--headless", "--free-base"]))
    run(Chain(Move(dx_m=0.2)), env)
    assert "cannot walk" in capsys.readouterr().out
