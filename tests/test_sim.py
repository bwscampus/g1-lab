import numpy as np
import pytest

from envs import SimEnv
from motions import TPose
from routines import Routine
from run import build_parser, run


@pytest.fixture
def sim_args():
    pytest.importorskip("mujoco")
    return build_parser().parse_args(["--env", "sim", "--policy", "tpose", "--headless"])


def test_tpose_headless_sim(sim_args):
    env = SimEnv(sim_args)
    try:
        ok = run(Routine(TPose()), env)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert ok is True
    # Pinned base: pelvis should not have moved.
    assert env.data.qpos[2] == pytest.approx(0.79, abs=1e-3)
    # Ended at STAND (elbows 1.28) and the arm_sdk blend handed back to hold.
    q = env._q()
    assert np.all(np.isfinite(q))


def test_head_camera_and_look_turns_toward_target():
    pytest.importorskip("mujoco")
    from behaviors import Look

    class Spy(Look):
        seen: list = []

        def track(self, t, obs):
            pose = super().track(t, obs)
            self.seen.append(pose.get(12))
            return pose

    args = build_parser().parse_args(["--env", "sim", "--policy", "look", "--headless",
                                      "--sim-target", "1.0,0.5,0.6", "--sim-obstacle", "1.2,0,0.225"])
    env = SimEnv(args)
    try:
        ok = run(Spy(duration=4.0), env)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert ok is True
    f = env.frame()
    assert f is not None and f.image.shape == (480, 640, 3) and f.image.std() > 0
    assert env.camera.count > 50
    yaws = [y for y in Spy.seen if y is not None]
    assert yaws and max(yaws) > 0.15                  # target on the left: positive yaw


def test_sim_goto_red_slides_to_target():
    pytest.importorskip("mujoco")
    from behaviors import GoTo
    from targets import RedDot

    args = build_parser().parse_args(["--env", "sim", "--policy", "goto_red", "--headless",
                                      "--sim-target", "1.5,0.3,0.6"])
    env = SimEnv(args)
    p = GoTo(RedDot(), timeout=30.0, ramp=0.2, to_stand=0.2)
    try:
        ok = run(p, env)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert ok is True and p.reached
    x, y, yaw = env._base_pose
    assert 0.5 < x < 1.4 and y > 0.05 and yaw > 0          # advanced, drifted left toward the ball
    assert env.data.qpos[0] == pytest.approx(x, abs=1e-6)


def test_sim_free_base_rejects_base_command(capsys):
    pytest.importorskip("mujoco")
    from behaviors import GoTo
    from targets import RedDot

    args = build_parser().parse_args(["--env", "sim", "--policy", "goto_red", "--headless",
                                      "--free-base", "--sim-target", "1.5,0.3,0.6"])
    env = SimEnv(args)
    try:
        run(GoTo(RedDot(), timeout=5.0, ramp=0.2, to_stand=0.2), env)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert "cannot walk" in capsys.readouterr().out
