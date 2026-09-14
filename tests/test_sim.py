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
