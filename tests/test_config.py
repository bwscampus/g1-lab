import numpy as np
import pytest

import config


def test_joint_table_is_consistent():
    assert len(config.JOINTS) == 29
    assert np.all(config.JOINT_LO < config.JOINT_HI)
    assert config.UPPER_BODY == list(range(12, 29))
    assert np.all(config.STAND_Q >= config.JOINT_LO) and np.all(config.STAND_Q <= config.JOINT_HI)


def test_joint_table_matches_menagerie():
    pytest.importorskip("mujoco")
    from envs.sim import load_model

    try:
        m = load_model()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert m.nu == 29
    for j in config.JOINTS:
        jid = m.actuator_trnid[j.index, 0]
        assert m.joint(jid).name == j.name + "_joint"
        lo, hi = m.jnt_range[jid]
        assert lo == pytest.approx(j.lo, abs=1e-4), j.name
        assert hi == pytest.approx(j.hi, abs=1e-4), j.name
    key = m.key("stand")
    assert np.allclose(key.qpos[7:], config.STAND_Q)
