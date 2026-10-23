import numpy as np
import pytest

from g1.core import config


def test_joint_table_is_consistent():
    assert len(config.JOINTS) == 29
    assert np.all(config.JOINT_LO < config.JOINT_HI)
    assert config.UPPER_BODY == list(range(12, 29))
    assert np.all(config.STAND_Q >= config.JOINT_LO) and np.all(config.STAND_Q <= config.JOINT_HI)


def test_joint_table_matches_the_twin_model():
    pytest.importorskip("mujoco")
    from g1.twin.model import load_model

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
    assert [m.actuator(i).name for i in range(m.nu)] == list(config.JOINT_NAMES)    # motors in DDS order
