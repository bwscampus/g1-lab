import numpy as np

from config import LEFT_ARM, UPPER_BODY
from envs import CheckEnv
from motions import TPose
from policy import Obs, Policy, Segment, SegmentPolicy
from routines import Routine
from run import build_parser, run


def make_check(*extra):
    args = build_parser().parse_args(["--env", "check", "--policy", "tpose", *extra])
    return CheckEnv(args)


def test_tpose_passes_check():
    env = make_check()
    assert run(Routine(TPose()), env) is True
    assert env.violations == []
    assert env.ticks == int(round(18.0 / 0.02))


class OutOfBounds(SegmentPolicy):
    name = "oob"
    joints = LEFT_ARM
    segments = (Segment({16: 3.0}, 1.0),)   # left_shoulder_roll limit is 2.2515


def test_out_of_bounds_fails():
    env = make_check()
    assert run(OutOfBounds(), env) is False
    assert any(v.kind == "above" and v.joint == 16 for v in env.violations)


class Jump(Policy):
    name = "jump"
    joints = UPPER_BODY

    def step(self, t, obs):
        if t > 0.1:
            return None
        out = obs.q.copy()
        out[18] = 2.0 if t < 0.05 else -1.0   # 3 rad in one 20 ms tick
        return self.action(out)


def test_velocity_violation():
    env = make_check()
    assert run(Jump(), env) is False
    assert any(v.kind == "velocity" for v in env.violations)


class BadWeight(Policy):
    name = "badweight"
    joints = UPPER_BODY

    def step(self, t, obs):
        return None if t > 0.05 else self.action(obs.q.copy(), weight=1.5)


def test_weight_out_of_range():
    env = make_check()
    assert run(BadWeight(), env) is False
    assert any(v.kind == "weight" for v in env.violations)


def test_weight_zero_masks_out_of_bounds_motion_speed():
    # With weight 0 the effective command never moves, so no velocity violation,
    # but the raw target is still checked against the limits.
    class W0(Policy):
        name = "w0"
        joints = LEFT_ARM

        def step(self, t, obs):
            if t > 0.05:
                return None
            out = obs.q.copy()
            out[18] = 5.0
            return self.action(out, weight=0.0)

    env = make_check()
    assert run(W0(), env) is False
    kinds = {v.kind for v in env.violations}
    assert kinds == {"above"}
    assert np.all(env.peak_vel == 0)


def test_weight_violation_prints_no_joint_name(capsys):
    env = make_check()
    run(BadWeight(), env)
    lines = [l for l in capsys.readouterr().out.splitlines() if "weight" in l and "t=" in l]
    assert lines and all(" -  " in l and "wrist" not in l for l in lines)
