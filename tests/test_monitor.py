import numpy as np
import pytest

from config import BASE_VEL_MAX, LEFT_ARM, STAND_Q, UPPER_BODY
from envs.monitor import JointMonitor, TooManyViolations, Violation
from policy import Action, Obs, Policy, Segment, SegmentPolicy
from routines import Routine
from run import run
from skills import TPose
from tests.doubles import sim_env


def act(q, joints=UPPER_BODY, **kw):
    return Action(q, list(joints), **kw)


# -- the monitor on its own ---------------------------------------------------------

def test_measured_angles_are_the_gate():
    m = JointMonitor()
    m.observe(STAND_Q)
    assert m.violations == []
    q = STAND_Q.copy()
    q[18] = 2.5                                   # left elbow limit is 2.094
    m.observe(q)
    v, = m.violations
    assert v.joint == 18 and v.kind == "above" and v.value == 2.5 and v.limit == pytest.approx(2.044, abs=1e-3)
    assert m.q_max[18] == 2.5 and m.q_min[18] == STAND_Q[18]
    q[18] = -1.5
    m.observe(q)
    assert m.violations[-1].kind == "below"
    assert 18 in m.rows() and str(m.violations[0]).startswith("t=  0.02s  left_elbow")


def test_commanded_targets_velocity_and_weight():
    m = JointMonitor()
    q = STAND_Q.copy()
    m.observe(q, act(q))
    assert m.violations == [] and m.commanded == set(UPPER_BODY)
    bad = q.copy()
    bad[16] = 3.0                                  # left shoulder roll limit is 2.2515
    m.observe(q, act(bad))
    kinds = {v.kind for v in m.violations}
    assert kinds == {"target_above", "velocity"}   # asking for it, and the jump to get there
    assert m.cmd_max[16] == 3.0
    before = len(m.violations)
    m.observe(q, act(q, weight=1.5))
    weight = [v for v in m.violations[before:] if v.kind == "weight"]
    assert len(weight) == 1 and weight[0].joint == -1
    assert " -  " in str(weight[0])                 # no joint name on a whole-body violation


def test_velocity_uses_the_blended_command():
    m = JointMonitor(max_vel=4.0)
    q = STAND_Q.copy()
    far = q.copy()
    far[18] = q[18] + 0.5                          # 25 rad/s if it applied in full
    m.observe(q, act(far, weight=0.0))             # ... but the weight hands the joint back
    assert m.violations == [] and m.peak_vel[18] == 0.0
    m.observe(q, act(far, weight=1.0))
    assert m.violations[-1].kind == "velocity" and m.peak_vel[18] == pytest.approx(25.0)


def test_base_limits_and_kinematic_pose():
    m = JointMonitor()
    q = STAND_Q.copy()
    for _ in range(50):                            # 1 s at 0.2 m/s
        m.observe(q, act(q, base=(0.2, 0.0, 0.0)))
    assert m.violations == [] and m.base_ticks == 50
    assert m.base_path == pytest.approx(0.2) and m.base_pose()[0] == pytest.approx(0.2)
    m.observe(q, act(q, base=(1.0, 0.0, 2.0)))
    assert {v.kind for v in m.violations} == {"base_vx", "base_vyaw"}
    assert m.base_peak[0] == 1.0


def test_strict_stops_early_and_report_only_does_not(capsys):
    m = JointMonitor(max_violations=3)
    q = STAND_Q.copy()
    q[18] = 2.5
    with pytest.raises(TooManyViolations):
        for _ in range(10):
            m.observe(q)
    assert len(m.violations) == 3 and m.report("sim") is False
    assert "FAIL" in capsys.readouterr().out

    soft = JointMonitor(strict=False, gate_targets=False, gate_velocity=False, gate_base=False)
    for _ in range(15):
        soft.observe(q)                            # never raises
    assert len(soft.violations) == 15 and soft.report("robot") is True
    out = capsys.readouterr().out
    assert "WARNING" in out and "and 3 more" in out             # violations are summarised
    soft2 = JointMonitor(strict=False, gate_targets=False, gate_velocity=False, gate_base=False)
    soft2.observe(q, act(q, weight=5.0, base=(9.0, 0, 0)))
    assert soft2.violations and all(v.kind == "above" for v in soft2.violations)   # gates off


# -- through a real sim run -----------------------------------------------------------

def test_tpose_passes_in_sim():
    env = sim_env()
    r = Routine(TPose())
    assert run(r, env) is True
    assert env.violations == [] and env.ticks == round(r.duration / 0.02)
    assert env.q_max[16] == pytest.approx(1.57, abs=0.05)        # measured tracks the command
    assert env.cmd_max[16] == pytest.approx(1.57)


class OutOfBounds(SegmentPolicy):
    name = "oob"
    joints = LEFT_ARM
    segments = (Segment({16: 3.0}, 3.0),)         # left_shoulder_roll limit is 2.2515


def test_out_of_bounds_fails_in_sim():
    env = sim_env()
    assert run(OutOfBounds(), env) is False
    kinds = {v.kind for v in env.violations}
    assert "target_above" in kinds                 # the policy asked for it
    assert any(v.joint == 16 for v in env.violations)


def test_physics_violation_needs_no_command():
    """A joint the policy never commands can still leave its bounds: with the
    base free the legs collapse under gravity, and only the measured angles see it."""
    env = sim_env("--free-base")
    p = SegmentPolicy([Segment({}, 6.0)], joints=[])
    ok = run(p, env)
    assert env.monitor.commanded == set()                         # nothing was commanded at all
    assert np.all(np.isfinite(env.q_min)) and np.all(np.isfinite(env.q_max))   # every joint watched
    assert ok is (env.violations == [])


class Jump(Policy):
    name = "jump"
    joints = UPPER_BODY

    def step(self, t, obs):
        if t > 0.1:
            return None
        out = obs.q.copy()
        out[18] = 2.0 if t < 0.05 else -1.0        # 3 rad in one 20 ms tick
        return self.action(out)


def test_velocity_violation():
    env = sim_env()
    assert run(Jump(), env) is False
    assert any(v.kind == "velocity" for v in env.violations)


class BadWeight(Policy):
    name = "badweight"
    joints = UPPER_BODY

    def step(self, t, obs):
        return None if t > 0.05 else self.action(obs.q.copy(), weight=1.5)


def test_weight_out_of_range(capsys):
    env = sim_env()
    assert run(BadWeight(), env) is False
    assert any(v.kind == "weight" for v in env.violations)
    lines = [l for l in capsys.readouterr().out.splitlines() if "weight" in l and "t=" in l]
    assert lines and all(" -  " in l and "wrist" not in l for l in lines)
