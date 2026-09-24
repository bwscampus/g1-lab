import numpy as np
import pytest

from config import LEFT_ARM, STAND_Q, UPPER_BODY
from skills import SixSeven, TPose
from policy import Obs, Segment, SegmentPolicy
from skills import Skill
from routines import POLICIES, ROUTINES, Routine, Selector, build_policy
from run import build_parser, main, run
from behaviors import Look


from tests.doubles import sim_env as make_check


def actions(policy):
    policy.reset(Obs(STAND_Q))
    out, n = [], 0
    while (a := policy.step(n * policy.dt, Obs(STAND_Q))) is not None:
        out.append(a)
        n += 1
    return out


def test_tpose_routine_passes_check():
    env = make_check()
    r = Routine(TPose())
    assert r.duration == 21.0                            # 5 s takeover + 11 s tpose + 5 s handback
    assert run(r, env) is True
    assert env.ticks == round(21.0 / 0.02)
    acts = actions(r)
    assert acts[500].q[16] == pytest.approx(1.57)        # t = 10 s, holding the T-pose


def test_sixseven_routine_passes_check():
    env = make_check()
    assert run(Routine(SixSeven()), env) is True
    assert env.ticks == round((10.0 + SixSeven().duration) / 0.02)


def test_demo_routine_continuous():
    env = make_check()
    r = ROUTINES["demo"]()
    assert run(r, env) is True
    assert not [v for v in env.violations if v.kind == "velocity"]
    assert env.ticks == round(r.duration / 0.02)


def test_bookend_weights():
    acts = actions(Routine(TPose(), SixSeven(reps=1)))
    assert acts[0].weight == 0.0
    assert acts[-1].weight < 0.01
    inside = acts[int(5.0 / 0.02) + 1: -int(5.0 / 0.02) - 1]   # strictly between the bookends
    assert all(a.weight == 1.0 for a in inside)


def test_pause_holds_pose():
    r1 = Routine(TPose(), SixSeven(reps=1), pause=1.0)
    r0 = Routine(TPose(), SixSeven(reps=1), pause=0.0)
    assert r1.duration - r0.duration == pytest.approx(1.0)
    assert not any("pause" in s.label for s in r0.segments)
    acts = actions(r1)
    t0 = 5.0 + TPose().duration                                  # pause starts after tpose
    i0 = int(round(t0 / 0.02))
    window = acts[i0 + 1: i0 + int(1.0 / 0.02)]
    assert all(np.array_equal(a.q, window[0].q) for a in window)


def test_start_goal_is_reserved_for_the_takeover():
    class Bad(Skill):
        name = "bad"

        def segments(self):
            return (Segment("start", 1.0),)

    with pytest.raises(ValueError):
        Routine(Bad())


def test_goal_outside_joints_rejected():
    p = SegmentPolicy([Segment({25: 0.0}, 1.0)], joints=LEFT_ARM)
    with pytest.raises(ValueError):
        p.reset(Obs(STAND_Q))


def test_build_policy():
    r = build_policy("tpose")
    assert isinstance(r, Routine) and len(r.skills) == 1 and r.name == "tpose"
    r = build_policy("tpose,sixseven")
    assert len(r.skills) == 2 and r.name == "tpose+sixseven"
    assert build_policy("demo").name == "demo"
    assert isinstance(build_policy("look"), Look)
    assert isinstance(build_policy("wave_on_red"), Selector)
    assert set(POLICIES) == {"look", "describe", "goto_red", "face_door", "wave_on_red", "wave_on_person"}
    with pytest.raises(KeyError):
        build_policy("nope")
    with pytest.raises(KeyError):
        build_policy("tpose,nope")


def test_labels_prefixed(capsys):
    actions(Routine(TPose()))
    out = capsys.readouterr().out
    assert "tpose: arms up" in out and "takeover:" in out and "handback:" in out


def test_skill_params():
    assert len(SixSeven(reps=5)._swings()) == 11
    assert TPose(hold=1.0).duration == 7.0               # rise 3 + hold 1 + lower 3
    assert TPose(hold=1.0, rise=1.0).duration == 3.0
    assert Routine(TPose()).joints == sorted(UPPER_BODY)


def test_cli_list_and_unknown(capsys):
    assert main(["--list"]) == 0
    assert "policies: describe, face_door, goto_red, look, wave_on_person, wave_on_red" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--env", "sim", "--policy", "nope"])
