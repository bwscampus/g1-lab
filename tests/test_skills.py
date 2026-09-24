import itertools
import math

import pytest

from config import BASE_VEL_MAX, STAND_Q
from policy import Obs, Segment, SegmentPolicy
from routines import Routine, build_policy
from run import build_parser, main, run
from skills import (SKILLS, STEP_MAX, WAIST_YAW, Handback, Skill, Takeover, describe_menu, menu,
                    parse_skill, skill_policy, skill_segments, validate_args)
from tests.doubles import sim_env

EXTREMES = {                       # (extreme args, cheap args for the pairwise test)
    "walk_forward": ([{"distance_m": 0.1}, {"distance_m": 3.0}], {"distance_m": 0.1}),
    "turn": ([{"angle_deg": -180.0}, {"angle_deg": 180.0}, {"angle_deg": 0.0}, {"angle_deg": 3.0}],
             {"angle_deg": 5.0}),
    "look": ([{"yaw_deg": -45.0}, {"yaw_deg": 45.0}], {"yaw_deg": 45.0}),
    "hold": ([{"seconds": 0.1}, {"seconds": 30.0}], {"seconds": 0.5}),
    "tpose": ([{}, {"hold": 0.5, "rise": 1.0}], {"hold": 0.5, "rise": 1.0}),
    "sixseven": ([{}, {"reps": 1, "settle": 1.0, "hold": 0.1}], {"reps": 1, "settle": 1.0, "hold": 0.1}),
}


def bookended(*skills, ramp=0.2, to_stand=0.5):
    """The skills with short bookends, as one policy."""
    parts = [Takeover(ramp=ramp, to_stand=to_stand), *skills,
             Handback(to_stand=to_stand, ramp=ramp)]
    segs, joints = [], set()
    for part in parts:
        joints.update(part.joints)
        segs.extend(skill_segments(part))
    return SegmentPolicy(segs, joints=sorted(joints))


def test_every_skill_passes_the_run_checks():
    for name, (extremes, _) in EXTREMES.items():
        for args in extremes:
            env = sim_env()
            assert run(bookended(SKILLS[name](**args)), env) is True, (name, args)
            assert env.violations == [], (name, args, env.violations)


def test_every_skill_pair_is_continuous():
    parts = [(n, cheap) for n, (_, cheap) in EXTREMES.items()]
    for (a, aa), (b, ba) in itertools.permutations(parts, 2):
        env = sim_env()
        assert run(bookended(SKILLS[a](**aa), SKILLS[b](**ba)), env) is True, (a, b)
        assert not [v for v in env.violations if v.kind == "velocity"], (a, b, env.violations)


def test_walk_and_turn_drive_the_base():
    env = sim_env()
    assert run(bookended(SKILLS["walk_forward"](distance_m=0.4)), env) is True
    assert env.base_path == pytest.approx(0.4, abs=1e-6) and env.base_peak[0] <= BASE_VEL_MAX[0]
    env = sim_env()
    assert run(bookended(SKILLS["turn"](angle_deg=-30)), env) is True
    assert env.base_pose()[2] == pytest.approx(math.radians(-30), abs=1e-6)
    env = sim_env()
    seg = Segment({}, 0.4, base=(1.0, 0.0, 0.0))
    assert run(SegmentPolicy([seg], joints=[WAIST_YAW]), env) is False
    assert {v.kind for v in env.violations} == {"base_vx"}


def test_look_holds_and_next_skill_recentres():
    p = bookended(SKILLS["look"](yaw_deg=40), SKILLS["walk_forward"](distance_m=0.1))
    p.reset(Obs(STAND_Q))
    yaws = []
    n = 0
    while (a := p.step(n * 0.02, Obs(STAND_Q))) is not None:
        yaws.append(a.q[WAIST_YAW]); n += 1
    assert max(yaws) == pytest.approx(math.radians(40), abs=1e-6)
    assert yaws[-1] == pytest.approx(0.0, abs=1e-6)


def test_validate_args():
    turn = SKILLS["turn"]
    assert validate_args(turn, {"angle_deg": "30"}) == ({"angle_deg": 30.0}, [])
    args, notes = validate_args(turn, {"angle_deg": 500})
    assert args == {"angle_deg": 180.0} and notes
    with pytest.raises(ValueError):
        validate_args(turn, {"angle": 30})
    with pytest.raises(ValueError):
        validate_args(turn, {})
    with pytest.raises(ValueError):
        validate_args(turn, {"angle_deg": "left"})
    assert validate_args(SKILLS["done"], {"found": "true"})[0] == {"found": True, "note": ""}
    assert validate_args(SKILLS["tpose"], {}) == ({"hold": 5.0, "rise": 3.0}, [])   # defaults fill in


def test_menu_and_parse():
    assert {s.name for s in menu(False)} == {"look", "hold", "tpose", "sixseven", "done"}
    assert [s.name for s in menu(True)] == [n for n in SKILLS if not SKILLS[n].internal]
    skill = parse_skill("turn:45")
    assert skill.name == "turn" and skill.args == {"angle_deg": 45.0}
    assert parse_skill("look:yaw_deg=-20").args == {"yaw_deg": -20.0}
    assert parse_skill("tpose").args == {"hold": 5.0, "rise": 3.0}        # defaults fill in
    assert parse_skill("tpose:1:2").args == {"hold": 1.0, "rise": 2.0}    # positional, in schema order
    with pytest.raises(KeyError):
        parse_skill("fly:1")
    with pytest.raises(ValueError):
        parse_skill("tpose:1:2:3")
    assert "walk_forward(distance_m: number [0.1, 3.0])  [needs --walk]" in describe_menu(menu())


def test_skill_chain_is_a_routine():
    r = build_policy("walk_forward:0.5,turn:45,tpose:0.5:1,sixseven:1:0.4:1:0.1")
    assert isinstance(r, Routine)
    assert [s.name for s in r.skills] == ["walk_forward", "turn", "tpose", "sixseven"]
    env = sim_env()
    assert run(r, env) is True and env.violations == [] and env.base_path == pytest.approx(0.5)
    with pytest.raises(ValueError):
        build_policy("done:true")
    with pytest.raises(ValueError):
        build_policy("takeover")                 # internal: not a movement
    with pytest.raises(KeyError):
        build_policy("nope:1")
    assert main(["--env", "sim", "--headless", "--policy", "turn:30,look:20,hold:1"]) == 0


def test_cli_lists_skills_and_gates_base(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "skills:" in out and "walk_forward(" in out
    args = ["--env", "sim", "--policy", "walk_forward:0.3", "--headless", "--free-base"]
    with pytest.raises(SystemExit):
        main(args)
    assert "cannot walk" in capsys.readouterr().err


def test_skill_policy_prefixes_labels(capsys):
    p = skill_policy(SKILLS["turn"](angle_deg=20))
    p.reset(Obs(STAND_Q))
    p.step(0.0, Obs(STAND_Q))
    assert "[turn] turn: turn +20 deg" in capsys.readouterr().out
