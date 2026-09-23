import itertools
import math

import pytest

from config import BASE_VEL_MAX, STAND_Q
from envs import CheckEnv
from motions import Handback, Takeover
from policy import Motion, Obs, Segment, SegmentPolicy
from routines import Routine, build_policy, motion_segments
from run import build_parser, main, run
from skills import (SKILLS, STEP_MAX, WAIST_YAW, Skill, describe_menu, menu, parse_skill, skill_policy,
                    validate_args)

EXTREMES = {
    "walk_forward": [{"distance_m": 0.1}, {"distance_m": 0.6}],
    "turn": [{"angle_deg": -68.0}, {"angle_deg": 68.0}, {"angle_deg": 0.0}, {"angle_deg": 3.0}],
    "look": [{"yaw_deg": -45.0}, {"yaw_deg": 45.0}],
    "hold": [{"seconds": 0.5}, {"seconds": 3.0}],
    "arms_up": [{}], "wave": [{}],
}


def check_env():
    return CheckEnv(build_parser().parse_args(["--env", "check", "--policy", "x"]))


def bookended(*motions):
    segs = []
    for part in (Takeover(), *motions, Handback()):
        segs.extend(motion_segments(part))
    return SegmentPolicy(segs, joints=sorted(set().union(*(m.joints for m in motions))) or Takeover().joints)


def test_every_skill_is_bounded_and_passes_check():
    for name, skill in SKILLS.items():
        if skill.terminal:
            continue
        for args in EXTREMES[name]:
            assert skill.max_duration(**args) <= STEP_MAX + 1e-9, (name, args)
            env = check_env()
            assert run(bookended(skill.build(**args)), env) is True, (name, args)
            assert env.violations == [], (name, args, env.violations)


def test_every_skill_pair_is_continuous():
    parts = [(n, EXTREMES[n][-1]) for n in SKILLS if not SKILLS[n].terminal]
    for (a, aa), (b, ba) in itertools.permutations(parts, 2):
        env = check_env()
        assert run(bookended(SKILLS[a].build(**aa), SKILLS[b].build(**ba)), env) is True, (a, b)
        assert not [v for v in env.violations if v.kind == "velocity"], (a, b, env.violations)


def test_walk_and_turn_drive_the_base_through_check():
    env = check_env()
    assert run(bookended(SKILLS["walk_forward"].build(distance_m=0.4)), env) is True
    assert env.base_path == pytest.approx(0.4, abs=1e-6) and env.base_peak[0] <= BASE_VEL_MAX[0]
    env = check_env()
    assert run(bookended(SKILLS["turn"].build(angle_deg=-30)), env) is True
    assert env.base_pose()[2] == pytest.approx(math.radians(-30), abs=1e-6)
    env = check_env()
    seg = Segment({}, 0.4, base=(1.0, 0.0, 0.0))
    assert run(SegmentPolicy([seg], joints=[WAIST_YAW]), env) is False
    assert {v.kind for v in env.violations} == {"base_vx"}


def test_look_holds_and_next_skill_recentres():
    p = bookended(SKILLS["look"].build(yaw_deg=40), SKILLS["walk_forward"].build(distance_m=0.1))
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
    assert args == {"angle_deg": 68.0} and notes
    with pytest.raises(ValueError):
        validate_args(turn, {"angle": 30})
    with pytest.raises(ValueError):
        validate_args(turn, {})
    with pytest.raises(ValueError):
        validate_args(turn, {"angle_deg": "left"})
    assert validate_args(SKILLS["done"], {"found": "true"})[0] == {"found": True}
    assert validate_args(SKILLS["arms_up"], {}) == ({}, [])


def test_menu_and_parse():
    assert {s.name for s in menu(False)} == {"look", "hold", "arms_up", "wave", "done"}
    assert [s.name for s in menu(True)] == list(SKILLS)
    skill, args = parse_skill("turn:45")
    assert skill.name == "turn" and args == {"angle_deg": 45.0}
    assert parse_skill("look:yaw_deg=-20")[1] == {"yaw_deg": -20.0}
    assert parse_skill("arms_up")[1] == {}
    with pytest.raises(KeyError):
        parse_skill("fly:1")
    with pytest.raises(ValueError):
        parse_skill("arms_up:1")
    assert "walk_forward(distance_m: number [0.1, 0.6])  [needs --walk]" in describe_menu(menu())


def test_skill_chain_is_a_routine():
    r = build_policy("walk_forward:0.5,turn:45,arms_up,sixseven")
    assert isinstance(r, Routine) and [m.name for m in r.motions] == ["walk", "turn", "arms_up", "sixseven"]
    env = check_env()
    assert run(r, env) is True and env.violations == [] and env.base_path == pytest.approx(0.5)
    with pytest.raises(ValueError):
        build_policy("done:true")
    with pytest.raises(KeyError):
        build_policy("tpose:1")
    assert main(["--env", "check", "--policy", "turn:30,look:20,hold:1"]) == 0


def test_cli_lists_skills_and_gates_base(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "skills:" in out and "walk_forward(" in out
    args = ["--env", "sim", "--policy", "walk_forward:0.3", "--headless", "--free-base"]
    with pytest.raises(SystemExit):
        main(args)
    assert "cannot walk" in capsys.readouterr().err


def test_skill_policy_prefixes_labels(capsys):
    p = skill_policy(SKILLS["turn"], {"angle_deg": 20})
    p.reset(Obs(STAND_Q))
    p.step(0.0, Obs(STAND_Q))
    assert "[turn] turn: turn +20 deg" in capsys.readouterr().out
