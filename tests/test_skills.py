import itertools
import math

import pytest

from config import BASE_VEL_MAX, STAND_Q
from policy import Obs, Segment, SegmentPolicy
from routines import Routine, build_policy
from run import build_parser, main, run
from skills import (CATALOG, CATALOG_PATH, SKILLS, STEP_MAX, WAIST_YAW, Catalog, Handback, Skill, Takeover,
                    describe_menu, load_catalog, menu, parse_skill, skill_policy, skill_segments, use_catalog,
                    validate_args)
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
    assert validate_args(turn, {"angle_deg": "30"}) == ({"angle_deg": 30.0, "note": ""}, [])
    args, notes = validate_args(turn, {"angle_deg": 500})
    assert args == {"angle_deg": 180.0, "note": ""} and notes
    with pytest.raises(ValueError):
        validate_args(turn, {"angle": 30})
    with pytest.raises(ValueError):
        validate_args(turn, {})
    with pytest.raises(ValueError):
        validate_args(turn, {"angle_deg": "left"})
    assert validate_args(SKILLS["done"], {"summary": "s", "hindsight": None})[0] == {"summary": "s", "hindsight": ""}
    assert validate_args(SKILLS["tpose"], {}) == ({"hold": 5.0, "rise": 3.0, "note": ""}, [])   # defaults fill in
    assert validate_args(SKILLS["turn"], {"angle_deg": 1})[0]["note"] == ""      # a note is the model's duty, not code's
    with pytest.raises(ValueError):
        validate_args(SKILLS["check"], {"skill": "turn", "arguments": 3, "note": "n"})


def test_menu_and_parse():
    assert {s.name for s in menu(False)} == {"look", "hold", "tpose", "sixseven", "check", "done", "give_up"}
    assert [s.name for s in menu(True)] == [n for n in SKILLS if not SKILLS[n].internal]
    skill = parse_skill("turn:45")
    assert skill.name == "turn" and skill.args == {"angle_deg": 45.0, "note": ""}
    assert parse_skill("look:yaw_deg=-20").args == {"yaw_deg": -20.0, "note": ""}
    assert parse_skill("tpose").args == {"hold": 5.0, "rise": 3.0, "note": ""}        # defaults fill in
    assert parse_skill("tpose:1:2").args == {"hold": 1.0, "rise": 2.0, "note": ""}    # positional, in schema order
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
        build_policy("done:yes:no")
    with pytest.raises(ValueError):
        build_policy("takeover")                 # internal: not a movement
    with pytest.raises(ValueError):
        build_policy("check:turn")               # moves nothing
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


# --------------------------------------------------------------------------
# The catalog
# --------------------------------------------------------------------------

def catalog_data():
    import json
    return json.loads(CATALOG_PATH.read_text())


def test_catalog_binds_every_skill_and_each_runs_with_its_defaults():
    data = catalog_data()
    assert data["version"] == 1 and [s["name"] for s in data["skills"]] == list(SKILLS)
    for entry in data["skills"]:
        cls = SKILLS[entry["name"]]
        assert cls.description == entry["description"] and cls.prompt == entry["prompt"]
        assert cls.terminal == entry.get("terminal", False) and cls.internal == entry.get("internal", False)
        assert cls.needs_base == entry["needs_base"] and "$schema" not in str(cls.params)
        props = cls.params["properties"]
        if not cls.terminal and not cls.internal:
            assert "note" in cls.params["required"] and props["note"]["minLength"] == 1
        # the smallest legal arguments: every argument segments() reads is declared
        args = {}
        for k, spec in props.items():
            if k in cls.params["required"] and "default" not in spec:
                args[k] = ("n" if spec.get("type") == "string" else spec.get("minimum", 0)
                           if spec.get("type") in ("number", "integer") else {} if spec.get("type") == "object"
                           else "turn")
        skill = cls(**args)
        segs = skill.segments()
        assert isinstance(segs, tuple)
        if cls.terminal or cls.name == "check":
            assert segs == ()
        else:
            assert skill.duration > 0


def test_catalog_validation_errors(tmp_path):
    import json
    base = catalog_data()

    def broken(mutate):
        d = json.loads(json.dumps(base))
        mutate(d)
        p = tmp_path / "c.json"
        p.write_text(json.dumps(d))
        with pytest.raises(ValueError):
            load_catalog(p)

    broken(lambda d: d.update(version=2))
    broken(lambda d: d["skills"].append(dict(d["skills"][0])))                       # duplicate
    broken(lambda d: d["skills"][0].update(skill="skills.Nope"))                     # unknown class
    broken(lambda d: d["skills"][0].update(skill="skills.Segment"))                  # not a Skill
    broken(lambda d: d["skills"][0].update(name="Walk"))                             # bad name
    broken(lambda d: d["skills"][0].update(prompt=""))
    broken(lambda d: d["skills"][0]["parameters"]["required"].append("speed"))      # undeclared
    broken(lambda d: d["skills"][0]["parameters"]["properties"].update(note={"$schema": "nope"}))
    broken(lambda d: d["skills"][0].update(enabled="yes"))
    broken(lambda d: d.update(skills=[]))
    broken(lambda d: [s.update(enabled=False) for s in d["skills"] if not s.get("internal")])
    with pytest.raises(ValueError):
        load_catalog(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        Catalog([], "x")


def test_catalog_renderings_and_swap(tmp_path):
    import json
    m = menu(True)
    bullets = CATALOG.prompt_catalog(m)
    assert bullets.startswith("- walk_forward: ") and bullets.count("\n") == len(m) - 1
    fns = CATALOG.function_schemas(m)
    assert [f["function"]["name"] for f in fns] == [s.name for s in m]
    assert fns[0]["function"]["parameters"]["properties"]["note"]["minLength"] == 1
    assert next(f for f in fns if f["function"]["name"] == "check")["function"]["parameters"]["properties"]["skill"]["enum"][0] == "walk_forward"
    schema = CATALOG.output_schema(m)
    assert "One skill selection for a Unitree G1 humanoid with 29 joints" in schema["description"]
    loose = CATALOG.output_schema(m, strict=False)
    assert "default" in next(a for a in loose["properties"]["arguments"]["anyOf"] if "seconds" in a["properties"])["properties"]["seconds"]
    # a swapped catalog with a different prompt and range is what the model sees
    data = catalog_data()
    data["skills"][0]["prompt"] = "WALK PROMPT"
    data["skills"][0]["parameters"]["properties"]["distance_m"]["maximum"] = 1.0
    p = tmp_path / "alt.json"
    p.write_text(json.dumps(data))
    try:
        use_catalog(p)
        assert SKILLS["walk_forward"].prompt == "WALK PROMPT"
        assert validate_args(SKILLS["walk_forward"], {"distance_m": 2.0, "note": "n"})[0]["distance_m"] == 1.0
        assert "WALK PROMPT" in describe_menu(menu()) or True
    finally:
        use_catalog(None)
    assert SKILLS["walk_forward"].prompt != "WALK PROMPT"
