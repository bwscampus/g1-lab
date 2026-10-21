import itertools
import json
import math

import pytest

from g1.cli import run
from g1.core import limits
from g1.core.action import Obs, Segment
from g1.core.config import BASE_VEL_MAX, STAND_Q
from g1.tools import (TOOLS, Chain, Tool, function_schemas, limit, load_module, menu, num, output_schema,
                      parameters, parse_chain, parse_tool, prefixed, prompt_catalog, unregister,
                      validate_args)
from g1.tools.arms import WAIST_YAW
from g1.tools.control import Handback, Takeover
from tests.doubles import sim_env

WP = [{"joints": {"left_shoulder_roll": 1.2, "left_elbow": 0.3}, "seconds": 1.5},
      {"joints": {"right_shoulder_roll": -1.2, "left_shoulder_roll": 0.2, "left_elbow": 1.28}, "seconds": 1.5}]
EXTREMES = {                       # (extreme args, cheap args for the pairwise test)
    "move": ([{"dx_m": -3.0, "dy_m": 1.5, "dyaw_deg": -180.0}, {"dx_m": 3.0, "dy_m": -1.5, "dyaw_deg": 180.0}, {}],
             {"dx_m": 0.1, "dyaw_deg": 5.0}),
    "arm_path": ([{"waypoints": WP}, {"waypoints": [{"joints": {"waist_yaw": 0.785}}]}],
                 {"waypoints": [{"joints": {"waist_yaw": -0.3}, "seconds": 1.0}]}),
    "walk_forward": ([{"distance_m": 0.1}, {"distance_m": 3.0}], {"distance_m": 0.1}),
    "turn": ([{"angle_deg": -180.0}, {"angle_deg": 180.0}, {"angle_deg": 0.0}, {"angle_deg": 3.0}],
             {"angle_deg": 5.0}),
    "look": ([{"yaw_deg": -45.0}, {"yaw_deg": 45.0, "seconds": 0.5}, {"yaw_deg": 145.0, "seconds": 3.0}, {"yaw_deg": -145.0, "seconds": 3.0}], {"yaw_deg": 45.0}),
    "point": ([{"down_deg": 0.0, "rise_s": 0.8, "hold_s": 0.5}, {"down_deg": 60.0}], {"hold_s": 0.5, "rise_s": 1.0}),
    "hold": ([{"seconds": 0.1}, {"seconds": 30.0}], {"seconds": 0.5}),
    "tpose": ([{}, {"hold_s": 0.5, "rise_s": 1.0}], {"hold_s": 0.5, "rise_s": 1.0}),
    "sixseven": ([{}, {"reps": 1, "settle_s": 1.0, "hold_s": 0.1}], {"reps": 1, "settle_s": 1.0, "hold_s": 0.1}),
}


def bookended(*tools, ramp=1.0):
    """The tools with short bookends, as one program."""
    parts = [Takeover(ramp_s=ramp), *tools, Handback(ramp_s=ramp)]
    segs, joints = [], set()
    for part in parts:
        joints.update(part.joints)
        segs.extend(prefixed(part))
    return Tool.of(segs, joints=sorted(joints), allows_start=True)


def play(program, q=STAND_Q):
    program.reset(Obs(q))
    out, n = [], 0
    while (a := program.step(n * 0.02, Obs(q))) is not None:
        out.append(a); n += 1
    return out


# --------------------------------------------------------------------------
# Every tool runs, every pair chains
# --------------------------------------------------------------------------

def test_every_tool_passes_the_run_checks():
    for name, (extremes, _) in EXTREMES.items():
        for args in extremes:
            env = sim_env()
            assert run(bookended(TOOLS[name](**args)), env) is True, (name, args)
            assert env.violations == [], (name, args, env.violations)


def test_every_tool_pair_is_continuous():
    parts = [(n, cheap) for n, (_, cheap) in EXTREMES.items()]
    for (a, aa), (b, ba) in itertools.permutations(parts, 2):
        env = sim_env()
        assert run(bookended(TOOLS[a](**aa), TOOLS[b](**ba)), env) is True, (a, b)
        assert not [v for v in env.violations if v.kind == "velocity"], (a, b, env.violations)


def test_walk_and_turn_drive_the_base():
    env = sim_env()
    assert run(bookended(TOOLS["walk_forward"](distance_m=0.4)), env) is True
    assert env.base_path == pytest.approx(0.4, abs=1e-6) and env.base_peak[0] <= BASE_VEL_MAX[0]
    env = sim_env()
    assert run(bookended(TOOLS["turn"](angle_deg=-30)), env) is True
    assert env.monitor.base_pose()[2] == pytest.approx(math.radians(-30), abs=1e-6)     # commanded, exactly
    assert env.base_pose()[2] == pytest.approx(math.radians(-30), abs=math.radians(10))  # the twin's truth
    env = sim_env()
    seg = Segment({}, 0.4, base=(1.0, 0.0, 0.0))
    assert run(Tool.of([seg], joints=[WAIST_YAW]), env) is False
    assert {v.kind for v in env.violations} == {"base_vx"}


def test_nothing_resets_the_pose_but_the_tool_itself():
    """No move to neutral at the start or the end, and walking keeps whatever the arms are doing."""
    assert Takeover().duration == limits.get("bookend_ramp_s") and Takeover().segments()[0].goal == "start"
    assert Handback().duration == limits.get("bookend_ramp_s") and Handback().segments()[0].goal == {}
    assert [s.label for s in Takeover().segments()] == ["taking control"]
    yaws = [a.q[WAIST_YAW] for a in play(bookended(TOOLS["look"](yaw_deg=40), TOOLS["walk_forward"](distance_m=0.1)))]
    assert max(yaws) == pytest.approx(math.radians(40), abs=1e-6)
    assert yaws[-1] == pytest.approx(math.radians(40), abs=1e-6)          # the walk did not recentre it
    env = sim_env()
    raised = TOOLS["arm_path"](waypoints=[{"joints": {"left_elbow": -0.4}, "seconds": 1.0}])
    assert run(bookended(raised, TOOLS["move"](dx_m=0.3, dyaw_deg=20), TOOLS["hold"](seconds=0.3)), env) is True
    assert env.violations == [] and env.cmd_min[18] == pytest.approx(-0.4, abs=1e-6)
    cmds = [a for a in play(bookended(raised, TOOLS["move"](dx_m=0.3)))]
    assert cmds[-1].q[18] == pytest.approx(-0.4, abs=1e-6) and cmds[-1].weight == pytest.approx(0.0, abs=1e-3)   # released there
    for name in ("move", "walk_forward", "turn"):
        assert all(s.goal == {} for s in TOOLS[name](**EXTREMES[name][1]).segments())


def test_move_ends_exactly_where_asked():
    for args, end in [({"dx_m": 1.0, "dy_m": 0.3, "dyaw_deg": -45.0}, (1.0, 0.3, -45.0)),
                      ({"dx_m": -0.5}, (-0.5, 0.0, 0.0)), ({"dyaw_deg": 90.0}, (0.0, 0.0, 90.0)), ({}, (0.0, 0.0, 0.0))]:
        env = sim_env()
        assert run(bookended(TOOLS["move"](**args)), env) is True and env.violations == [], args
        x, y, yaw = env.monitor.base_pose()
        assert (x, y, math.degrees(yaw)) == pytest.approx(end, abs=1e-6)                  # commanded, exactly
        x, y, yaw = env.base_pose()                                                     # walked: the twin's truth
        assert (x, y) == pytest.approx(end[:2], abs=0.15) and math.degrees(yaw) == pytest.approx(end[2], abs=10), args
    segs = TOOLS["move"](dx_m=1.0, dy_m=0.3, dyaw_deg=-45.0).segments()
    assert len(segs) == 2 and segs[0].base[2] == 0.0 and segs[1].base[:2] == (0.0, 0.0)   # translate, then turn
    assert validate_args(TOOLS["move"], {"dx_m": 9})[0]["dx_m"] == 3.0


def test_point_moves_only_the_right_arm_and_lowers_it():
    from g1.core.poses import STAND
    from g1.tools.arms import R_ARM
    p = TOOLS["point"](down_deg=0.0)
    segs = p.segments()
    assert all(set(s.goal) == set(R_ARM) for s in segs)
    assert segs[0].goal[R_ARM[0]] < -1.57 and segs[0].goal[R_ARM[3]] == pytest.approx(1.57)   # level, aimed above the sag
    assert TOOLS["point"](down_deg=60.0).segments()[0].goal[R_ARM[0]] > segs[0].goal[R_ARM[0]]   # steeper aims lower
    assert segs[-1].goal == {j: STAND[j] for j in R_ARM}
    actions = play(bookended(TOOLS["look"](yaw_deg=30), p))
    assert actions[-1].q[WAIST_YAW] == pytest.approx(math.radians(30), abs=1e-6)              # the waist stays


def test_sit_calls_setfsmid_3_once_and_the_twin_sits_on_the_chair_behind(capsys):
    s = TOOLS["sit"]()
    cmds = [a.command for a in play(s)]
    assert cmds.count(("Sit", {})) == 1 and cmds[0] is None and play(s)[-1].weight == pytest.approx(1.0, abs=0.01)
    env = sim_env(mode="standing", scene="room", sim_objects=[("chair", -0.37, 0.0, None)])   # seat edge 0.1 m behind
    assert run(bookended(s), env) is True and env.violations == []
    assert "seated on chair1" in env.twin.summary() and env.seated              # the twin plays no stand-up
    fsms = [c["parameter"] for c in env.twin.services.log if c["api"] == 7101]
    assert fsms[-2:] == ['{"data": 3}', '{"data": 801}']                         # sat, then the teardown's stand-up
    assert "still seated" in capsys.readouterr().out


def test_arm_path_waypoints_in_order_and_rejections():
    env = sim_env()
    tool = TOOLS["arm_path"](waypoints=WP)
    assert run(bookended(tool), env) is True and env.violations == []
    assert env.cmd_max[16] == pytest.approx(1.2, abs=1e-6) and env.cmd_min[23] == pytest.approx(-1.2, abs=1e-6)
    left = [a.q[16] for a in play(tool)]
    assert max(left[:75]) == pytest.approx(1.2, abs=1e-3) and left[-1] == pytest.approx(0.2, abs=1e-3)   # in order
    for bad in [{"waypoints": []}, {"waypoints": [{"joints": {}}]}, {"waypoints": [{"joints": {"nope": 0.1}}]},
                {"waypoints": [{"joints": {"left_elbow": 3.0}}]}, {"waypoints": [{"joints": {"waist_yaw": "x"}}]},
                {"waypoints": [{"joints": {"waist_yaw": 0.1}, "seconds": 0.1}]}, {"waypoints": "up"}]:
        with pytest.raises(ValueError):
            TOOLS["arm_path"](**bad)
    schema = parameters(TOOLS["arm_path"], menu(True))
    joints = schema["properties"]["waypoints"]["items"]["properties"]["joints"]
    assert joints["properties"]["left_elbow"] == {"type": "number", "minimum": -1.047, "maximum": 2.094}
    assert joints["additionalProperties"] is False and len(joints["properties"]) == 17


class DampGesture(TOOLS["wave_hand"]):
    name = "damp"
    command = "Damp"


class NoLoco(TOOLS["wave_hand"]):
    name = "noloco"
    needs_loco = False


def test_gestures_emit_one_onboard_call_and_the_twin_waves(capsys):
    w = TOOLS["wave_hand"](turn_flag=True, seconds=2.0)
    cmds = [a.command for a in play(w)]
    assert cmds.count(("WaveHand", {"turn_flag": True})) == 1 and cmds[0] is None       # once, after the handover
    env = sim_env()
    assert run(bookended(w), env) is True and env.base_ticks == 0
    tasks = [c for c in env.twin.services.log if c["api"] == 7106]
    assert [(c["service"], c["parameter"], c["code"]) for c in tasks] == [("sport", '{"data": 1}', 0)]   # SetArmTask, as the SDK sends it
    assert env.q_min[23] < -0.8                                  # the onboard wave raised the right arm
    assert not any(t.needs_loco for t in menu(True)) and [t.name for t in menu(True, True) if t.needs_loco] == ["wave_hand", "shake_hand", "sit"]
    with pytest.raises(ValueError, match="allowed onboard method"):
        DampGesture.check_definition()
    with pytest.raises(ValueError, match="needs_loco"):
        NoLoco.check_definition()
    assert [t["function"]["name"] for t in function_schemas(menu(True))] == ["move", "arm_path", "look", "point", "hold", "check", "say",
                                                                           "done", "give_up"]
    assert [t["function"]["name"] for t in function_schemas(menu(True, True))][6:8] == ["wave_hand", "shake_hand"]


# --------------------------------------------------------------------------
# Arguments, the menu, the CLI forms
# --------------------------------------------------------------------------

def test_validate_args():
    turn = TOOLS["turn"]
    assert validate_args(turn, {"angle_deg": "30"}) == ({"angle_deg": 30.0, "note": ""}, [])
    args, notes = validate_args(turn, {"angle_deg": 500})
    assert args == {"angle_deg": 180.0, "note": ""} and notes
    with pytest.raises(ValueError):
        validate_args(turn, {"angle": 30})
    with pytest.raises(ValueError):
        validate_args(turn, {})
    with pytest.raises(ValueError):
        validate_args(turn, {"angle_deg": "left"})
    assert validate_args(TOOLS["done"], {"summary": "s", "hindsight": None})[0] == {"summary": "s", "hindsight": ""}
    assert validate_args(TOOLS["tpose"], {}) == ({"hold_s": 5.0, "rise_s": 3.0, "note": ""}, [])   # defaults fill in
    assert validate_args(TOOLS["turn"], {"angle_deg": 1})[0]["note"] == ""      # a note is the model's duty, not code's
    with pytest.raises(ValueError):
        validate_args(TOOLS["check"], {"tool": "turn", "arguments": 3, "note": "n"})


def test_menu_and_parse():
    assert [t.name for t in menu(False)] == ["arm_path", "look", "point", "hold", "check", "say", "done", "give_up"]
    assert [t.name for t in menu(True)] == ["move", "arm_path", "look", "point", "hold", "check", "say", "done",
                                            "give_up"]
    assert [t.name for t in menu(True, True)] == ["move", "arm_path", "look", "point", "hold", "check", "wave_hand",
                                                 "shake_hand", "sit", "say", "done", "give_up"]
    assert "sit" not in [t.name for t in menu(False, True)]                    # never on the gantry
    assert {n for n in TOOLS if not TOOLS[n].visible} == {"walk_forward", "turn", "tpose", "sixseven",
                                                          "takeover", "handback"}
    m = parse_tool("move:1:0.3:-45")
    assert m.args == {"dx_m": 1.0, "dy_m": 0.3, "dyaw_deg": -45.0, "note": ""}
    a = parse_tool('arm_path:waypoints=[{"joints":{"left_elbow":-0.4},"seconds":1.5}]')
    assert a.waypoints == [{"joints": {"left_elbow": -0.4}, "seconds": 1.5}] and a.duration == 1.5
    with pytest.raises(ValueError):
        parse_tool("arm_path:waypoints=notjson")
    tool = parse_tool("turn:45")
    assert tool.name == "turn" and tool.args == {"angle_deg": 45.0, "note": ""}
    assert parse_tool("look:yaw_deg=-20").args == {"yaw_deg": -20.0, "seconds": 1.5, "note": ""}
    assert parse_tool("tpose").args == {"hold_s": 5.0, "rise_s": 3.0, "note": ""}        # defaults fill in
    assert parse_tool("tpose:1:2").args == {"hold_s": 1.0, "rise_s": 2.0, "note": ""}    # positional, in params order
    with pytest.raises(KeyError):
        parse_tool("fly:1")
    with pytest.raises(ValueError):
        parse_tool("tpose:1:2:3")


def test_chain_seeds_each_tool_from_the_last_command():
    c = parse_chain("walk_forward:0.5,turn:45,tpose:0.5:1,sixseven:1:0.4:1:0.1")
    assert isinstance(c, Chain) and c.name == "walk_forward+turn+tpose+sixseven"
    assert [t.name for t in c.tools] == ["walk_forward", "turn", "tpose", "sixseven"]     # presets still chain
    env = sim_env()
    assert run(c, env) is True and env.violations == [] and env.base_path == pytest.approx(0.5)
    c = parse_chain('move:1:0.3:-45,arm_path:waypoints=[{"joints":{"waist_yaw":0.5},"seconds":1}]')
    assert [t.name for t in c.tools] == ["move", "arm_path"]
    # the boundary between tools is continuous: the second starts where the first's command ended
    actions = play(parse_chain("look:30:0.5,look:-30:0.5", pause=0.0))
    yaw = [a.q[WAIST_YAW] for a in actions]
    steps = [abs(b - a) for a, b in zip(yaw, yaw[1:])]
    assert max(yaw) == pytest.approx(math.radians(30), abs=1e-6) and min(yaw) == pytest.approx(math.radians(-30), abs=1e-6)
    assert max(steps) < 0.08                                            # no jump anywhere in the stream
    with pytest.raises(ValueError):
        parse_chain("done:yes:no")
    with pytest.raises(ValueError):
        parse_chain("check:turn")               # moves nothing
    with pytest.raises(KeyError):
        parse_chain("nope:1")




def test_player_prefixes_labels(capsys):
    play(Tool.of(prefixed(TOOLS["turn"](angle_deg=20)), name="turn"))
    assert "[turn] turn: turn +20 deg" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The definition rules: metadata in the class, limits filled in, flexible parameters
# --------------------------------------------------------------------------

def test_every_tool_is_well_defined_and_runs_with_its_defaults():
    for name, cls in TOOLS.items():
        assert cls.name == name and cls.kind in ("motion", "query", "terminal")
        cls.check_definition()
        if cls.visible:
            assert cls.prompt.strip() and "{limit:" not in cls.describe()        # limits are filled in, never shown raw
        schema = cls.schema()
        assert "$limit" not in json.dumps(schema) and "Limit" not in json.dumps(schema, default=repr)
        if cls.kind != "terminal":
            assert "note" in schema["required"] and schema["properties"]["note"]["minLength"] == 1
        else:
            assert "note" not in schema["properties"]
        # the smallest legal arguments: every argument segments() reads is declared
        args = {}
        for k, spec in schema["properties"].items():
            if k in schema["required"] and "default" not in spec:
                args[k] = ("n" if spec.get("type") == "string" else spec.get("minimum", 0)
                           if spec.get("type") in ("number", "integer") else {} if spec.get("type") == "object"
                           else [{"joints": {"waist_yaw": 0.0}}] if spec.get("type") == "array" else "move")
        tool = cls(**args)
        segs = tool.segments()
        assert isinstance(segs, tuple)
        if cls.kind != "motion":
            assert segs == ()
        else:
            assert tool.duration > 0


def test_parameters_are_numbers_with_ranges_not_switches():
    """A continuous quantity is a number with a range, in seconds/metres/degrees;
    enums are only for genuinely discrete choices (the menu's tool names)."""
    for cls in TOOLS.values():
        for key, spec in cls.schema()["properties"].items():
            if key == "note":
                continue
            kind = spec.get("type")
            if kind in ("number", "integer"):
                assert spec["minimum"] < spec["maximum"], (cls.name, key)
                if kind == "number":
                    assert key.endswith(("_s", "_m", "_deg", "_rad")) or key == "seconds", (cls.name, key)
            assert "enum" not in spec or (cls.name, key) == ("check", "tool"), (cls.name, key)
    with pytest.raises(ValueError, match="minimum and maximum"):
        type("Bad", (Tool,), {"name": "bad", "prompt": "p", "params": {"x_s": {"type": "number"}}}).check_definition()


def test_limits_are_read_when_the_schema_is(tmp_path):
    data = json.loads(limits.PATH.read_text())
    data["limits"]["move_dx_max_m"]["value"] = 1.0
    p = tmp_path / "limits.json"
    p.write_text(json.dumps(data))
    try:
        limits.use(p)
        assert TOOLS["move"].schema()["properties"]["dx_m"]["maximum"] == 1.0
        assert "up to 1 m either way" in TOOLS["move"].describe()
        assert validate_args(TOOLS["move"], {"dx_m": 2.0})[0]["dx_m"] == 1.0       # clamped
    finally:
        limits.use(None)
    assert TOOLS["move"].schema()["properties"]["dx_m"]["maximum"] == limits.get("move_dx_max_m")
    with pytest.raises(ValueError, match="unknown limit"):
        limit("nope")


def test_renderings():
    m = menu(True)
    bullets = prompt_catalog(m)
    assert bullets.startswith("- move: ") and bullets.count("\n") == len(m) - 1
    fns = function_schemas(m)
    assert [f["function"]["name"] for f in fns] == [t.name for t in m]
    assert fns[0]["function"]["parameters"]["properties"]["note"]["minLength"] == 1
    assert next(f for f in fns if f["function"]["name"] == "check")["function"]["parameters"]["properties"]["tool"]["enum"][0] == "move"
    schema = output_schema(m)
    assert "One tool selection for a Unitree G1 humanoid with 29 joints" in schema["description"]
    loose = output_schema(m, strict=False)
    assert "default" in next(a for a in loose["properties"]["arguments"]["anyOf"] if "seconds" in a["properties"])["properties"]["seconds"]


def test_definition_errors_name_the_problem(tmp_path):
    def bad(body, match):
        p = tmp_path / "bad.py"
        p.write_text("from g1.tools.base import Tool, num\n" + body)
        with pytest.raises(ValueError, match=match):
            load_module(p)

    bad("class A(Tool):\n    name = 'Bad Name'\n    prompt = 'p'\n", "snake_case")
    bad("class A(Tool):\n    name = 'a'\n", "needs a prompt")
    bad("class A(Tool):\n    name = 'a'\n    prompt = 'p'\n    kind = 'reflex'\n", "kind")
    bad("class A(Tool):\n    name = 'a'\n    prompt = 'p'\n    params = {'x': 3}\n", "needs a type")
    bad("class A(Tool):\n    name = 'a'\n    prompt = 'p {limit:nope}'\n", "unknown limit")
    bad("class A(Tool):\n    name = 'move'\n    prompt = 'p'\n", "already defined")
    bad("x = 1\n", "no Tool subclass")
    assert "move" in TOOLS and TOOLS["move"].__module__ == "g1.tools.move"    # the built-in survived the clash


BOW = """
import math
from g1.core.action import Segment
from g1.tools.base import Tool, num


class Bow(Tool):
    name = "bow"
    prompt = "Bow from the waist and come back up."
    params = {"angle_deg": num(5.0, 25.0, 15.0, "how far forward"), "down_s": num(0.5, 3.0, 1.5, "time to bend")}

    def segments(self):            # the waist only: every other joint stays where it is
        return (Segment({14: math.radians(self.angle_deg)}, self.down_s, label="bending"),
                Segment({14: 0.0}, self.down_s, label="back up"))
"""


def test_a_tool_file_is_discovered_and_runs(tmp_path):
    """A file with a Tool subclass: load it, see it in the menu, run it in sim, chain it."""
    p = tmp_path / "bow.py"
    p.write_text(BOW)
    (cls,) = load_module(p)
    try:
        assert cls.name == "bow" and "bow" in TOOLS and cls in menu(False)
        assert cls.schema()["properties"]["down_s"]["default"] == 1.5 and "note" in cls.schema()["required"]
        assert "bow" in [f["function"]["name"] for f in function_schemas(menu(True))]
        env = sim_env()
        assert run(parse_chain("bow:10:1,hold:0.5,bow:angle_deg=5", pause=0.5), env) is True
        assert env.violations == [] and env.cmd_max[14] == pytest.approx(math.radians(10), abs=1e-6)
        assert env.q_max[14] > math.radians(12)            # the waist bent, further than asked: kp 60 sags under the torso
        assert env.cmd_min[16] == env.cmd_max[16]                              # and the arms were left alone
        assert parse_tool("bow").duration == pytest.approx(3.0)
    finally:
        unregister("bow")
    assert "bow" not in TOOLS




# --------------------------------------------------------------------------
# The speaker
# --------------------------------------------------------------------------

def test_say_speaks_once_and_waits_for_the_text(capsys):
    from g1.agent.decider import Decision, ProtocolError, build_context, AgentTurn
    from g1.core.config import AUDIO_METHODS
    say = TOOLS["say"]
    assert say in menu(False) and "say" in [f["function"]["name"] for f in function_schemas(menu(False))]
    s = say(text="I found the mug on the floor")                       # 28 chars
    assert s.seconds == pytest.approx(max(limits.get("say_min_s"), 28 / limits.get("speech_chars_per_s")))
    assert say(text="hi").seconds == pytest.approx(limits.get("say_min_s"))
    assert say(text="hi", pause_s=2.5).seconds == pytest.approx(limits.get("say_min_s") + 2.5)
    cmds = [a.command for a in play(s)]
    assert cmds[0] == ("TtsMaker", {"text": "I found the mug on the floor", "speaker_id": int(limits.get("tts_speaker_id"))})
    assert cmds.count(cmds[0]) == 1 and "TtsMaker" in AUDIO_METHODS
    long = say(text="x" * 500)
    assert len(long.text) == limits.get("say_max_chars") and long.notes            # code and the CLI: clamped with a note
    with pytest.raises(ValueError):
        say(text="   ")
    ctx = build_context(menu(False), can_walk=False, max_decisions=5)
    turn = AgentTurn("{}", {})
    d = Decision.parse('{"name": "say", "arguments": {"text": "hello", "note": "a person ahead"}}', turn, ctx)
    assert d.name == "say" and d.arguments["pause_s"] == 0.0
    with pytest.raises(ProtocolError):                                            # the model: rejected, not clamped
        Decision.parse('{"name": "say", "arguments": {"text": "%s", "note": "n"}}' % ("x" * 500), turn, ctx)
    with pytest.raises(ProtocolError):
        Decision.parse('{"name": "say", "arguments": {"text": "", "note": "n"}}', turn, ctx)
    # sim speaks by printing and does not abort, unlike a gesture
    env = sim_env()
    assert run(bookended(s, TOOLS["hold"](seconds=0.2)), env) is True and env.violations == [] and env.base_ticks == 0
    out = capsys.readouterr().out
    assert "sim: says 'I found the mug on the floor'" in out and "robot-only" not in out


class Loud(TOOLS["say"]):
    name = "loud"
    command = "PlayStream"


class Silent(TOOLS["say"]):
    name = "silent"
    needs_loco = False          # an audio command needs no LocoClient


def test_audio_allow_list():
    with pytest.raises(ValueError, match="allowed onboard method"):
        Loud.check_definition()
    Silent.check_definition()
    with pytest.raises(ValueError, match="needs_loco"):
        NoLoco.check_definition()          # a LocoClient gesture still must say so
