"""Routines: reference-motion timelines (upper body, base, onboard), validated, compiled to the
segments every tool plays, dry-run before they move, played on their own or inside a task."""
import json

import numpy as np
import pytest

from g1.core import limits
from g1.core.action import Obs
from g1.core.config import STAND_Q, UPPER_BODY, joint_index
from g1.core.poses import STAND
from g1.routines import RoutineError, compile_segments, load_routine, routine_names, run_spec
from g1.tools import TOOLS, Tool
from g1.tools.routines import RoutineTool, routine_program


def write(tmp_path, tracks, name="r", notes="a test routine"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "routine.json").write_text(json.dumps({"notes": notes, "modes": ["standing"] if "base" in tracks else ["gantry", "standing"], "tracks": tracks}))
    return d


def keyframes(*pairs):
    return {"upper_body": {"keyframes": [{"t": t, "joints": j} for t, j in pairs]}}


def play(segments):
    tool = Tool.of(segments, joints=UPPER_BODY)
    tool.reset(Obs(STAND_Q.copy()))
    out, n = [], 0
    while (a := tool.step(n * 0.02, Obs(STAND_Q))) is not None:
        out.append(a)
        n += 1
    return out


# -- the shipped routines ---------------------------------------------------------------------

def test_the_shipped_routines_load_and_tpose_is_the_old_preset():
    assert {"tpose", "sixseven"} <= set(routine_names())
    segs = compile_segments(load_routine("tpose"))
    assert [s.duration for s in segs] == [3.0, 5.0, 3.0] and [s.label for s in segs] == ["arms up", "holding T-pose", "arms down"]
    l_roll, r_roll = joint_index("left_shoulder_roll"), joint_index("right_shoulder_roll")
    assert segs[0].goal[l_roll] == pytest.approx(1.57) and segs[0].goal[r_roll] == pytest.approx(-1.57)
    assert segs[1].goal == segs[0].goal and segs[2].goal == pytest.approx(STAND)        # held, then the stand pose
    assert all(s.ease == "smooth" and s.base is None and s.command is None for s in segs)
    assert load_routine("tpose").upper_body_only and "T" in load_routine("tpose").notes
    six = compile_segments(load_routine("sixseven"))
    assert len(six) == 11 and sum(s.duration for s in six) == pytest.approx(11.8)


# -- validation -----------------------------------------------------------------------------

@pytest.mark.parametrize("tracks, match", [
    (keyframes((1.0, {"left_toe": 0.1})), "unknown joint"),
    (keyframes((1.0, {"left_knee": 0.1})), "not an upper-body joint"),
    (keyframes((1.0, {"waist_yaw": 9.0})), "outside its limits"),
    (keyframes((1.0, {"waist_yaw": 0.1}), (1.0, {"waist_yaw": 0.2})), "must come after"),
    ({**keyframes((1.0, {"waist_yaw": 0.1})), "base": [{"t0": 0.0, "t1": 1.0, "vx": 5.0}]}, "over the base ceilings"),
    ({**keyframes((1.0, {"waist_yaw": 0.1}), (2.0, {})), "base": [{"t0": 0.5, "t1": 2.0, "vx": 0.1}]}, "keyframe times"),
    ({**keyframes((1.0, {"waist_yaw": 0.1})), "onboard": [{"t": 0.5, "call": "WaveHand", "seconds": 3.0}]}, "keyframe time"),
    ({"base": [{"t0": 0.0, "t1": 4.0, "vx": 0.1}], "onboard": [{"t": 2.0, "call": "WaveHand", "seconds": 3.0}]},
     "inside a base window"),
    ({"onboard": [{"t": 0.0, "call": "Sit", "seconds": 3.0}]}, "onboard"),
    ({}, "tracks"),
    ({"upper_body": {"keyframes": [{"t": 0, "joints": {}}]}}, "keyframes"),
])
def test_a_bad_routine_is_refused_at_load(tmp_path, tracks, match):
    with pytest.raises(RoutineError, match=match):
        load_routine(write(tmp_path, tracks))


def test_too_many_keyframes_or_too_long_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(limits.LIMITS["routine_keyframes_max"], "value", 2)
    with pytest.raises(RoutineError, match="routine_keyframes_max"):
        load_routine(write(tmp_path, keyframes((1.0, {}), (2.0, {}), (3.0, {}))))
    monkeypatch.setitem(limits.LIMITS["routine_duration_max_s"], "value", 2.5)
    with pytest.raises(RoutineError, match="routine_duration_max_s"):
        load_routine(write(tmp_path, keyframes((1.0, {}), (3.0, {})), name="long"))


def test_unknown_routines_name_the_known_ones(tmp_path):
    with pytest.raises(RoutineError, match="routines: .*tpose"):
        load_routine("no_such_routine")


# -- compilation --------------------------------------------------------------------------------

def test_dense_samples_are_reached_smoothly_then_played_linearly(tmp_path):
    d = write(tmp_path, {"upper_body": {"samples": "motion.npz"}})
    t = np.array([1.0, 1.1, 1.2, 1.3])
    np.savez(d / "motion.npz", t=t, q=np.array([[0.0], [0.1], [0.2], [0.3]]), joints=np.array(["waist_yaw"]))
    segs = compile_segments(load_routine(d))
    assert [s.ease for s in segs] == ["smooth", "linear", "linear", "linear"]
    assert [round(s.duration, 6) for s in segs] == [1.0, 0.1, 0.1, 0.1]
    yaw = [a.q[joint_index("waist_yaw")] for a in play(segs)]
    between = yaw[57]                                                # t = 1.14 s, 40% from the 1.1 s sample to 1.2 s
    assert between == pytest.approx(0.14, abs=1e-6)                   # linear between samples
    np.savez(d / "motion.npz", t=np.array([1.0, 1.001]), q=np.array([[0.0], [0.1]]), joints=np.array(["waist_yaw"]))
    with pytest.raises(RoutineError, match="routine_sample_rate_max_hz"):
        load_routine(d)


def test_base_windows_and_gestures_land_on_their_ticks(tmp_path):
    tracks = {**keyframes((1.0, {"waist_yaw": 0.2}), (2.0, {"waist_yaw": 0.2}), (3.0, {"waist_yaw": 0.0})),
              "base": [{"t0": 1.0, "t1": 2.0, "vx": 0.2, "label": "a step"}],
              "onboard": [{"t": 2.0, "call": "WaveHand", "seconds": 3.0, "args": {"turn_flag": False}}]}
    r = load_routine(write(tmp_path, tracks))
    assert r.uses_base and r.uses_onboard and not r.upper_body_only
    segs = compile_segments(r)
    ramp = limits.get("gesture_ramp_s")
    assert [round(s.duration, 6) for s in segs] == [1.0, 1.0, ramp, 3.0, ramp, 1.0]
    assert [s.base for s in segs] == [None, (0.2, 0.0, 0.0), None, None, None, None] and segs[1].label == "a step"
    assert segs[3].command == ("WaveHand", {"turn_flag": False}) and segs[3].weight(0.5) == 0.0
    assert segs[-1].goal == {joint_index("waist_yaw"): 0.0}          # the timeline goes on after the gesture
    tool = RoutineTool(routine=str(r.dir))
    assert tool.needs_base and tool.needs_loco


def test_a_base_only_routine_holds_the_arms(tmp_path):
    segs = compile_segments(load_routine(write(tmp_path, {"base": [{"t0": 0.0, "t1": 2.0, "vyaw": 0.3}]})))
    assert len(segs) == 1 and segs[0].goal == {} and segs[0].base == (0.0, 0.0, 0.3) and segs[0].duration == 2.0


# -- playing ---------------------------------------------------------------------------------

def test_a_full_body_routine_is_kept_but_never_played(tmp_path):
    d = write(tmp_path, {"full_body": {"samples": "walk.npz"}})
    assert load_routine(d).full_body
    with pytest.raises(RoutineError, match="tracking controller"):
        RoutineTool(routine=str(d))


def test_the_dry_run_refuses_a_routine_before_anything_moves(tmp_path):
    from g1.cli import run
    from tests.doubles import sim_env
    d = write(tmp_path, keyframes((0.1, {"waist_yaw": 1.0})))           # far too fast for the speed gate
    env = sim_env()
    with pytest.raises(SystemExit, match="refused before it moved"):
        run(routine_program(str(d)), env)
    assert getattr(env, "ticks", 0) == 0                                 # not one command was sent


def test_play_routine_offers_upper_body_routines_only(tmp_path, monkeypatch):
    schema = TOOLS["play_routine"].schema()["properties"]["routine"]
    assert set(schema["enum"]) >= {"tpose", "sixseven"}
    with pytest.raises(ValueError, match="not a routine a task may play"):
        TOOLS["play_routine"](routine="no_such_routine")
    assert "tpose" in TOOLS["play_routine"].describe()


def test_a_routine_runs_from_g1_run_and_records_under_routines(tmp_path):
    from g1.cli import run_task
    from tests.doubles import sim_env
    spec = run_spec(write(tmp_path, keyframes((0.5, {"waist_yaw": 0.1}), (1.0, {"waist_yaw": 0.0})), name="nod"))
    assert spec["routine"] == "nod" and spec["instruction"] is None and spec["dir"].name == "nod"
    assert run_task(spec, sim=True, mode=None, env=sim_env()) == 0
    (run_dir,) = (spec["dir"] / "runs").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "_sim_nod" in run_dir.name and summary["outcome"] == "success" and summary["instruction"] == "nod"


def test_g1_run_falls_back_to_routines_and_names_both(capsys):
    from g1.cli import main
    with pytest.raises(SystemExit, match="needs --mode"):
        main(["run", "tpose"])                                           # a routine, found by name
    with pytest.raises(SystemExit):
        main(["run", "no_such_thing", "--sim"])
    err = capsys.readouterr().err
    assert "routines: " in err and "tpose" in err and "tasks: " in err


def test_a_routine_declares_its_modes_and_one_that_drives_the_base_is_standing_only(tmp_path):
    arms = {"upper_body": {"keyframes": [{"t": 1.0, "joints": {"waist_yaw": 0.1}}]}}
    d = write(tmp_path, arms)
    assert load_routine(d).modes == ("gantry", "standing")
    (d / "routine.json").write_text(json.dumps({"tracks": arms}))
    with pytest.raises(RoutineError, match="modes"):
        load_routine(d)
    base = {**arms, "base": [{"t0": 0.0, "t1": 1.0, "vx": 0.2}]}
    (d / "routine.json").write_text(json.dumps({"modes": ["gantry", "standing"], "tracks": base}))
    with pytest.raises(RoutineError, match="standing mode only"):
        load_routine(d)
    from g1.agent.task import mode_refusal
    spec = run_spec(write(tmp_path, base, name="walker"))
    assert spec["modes"] == ["standing"] and "standing mode only, not gantry" in mode_refusal(spec, "gantry")


def test_every_shipped_task_and_routine_says_its_modes():
    from g1.agent.task import load_task
    from g1.routines import routine_names
    for name in routine_names():
        assert load_routine(name).modes
    for d in (limits.ROOT / "tasks").iterdir():
        if (d / "task.json").is_file():
            assert load_task(d)["modes"], d
    assert load_task(limits.ROOT / "tasks" / "find_the_mug")["modes"] == ["standing"]      # it walks
