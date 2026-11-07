"""g1 start's engine (g1/session.py): the model chooses for chat and speech, skills run the one way,
the mode decides what may run, the settings switch on and off, and a stop always ends at neutral."""
import json
from pathlib import Path

import pytest

from g1 import session as session_module
from g1.console import parse
from g1.envs.moves import Moves, inventory
from g1.microphone import Microphone
from g1.session import (Command, RoutineInfo, ScriptedUI, Session, TaskInfo, routine_menu, system_prompt, talk_menu,
                        task_menu, tools)
from g1.vlm import ToolCall
from tests.envs.test_moves import Arm, Loco

MIMIC = {"fsm_id": 550, "api_id": 1, "motion": {
    "101": {"duration": 0.02, "name": "stance"},
    "401": {"duration": 3.08, "name": "vq_Y-qishen-faceup_001_get_up"},
    "516": {"duration": 4.92, "name": "bow"},
    "604": {"duration": 17.92, "name": "vq_YSKJ_JIXIEWU3_001_XX_b1f11bf1_G1_50hz"},
    "553": {"duration": 6.02, "name": "vq_shuangshoudazhaohu"}}}
ROUTINES = {"tpose": RoutineInfo("arms out in a T"),
            "walkabout": RoutineInfo("a step forward", uses_base=True, modes=("standing",))}
TASKS = {"find_it": TaskInfo("find the laptop and point at it"), "walk_to_it": TaskInfo("walk to it", modes=("standing",)),
         "point_on_request": TaskInfo("point at what you hear", hears=True)}


class Model:
    """Answers each request with the next scripted (name, arguments); an Exception is raised."""
    def __init__(self, *replies):
        self.replies, self.asked, self.tools = list(replies), [], []

    def complete(self, messages, *, tools):
        self.asked.append(messages)
        self.tools.append([t["function"]["name"] for t in tools])
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return ToolCall("call_1", r[0], json.dumps(r[1]))


class Audio:
    def __init__(self):
        self.said = []

    def TtsMaker(self, text, speaker_id):
        self.said.append(text)
        return 0


class Runner:
    """Records each routine or task run (kind, name, mode, settings; a task's slots in ``fills``) and
    returns a result."""
    def __init__(self, result=None, raises=None, check=None):
        self.runs, self.result, self.raises, self.check = [], result or {"ok": True, "outcome": "success"}, raises, check
        self.fills = []

    def __call__(self, kind, name, mode, settings, fill=None):
        self.runs.append((kind, name, mode, settings))
        if fill:
            self.fills.append(fill)
        if self.check:
            self.check()
        if self.raises:
            raise self.raises
        return self.result


def make(*replies, fsm=None, mode="standing", runner=None, ui=None, sleep_hook=None, **kw):
    """A session on fakes. ``fsm``: what the robot reports (default 200: both modes allow it)."""
    fsm = 200 if fsm is None else fsm
    clock = {"t": 0.0}

    def sleep(dt):
        clock["t"] += dt
        if sleep_hook is not None:
            sleep_hook(clock["t"])
    now = lambda: clock["t"]
    loco = Loco(fsm)
    loco.GetMimicMotion = lambda: (0, MIMIC)
    moves = Moves(loco, Arm(), sleep=sleep, now=now, confirm=lambda q: True)
    mic = Microphone(now)
    menu = talk_menu(inventory(loco, Arm(), []))
    s = Session(mic, Audio(), moves, Model(*replies), menu, mode=mode, routines=dict(ROUTINES), tasks=kw.pop("tasks", dict(TASKS)),
                runner=runner if runner is not None else Runner(), ui=ui or ScriptedUI(), sleep=sleep, now=now, **kw)
    s._refresh_fsm()                                              # as run() does first
    s.mic_on = True                                               # as /mic on (it starts off)
    return s, mic, clock


def heard(mic, text):
    mic.on_message(json.dumps({"text": text, "language": "<|en|>"}))


# -- the menu ------------------------------------------------------------------------------------

def test_only_gestures_and_dances_are_built_ins():
    s, _, _ = make()
    assert set(s.menu) == {516, 553, 604}                        # never the stance or a get-up
    assert s.menu[604]["name"] == "mechanical dance" and s.menu[553]["name"] == "greet with both hands"
    move = next(x for x in tools(s.menu) if x["function"]["name"] == "move")["function"]
    assert move["parameters"]["properties"]["move_id"]["enum"] == [516, 553, 604]
    assert s.resolve_move("bow") == 516 and s.resolve_move("516") == 516 and s.resolve_move("greet") == 553
    assert s.resolve_move("401") is None and s.resolve_move("dance") is None or s.resolve_move("dance") == 604


def test_the_menus_offer_the_routines_and_tasks_that_exist():
    from g1.routines import routine_names
    routines, tasks = routine_menu(), task_menu()
    assert set(routines) == set(routine_names()) and "T" in routines["tpose"].notes
    assert set(tasks) == {"point_at"}                                              # the rest say "session": false
    assert tasks["point_at"].instruction.startswith("find the {object}") and "object" in tasks["point_at"].fill


def test_the_model_is_offered_what_the_mode_allows():
    s, _, _ = make()
    names = [t["function"]["name"] for t in s.model_tools()]
    assert names == ["say", "move", "play_routine", "run_task", "remember", "ignore"]          # never a walk
    g, _, _ = make(mode="gantry")
    tools_ = {t["function"]["name"]: t["function"] for t in g.model_tools()}
    assert "move" not in tools_                                                     # hanging: no built-ins
    assert tools_["play_routine"]["parameters"]["properties"]["name"]["enum"] == ["tpose"]   # no base routine
    assert set(tools_["run_task"]["parameters"]["properties"]["name"]["enum"]) == {"find_it", "point_on_request"}
    s.set_mic(False)
    assert s.offered()[1].keys() == {"find_it", "walk_to_it", "point_on_request"}   # a hearing task hears with the mic off too
    assert "Answer with exactly one tool call" in system_prompt()


def test_skills_lists_only_what_this_mode_allows():
    g, _, _ = make(mode="gantry")
    text = "\n".join(g.skills())
    assert "mode: gantry" in text and "built-ins: (none in gantry mode)" in text
    assert "/move" not in text and "walkabout" not in text and "walk_to_it" not in text
    assert "✗" not in text
    assert "· /routine tpose" in text and "· /task find_it" in text and "· /task point_on_request" in text
    g.set_mic(False)
    assert "point_on_request" in "\n".join(g.skills())                             # the mic setting is chat's only
    s, _, _ = make()
    text = "\n".join(s.skills())
    assert all(f"/move {m}" in text for m in (516, 553, 604)) and "/walk" not in text and "· /routine walkabout" in text


def test_with_the_speaker_off_replies_are_printed_never_said_and_runs_follow_it(capsys):
    runner = Runner()
    s, _, clock = make(("say", {"text": "I am a robot"}), runner=runner)
    s.handle(Command("speaker", ["off"]))
    assert "speaker off" in s.status()
    t0 = clock["t"]
    s.chat("who are you", "typed")
    s.handle(Command("say", text="hello"))
    s.handle(Command("task", ["find_it"]))
    out = capsys.readouterr().out
    assert s.audio.said == [] and clock["t"] == t0                                # nothing sent, nothing waited
    assert "robot: I am a robot" in out and "robot: hello" in out
    assert runner.runs[0][3]["speaker"] is False                                   # the task's say is printed too
    s.handle(Command("speaker", ["on"]))
    s.handle(Command("say", text="hello"))
    s.handle(Command("task", ["find_it"]))
    assert s.audio.said[0] == "hello" and runner.runs[1][3]["speaker"] is True and "speaker on" in s.status()


def test_chat_says_moves_ignores_and_survives_a_model_error():
    s, _, _ = make(("say", {"text": "I am about one point three metres tall."}),
                   ("move", {"move_id": 604, "say": "Here is my robot dance!"}),
                   ("ignore", {"reason": "side chatter"}),
                   ("move", {"move_id": 401}),                      # not allowed: nothing moves
                   RuntimeError("network down"))
    s.chat("how tall are you")
    s.chat("can you dance")
    assert s.moves.loco.calls == [550604, 200] and s.moves.loco.fsm == 200     # 802 after it: back to the caller's 200
    s.chat("no I was talking to her")
    s.chat("lie down")
    s.chat("what is your name")
    assert s.moves.loco.calls == [550604, 200]
    assert s.audio.said == ["I am about one point three metres tall.", "Here is my robot dance!",
                            "Sorry, I could not think of an answer just now."]
    assert s.counts["moves"] == 1 and s.counts["ignored"] == 2 and s.counts["errors"] == 1 and s.counts["typed"] == 5
    last = s.model.asked[-1]
    assert last[0]["role"] == "system" and last[-1] == {"role": "user", "content": "what is your name"}
    assert {"role": "assistant", "content": "[played move 604: mechanical dance]"} in last   # it remembers


def test_it_never_walks_and_says_so():
    s, _, _ = make(("walk", {"forward_m": 1.0}), ("say", {"text": "I stay where I am, sorry!"}))
    s.chat("come forward")                                          # a walk call is not a tool: ignored, nothing moves
    s.chat("walk over here")
    assert "You cannot walk" in system_prompt() and s.moves.loco.calls == [] and s.counts["ignored"] == 1
    assert s.audio.said == ["I stay where I am, sorry!"]
def test_a_routine_and_a_task_the_model_chooses_are_confirmed_run_and_reported():
    runner = Runner(result={"ok": True, "outcome": "success", "summary": "I found it."})
    ui = ScriptedUI(True, True)
    s, _, _ = make(("play_routine", {"name": "tpose", "say": "Watch my T-pose!"}), ("run_task", {"name": "find_it"}),
                   runner=runner, ui=ui)
    s.chat("do a T-pose")
    s.chat("can you find the laptop")
    assert len(ui.asked) == 2 and "tpose routine" in ui.asked[0] and "standing mode" in ui.asked[0]
    assert [(k, n, m) for k, n, m, _ in runner.runs] == [("routine", "tpose", "standing"), ("task", "find_it", "standing")]
    assert runner.runs[0][3] == {"microphone": s.mic, "camera_shared": None, "speaker": True,   # the mic; no camera
                                 "loco_shared": s.moves.loco, "audio_shared": s.audio}         # and its clients
    assert s.audio.said == ["Watch my T-pose!", "Done.", "I will find the laptop and point at it.", "I found it."]


def test_no_at_the_keyboard_moves_nothing():
    runner = Runner()
    s, _, _ = make(("play_routine", {"name": "tpose"}), runner=runner, ui=ScriptedUI(False))
    s.chat("do a T-pose")
    assert runner.runs == [] and s.audio.said[-1] == "Okay, I will not do that." and s.moves.loco.calls == []


# -- direct commands ----------------------------------------------------------------------------

def test_a_skill_typed_runs_without_asking():
    runner, ui = Runner(), ScriptedUI()
    s, _, _ = make(runner=runner, ui=ui)
    s.handle(Command("routine", ["tpose"]))
    s.handle(Command("task", ["find_it"]))
    s.handle(Command("move", ["bow"]))
    s.handle(Command("say", text="hello everyone"))
    assert ui.asked == [] and [r[:2] for r in runner.runs] == [("routine", "tpose"), ("task", "find_it")]
    assert 550516 in s.moves.loco.calls
    assert "hello everyone" in s.audio.said and s.counts["model" if False else "typed"] == 0   # no model call


def test_unknown_or_impossible_skills_are_refused_and_said_why(capsys):
    s, _, _ = make(mode="gantry")
    s.handle(Command("move", ["bow"]))
    s.handle(Command("routine", ["walkabout"]))
    s.handle(Command("task", ["walk_to_it"]))
    s.handle(Command("routine", ["nope"]))
    out = capsys.readouterr().out
    assert out.count("refused:") == 4 and "need standing mode" in out and "no routine 'nope'" in out
    assert "routine walkabout runs in standing mode only: /mode standing" in out
    assert "task walk_to_it runs in standing mode only" in out
    assert s.moves.loco.calls == [] and s.counts["refused"] == 4


def test_a_refusal_keeps_the_session_going():
    s, _, _ = make(("say", {"text": "Still here."}), runner=Runner(raises=SystemExit("--mode standing needs a robot standing")))
    s.handle(Command("routine", ["tpose"]))
    s.chat("are you ok")
    assert s.audio.said == ["Sorry, I could not do the tpose routine.", "Still here."] and s.counts["errors"] == 1


# -- mode and settings --------------------------------------------------------------------------

def test_with_the_mic_off_speech_is_not_chat_but_a_hearing_task_still_hears(capsys):
    s, mic, clock = make()
    s.set_mic(False)
    heard(mic, "do a dance")
    clock["t"] += 5.0
    assert s.heard() is None
    s.handle(Command("task", ["point_on_request"]))
    assert s.runner.runs[-1][3]["microphone"] is mic and "say or type \"stop\"" in capsys.readouterr().out
    s.set_mic(True)
    heard(mic, "hello")
    assert s.heard() is None                                      # not yet quiet long enough
    clock["t"] += 5.0
    assert s.heard() == "hello"                                   # what was said while off is never answered


def test_the_camera_and_lidar_switch_on_and_off_and_runs_share_the_camera():
    opened, stopped = [], []

    class Cam:
        def stop(self):
            stopped.append("camera")

    class View:
        url, port = "http://127.0.0.1:8765/", 8765

        def stop(self):
            stopped.append("view")

    class Lidar:
        url = "http://127.0.0.1:8766/"

        def start(self):
            opened.append("lidar")

        def stop(self):
            stopped.append("lidar")
    runner = Runner()
    s, _, _ = make(runner=runner, camera_factory=lambda mode: opened.append(f"camera {mode}") or Cam(),
                   viewer_factory=lambda cam: View(), lidar_factory=Lidar)
    s.set_camera(True)
    s.set_lidar(True)
    assert "camera http://127.0.0.1:8765/" in s.status() and "lidar http://127.0.0.1:8766/" in s.status()
    s.handle(Command("task", ["find_it"]))
    assert runner.runs[0][3]["camera_shared"] is s.camera          # the run reuses the session's camera
    s.set_camera(False)
    s.set_lidar(False)
    assert opened == ["camera standing", "lidar"] and stopped == ["view", "camera", "lidar"]
    assert "camera off" in s.status() and "lidar off" in s.status()


def test_a_lidar_that_cannot_start_leaves_it_off(capsys):
    class Broken:
        def start(self):
            raise RuntimeError("lidar_driver is off; switch it on in the Unitree app")
    s, _, _ = make(lidar_factory=Broken)
    s.set_lidar(True)
    assert s.lidar is None and "lidar_driver is off" in capsys.readouterr().out


# -- stopping -------------------------------------------------------------------------------------

def test_a_stop_ends_a_built_in_move_at_neutral():
    fired = []

    def ctrl_c_once(now):
        if not fired and now > 0.6:
            fired.append(now)
            raise KeyboardInterrupt
    s, _, _ = make(fsm=501, sleep_hook=ctrl_c_once)
    s.handle(Command("move", ["bow"]))
    assert s.moves.loco.calls[-1] == 501 and s.counts["stopped"] == 1         # back to the FSM it started from


def test_a_stopped_routine_is_said_and_the_session_goes_on():
    s, _, _ = make(("say", {"text": "Ready again."}), runner=Runner(raises=KeyboardInterrupt()))
    s.handle(Command("routine", ["tpose"]))
    s.chat("hello")
    assert s.counts["stopped"] == 1 and s.audio.said == ["Ready again."]
    s2, _, _ = make(runner=Runner(result={"ok": False, "outcome": "failed", "stopped": True}))
    s2.handle(Command("routine", ["tpose"]))
    assert s2.audio.said == ["I stopped."]


def test_run_takes_events_and_speech_until_quit_and_shuts_down(capsys):
    import threading
    stopped = []

    class Lidar:
        url = "x"

        def start(self):
            pass

        def stop(self):
            stopped.append("lidar")
    s, mic, clock = make(("say", {"text": "Hi!"}), lidar_factory=Lidar)
    s.set_lidar(True)
    heard(mic, "hello robot")
    clock["t"] += 5.0                                                  # quiet long enough to answer
    s.events.put(Command("status"))
    done = threading.Event()

    def watch():                                                       # quit once it has answered
        for _ in range(500):
            if s.audio.said:
                break
            threading.Event().wait(0.01)
        s.quit.set()
        done.set()
    threading.Thread(target=watch, daemon=True).start()
    assert s.run() == 0 and done.wait(5)
    out = capsys.readouterr().out
    assert s.audio.said == ["Hi!"] and "heard: hello robot" in out and stopped == ["lidar"] and "session:" in out


def test_the_session_reaches_the_arms_only_through_run_once():
    """Routines and tasks run only through cli.run_once (the env's takeover, dry-run and safe
    return); the session never touches arm_sdk, the FSM or a tool itself."""
    import ast
    tree = ast.parse(Path(session_module.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert {m for m in imported if m.startswith(("g1.agent", "g1.tools", "g1.envs.robot", "g1.cli", "g1.routines"))} \
        == {"g1.agent.task", "g1.cli"}, imported
    froms = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("g1.cli", "run_once") in froms and not any(m == "g1.cli" and name != "run_once" for m, name in froms)
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for forbidden in ("arm", "arm_sdk", "switch_to", "Move", "SetFsmId", "ExecuteAction", "base", "TOOLS"):
        assert forbidden not in used, forbidden
    calls = {n.args[1].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "call"
             and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)}
    assert calls == {"TtsMaker"}, calls


def test_a_routine_runs_for_real_on_the_twin_through_the_session(tmp_path):
    """The real routine through run_once on the in-process twin, as the robot runner does."""
    from g1.cli import run_once
    from g1.routines import run_spec
    from g1.session import result_of
    from tests.agent.test_agent import Scripted
    from tests.doubles import scripted_sim
    envs = []

    def runner(kind, name, mode, settings):
        env = scripted_sim(Scripted())
        envs.append(env)
        spec = {**run_spec(name), "dir": tmp_path / name}
        return result_of(*run_once(spec, sim=True, mode=None, env=env, ask=False))
    s, _, _ = make(runner=runner)
    s.handle(Command("routine", ["tpose"]))
    assert s.audio.said == ["Done."] and envs[0].actions[-1].weight == pytest.approx(0.0, abs=1e-3)


# -- a task's slot (point at X) ---------------------------------------------------------------------

POINT = TaskInfo("find the {object} and point at it", fill={"object": {"note": "what to point at", "default": "laptop"}})


def test_the_model_fills_a_tasks_slot_and_the_question_and_runner_get_it(capsys):
    runner, ui = Runner(), ScriptedUI(True)
    s, _, _ = make(("run_task", {"name": "point_at", "object": "water bottle", "say": "Looking for it!"}),
                   ("run_task", {"name": "point_at"}), runner=runner, ui=ui, tasks={**TASKS, "point_at": POINT})
    run_task = next(t for t in s.model_tools() if t["function"]["name"] == "run_task")["function"]
    assert "what to point at" in run_task["parameters"]["properties"]["object"]["description"]
    s.chat("point at the water bottle")
    assert "find the water bottle and point at it" in ui.asked[0]
    assert runner.runs[0][:3] == ("task", "point_at", "standing") and runner.fills == [{"object": "water bottle"}]
    assert "[task point_at (water bottle): success]" in [h["content"] for h in s.history]
    s.chat("point at it")                                           # no object: asked again, nothing runs
    assert len(runner.runs) == 1 and s.audio.said[-1] == "What should I point at? Please tell me again."


def test_a_typed_task_takes_the_rest_of_the_line_as_its_slot(capsys):
    runner = Runner()
    s, _, _ = make(runner=runner, tasks={**TASKS, "point_at": POINT})
    s.handle(parse("/task point_at the teacher's {red} cup"))
    s.handle(parse("/task point_at"))
    assert runner.fills == [{"object": "the teacher's red cup"}] and "needs its object" in capsys.readouterr().out


# -- look_up ---------------------------------------------------------------------------------------

class Searching(Model):
    provider = "openrouter"

    def __init__(self, *replies, found="It is sunny and 18 degrees in Boston today.", wait=None):
        super().__init__(*replies)
        self.found, self.wait, self.searched = found, wait, []

    def search(self, query, timeout=None):
        self.searched.append(query)
        if self.wait is not None:
            self.wait.wait(5)
        if isinstance(self.found, Exception):
            raise self.found
        return self.found


def test_look_up_says_its_line_searches_then_answers_from_the_results():
    s, _, _ = make()
    s.model = Searching(("look_up", {"query": "Boston weather today", "say": "Let me check."}),
                        ("say", {"text": "It is sunny and eighteen degrees."}))
    assert "look_up" in [t["function"]["name"] for t in s.model_tools()]
    s.chat("what's the weather in Boston")
    assert s.model.searched == ["Boston weather today"]
    assert s.audio.said == ["Let me check.", "It is sunny and eighteen degrees."] and s.counts["looked_up"] == 1
    second = s.model.asked[1]
    assert "sunny and 18 degrees" in second[-1]["content"] and "look_up" not in s.model.tools[1]   # one look-up a turn


def test_look_up_is_bounded_said_when_it_fails_and_only_offered_with_search(monkeypatch):
    import threading
    s, _, _ = make()
    assert "look_up" not in [t["function"]["name"] for t in s.model_tools()]   # a model without search
    hold = threading.Event()
    monkeypatch.setitem(session_module.limits.LIMITS["look_up_timeout_s"], "value", 0.05)
    s.model = Searching(("look_up", {"query": "news"}), wait=hold)
    s.chat("any news today")
    hold.set()
    s.model = Searching(("look_up", {"query": "news"}), found=RuntimeError("quota"))
    s.chat("any news today")
    assert s.audio.said == ["Sorry, I could not look that up right now."] * 2 and len(s.model.asked) == 1



def test_a_long_reply_is_spoken_whole_in_sentence_pieces(monkeypatch, capsys):
    monkeypatch.setitem(session_module.limits.LIMITS["say_max_chars"], "value", 40)
    text = ("Brentwood School is in Los Angeles. It has two campuses! The upper school offers many courses in "
            "science and the arts.")
    pieces = session_module.speech_pieces(text, 40)
    assert " ".join(pieces) == text and all(len(p) <= 40 for p in pieces)
    assert pieces[:2] == ["Brentwood School is in Los Angeles.", "It has two campuses!"]
    s, _, _ = make()
    s.speak(text)
    assert " ".join(s.audio.said) == text and len(s.audio.said) == len(pieces) and f"robot: {text}" in capsys.readouterr().out


# -- the FSM: modes block, /fsm switches, the start brings it up ------------------------------------

def test_the_start_brings_it_up_from_limp_in_both_modes_and_quit_changes_no_fsm(capsys):
    for mode in ("gantry", "standing"):
        ui = ScriptedUI(True)
        end = 200 if mode == "gantry" else 802                     # standing ends where stored moves play
        s, _, _ = make(fsm=0, mode=mode, ui=ui)
        assert s.start() and f"0 (zero torque) -> 1 (damp) -> 4 (locked stand) -> {end}" in ui.asked[0]
        assert s.moves.loco.calls == [1, 4, end] and s.fsm == end
        s.shutdown()
        assert s.moves.loco.calls == [1, 4, end] and f"left in FSM {end}" in capsys.readouterr().out
    s, _, _ = make(fsm=501, ui=ScriptedUI(True))                     # standing already: left as it is
    assert s.start() and s.moves.loco.calls == []
    s, _, _ = make(fsm=1, mode="gantry", ui=ScriptedUI(False))       # declined: not started, nothing switched
    assert not s.start() and s.moves.loco.calls == []


def test_fsm_switches_now_unless_the_mode_blocks_it(capsys):
    ui = ScriptedUI(False, True)
    s, _, _ = make(ui=ui)                                            # standing, in 200
    s.handle(Command("fsm", ["501"]))
    out = capsys.readouterr().out
    assert s.moves.loco.calls == [501] and s.fsm == 501 and "FSM 501 (walk, 3-DoF waist) after" in out
    s.handle(Command("fsm", ["1"]))                                  # standing blocks damp: never asked, never sent
    out = capsys.readouterr().out
    assert s.moves.loco.calls == [501] and ui.asked == [] and "standing mode blocks FSM 1" in out and "/mode gantry" in out
    s.handle(Command("fsm", ["3"]))                                  # sit: asked, declined
    assert s.moves.loco.calls == [501] and "sit" in ui.asked[0] and "not switched" in capsys.readouterr().out
    s.moves.loco.SetFsmId = lambda fsm: 0                           # accepted, never entered (501 -> 200 on our G1)
    s.handle(Command("fsm", ["200"]))
    assert "did not enter 200 (main operation)" in capsys.readouterr().out and s.fsm == 501
    g, _, _ = make(mode="gantry")
    g.handle(Command("fsm", ["501"]))
    assert g.moves.loco.calls == [] and "gantry mode blocks FSM 501" in capsys.readouterr().out


def test_mode_never_changes_the_fsm_and_is_refused_from_one_it_blocks(capsys):
    ui = ScriptedUI(True, True)
    s, _, _ = make(ui=ui)                                            # standing, in 200
    s.set_mode("gantry")
    assert s.mode == "gantry" and "hanging" in ui.asked[0] and s.moves.loco.calls == []
    s.moves.loco.fsm = 1
    s.set_mode("standing")                                          # damp: standing blocks it
    assert s.mode == "gantry" and len(ui.asked) == 1 and "standing mode blocks" in capsys.readouterr().out
    s2, _, _ = make(fsm=501, ui=ScriptedUI(True))
    s2.set_mode("gantry")                                           # 501: gantry blocks it
    assert s2.mode == "standing" and "gantry mode blocks" in capsys.readouterr().out


def test_the_mic_starts_off_and_the_speaker_on():
    s = Session(Microphone(lambda: 0.0), Audio(), None, None, {})
    assert not s.mic_on and s.speaker_on and "mic off" in s.status() and "speaker on" in s.status()


def test_built_ins_switch_as_they_need_and_go_back_and_runs_stay_in_the_fsm(capsys):
    runner = Runner()
    s, _, _ = make(fsm=501, runner=runner)
    s.handle(Command("move", ["bow"]))                              # plays from 501, ends in 802: back to 501
    assert s.moves.loco.calls == [550516, 501] and s.fsm == 501
    s.handle(Command("routine", ["tpose"]))                         # a run: no FSM change from the session
    assert s.moves.loco.calls == [550516, 501] and runner.runs[0][:2] == ("routine", "tpose") and s.fsm == 501
    s.moves.loco.fsm = 4
    s.handle(Command("move", ["bow"]))                              # 4 does not play moves: 200 first, then back to 4
    assert s.moves.loco.calls[-3:] == [200, 550516, 4]


def test_a_blocked_fsm_refuses_a_run_and_gantry_runs_from_damp(capsys):
    runner = Runner()
    g, _, _ = make(fsm=1, mode="gantry", runner=runner)             # gantry from damp: the run brings it up itself
    g.handle(Command("task", ["find_it"]))
    assert g.moves.loco.calls == [] and runner.runs[-1][:3] == ("task", "find_it", "gantry")
    s2, _, _ = make(fsm=0, runner=runner)                           # limp while standing: blocked
    s2.handle(Command("routine", ["tpose"]))
    assert len(runner.runs) == 1 and "standing mode blocks" in capsys.readouterr().out


def test_the_menus_list_what_the_mode_allows_whatever_the_fsm(capsys):
    s, _, _ = make(fsm=1)                                           # standing, but damp: nothing may run now
    text = "\n".join(s.skills())
    assert "/move 516" in text and "/routine tpose" in text and "/task find_it" in text and "(none" not in text
    assert {"move", "play_routine", "run_task"} <= {t["function"]["name"] for t in s.model_tools()}
    s.handle(Command("routine", ["tpose"]))                         # running one still checks the FSM
    assert s.runner.runs == [] and "standing mode blocks" in capsys.readouterr().out


def test_a_line_typed_while_a_hearing_task_runs_reaches_it_and_is_not_chat_afterwards():
    from g1.console import Console
    seen, out = [], []
    s = None

    def task_hears(kind, name, mode, settings, fill=None):
        c.submit("stop now")                                        # typed while it runs
        seen.append([h.text for h in settings["microphone"].since(0)])
        return {"ok": True, "outcome": "success"}
    s, mic, clock = make(runner=task_hears)
    c = Console(s, interrupt=lambda: None, out=out.append)
    s.busy.set()                                                    # as _busy does while the task runs
    s.handle(Command("task", ["point_on_request"]))
    assert seen == [["stop now"]] and "to point_on_request: stop now" in out and s.events.empty()
    clock["t"] += 5.0
    assert s.heard() is None and not s.listening_task                # not answered as chat once it ended
    s.busy.clear()
    c.submit("hello")                                               # nothing listening: chat, queued as usual
    assert s.events.get_nowait().text == "hello"


def test_an_unknown_fsm_never_refuses_a_skill_and_runs_get_the_sessions_clients():
    runner = Runner()
    s, _, _ = make(runner=runner)
    s.moves.loco.GetFsmId = lambda: (3104, None)                     # no reply (the 3104s seen on the robot)
    s.handle(Command("routine", ["tpose"]))
    assert runner.runs[0][:2] == ("routine", "tpose")
    settings = runner.runs[0][3]
    assert settings["loco_shared"] is s.moves.loco and settings["audio_shared"] is s.audio


def test_fsm_without_an_id_lists_every_fsm_and_marks_where_it_is_and_what_the_mode_blocks():
    from g1.envs import sdk
    s, _, _ = make(fsm=501)
    lines = s.fsm_lines()
    assert len([l for l in lines if l[:6].strip().lstrip(">").strip().isdigit()]) == len(sdk.FSM_INFO)
    here = next(l for l in lines if l.startswith(" >"))
    assert "501" in here and "walk, 3-DoF waist" in here and "walk_fsm" in here
    assert "[blocked in standing]" in next(l for l in lines if l.split()[0] == "1")
    assert s.moves.loco.calls == []                                 # listing never switches


def test_the_arms_up_and_down_routines_load_and_hold_the_six_seven_pose():
    from g1.routines import compile_segments, load_routine
    up, down = load_routine("arms_up"), load_routine("arms_down")
    assert up.points[0][1] == up.points[-1][1] and up.upper_body_only and down.upper_body_only
    assert sum(seg.duration for seg in compile_segments(up)) <= session_module.limits.get("routine_duration_max_s")
