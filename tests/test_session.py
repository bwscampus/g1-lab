"""g1 start's engine (g1/session.py): the model chooses for chat and speech, skills run the one way,
the mode decides what may run, the settings switch on and off, and a stop always ends at neutral."""
import json
from pathlib import Path

import pytest

from g1 import session as session_module
from g1.console import parse
from g1.envs.moves import Moves, inventory
from g1.microphone import Microphone
from g1.session import (Command, rest_fsm, RoutineInfo, ScriptedUI, Session, TaskInfo, routine_menu, system_prompt, talk_menu,
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
    """A session on fakes. ``fsm``: what the robot reports (default: the mode's resting FSM)."""
    fsm = rest_fsm(mode) if fsm is None else fsm
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
    assert set(tasks) == {"point_at", "point_at_the_laptop"}                       # the rest say "session": false
    assert tasks["point_at"].instruction.startswith("find the {object}") and "object" in tasks["point_at"].fill


def test_the_model_is_offered_what_the_mode_allows():
    s, _, _ = make()
    names = [t["function"]["name"] for t in s.model_tools()]
    assert names == ["say", "move", "walk", "play_routine", "run_task", "remember", "ignore"]
    g, _, _ = make(mode="gantry")
    tools_ = {t["function"]["name"]: t["function"] for t in g.model_tools()}
    assert "move" not in tools_ and "walk" not in tools_                           # hanging: no built-ins
    assert tools_["play_routine"]["parameters"]["properties"]["name"]["enum"] == ["tpose"]   # no base routine
    assert set(tools_["run_task"]["parameters"]["properties"]["name"]["enum"]) == {"find_it", "point_on_request"}
    s.set_mic(False)
    assert s.offered()[1].keys() == {"find_it", "walk_to_it"}                       # a hearing task needs the mic
    assert "Answer with exactly one tool call" in system_prompt()


def test_skills_lists_only_what_this_mode_allows():
    g, _, _ = make(mode="gantry")
    text = "\n".join(g.skills())
    assert "mode: gantry" in text and "built-ins: (none in gantry mode)" in text
    assert "/move" not in text and "/walk" not in text and "walkabout" not in text and "walk_to_it" not in text
    assert "✗" not in text
    assert "· /routine tpose" in text and "· /task find_it" in text and "· /task point_on_request" in text
    g.set_mic(False)
    assert "point_on_request" not in "\n".join(g.skills())                         # a hearing task needs the mic
    s, _, _ = make()
    text = "\n".join(s.skills())
    assert all(f"/move {m}" in text for m in (516, 553, 604)) and "/walk" in text and "· /routine walkabout" in text


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
    assert s.moves.loco.calls == [550604, 501] and s.moves.loco.fsm == 501      # 802 after the dance: back to rest
    s.chat("no I was talking to her")
    s.chat("lie down")
    s.chat("what is your name")
    assert s.moves.loco.calls == [550604, 501]
    assert s.audio.said == ["I am about one point three metres tall.", "Here is my robot dance!",
                            "Sorry, I could not think of an answer just now."]
    assert s.counts["moves"] == 1 and s.counts["ignored"] == 2 and s.counts["errors"] == 1 and s.counts["typed"] == 5
    last = s.model.asked[-1]
    assert last[0]["role"] == "system" and last[-1] == {"role": "user", "content": "what is your name"}
    assert {"role": "assistant", "content": "[played move 604: mechanical dance]"} in last   # it remembers


def test_a_walk_is_capped_and_said_first():
    s, _, _ = make(("walk", {"turn_deg": 0, "forward_m": 5.0, "side_m": 0, "say": "Stepping forward!"}),
                   ("walk", {"turn_deg": 0, "forward_m": 0, "side_m": 0}))
    s.chat("come forward")
    vel = [c for c in s.moves.loco.calls if isinstance(c, tuple) and c[1] > 0]
    speed, poll = session_module.limits.get("walk_speed"), session_module.limits.get("base_send_period_s")
    assert len(vel) <= session_module.limits.get("talk_walk_max_m") / speed / poll + 2   # 5 m asked, the cap walked
    assert s.audio.said == ["Stepping forward!"] and s.counts["walks"] == 1
    s.chat("walk")                                                                      # nothing to walk: ignored
    assert s.counts["walks"] == 1 and s.counts["ignored"] == 1


def test_a_routine_and_a_task_the_model_chooses_are_confirmed_run_and_reported():
    runner = Runner(result={"ok": True, "outcome": "success", "summary": "I found it."})
    ui = ScriptedUI(True, True)
    s, _, _ = make(("play_routine", {"name": "tpose", "say": "Watch my T-pose!"}), ("run_task", {"name": "find_it"}),
                   runner=runner, ui=ui)
    s.chat("do a T-pose")
    s.chat("can you find the laptop")
    assert len(ui.asked) == 2 and "tpose routine" in ui.asked[0] and "standing mode" in ui.asked[0]
    assert [(k, n, m) for k, n, m, _ in runner.runs] == [("routine", "tpose", "standing"), ("task", "find_it", "standing")]
    assert runner.runs[0][3] == {"microphone": s.mic, "camera_shared": None, "speaker": True}   # the mic; no camera
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
    s.handle(Command("walk", ["0.2"]))
    s.handle(Command("say", text="hello everyone"))
    assert ui.asked == [] and [r[:2] for r in runner.runs] == [("routine", "tpose"), ("task", "find_it")]
    assert 550516 in s.moves.loco.calls and any(isinstance(c, tuple) for c in s.moves.loco.calls)
    assert "hello everyone" in s.audio.said and s.counts["model" if False else "typed"] == 0   # no model call


def test_unknown_or_impossible_skills_are_refused_and_said_why(capsys):
    s, _, _ = make(mode="gantry")
    s.handle(Command("move", ["bow"]))
    s.handle(Command("walk", ["0.5"]))
    s.handle(Command("routine", ["walkabout"]))
    s.handle(Command("task", ["walk_to_it"]))
    s.handle(Command("routine", ["nope"]))
    out = capsys.readouterr().out
    assert out.count("refused:") == 5 and "need standing mode" in out and "no routine 'nope'" in out
    assert "routine walkabout runs in standing mode only: /mode standing" in out
    assert "task walk_to_it runs in standing mode only" in out
    assert s.moves.loco.calls == [] and s.counts["refused"] == 5


def test_a_built_in_goes_back_to_walk_mode_at_once_and_skills_start_only_from_the_resting_fsm(capsys):
    s = None
    seen = []
    runner = Runner(check=lambda: seen.append(s.moves.loco.fsm))
    s, _, _ = make(runner=runner)
    s.handle(Command("move", ["bow"]))
    assert s.moves.loco.calls == [550516, 501] and s.fsm == 501          # a stored move ends in 802: back to rest
    s.handle(Command("routine", ["tpose"]))
    assert seen == [501]
    s.moves.loco.fsm = 802                                                  # left there somehow (the remote)
    s.handle(Command("routine", ["tpose"]))
    s.handle(Command("move", ["bow"]))
    out = capsys.readouterr().out
    assert len(runner.runs) == 1 and out.count("not standing mode's resting FSM 501") == 2 and "/mode standing" in out
    assert "(rests in 501)" in s.status()
    s.set_mode("standing")                                                  # the same mode: back to its rest
    assert s.moves.loco.fsm == 802                                          # not confirmed yet: untouched
    s.confirmed = "standing"
    s.set_mode("standing")
    assert s.moves.loco.fsm == 501
    s.handle(Command("routine", ["tpose"]))
    assert len(runner.runs) == 2


def test_in_gantry_a_task_runs_in_gantry_mode_from_200_and_never_from_limp(capsys):
    runner = Runner()
    s, _, _ = make(mode="gantry", runner=runner)
    s.handle(Command("task", ["find_it"]))
    assert runner.runs[0][2] == "gantry" and s.moves.loco.calls == []      # it rests in 200: nothing to switch
    s.moves.loco.fsm = 1
    s.handle(Command("task", ["find_it"]))
    assert len(runner.runs) == 1 and "resting FSM 200" in capsys.readouterr().out


def test_a_refusal_keeps_the_session_going():
    s, _, _ = make(("say", {"text": "Still here."}), runner=Runner(raises=SystemExit("--mode standing needs a robot standing")))
    s.handle(Command("routine", ["tpose"]))
    s.chat("are you ok")
    assert s.audio.said == ["Sorry, I could not do the tpose routine.", "Still here."] and s.counts["errors"] == 1


# -- mode and settings --------------------------------------------------------------------------

def test_switching_mode_asks_then_goes_to_the_new_resting_fsm():
    ui = ScriptedUI(True, True, False)
    s, _, _ = make(ui=ui)
    s.set_mode("gantry")
    assert s.mode == "gantry" and "hanging" in ui.asked[0] and s.moves.loco.calls == [200] and s.fsm == 200
    s.set_mode("standing")
    assert s.mode == "standing" and "standing on its own feet" in ui.asked[1] and s.moves.loco.calls == [200, 501]
    s.set_mode("gantry")                                          # declined: nothing switches
    assert s.mode == "standing" and s.moves.loco.calls == [200, 501]


def test_the_start_asks_then_rests_and_quit_parks(capsys):
    ui = ScriptedUI(True)
    s, _, _ = make(fsm=1, mode="gantry", ui=ui)                   # hanging limp
    assert s.start() and "hanging in the gantry" in ui.asked[0]
    assert s.moves.loco.calls == [4, 200] and s.fsm == 200        # damp -> locked stand -> main operation
    s.shutdown()
    assert s.moves.loco.calls[-1] == 1 and "damping" in capsys.readouterr().out   # it hangs: limp at the end
    s, _, _ = make(fsm=802, ui=ScriptedUI(True))
    assert s.start() and s.moves.loco.calls == [501]
    s.moves.loco.fsm = 802
    s.shutdown()
    assert s.moves.loco.calls == [501, 501] and "left standing in FSM 501" in capsys.readouterr().out   # never damped
    s, _, _ = make(mode="gantry", ui=ScriptedUI(False))
    assert not s.start() and s.confirmed is None
    s.shutdown()                                                  # no one said it hangs: never damped
    assert s.moves.loco.calls == [] and "not damped" in capsys.readouterr().out


def test_standing_is_refused_when_the_robot_does_not_stand(capsys):
    ui = ScriptedUI(True)
    s, _, _ = make(fsm=1, mode="gantry", ui=ui)
    s.set_mode("standing")
    assert s.mode == "gantry" and ui.asked == [] and "not standing" in capsys.readouterr().out


def test_with_the_mic_off_speech_is_ignored_and_a_hearing_task_refused(capsys):
    s, mic, clock = make()
    s.set_mic(False)
    heard(mic, "do a dance")
    clock["t"] += 5.0
    assert s.heard() is None
    s.handle(Command("task", ["point_on_request"]))
    assert "/mic on first" in capsys.readouterr().out
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

def test_a_stop_ends_a_built_in_move_and_a_walk_at_neutral():
    fired = []

    def ctrl_c_once(now):
        if not fired and now > 0.6:
            fired.append(now)
            raise KeyboardInterrupt
    s, _, _ = make(sleep_hook=ctrl_c_once)
    s.handle(Command("move", ["bow"]))
    assert s.moves.loco.calls[-1] == 501 and s.counts["stopped"] == 1         # back to the FSM it started from
    fired.clear()
    s.handle(Command("walk", ["1.0"]))
    assert s.moves.loco.calls[-1] == ("vel", 0.0, 0.0, 0.0) and s.counts["stopped"] == 2   # zero velocity


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
