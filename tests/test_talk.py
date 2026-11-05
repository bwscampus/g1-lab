import json

import pytest
from pathlib import Path

from g1 import talk
from g1.envs.moves import Moves, inventory
from g1.microphone import Microphone
from g1.talk import Keys, Talk, talk_menu, tools
from g1.vlm import ToolCall
from tests.envs.test_moves import Arm, Loco

MIMIC = {"fsm_id": 550, "api_id": 1, "motion": {
    "101": {"duration": 0.02, "name": "stance"},
    "401": {"duration": 3.08, "name": "vq_Y-qishen-faceup_001_get_up"},
    "516": {"duration": 4.92, "name": "bow"},
    "604": {"duration": 17.92, "name": "vq_YSKJ_JIXIEWU3_001_XX_b1f11bf1_G1_50hz"},
    "553": {"duration": 6.02, "name": "vq_shuangshoudazhaohu"}}}


class Model:
    """Answers each request with the next scripted (name, arguments); an Exception is raised."""
    def __init__(self, *replies):
        self.replies, self.asked = list(replies), []

    def complete(self, messages, *, tools):
        self.asked.append(messages)
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


class ScriptedKeys(Keys):
    """The keyboard, scripted: each y/N question takes the next answer (no stdin thread)."""
    def __init__(self, *answers):
        super().__init__(interrupt=lambda: self.interrupted.append(True))
        self.scripted, self.asked, self.interrupted = list(answers), [], []

    def confirm(self, question):
        self.asked.append(question)
        return bool(self.scripted.pop(0)) if self.scripted else False


def make(*replies, fsm=501, routines=None, tasks=None, runner=None, keys=None, sleep_hook=None):
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
    t = Talk(mic, Audio(), moves, Model(*replies), menu, sleep=sleep, now=now, routines=routines, tasks=tasks,
             runner=runner, keys=keys or ScriptedKeys())
    return t, mic, clock


def heard(mic, text):
    mic.on_message(json.dumps({"text": text, "language": "<|en|>"}))


def test_only_gestures_and_dances_are_offered():
    t, _, _ = make()
    assert set(t.menu) == {516, 553, 604}                        # never the stance or a get-up
    assert t.menu[604]["name"] == "mechanical dance" and t.menu[553]["name"] == "greet with both hands"
    move = next(x for x in tools(t.menu) if x["function"]["name"] == "move")["function"]
    assert move["parameters"]["properties"]["move_id"]["enum"] == [516, 553, 604]
    assert {x["function"]["name"] for x in t.tools} == {"say", "move", "walk", "ignore"}


def test_listen_joins_what_is_said_until_a_pause():
    t, mic, clock = make()
    heard(mic, "hello robot")
    clock["t"] += 0.3
    heard(mic, "how tall are you")
    text, seq = t.listen(0)
    assert text == "hello robot how tall are you" and seq == 2


def test_say_move_ignore_and_a_model_error():
    t, mic, _ = make(("say", {"text": "I am about one point three metres tall."}),
                     ("move", {"move_id": 604, "say": "Here is my robot dance!"}),
                     ("ignore", {"reason": "side chatter"}),
                     ("move", {"move_id": 401}),                      # not allowed: nothing moves
                     RuntimeError("network down"))
    t.answer("how tall are you")
    t.answer("can you dance")
    assert t.moves.loco.calls == [550604] and t.moves.loco.fsm == 802
    t.answer("no I was talking to her")
    t.answer("lie down")
    t.answer("what is your name")
    assert t.moves.loco.calls == [550604]
    assert t.audio.said == ["I am about one point three metres tall.", "Here is my robot dance!",
                            "Sorry, I could not think of an answer just now."]
    assert t.counts == {"heard": 0, "said": 3, "moves": 1, "walks": 0, "routines": 0, "tasks": 0, "stopped": 0,
                        "ignored": 2, "errors": 1}
    last = t.model.asked[-1]
    assert last[0]["role"] == "system" and last[-1] == {"role": "user", "content": "what is your name"}
    assert {"role": "assistant", "content": "[played move 604: mechanical dance]"} in last   # it remembers


def test_talk_chooses_but_never_reaches_the_arms_itself():
    """Talk runs routines and tasks only through cli.run_once (the env's takeover, dry-run and safe
    return), names them only through g1.routines and g1.agent.task, and never touches arm_sdk,
    the FSM or a tool itself."""
    import ast
    tree = ast.parse(Path(talk.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert {m for m in imported if m.startswith(("g1.agent", "g1.tools", "g1.envs.robot", "g1.cli", "g1.routines"))} \
        == {"g1.agent.task", "g1.cli"}, imported
    froms = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("g1.cli", "run_once") in froms and not any(m == "g1.cli" and name != "run_once" for m, name in froms)
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for forbidden in ("arm", "arm_sdk", "switch_to", "Move", "SetFsmId", "ExecuteAction", "base", "TOOLS"):
        assert forbidden not in used, forbidden
    # the only robot calls it makes itself: speech; moves go through Moves.play, the FSM through Moves.switch
    calls = {n.args[1].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "call"
             and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)}
    assert calls == {"TtsMaker"}, calls


def test_walk_is_capped_and_said_first():
    t, _, _ = make(("walk", {"turn_deg": 0, "forward_m": 5.0, "side_m": 0, "say": "Stepping forward!"}),
                   ("walk", {"turn_deg": 0, "forward_m": 0, "side_m": 0}))
    t.answer("come forward")
    vel = [c for c in t.moves.loco.calls if isinstance(c, tuple) and c[1] > 0]
    speed, poll = talk.limits.get("walk_speed"), talk.limits.get("base_send_period_s")
    assert len(vel) <= talk.limits.get("talk_walk_max_m") / speed / poll + 2             # 5 m asked, the cap walked
    assert t.audio.said == ["Stepping forward!"] and t.counts["walks"] == 1
    t.answer("walk")                                                                     # nothing to walk: ignored
    assert t.counts["walks"] == 1 and t.counts["ignored"] == 1
    assert "1 m" in talk.system_prompt() and "180 degrees" in talk.system_prompt()


# -- the supervisor: routines and tasks, confirmed, run, stopped ---------------------------------

def twin_runner(tmp_path, monkeypatch, *, interrupt_at=None, done_summary="I found it and pointed at it."):
    """Runs the real routine or a small task on the in-process twin, as talk's robot runner does on
    the robot (cli.run_once, kept unreviewed); ``interrupt_at`` presses Ctrl-C that far into it."""
    from g1.agent import agent as agent_module
    from g1.agent.task import load_task
    from g1.cli import run_once
    from g1.routines import run_spec
    from g1.talk import result_of
    from g1.tools import Chain
    from g1.tools.routines import RoutineTool
    from tests.agent.test_agent import Scripted, Sequence
    from tests.doubles import scripted_sim
    from tests.test_cli import Interrupting
    seen = {"envs": [], "runs": []}

    def run(kind, name):
        env = scripted_sim(Scripted())
        seen["envs"].append(env)
        if kind == "routine":
            spec = run_spec(name)
            spec = {**spec, "dir": tmp_path / "routines" / name}          # record under tmp, not the repo
            program = Interrupting(Chain(RoutineTool(routine=name)), at=interrupt_at) if interrupt_at else None
        else:
            d = tmp_path / "tasks" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "task.json").write_text(json.dumps({"instruction": "find the laptop and point at it"}))
            spec = load_task(d)
            dec = Sequence([("hold", {"seconds": 0.3, "note": "looking"}), ("done", {"summary": done_summary, "hindsight": ""})])
            monkeypatch.setattr(agent_module.VLMDecider, "from_env", classmethod(lambda cls, *a, **k: dec))
            program = None
        code, folder = run_once(spec, sim=True, mode=None, env=env, ask=False, program=program)
        seen["runs"].append(folder)
        return result_of(code, folder)
    return run, seen


def test_the_menu_offers_the_routines_and_tasks_that_exist():
    from g1.routines import routine_names
    routines, tasks = talk.routine_menu(), talk.task_menu()
    assert set(routines) == set(routine_names()) and "T" in routines["tpose"]
    assert {"find_the_mug", "point_at_the_laptop"} <= set(tasks) and "laptop" in tasks["point_at_the_laptop"]
    names = {x["function"]["name"]: x["function"] for x in tools({}, routines, tasks)}
    assert names["play_routine"]["parameters"]["properties"]["name"]["enum"] == list(routines)
    assert names["run_task"]["parameters"]["properties"]["name"]["enum"] == list(tasks)
    t, _, _ = make()                                                  # no runner: nothing to run them with
    assert {x["function"]["name"] for x in t.tools} == {"say", "move", "walk", "ignore"}


def test_a_routine_and_a_task_are_confirmed_run_and_reported(tmp_path, monkeypatch):
    runner, seen = twin_runner(tmp_path, monkeypatch)
    keys = ScriptedKeys(True, True)
    t, _, _ = make(("play_routine", {"name": "tpose", "say": "Watch my T-pose!"}),
                   ("run_task", {"name": "find_it"}),
                   routines={"tpose": "arms out in a T"}, tasks={"find_it": "find the laptop and point at it"},
                   runner=runner, keys=keys)
    t.answer("do a T-pose")
    t.answer("can you find the laptop")
    assert len(keys.asked) == 2 and "tpose routine" in keys.asked[0] and "takes over the arms" in keys.asked[0]
    assert t.audio.said == ["Watch my T-pose!", "Done.", "I will find the laptop and point at it.",
                            "I found it and pointed at it."]                          # the task's done summary
    routine_run, task_run = seen["runs"]
    assert routine_run.name.endswith("_sim_tpose") and task_run.name.endswith("_unreviewed")   # kept, never asked
    assert seen["envs"][0].actions[-1].weight == pytest.approx(0.0, abs=1e-3)   # handed back (the teardown releases)
    assert t.counts["routines"] == 1 and t.counts["tasks"] == 1 and t.counts["errors"] == 0
    assert {"role": "assistant", "content": "[routine tpose: success]"} in t.history


def test_n_at_the_keyboard_moves_nothing():
    ran = []
    t, _, _ = make(("play_routine", {"name": "tpose"}), routines={"tpose": "T"}, runner=lambda k, n: ran.append(n),
                   keys=ScriptedKeys(False))
    t.answer("do a T-pose")
    assert ran == [] and t.audio.said[-1] == "Okay, I will not do that." and t.moves.loco.calls == []


def test_after_a_built_in_move_it_goes_back_to_walk_mode_before_a_routine():
    fsm_at_start = []
    t, _, _ = make(("play_routine", {"name": "tpose"}), fsm=802, routines={"tpose": "T"},
                   runner=lambda k, n: fsm_at_start.append(t.moves.loco.fsm) or {"ok": True, "outcome": "success"},
                   keys=ScriptedKeys(True))
    t.answer("do a T-pose")
    assert fsm_at_start == [501] and 501 in t.moves.loco.calls        # 802 is not a standing-run state


def test_a_refusal_keeps_talk_listening():
    def refuse(kind, name):
        raise SystemExit("--mode standing needs a robot standing")
    t, _, _ = make(("play_routine", {"name": "tpose"}), ("say", {"text": "Still here."}), routines={"tpose": "T"},
                   runner=refuse, keys=ScriptedKeys(True))
    t.answer("do a T-pose")
    t.answer("are you ok")
    assert t.audio.said == ["I will play my tpose routine.", "Sorry, I could not do the tpose routine.", "Still here."]
    assert t.counts["errors"] == 1


def test_ctrl_c_stops_a_routine_to_neutral_and_talk_goes_on(tmp_path, monkeypatch):
    runner, seen = twin_runner(tmp_path, monkeypatch, interrupt_at=3.5)          # mid-raise
    t, _, _ = make(("play_routine", {"name": "tpose"}), ("say", {"text": "Ready again."}), routines={"tpose": "T"},
                   runner=runner, keys=ScriptedKeys(True))
    t.answer("do a T-pose")
    env = seen["envs"][0]
    assert env.actions[-1].weight == 0.0 and env.violations == []      # the safe return ran to the end
    t.answer("hello")
    assert t.audio.said[-2:] == ["I stopped.", "Ready again."] and t.counts["stopped"] == 1


def test_ctrl_c_stops_a_built_in_move_and_a_walk_to_neutral():
    fired = []

    def ctrl_c_once(now):
        if not fired and now > 0.6:
            fired.append(now)
            raise KeyboardInterrupt
    t, _, _ = make(("move", {"move_id": 516}), ("walk", {"turn_deg": 0, "forward_m": 1.0, "side_m": 0}),
                   sleep_hook=ctrl_c_once)
    t.answer("bow")
    assert t.moves.loco.calls[-1] == 501 and t.counts["stopped"] == 1  # back to the FSM it started from
    fired.clear()
    t.answer("come forward")
    assert t.moves.loco.calls[-1] == ("vel", 0.0, 0.0, 0.0) and t.counts["stopped"] == 2   # zero velocity


def test_ctrl_c_with_nothing_running_does_not_quit(capsys):
    keys = ScriptedKeys()
    presses = []

    def hook(now):
        if not presses:
            presses.append(now)
            raise KeyboardInterrupt                                    # Ctrl-C while it only listens
        if len(presses) == 1 and now > 1.0:
            presses.append(now)
            keys.request_quit("q")
    t, _, _ = make(keys=keys, sleep_hook=hook)
    assert t.run() == 0 and len(presses) == 2                          # it kept listening until q
    out = capsys.readouterr().out
    assert "nothing is running now; press q then Enter to quit talk" in out and "ending talk" in out


def test_q_stops_what_runs_first_then_quits_and_answers_no():
    keys = ScriptedKeys()
    keys.busy.set()
    keys.request_quit("q")
    assert keys.quit.is_set() and keys.interrupted == [True]           # the running skill is stopped (SIGINT)
    keys.request_quit("q")
    assert keys.interrupted == [True]                                   # once
    idle = Keys(interrupt=lambda: pytest.fail("nothing runs: nothing to stop"))
    idle.request_quit("q")
    assert idle.quit.is_set() and idle.confirm("start?") is False      # a quitting talk starts nothing


def test_the_keyboard_thread_reads_q_and_answers(monkeypatch):
    import queue
    lines = queue.Queue()
    keys = Keys(lines, interrupt=lambda: None)
    keys.start()
    import threading
    got = []
    asker = threading.Thread(target=lambda: got.append(keys.confirm("go?")))
    asker.start()
    import time as _time
    _time.sleep(0.05)
    lines.put("y\n")
    asker.join(2)
    assert got == [True] and not keys.quit.is_set()
    lines.put("q\n")
    for _ in range(100):
        if keys.quit.is_set():
            break
        _time.sleep(0.01)
    assert keys.quit.is_set()
    lines.put(None)


def test_sigterm_and_sighup_act_like_q(monkeypatch):
    import os
    import signal
    keys = ScriptedKeys()
    keys.busy.set()
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        talk.quit_on_signals(keys)
        os.kill(os.getpid(), signal.SIGTERM)
        assert keys.quit.is_set() and keys.interrupted == [True]        # stopped first, through SIGINT
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def test_the_default_stop_is_sigint(monkeypatch):
    import os
    import signal
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append(sig))
    keys = Keys()
    keys.busy.set()
    keys.request_quit("q")
    assert sent == [signal.SIGINT]                                     # the one stop path the returns are shielded on
