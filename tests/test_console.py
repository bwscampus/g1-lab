"""g1 start's terminal (g1/console.py): the commands, what Esc and Ctrl-C do, the y/N questions, and
the prompt_toolkit app driven by piped keystrokes."""
import threading
import time

import pytest

from g1.console import COMMANDS, Console, ParseError, completions, help_lines, parse
from g1.session import Command, RoutineInfo, Session, TaskInfo


# -- parsing -------------------------------------------------------------------------------------

@pytest.mark.parametrize("line, name, args, text", [
    ("do a T-pose", "chat", [], "do a T-pose"),
    ("/speaker off", "speaker", ["off"], ""), ("/speaker ON", "speaker", ["on"], ""),
    ("/routine tpose", "routine", ["tpose"], ""),
    ("/task find_the_mug", "task", ["find_the_mug"], ""),
    ("/move greet with both hands", "move", ["greet with both hands"], ""),
    ("/move 516", "move", ["516"], ""),
    ("/walk 0.3", "walk", ["0.3"], ""),
    ("/walk 0.3 -0.1 45", "walk", ["0.3", "-0.1", "45"], ""),
    ("/say hello everyone", "say", [], "hello everyone"),
    ("/mode Gantry", "mode", ["gantry"], ""),
    ("/mic off", "mic", ["off"], ""),
    ("/camera on", "camera", ["on"], ""),
    ("/lidar on", "lidar", ["on"], ""),
    ("/skills", "skills", [], ""), ("/status", "status", [], ""), ("/stop", "stop", [], ""),
    ("/help", "help", [], ""), ("/QUIT", "quit", [], ""),
])
def test_every_command_parses(line, name, args, text):
    cmd = parse(line)
    assert (cmd.name, cmd.args, cmd.text) == (name, args, text)


@pytest.mark.parametrize("line, match", [
    ("/fly", "unknown command /fly"), ("/routine", "usage: /routine NAME"), ("/routine a b", "usage"),
    ("/walk", "usage: /walk"), ("/walk fast", "numbers"), ("/walk 1 2 3 4", "usage"), ("/mode sitting", "usage"),
    ("/mic maybe", "usage"), ("/say", "usage"), ("/chat hi", "unknown command /chat"),
    ("/speaker maybe", "usage"), ("/skills now", "usage"), ("   ", "^$"),
])
def test_a_bad_line_says_why(line, match):
    with pytest.raises(ParseError, match=match):
        parse(line)


def test_help_covers_every_command_and_the_keys():
    text = "\n".join(help_lines())
    assert all(usage in text for usage, _ in COMMANDS.values()) and "Esc stops" in text and "L2+B" in text
    assert "/chat" not in text and "/speaker on|off" in text


# -- the console's logic ----------------------------------------------------------------------------

def console(busy=False, **kw):  # kw: Session settings (mode=...)
    s = Session(None, None, None, None, {516: {"name": "bow", "duration": 4.9}},
                routines={"tpose": RoutineInfo("arms out in a T"), "walkabout": RoutineInfo("a step", uses_base=True)},
                tasks={"find_it": TaskInfo("find it")}, runner=lambda *a: {"ok": True}, **kw)
    if busy:
        s.busy.set()
    out, stops = [], []
    clock = {"t": 0.0}
    c = Console(s, interrupt=lambda: stops.append(True), out=out.append, clock=lambda: clock["t"])
    return c, s, out, stops, clock


def test_lines_go_to_the_session_and_info_is_answered_at_once():
    c, s, out, _, _ = console()
    c.submit("do a T-pose")
    c.submit("/routine tpose")
    c.submit("/skills")
    c.submit("/help")
    c.submit("/fly")
    assert [s.events.get_nowait().name for _ in range(2)] == ["chat", "routine"] and s.events.empty()
    text = "\n".join(out)
    assert "/routine tpose" in text and "/quit" in text and "unknown command /fly" in text


def test_esc_stops_only_what_runs():
    c, s, out, stops, _ = console()
    c.escape()
    assert stops == []                                              # idle: nothing to stop
    s.busy.set()
    c.escape()
    assert stops == [True] and "stopping" in out[-1]


def test_ctrl_c_clears_stops_or_exits_on_the_second_press():
    c, s, out, stops, clock = console()
    assert c.ctrl_c("half a line") == "clear"
    assert c.ctrl_c("") == "hint" and "again to exit" in out[-1]
    clock["t"] += 5.0
    assert c.ctrl_c("") == "hint"                                   # too late: a fresh first press
    clock["t"] += 1.0
    assert c.ctrl_c("") == "quit" and s.quit.is_set() and stops == []
    c2, s2, _, stops2, _ = console(busy=True)
    assert c2.ctrl_c("") == "stopped" and stops2 == [True] and not s2.quit.is_set()   # a reflex Ctrl-C stops the robot


def test_quit_while_busy_stops_first_and_mode_waits_for_esc():
    c, s, out, stops, _ = console(busy=True)
    c.submit("/mode gantry")
    assert s.events.empty() and "Esc to stop it first" in out[-1]
    c.submit("/camera on")
    assert s.events.get_nowait().name == "camera" and "queued" in out[-1]
    c.submit("/quit")
    assert s.quit.is_set() and stops == [True] and "back to neutral" in out[-1]


def test_a_question_takes_the_next_line_and_esc_answers_no():
    c, s, out, _, _ = console()
    got = []
    t = threading.Thread(target=lambda: got.append(c.confirm("Start the tpose routine?")))
    t.start()
    for _ in range(100):
        if c.asking:
            break
        time.sleep(0.01)
    c.submit("y")
    t.join(2)
    assert got == [True] and "[y/N]" in out[-1] and s.events.empty()     # the answer is not a chat line
    t = threading.Thread(target=lambda: got.append(c.confirm("Start?")))
    t.start()
    for _ in range(100):
        if c.asking:
            break
        time.sleep(0.01)
    c.escape()
    t.join(2)
    assert got == [True, False]


def test_completion_offers_commands_then_names():
    c, _, _, _, _ = console()
    assert ("/routine", COMMANDS["routine"][1]) in completions(c, "/ro")
    assert completions(c, "/routine t") == [("tpose", "arms out in a T")]
    assert [n for n, _ in completions(c, "/routine ")] == ["tpose", "walkabout"]
    assert completions(c, "/task ") == [("find_it", "find it")]
    assert completions(c, "/move b") == [("bow", "built-in 516")]
    assert [n for n, _ in completions(c, "/mode ")] == ["standing", "gantry"]
    assert completions(c, "hello") == []
    g, _, _, _, _ = console(mode="gantry")                          # only what the mode allows
    assert [n for n, _ in completions(g, "/routine ")] == ["tpose"] and completions(g, "/move b") == []
    assert [n for n, _ in completions(g, "/speaker ")] == ["on", "off"]


# -- the app, keystrokes piped in -----------------------------------------------------------------

def test_the_app_takes_typed_lines_esc_and_ctrl_c_twice(tmp_path):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from g1.console import build_app
    c, s, out, stops, clock = console()
    with create_pipe_input() as pipe:
        app = build_app(c, input=pipe, output=DummyOutput(), history=tmp_path / "history")
        done = []
        runner = threading.Thread(target=lambda: done.append(app.run(handle_sigint=False)), daemon=True)
        runner.start()
        time.sleep(0.3)
        pipe.send_text("/routine tpose\r")
        pipe.send_text("say hi\r")
        time.sleep(0.3)
        s.busy.set()
        pipe.send_text("\x1b")                                       # Esc: stops the running skill
        for _ in range(100):
            if stops:
                break
            time.sleep(0.02)
        s.busy.clear()
        pipe.send_text("\x03")                                       # Ctrl-C, empty line: the hint
        time.sleep(0.2)
        pipe.send_text("\x03")                                       # again: exit
        runner.join(5)
    assert not runner.is_alive() and s.quit.is_set()
    assert [s.events.get_nowait() for _ in range(2)] == [Command("routine", ["tpose"]), Command("chat", text="say hi")]
    assert stops == [True] and any("again to exit" in o for o in out)
    assert (tmp_path / "history").read_text().count("/routine tpose") == 1
