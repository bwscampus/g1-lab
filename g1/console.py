"""``g1 start``: the robot in one interactive session, in a Claude-Code-style terminal.

    g1 start --mode standing|gantry          the robot only (UNITREE_IFACE)

The robot listens (its speech recognition) and you type. Plain text is chat, as if you had said it:
the model chooses (an answer, a built-in, a routine or a task; ``g1/session.py``). ``/`` commands
call things directly:

    /skills                     the built-ins, routines and tasks this mode allows
    /routine NAME · /task NAME  run one now (no y/N: you asked for it)
    /move ID|NAME               a built-in move now (standing mode)
    /say TEXT                   the robot says it
    /mode gantry|standing       switch mode (y/N); never changes the FSM (gantry allows 0, 1, 4, 200; standing all but 0, 1)
    /fsm ID                     switch the FSM now, if the mode allows it
    /mic on|off · /speaker on|off · /camera on|off · /lidar on|off
    /memory · /soul             what the robot remembers, who it is (knowledge/)
    /status · /stop · /help · /quit

Keys: **Esc stops** what the robot is doing (to its safe neutral; the session goes on). **Ctrl-C**
clears the line; with a skill running it stops it like Esc; on an empty line twice within
``console_ctrl_c_window_s`` it exits (stopping anything running first), as does ``/quit``, Ctrl-D on an
empty line, SIGTERM and SIGHUP. Every stop is SIGINT to this process: skills run on the main thread,
where the safe returns are shielded on it, so nothing cuts a return short. The remote's L2+B (damp) is
the emergency stop, always.

The terminal runs in its own thread (prompt_toolkit), output scrolling above the input box
(``patch_stdout``), with the status line under it and ``/`` completion.
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from g1.core import limits
from g1.session import MODES, Command

HISTORY = Path.home() / ".g1_history"

COMMANDS: dict[str, tuple[str, str]] = {
    "skills": ("/skills", "the built-ins, routines and tasks this mode allows"),
    "routine": ("/routine NAME", "play a routine now"),
    "task": ("/task NAME [WHAT]", "run a task now (WHAT fills its slot: /task point_at water bottle)"),
    "move": ("/move ID|NAME", "play a built-in move now (standing)"),
    "say": ("/say TEXT", "the robot says it"),
    "mode": ("/mode gantry|standing", "switch mode (y/N); never changes the FSM. gantry allows FSM 0, 1, 4, 200; standing all but 0, 1"),
    "mic": ("/mic on|off", "handle what the robot hears like chat, or ignore it"),
    "speaker": ("/speaker on|off", "say replies out loud, or only here (tasks' say too)"),
    "camera": ("/camera on|off", "the head camera, live at http://127.0.0.1:<view_port>; tasks use it"),
    "lidar": ("/lidar on|off", "SLAM mapping and its live view; off saves the map"),
    "memory": ("/memory", "what the robot remembers (knowledge/MEMORY.md, the newest)"),
    "soul": ("/soul", "who the robot is (knowledge/SOUL.md; edit it in an editor)"),
    "fsm": ("/fsm ID", "switch the robot to FSM ID now (SetFsmId; refused if the mode blocks it; y/N for damp, sit, stand-up)"),
    "status": ("/status", "the link, the FSM and the battery (read-only)"),
    "stop": ("/stop", "stop what the robot is doing (Esc)"),
    "help": ("/help", "this list"),
    "quit": ("/quit", "stop anything running, then exit (Ctrl-C twice)"),
}
CHOICES = {"mode": MODES, "mic": ("on", "off"), "speaker": ("on", "off"), "camera": ("on", "off"),
           "lidar": ("on", "off")}


class ParseError(ValueError):
    pass


def parse(line: str) -> Command:
    """One typed line as a ``Command``: plain text is chat; ``/name args`` is a command."""
    text = line.strip()
    if not text:
        raise ParseError("")
    if not text.startswith("/"):
        return Command("chat", text=text)
    head, _, rest = text[1:].partition(" ")
    name, rest = head.lower(), rest.strip()
    if name not in COMMANDS:
        raise ParseError(f"unknown command /{name}; /help lists them")
    usage = COMMANDS[name][0]
    if name == "say":
        if not rest:
            raise ParseError(f"usage: {usage}")
        return Command(name, text=rest)
    if name == "task":                             # the rest of the line is the task's slot, as typed
        task, _, what = rest.partition(" ")
        if not task:
            raise ParseError(f"usage: {usage}")
        return Command(name, [task], text=what.strip())
    try:
        args = shlex.split(rest)
    except ValueError as e:
        raise ParseError(f"{usage}: {e}") from None
    if name == "routine":
        if len(args) != 1:
            raise ParseError(f"usage: {usage}")
    elif name == "move":
        if not args:
            raise ParseError(f"usage: {usage}")
        args = [" ".join(args)]
    elif name == "fsm":
        if len(args) != 1 or not args[0].isdigit():
            raise ParseError(f"usage: {usage} (a number, e.g. /fsm 200)")
    elif name in CHOICES:
        if len(args) != 1 or args[0].lower() not in CHOICES[name]:
            raise ParseError(f"usage: {usage}")
        args = [args[0].lower()]
    elif args:
        raise ParseError(f"usage: {usage}")
    return Command(name, args)


def help_lines() -> list[str]:
    width = max(len(u) for u, _ in COMMANDS.values())
    return ([f"  {u.ljust(width)}  {d}" for u, d in COMMANDS.values()]
            + ["  Esc stops what the robot is doing · Ctrl-C clears the line (stops a running skill) · "
               "Ctrl-C twice on an empty line exits · the remote's L2+B is the emergency stop"])


class Console:
    """The terminal's logic, apart from the screen: what a submitted line, Esc and Ctrl-C do, and the
    y/N questions the session asks (``confirm``, answered by the next submitted line). The
    prompt_toolkit app (``build_app``) only calls these."""

    def __init__(self, session, *, interrupt: Optional[Callable[[], None]] = None,
                 out: Callable[[str], None] = print, clock: Callable[[], float] = time.monotonic) -> None:
        self.session = session
        session.ui = self
        self.interrupt = interrupt or (lambda: os.kill(os.getpid(), signal.SIGINT))
        self.out, self.clock = out, clock
        self._pending: Optional[tuple[threading.Event, list]] = None
        self._lock = threading.Lock()
        self._last_ctrl_c: Optional[float] = None
        self.app = None
        self._thread: Optional[threading.Thread] = None

    # -- questions ------------------------------------------------------------------------------------
    def confirm(self, question: str) -> bool:
        """Ask y/N; the next line typed answers it (Esc or quitting answers no)."""
        if self.session.quit.is_set():
            return False
        done, answer = threading.Event(), []
        with self._lock:
            self._pending = (done, answer)
        self.out(f"{question} [y/N]")
        while not done.wait(0.1):
            if self.session.quit.is_set():
                break
        with self._lock:
            self._pending = None
        return bool(answer) and answer[0] in ("y", "yes") and not self.session.quit.is_set()

    def _answer(self, text: str) -> bool:
        with self._lock:
            if self._pending is None:
                return False
            done, answer = self._pending
            answer.append(text.strip().lower())
            done.set()
            return True

    @property
    def asking(self) -> bool:
        return self._pending is not None

    # -- input ----------------------------------------------------------------------------------------
    def submit(self, text: str) -> None:
        if self._answer(text):
            return
        try:
            cmd = parse(text)
        except ParseError as e:
            if str(e):
                self.out(str(e))
            return
        busy = self.session.busy.is_set()
        if cmd.name == "help":
            for line in help_lines():
                self.out(line)
        elif cmd.name in ("skills", "memory", "soul"):
            lines = {"skills": self.session.skills, "memory": self.session.memory_lines,
                     "soul": self.session.soul_lines}[cmd.name]()
            for line in lines:
                self.out(line)
        elif cmd.name == "stop":
            self.stop()
        elif cmd.name == "quit":
            self.request_quit("/quit")
        elif cmd.name in ("mode", "fsm") and busy:
            self.out(f"a skill is running: Esc to stop it first, then /{cmd.name}")
        else:
            if busy:
                self.out(f"queued: /{cmd.name} runs when the current skill ends (Esc stops it)")
            self.session.events.put(cmd)

    def stop(self) -> bool:
        """Esc, /stop: stop the running skill (SIGINT). Nothing running: nothing to do."""
        if self.session.busy.is_set():
            self.out("stopping: back to neutral")
            self.interrupt()
            return True
        return False

    def escape(self) -> None:
        if self._answer("n"):
            return
        self.stop()

    def ctrl_c(self, line: str) -> str:
        """``clear`` (the line had text), ``stopped`` (a skill was running), ``hint`` or ``quit``."""
        if line:
            return "clear"
        if self.stop():
            return "stopped"
        now = self.clock()
        if self._last_ctrl_c is not None and now - self._last_ctrl_c <= limits.get("console_ctrl_c_window_s"):
            self.request_quit("Ctrl-C")
            return "quit"
        self._last_ctrl_c = now
        self.out("press Ctrl-C again to exit")
        return "hint"

    def request_quit(self, why: str) -> None:
        """Exit: stop anything running first (SIGINT), answer any question no, end the session."""
        if self.session.quit.is_set():
            return
        busy = self.session.busy.is_set()
        self.out(f"{why}: exiting" + (" once the robot is back to neutral" if busy else ""))
        self.session.quit.set()
        self._answer("n")
        if busy:
            self.interrupt()

    # -- the screen ------------------------------------------------------------------------------------
    def start(self, **app_kw) -> None:
        self.app = build_app(self, **app_kw)
        self._thread = threading.Thread(target=self._run_app, name="g1-console", daemon=True)
        self._thread.start()

    def _run_app(self) -> None:
        try:
            self.app.run(handle_sigint=False, set_exception_handler=False)
        except Exception as e:                       # the terminal went away: the session ends cleanly
            self.out(f"console: {e}")
        finally:
            self.request_quit("console closed")

    def close(self) -> None:
        app, self.app = self.app, None
        if app is not None and app.is_running:
            app.loop.call_soon_threadsafe(app.exit) if app.loop else None
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def completions(console: Console, text: str) -> list[tuple[str, str]]:
    """What Tab offers for ``text`` (the line so far): ``(completion, description)``."""
    if not text.startswith("/"):
        return []
    head, space, rest = text[1:].partition(" ")
    if not space:
        return [(f"/{n}", d) for n, (_, d) in COMMANDS.items() if n.startswith(head.lower())]
    s = console.session
    options: dict[str, str] = {}
    if head == "routine":                          # only what this mode allows, as /skills
        options = {n: r.notes for n, r in s.routines.items() if s.why_not("routine", n) is None}
    elif head == "task":
        options = {n: t.instruction for n, t in s.tasks.items() if s.why_not("task", n) is None}
    elif head == "move":
        options = {m["name"]: f"built-in {mid}" for mid, m in s.menu.items() if s.why_not("move", str(mid)) is None}
    elif head in CHOICES:
        options = {c: "" for c in CHOICES[head]}
    return [(n, d) for n, d in options.items() if n.lower().startswith(rest.lower())]


def build_app(console: Console, *, input=None, output=None, history: Optional[Path] = HISTORY):
    """The prompt_toolkit application: an input box with a status line under it and a completion
    menu; output printed elsewhere scrolls above it (``patch_stdout``)."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.menus import CompletionsMenu
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame, TextArea

    class Slash(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            word = text.rsplit(" ", 1)[-1] if " " in text else text
            for value, meta in completions(console, text):
                yield Completion(value, start_position=-len(word), display_meta=meta)

    try:
        hist = FileHistory(str(history)) if history is not None else InMemoryHistory()
    except OSError:
        hist = InMemoryHistory()
    box = TextArea(prompt="> ", multiline=False, wrap_lines=True, completer=Slash(), complete_while_typing=True,
                   history=hist)

    def accept(buffer) -> bool:
        console.submit(buffer.text)
        return False                                    # clear the line

    box.accept_handler = accept
    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _(event):
        if box.buffer.complete_state is not None:
            box.buffer.cancel_completion()
        else:
            console.escape()

    @kb.add("c-c")
    def _(event):
        if console.ctrl_c(box.text) == "clear":
            box.text = ""

    @kb.add("c-d")
    def _(event):
        if not box.text:
            console.request_quit("Ctrl-D")

    def status():
        s = console.session
        hint = "y/N?" if console.asking else "Esc stops · Ctrl-C ×2 exits · /help"
        return [("class:status", f"  {s.status()}   ({hint})")]

    root = FloatContainer(
        HSplit([Frame(box), Window(FormattedTextControl(status), height=1)]),
        floats=[Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=12, scroll_offset=1))])
    app = Application(layout=Layout(root, focused_element=box), key_bindings=kb, full_screen=False,
                      refresh_interval=limits.get("console_refresh_s"), input=input, output=output,
                      style=Style.from_dict({"status": "fg:ansibrightblack", "frame.border": "fg:ansibrightblack"}))

    app.ttimeoutlen = limits.get("console_escape_s")     # a lone Esc counts after this long, not 0.5 s

    def watch_quit():                                    # the session ended: close the screen
        console.session.quit.wait()
        try:
            app.loop.call_soon_threadsafe(app.exit) if app.loop else None
        except RuntimeError:
            pass
    threading.Thread(target=watch_quit, name="g1-console-quit", daemon=True).start()
    return app


def quit_on_signals(console: Console) -> None:
    """SIGTERM and SIGHUP (a kill, a dropped ssh) exit like /quit, stopping to neutral first."""
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, lambda signum, frame: console.request_quit(signal.Signals(signum).name))


def start_main(argv: list[str]) -> int:
    from types import SimpleNamespace

    from g1.envs.link import dds_domain, is_loopback
    from g1.envs.moves import connect
    from g1.microphone import subscribe
    from g1.session import Session, robot_runner, routine_menu, talk_menu, task_menu
    from g1.vlm import VLMClient, load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(prog="g1 start", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, required=True,
                   help="standing: on its own feet, we command the legs (built-in moves); FSM 0 and 1 blocked. "
                        "gantry: hanging, the legs never commanded; only FSM 0, 1, 4, 200. Both start with the bring-up "
                        "0 -> 1 -> 4 -> 200 (y/N); /fsm ID switches the FSM, /mode the mode; neither mode changes the FSM")
    a = p.parse_args(argv)
    iface = os.environ.get("UNITREE_IFACE")
    if not iface:
        p.error("set UNITREE_IFACE=<iface or IP> in .env")
    if is_loopback(iface):
        p.error("g1 start is the robot's: its speech recognition, voice and built-ins (the twin has none yet)")
    verbose = os.environ.get("G1_VERBOSE", "") not in ("", "0")
    try:
        model = VLMClient.from_env()
    except Exception as e:
        print(f"warning: no model ({e}): chat is off; call skills directly")
        model = None
    moves, link = connect(iface, verbose)
    audio = link.audio()
    audio.SetTimeout(limits.get("rpc_timeout_s"))
    audio.Init()
    mic = subscribe(link, time.monotonic)
    if mic is None:
        print("warning: no microphone (the robot's speech recognition did not subscribe): typed chat only")
    menu = talk_menu(moves.inventory())

    def open_camera(mode: str):
        from g1.cli import run_settings
        from g1.envs.robot import RobotEnv
        return RobotEnv._open_camera(SimpleNamespace(args=run_settings(sim=False, mode=mode)))

    def view(camera):
        from g1.camera import Viewer
        v = Viewer(camera, title="g1 head camera")
        try:
            v.start()
        except OSError:
            v.port = 0
            v.start()
        return v

    def lidar():
        from g1.lidar import LidarSession
        return LidarSession(link, iface, dds_domain(iface), print_lines=False, verbose=verbose)

    def status():
        from g1.envs.robot import status_main
        status_main([])

    from g1.knowledge import Knowledge
    knowledge = Knowledge()
    for note in knowledge.ensure():
        print(note)
    session = Session(mic, audio, moves, model, menu, mode=a.mode, routines=routine_menu(), tasks=task_menu(),
                      runner=robot_runner(), verbose=verbose, camera_factory=open_camera, viewer_factory=view,
                      lidar_factory=lidar, status_probe=status, knowledge=knowledge)
    console = Console(session)
    quit_on_signals(console)
    from prompt_toolkit.patch_stdout import patch_stdout
    with patch_stdout(raw=True):
        console.start()
        try:
            print(f"g1 start: {a.mode} mode; model {model.model if model else 'none'}; "
                  f"{len(menu)} built-in moves, {len(session.routines)} routines, {len(session.tasks)} tasks. "
                  "The mic is off: /mic on. /help")
            moves.confirm = lambda question: True    # the session asks (start, /mode); FSM switches follow its yes
            if session.start():
                session.run()
            else:
                print("not started")
        except SystemExit as e:
            print(e)
            return 1
        finally:
            session.quit.set()
            console.close()
    return 0
