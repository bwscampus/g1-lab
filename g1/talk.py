"""g1 talk: the supervisor. The robot stands ready, listens, and for each thing it hears chooses
what to do from a fixed menu, or just answers.

    g1 talk                     the robot only (UNITREE_IFACE), voice interaction on in the Unitree app

Each turn: what was said (the robot's own speech recognition on ``rt/audio_msg``, through
``g1/microphone.py``) goes to the model (``configs/model.json``), which answers with exactly one call:

* ``say``: an answer, spoken with the robot's text-to-speech (``TtsMaker``); no action.
* ``move``: a **built-in** (one of Unitree's stored dances and gestures, the allow-list below,
  played by ``g1 moves``'s player) and ``walk``: a built-in walk (a short turn, walk or sidestep,
  each capped by a ``talk_*_max`` limit, walked by ``Moves.walk``).
* ``play_routine``: a **routine** (``routines/``: our joint motion, start to finish, no model).
* ``run_task``: a **task** (``tasks/``: an instruction the vision model works through with the camera).
* ``ignore``: chatter not meant for it.

It only ever chooses: it never writes a task or a routine. A routine or a task takes over the arms,
so each one is confirmed at the keyboard (y/N); built-ins are confirmed once, at the start. A routine
or task runs through ``cli.run_once`` in standing mode (it goes to FSM 200 and back to walk mode),
sharing talk's microphone, and is kept unreviewed (no questions mid-conversation).

Stopping and quitting (the remote's L2+B damp is the emergency stop, always):

* **Ctrl-C stops** what the robot is doing, to a safe neutral, and talk goes on listening: a built-in
  move goes back to the FSM it started from, a walk to zero velocity, a routine or task through the
  safe return (the arms to the stand pose, the weight to 0, the base stopped). Nothing cuts that
  return short. With nothing running, Ctrl-C says how to quit; it never quits.
* **q then Enter quits**: anything running is first stopped to neutral by the same path. SIGTERM and
  SIGHUP (a kill, a dropped ssh) act like q. Every stop goes through SIGINT, so the shields that
  protect a safe return protect it from these too.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import sys
import threading
import time
from typing import Callable, Optional

from g1.core import limits
from g1.envs import sdk
from g1.envs.moves import PLAY_FSMS, Moves, connect, gloss

# The moves talk may play: the gestures and dances (mimic ids 5xx, 6xx). Never the get-up and
# lie-down motions (4xx) or the stance (101), whatever the robot lists.
TALK_MOVES = range(500, 700)

SYSTEM = """You are a Unitree G1, a humanoid robot, at a school assembly. Students and teachers talk to you.
Each turn you get what was just said near you, as your speech recognition heard it (it can mishear: answer the
most likely meaning). Answer with exactly one tool call:
- say: a spoken answer. One to three short sentences, friendly and clear for school students. Plain words only:
  no lists, no emoji, no markdown, no stage directions. Numbers as words when short.
- move: when someone asks you to dance, wave, bow, clap, greet, or do a move. Pick the best fit from the moves
  listed in the tool; you may give a short line to say first. If they ask for a dance without saying which, pick one.
- walk: when someone asks you to walk, step, come forward, back up, step aside or turn around. Short moves only:
  forward or back up to {walk_max:g} m, sideways up to {side_max:g} m, turning up to {turn_max:g} degrees. Say a short
  warning first ("Stepping forward!"). You cannot see while you talk, so never walk unless asked, and never toward people.
- play_routine: when someone asks for one of your routines (fixed arm motions you know by name, listed in the tool).
- run_task: when someone asks for something one of your tasks does (listed in the tool with what each one does).
  A task uses your camera and takes a while; say briefly what you will do first.
- ignore: when what you heard is not meant for you (side conversations, noise, half a word).
You can only talk, play those moves, routines and tasks, and take those short walks: you cannot make up new ones.
If asked for something else, say so kindly. A person at the keyboard approves each routine and task before it starts.
Keep everything school-appropriate and decline anything unsafe or unkind politely.
You are about 1.3 m tall and made by Unitree Robotics; you run on your own onboard computer.
Always answer in English, whatever language you hear or are asked to speak."""


def system_prompt() -> str:
    return SYSTEM.format(walk_max=limits.get("talk_walk_max_m"), side_max=limits.get("talk_side_max_m"),
                         turn_max=limits.get("talk_turn_max_deg"))


def move_name(m: dict) -> str:
    """A stored move's name for people and the model: its English gloss, else its tidied name."""
    if gloss(m["name"]):
        return gloss(m["name"])
    name = re.sub(r"_G1_50hz$|_v1$|_\d{3}$", "", m["name"]).removeprefix("vq_")
    return name.replace("_", " ")


def talk_menu(inv: dict) -> dict[int, dict]:
    """The allow-listed moves this robot has: ``{id: {"name", "duration"}}``."""
    return {mid: {"name": move_name(m), "duration": float(m.get("duration", 0))}
            for mid, m in sorted(inv["mimic"].items()) if mid in TALK_MOVES}


def routine_menu() -> dict[str, str]:
    """The routines talk may play: ``{name: the first line of its notes}`` (a full-body one is left out)."""
    from g1 import routines
    out = {}
    for name in routines.routine_names():
        try:
            r = routines.load_routine(name)
        except routines.RoutineError as e:
            print(f"warning: routine {name} skipped: {e}")
            continue
        if not r.full_body:
            out[name] = (r.notes.strip().splitlines() or [name])[0]
    return out


def task_menu() -> dict[str, str]:
    """The tasks talk may run: ``{name: its instruction}``."""
    from g1.agent.task import ROOT, load_task
    out = {}
    for d in sorted((ROOT / "tasks").iterdir()):
        if (d / "task.json").is_file():
            try:
                out[d.name] = load_task(d)["instruction"]
            except ValueError as e:
                print(f"warning: task {d.name} skipped: {e}")
    return out


def tools(menu: dict[int, dict], routines: Optional[dict[str, str]] = None,
          tasks: Optional[dict[str, str]] = None) -> list[dict]:
    listing = "; ".join(f"{mid} {m['name']} ({m['duration']:.0f} s)" for mid, m in menu.items())
    out = [
        {"type": "function", "function": {
            "name": "say", "description": "Speak an answer out loud.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "what to say: one to three short sentences"}},
                "required": ["text"]}}},
        {"type": "function", "function": {
            "name": "move", "description": f"Play one of your stored moves. The moves: {listing}.",
            "parameters": {"type": "object", "properties": {
                "move_id": {"type": "integer", "enum": list(menu), "description": "the move's number"},
                "say": {"type": "string", "description": "optional: a short line to say before the move"}},
                "required": ["move_id"]}}},
        {"type": "function", "function": {
            "name": "walk", "description": "Take a short walk: turn first, then walk forward or back, then sidestep. "
                                           "Leave the parts you do not need at 0.",
            "parameters": {"type": "object", "properties": {
                "turn_deg": {"type": "number", "minimum": -limits.get("talk_turn_max_deg"),
                             "maximum": limits.get("talk_turn_max_deg"), "description": "turn: positive left, negative right"},
                "forward_m": {"type": "number", "minimum": -limits.get("talk_walk_max_m"),
                              "maximum": limits.get("talk_walk_max_m"), "description": "walk: positive forward, negative back"},
                "side_m": {"type": "number", "minimum": -limits.get("talk_side_max_m"),
                           "maximum": limits.get("talk_side_max_m"), "description": "sidestep: positive left, negative right"},
                "say": {"type": "string", "description": "a short warning to say before walking"}},
                "required": ["turn_deg", "forward_m", "side_m"]}}},
    ]
    if routines:
        listing = "; ".join(f"{n} ({notes})" for n, notes in routines.items())
        out.append({"type": "function", "function": {
            "name": "play_routine", "description": f"Play one of your routines start to finish. The routines: {listing}.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "enum": list(routines), "description": "the routine"},
                "say": {"type": "string", "description": "a short line to say before it"}},
                "required": ["name"]}}})
    if tasks:
        listing = "; ".join(f"{n}: {instruction}" for n, instruction in tasks.items())
        out.append({"type": "function", "function": {
            "name": "run_task", "description": f"Run one of your tasks with your camera until it is done. The tasks: {listing}.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "enum": list(tasks), "description": "the task"},
                "say": {"type": "string", "description": "a short line to say before it: what you will do"}},
                "required": ["name"]}}})
    out.append({"type": "function", "function": {
        "name": "ignore", "description": "Say nothing: what was heard was not meant for you.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "a few words, for the log"}},
            "required": ["reason"]}}})
    return out


def _clamp(value, cap: float) -> float:
    try:
        return max(-cap, min(cap, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


class Keys:
    """The keyboard, read on its own thread: ``q`` (then Enter) asks to quit, any other line answers
    the pending y/N question. Asking to quit while something runs stops it first, through SIGINT,
    the one stop path the safe returns are shielded on. ``lines`` (tests) replaces stdin."""

    def __init__(self, lines: Optional[queue.Queue] = None, *, interrupt: Callable[[], None] = None) -> None:
        self.lines: queue.Queue = lines if lines is not None else queue.Queue()
        self.answers: queue.Queue = queue.Queue()
        self.quit = threading.Event()
        self.busy = threading.Event()             # something is running: quitting stops it first
        self.interrupt = interrupt or (lambda: os.kill(os.getpid(), signal.SIGINT))
        self._thread: Optional[threading.Thread] = None

    def start(self, stdin=None) -> None:
        if stdin is not None:
            def read():
                for line in stdin:
                    self.lines.put(line)
                self.lines.put(None)
            threading.Thread(target=read, name="talk-stdin", daemon=True).start()
        self._thread = threading.Thread(target=self._dispatch, name="talk-keys", daemon=True)
        self._thread.start()

    def _dispatch(self) -> None:
        while True:
            line = self.lines.get()
            if line is None:
                return
            if line.strip().lower() in ("q", "quit", "exit"):
                self.request_quit("q")
            else:
                self.answers.put(line.strip().lower())

    def request_quit(self, why: str) -> None:
        if self.quit.is_set():
            return
        self.quit.set()
        print(f"\n{why}: quitting" + (" once the robot is back to neutral" if self.busy.is_set() else ""), flush=True)
        if self.busy.is_set():
            self.interrupt()
        self.answers.put("n")                     # a pending question is answered no

    def confirm(self, question: str) -> bool:
        while not self.answers.empty():           # nothing typed before the question answers it
            self.answers.get_nowait()
        if self.quit.is_set():
            return False
        print(f"{question} [y/N] ", end="", flush=True)
        answer = self.answers.get()
        return answer in ("y", "yes") and not self.quit.is_set()


class Talk:
    """The conversation loop over a Microphone, the robot's AudioClient, a ``Moves`` player, a model
    client (``complete(messages, tools=...) -> ToolCall``), the routines and tasks it may choose and
    a ``runner(kind, name) -> dict`` that runs one on the robot. The clock is injectable for tests."""

    def __init__(self, mic, audio, moves: Moves, model, menu: dict[int, dict], *, verbose: bool = False,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.monotonic,
                 routines: Optional[dict[str, str]] = None, tasks: Optional[dict[str, str]] = None,
                 runner: Optional[Callable[[str, str], dict]] = None, keys: Optional[Keys] = None) -> None:
        self.mic, self.audio, self.moves, self.model, self.menu = mic, audio, moves, model, menu
        self.verbose, self.sleep, self.now = verbose, sleep, now
        self.routines, self.tasks = dict(routines or {}), dict(tasks or {})
        self.runner = runner
        self.keys = keys or Keys()
        self.tools = tools(menu, self.routines if runner else None, self.tasks if runner else None)
        self.history: list[dict] = []           # user / assistant text turns, newest last
        self.counts = {"heard": 0, "said": 0, "moves": 0, "walks": 0, "routines": 0, "tasks": 0, "stopped": 0,
                       "ignored": 0, "errors": 0}

    # -- hearing ---------------------------------------------------------------------------------
    def listen(self, seq: int) -> Optional[tuple[str, int]]:
        """Block until someone has said something and then been quiet for ``talk_quiet_s``;
        everything said since ``seq``, joined, and the new seq. None once quitting."""
        while not self.keys.quit.is_set():
            heard = self.mic.since(seq)
            if heard and self.now() - heard[-1].stamp >= limits.get("talk_quiet_s"):
                return " ".join(h.text for h in heard), heard[-1].seq
            self.sleep(limits.get("talk_poll_s"))
        return None

    # -- acting ----------------------------------------------------------------------------------
    def speak(self, text: str) -> None:
        text = text.strip()[: int(limits.get("say_max_chars"))]
        if not text:
            return
        print(f"robot: {text}")
        code, _ = sdk.call(self.audio, "TtsMaker", text, int(limits.get("tts_speaker_id")),
                           log=self.moves.log, verbose=self.verbose, label="AudioClient")
        if code != 0:
            print(f"warning: TtsMaker failed ({sdk.explain(code)}); voice interaction on in the app?")
            return
        self.counts["said"] += 1
        # The robot speaks on its own; wait out the words, then until its speaker reports quiet.
        self.sleep(max(limits.get("say_min_s"), len(text) / limits.get("speech_chars_per_s")))
        t0 = self.now()
        while self.mic.speaking and self.now() - t0 < limits.get("talk_speech_timeout_s"):
            self.sleep(limits.get("talk_poll_s"))

    def _remember(self, heard: str, what: str) -> None:
        self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": what}]

    def _busy(self, action: Callable[[], object]) -> object:
        """Run one skill with the keyboard's quit wired to stop it."""
        self.keys.busy.set()
        try:
            return action()
        finally:
            self.keys.busy.clear()

    def answer(self, heard: str) -> None:
        """One turn: ask the model about ``heard`` and do what it chose. A skill stopped by Ctrl-C
        (or by quitting) has already gone back to its neutral when this returns."""
        messages = ([{"role": "system", "content": system_prompt()}]
                    + self.history[-2 * int(limits.get("talk_history_turns")):]
                    + [{"role": "user", "content": heard}])
        t0 = self.now()
        try:
            call = self.model.complete(messages, tools=self.tools)
            args = json.loads(call.arguments or "{}")
        except KeyboardInterrupt:
            raise
        except Exception as e:                  # the network, the provider, a bad reply: say so and go on
            self.counts["errors"] += 1
            print(f"model error after {self.now() - t0:.1f} s: {type(e).__name__}: {e}")
            self.speak("Sorry, I could not think of an answer just now.")
            return
        print(f"  ({call.name} after {self.now() - t0:.1f} s)")
        if call.name == "move" and int(args.get("move_id", -1)) in self.menu:
            mid = int(args["move_id"])
            self._remember(heard, f"[played move {mid}: {self.menu[mid]['name']}]")
            if args.get("say"):
                self.speak(str(args["say"]))
            self.counts["moves"] += 1
            inv_move = {"kind": "mimic", "id": mid, "label": self.menu[mid]["name"],
                        "duration": self.menu[mid]["duration"]}
            if self._busy(lambda: self.moves.play([inv_move])) == 130:
                self._stopped("the move")
            return
        if call.name == "walk":
            amounts = {k: _clamp(args.get(k), limits.get(cap)) for k, cap in
                       (("turn_deg", "talk_turn_max_deg"), ("forward_m", "talk_walk_max_m"), ("side_m", "talk_side_max_m"))}
            if any(amounts.values()):
                what = ", ".join(f"{k} {v:+g}" for k, v in amounts.items() if v)
                self._remember(heard, f"[walked: {what}]")
                if args.get("say"):
                    self.speak(str(args["say"]))
                self.counts["walks"] += 1
                if self._busy(lambda: self.moves.walk(**amounts)) == 130:
                    self._stopped("the walk")
                return
        if call.name == "play_routine" and self.runner and args.get("name") in self.routines:
            self.run_on_robot("routine", args["name"], heard, str(args.get("say") or ""))
            return
        if call.name == "run_task" and self.runner and args.get("name") in self.tasks:
            self.run_on_robot("task", args["name"], heard, str(args.get("say") or ""))
            return
        if call.name == "say" and str(args.get("text", "")).strip():
            self._remember(heard, args["text"])
            self.speak(str(args["text"]))
            return
        self.counts["ignored"] += 1
        print(f"  (ignored: {args.get('reason') or call.name})")

    def _stopped(self, what: str) -> None:
        self.counts["stopped"] += 1
        print(f"stopped {what}: the robot is back to neutral")

    def run_on_robot(self, kind: str, name: str, heard: str, say: str = "") -> None:
        """A routine or a task, confirmed at the keyboard, run in standing mode from walk mode."""
        what = (f"the {name} routine ({self.routines[name]})" if kind == "routine"
                else f"the task {name}: {self.tasks[name]}")
        self.speak(say or (f"I will play my {name} routine." if kind == "routine" else f"I will {self.tasks[name]}."))
        if not self.keys.confirm(f"{what}; it takes over the arms (standing mode). Area clear, spotter and remote "
                                 f"ready. Start?"):
            self._remember(heard, f"[{kind} {name}: not approved at the keyboard]")
            self.speak("Okay, I will not do that.")
            return
        fsm = self.moves.fsm()
        if fsm not in sdk.STANDING_FSMS:             # after a built-in move it stands in 802: back to walk mode
            print(f"the robot is in FSM {sdk.fsm_name(fsm)}; walk mode first")
            self.moves.switch(int(limits.get("walk_fsm")))
        self.counts["routines" if kind == "routine" else "tasks"] += 1
        try:
            result = self._busy(lambda: self.runner(kind, name))
        except KeyboardInterrupt:
            self._stopped(f"{kind} {name}")
            self._remember(heard, f"[{kind} {name}: stopped]")
            return
        except (SystemExit, Exception) as e:        # refused or failed: the env's teardown has already run
            self.counts["errors"] += 1
            print(f"{kind} {name} failed: {e}")
            self._remember(heard, f"[{kind} {name}: could not run]")
            self.speak(f"Sorry, I could not do the {name} {kind}.")
            return
        if result.get("stopped"):
            self._stopped(f"{kind} {name}")
        outcome = "stopped" if result.get("stopped") else (result.get("outcome") or "unknown")
        self._remember(heard, f"[{kind} {name}: {outcome}]")
        if result.get("stopped"):
            self.speak("I stopped.")
        elif kind == "task" and result.get("summary"):
            self.speak(str(result["summary"]))
        else:
            self.speak("Done." if result.get("ok") else "That did not go as planned.")

    def run(self) -> int:
        print(f"listening: it answers, plays built-ins ({len(self.menu)} moves), routines ({len(self.routines)}) and "
              f"tasks ({len(self.tasks)}) when asked. Ctrl-C stops what the robot is doing; q then Enter quits.")
        seq = self.mic.seq
        try:
            while not self.keys.quit.is_set():
                try:
                    got = self.listen(seq)
                    if got is None:
                        break
                    heard, seq = got
                    self.counts["heard"] += 1
                    print(f"\nheard: {heard}")
                    self.answer(heard)
                except KeyboardInterrupt:           # Ctrl-C with nothing running (or between steps): never quits
                    if not self.keys.quit.is_set():
                        print("\nnothing is running now; press q then Enter to quit talk")
                seq = self.mic.seq                    # nothing said while it spoke or moved is answered
        finally:
            print("talk: " + ", ".join(f"{k} {v}" for k, v in self.counts.items())
                  + f"; microphone: {self.mic.counts}")
        print("ending talk")
        return 0


def robot_runner(mic) -> Callable[[str, str], dict]:
    """Runs a routine or task on the robot through ``cli.run_once``: standing mode, talk's microphone,
    kept unreviewed (no questions mid-conversation)."""
    def run(kind: str, name: str) -> dict:
        from g1 import routines
        from g1.agent.task import find, load_task
        from g1.cli import run_once
        spec = routines.run_spec(name) if kind == "routine" else load_task(find(name))
        code, folder = run_once(spec, sim=False, mode="standing", ask=False, settings={"microphone": mic})
        return result_of(code, folder)
    return run


def result_of(code: int, folder) -> dict:
    """What talk says after a run: ``ok``, ``outcome``, ``stopped`` and the model's done summary."""
    summary = {}
    if folder is not None:
        try:
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    closing = summary.get("closing") or {}
    interrupt = summary.get("interrupt") or {}
    return {"ok": code == 0, "folder": folder, "outcome": summary.get("outcome"),
            "stopped": summary.get("end_reason") == "interrupted" or interrupt.get("trigger") == "ctrl_c",
            "summary": closing.get("summary") if summary.get("model_outcome") == "success" else None}


def quit_on_signals(keys: Keys) -> None:
    """SIGTERM and SIGHUP (a kill, a dropped ssh) act like q: quit, stopping to neutral first."""
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, lambda signum, frame: keys.request_quit(signal.Signals(signum).name))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="g1 talk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    iface = os.environ.get("UNITREE_IFACE")
    if not iface:
        p.error("set UNITREE_IFACE=<iface or IP> in .env")
    from g1.envs.link import is_loopback
    if is_loopback(iface):
        p.error("g1 talk is the robot's: its speech recognition, voice and stored moves (the twin has none)")
    from g1.microphone import subscribe
    from g1.vlm import VLMClient
    verbose = os.environ.get("G1_VERBOSE", "") not in ("", "0")
    try:
        model = VLMClient.from_env()            # fails here, before the robot is touched, without a key
    except Exception as e:
        p.error(f"the model: {e}")
    keys = Keys()
    keys.start(sys.stdin)
    quit_on_signals(keys)
    moves, link = connect(iface, verbose)
    moves.confirm = keys.confirm
    audio = link.audio()
    audio.SetTimeout(limits.get("rpc_timeout_s"))
    audio.Init()
    mic = subscribe(link, time.monotonic)
    if mic is None:
        raise SystemExit("no microphone: cannot listen")
    menu = talk_menu(moves.inventory())
    if not menu:
        raise SystemExit("the robot lists none of the allowed moves (5xx, 6xx); g1 moves list")
    routines, tasks = routine_menu(), task_menu()
    print(f"model: {model.model} on {model.provider}; routines: {', '.join(routines) or 'none'}; "
          f"tasks: {', '.join(tasks) or 'none'}")
    fsm = moves.fsm()
    if fsm not in PLAY_FSMS:
        print(f"the robot is in FSM {sdk.fsm_name(fsm)}; standing it up for talk mode")
        moves.switch(int(limits.get("walk_fsm")))
    else:
        print(f"ready: FSM {sdk.fsm_name(fsm)}")
    if not keys.confirm("talk mode plays built-in moves and takes short walks when asked, without asking again "
                        "(routines and tasks are asked each time): area clear, spotter and remote ready?"):
        raise SystemExit("not started")
    moves.confirm = lambda question: True
    return Talk(mic, audio, moves, model, menu, verbose=verbose, routines=routines, tasks=tasks,
                runner=robot_runner(mic), keys=keys).run()
