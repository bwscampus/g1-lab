"""``g1 start``'s engine: the robot in one session, driven by what it hears and what you type.

For each thing said (``rt/audio_msg``, the robot's speech recognition) or typed as chat, the model
(``configs/model.json``) makes exactly one choice:

* ``say``: an answer, spoken with ``TtsMaker``; no action.
* ``move``: a **built-in** stored dance or gesture (ids 5xx/6xx), played by ``g1 moves``'s player;
  ``walk``: a built-in walk (turn, walk, sidestep, each capped by a ``talk_*_max`` limit).
* ``play_routine``: a **routine** (``routines/``); ``run_task``: a **task** (``tasks/``).
* ``ignore``: not meant for it.

It only chooses, never writes a routine or task, and a routine or task it chooses is confirmed y/N
first. You can also call any skill yourself (``g1/console.py``: ``/routine``, ``/task``, ``/move``,
``/walk``, ``/say``), which runs without asking. Everything runs here, on the main thread, one event
at a time (``events``), so Ctrl-C, Esc and quitting all stop a skill the one way the safe returns are
shielded on: SIGINT.

The **mode** decides what may run and how: ``standing`` (everything; routines and tasks run in
``--mode standing``) or ``gantry`` (no built-ins or base motion; routines and tasks run in ``--mode
gantry``: the bring-up, then damp at the end). Switching mode never moves the robot. The settings:
the **mic** (speech handled like chat, or ignored), the **camera** (opened once and served at
``view_port``; runs reuse it) and the **lidar** (a SLAM mapping session and its view, map saved when
it ends).
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from g1.core import limits
from g1.envs import sdk
from g1.envs.moves import PLAY_FSMS, Moves, gloss

# The moves the model may play: the gestures and dances (mimic ids 5xx, 6xx). Never the get-up and
# lie-down motions (4xx) or the stance (101), whatever the robot lists.
TALK_MOVES = range(500, 700)
MODES = ("standing", "gantry")

SYSTEM = """You are a Unitree G1, a humanoid robot, at a school assembly. Students and teachers talk to you.
Each turn you get what was just said near you (by voice, as your speech recognition heard it, or typed by the
person at your keyboard; speech can be misheard: answer the most likely meaning). Answer with exactly one tool call:
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
You can only talk and choose from those moves, walks, routines and tasks: you cannot make up new ones, and the
tools offered can change (hanging in a gantry you cannot dance or walk). If asked for something you cannot do, say so
kindly. A person at the keyboard approves each routine and task before it starts.
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
    """The allow-listed built-in moves this robot has: ``{id: {"name", "duration"}}``."""
    return {mid: {"name": move_name(m), "duration": float(m.get("duration", 0))}
            for mid, m in sorted(inv["mimic"].items()) if mid in TALK_MOVES}


@dataclass
class RoutineInfo:
    notes: str
    uses_base: bool = False
    uses_onboard: bool = False


@dataclass
class TaskInfo:
    instruction: str
    hears: bool = False


def routine_menu() -> dict[str, RoutineInfo]:
    """The routines the session may play (a full-body one is left out)."""
    from g1 import routines
    out = {}
    for name in routines.routine_names():
        try:
            r = routines.load_routine(name)
        except routines.RoutineError as e:
            print(f"warning: routine {name} skipped: {e}")
            continue
        if not r.full_body:
            out[name] = RoutineInfo((r.notes.strip().splitlines() or [name])[0], r.uses_base, r.uses_onboard)
    return out


def task_menu() -> dict[str, TaskInfo]:
    """The tasks the session may run."""
    from g1.agent.task import ROOT, load_task
    out = {}
    for d in sorted((ROOT / "tasks").iterdir()):
        if (d / "task.json").is_file():
            try:
                t = load_task(d)
            except ValueError as e:
                print(f"warning: task {d.name} skipped: {e}")
                continue
            out[d.name] = TaskInfo(t["instruction"], bool(t.get("hears")))
    return out


def tools(menu: dict[int, dict], routines: Optional[dict] = None, tasks: Optional[dict] = None,
          *, builtins: bool = True) -> list[dict]:
    """The model's tools for this session: say and ignore always; the built-ins when standing; the
    routines and tasks given (already filtered for the mode)."""
    out = [{"type": "function", "function": {
        "name": "say", "description": "Speak an answer out loud.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "what to say: one to three short sentences"}},
            "required": ["text"]}}}]
    if builtins and menu:
        listing = "; ".join(f"{mid} {m['name']} ({m['duration']:.0f} s)" for mid, m in menu.items())
        out += [
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
                    "required": ["turn_deg", "forward_m", "side_m"]}}}]
    if routines:
        listing = "; ".join(f"{n} ({_notes(r)})" for n, r in routines.items())
        out.append({"type": "function", "function": {
            "name": "play_routine", "description": f"Play one of your routines start to finish. The routines: {listing}.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "enum": list(routines), "description": "the routine"},
                "say": {"type": "string", "description": "a short line to say before it"}},
                "required": ["name"]}}})
    if tasks:
        listing = "; ".join(f"{n}: {_instruction(t)}" for n, t in tasks.items())
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


def _notes(r) -> str:
    return r.notes if isinstance(r, RoutineInfo) else str(r)


def _instruction(t) -> str:
    return t.instruction if isinstance(t, TaskInfo) else str(t)


def _clamp(value, cap: float) -> float:
    try:
        return max(-cap, min(cap, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


@dataclass
class Command:
    """One thing to do, from the console (typed) or the session itself."""
    name: str                      # chat | routine | task | move | walk | say | mode | mic | camera | lidar | status
    args: list[str] = field(default_factory=list)
    text: str = ""                 # chat and say: the words


class ScriptedUI:
    """The session's view of the terminal when there is none (tests): y/N answers from a list."""
    def __init__(self, *answers: bool) -> None:
        self.answers, self.asked = list(answers), []

    def confirm(self, question: str) -> bool:
        self.asked.append(question)
        return bool(self.answers.pop(0)) if self.answers else False


class Session:
    """The robot in one session: built-ins through ``Moves``, routines and tasks through ``runner``
    (``runner(kind, name, mode, settings) -> dict``, ``cli.run_once`` on the robot), the model
    choosing for chat and speech. ``ui.confirm(question) -> bool`` asks y/N. The clock is injectable."""

    def __init__(self, mic, audio, moves: Moves, model, menu: dict[int, dict], *, mode: str = "standing",
                 routines: Optional[dict] = None, tasks: Optional[dict] = None,
                 runner: Optional[Callable[..., dict]] = None, ui=None, verbose: bool = False,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.monotonic,
                 camera_factory: Optional[Callable[[str], Any]] = None, viewer_factory: Optional[Callable] = None,
                 lidar_factory: Optional[Callable[[], Any]] = None, status_probe: Optional[Callable[[], None]] = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mic, self.audio, self.moves, self.model, self.menu = mic, audio, moves, model, menu
        self.mode = mode
        self.routines = {k: (v if isinstance(v, RoutineInfo) else RoutineInfo(str(v))) for k, v in (routines or {}).items()}
        self.tasks = {k: (v if isinstance(v, TaskInfo) else TaskInfo(str(v))) for k, v in (tasks or {}).items()}
        self.runner, self.ui = runner, ui or ScriptedUI()
        self.verbose, self.sleep, self.now = verbose, sleep, now
        self.camera_factory, self.viewer_factory = camera_factory, viewer_factory
        self.lidar_factory, self.status_probe = lidar_factory, status_probe
        self.events: queue.Queue = queue.Queue()
        self.quit = threading.Event()
        self.busy = threading.Event()
        self.running = ""                       # what runs now, for the status line
        self.fsm: Optional[int] = None
        self.mic_on = mic is not None
        self.camera = None
        self.viewer = None
        self.lidar = None
        self.history: list[dict] = []           # user / assistant text turns, newest last
        self.counts = {"heard": 0, "typed": 0, "said": 0, "moves": 0, "walks": 0, "routines": 0, "tasks": 0,
                       "stopped": 0, "refused": 0, "ignored": 0, "errors": 0}
        self._seq = mic.seq if mic is not None else 0

    # -- what may run ------------------------------------------------------------------------------
    def why_not(self, kind: str, name: str = "") -> Optional[str]:
        """Why ``kind name`` may not run now (None: it may)."""
        if kind in ("move", "walk"):
            if self.mode != "standing":
                return "built-ins (moves, walks) need standing mode: /mode standing"
            if kind == "move" and name and self.resolve_move(name) is None:
                return f"no built-in move {name!r} (/skills lists them)"
        if kind == "routine":
            if name not in self.routines:
                return f"no routine {name!r} (/skills lists them)"
            if self.mode == "gantry" and self.routines[name].uses_base:
                return f"routine {name} drives the base: standing mode only"
        if kind == "task":
            if name not in self.tasks:
                return f"no task {name!r} (/skills lists them)"
            if self.tasks[name].hears and not self.mic_on:
                return f"task {name} listens: /mic on first"
        if kind in ("routine", "task") and self.runner is None:
            return "no runner for routines and tasks"
        return None

    def resolve_move(self, name: str) -> Optional[int]:
        """A built-in move by id or by (part of) its name."""
        try:
            mid = int(name)
            return mid if mid in self.menu else None
        except ValueError:
            pass
        want = name.strip().lower()
        hits = [mid for mid, m in self.menu.items() if want == m["name"].lower()] or \
               [mid for mid, m in self.menu.items() if want in m["name"].lower()]
        return hits[0] if len(hits) == 1 else None

    def offered(self) -> tuple[dict, dict]:
        """The routines and tasks the model is offered in this mode."""
        routines = {n: r for n, r in self.routines.items() if self.why_not("routine", n) is None}
        tasks = {n: t for n, t in self.tasks.items() if self.why_not("task", n) is None}
        return routines, tasks

    def model_tools(self) -> list[dict]:
        routines, tasks = self.offered()
        return tools(self.menu, routines, tasks, builtins=self.mode == "standing")

    def skills(self) -> list[str]:
        """``/skills``: every skill, marked available (·) or not (✗ and why)."""
        lines = [f"mode: {self.mode}", "built-ins:"]
        mark = lambda why: "  · " if why is None else "  ✗ "
        for mid, m in self.menu.items():
            why = self.why_not("move", str(mid))
            lines.append(f"{mark(why)}/move {mid}  {m['name']} ({m['duration']:.0f} s)" + (f"  — {why}" if why else ""))
        why = self.why_not("walk")
        lines.append(f"{mark(why)}/walk FWD [SIDE [TURN]]  a short walk" + (f"  — {why}" if why else ""))
        lines.append("routines:")
        for n, r in self.routines.items():
            why = self.why_not("routine", n)
            lines.append(f"{mark(why)}/routine {n}  {r.notes}" + (f"  — {why}" if why else ""))
        lines.append("tasks:")
        for n, t in self.tasks.items():
            why = self.why_not("task", n)
            lines.append(f"{mark(why)}/task {n}  {t.instruction}" + (f"  — {why}" if why else ""))
        return lines

    # -- the status line -----------------------------------------------------------------------------
    def status(self) -> str:
        parts = [self.mode]
        if self.fsm is not None:
            parts.append(f"FSM {self.fsm}")
        parts.append(f"mic {'on' if self.mic_on else 'off'}")
        parts.append(f"camera {self.viewer.url if self.viewer is not None else 'off'}")
        lidar_url = getattr(self.lidar, "url", None) if self.lidar is not None else None
        parts.append(f"lidar {lidar_url or 'off'}")
        parts.append(f"running {self.running}" if self.running else "idle")
        return " · ".join(parts)

    # -- speaking ------------------------------------------------------------------------------------
    def speak(self, text: str) -> None:
        text = text.strip()[: int(limits.get("say_max_chars"))]
        if not text:
            return
        print(f"robot: {text}")
        if self.audio is None:
            return
        code, _ = sdk.call(self.audio, "TtsMaker", text, int(limits.get("tts_speaker_id")),
                           log=self.moves.log, verbose=self.verbose, label="AudioClient")
        if code != 0:
            print(f"warning: TtsMaker failed ({sdk.explain(code)}); voice interaction on in the app?")
            return
        self.counts["said"] += 1
        # The robot speaks on its own; wait out the words, then until its speaker reports quiet.
        self.sleep(max(limits.get("say_min_s"), len(text) / limits.get("speech_chars_per_s")))
        t0 = self.now()
        while self.mic is not None and self.mic.speaking and self.now() - t0 < limits.get("talk_speech_timeout_s"):
            self.sleep(limits.get("talk_poll_s"))

    def _remember(self, heard: str, what: str) -> None:
        self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": what}]

    # -- hearing ---------------------------------------------------------------------------------------
    def heard(self) -> Optional[str]:
        """What has been said since the last turn, once the speaker has been quiet for ``talk_quiet_s``
        (None if nothing, or the mic is off)."""
        if self.mic is None:
            return None
        got = self.mic.since(self._seq)
        if not got:
            return None
        if not self.mic_on:
            self._seq = got[-1].seq                 # said while the mic is off: never answered
            return None
        if self.now() - got[-1].stamp < limits.get("talk_quiet_s"):
            return None
        self._seq = got[-1].seq
        return " ".join(h.text for h in got)

    # -- the model ----------------------------------------------------------------------------------
    def chat(self, text: str, source: str = "typed") -> None:
        """``text`` as if said: the model chooses, and its choice runs."""
        self.counts["heard" if source == "heard" else "typed"] += 1
        if self.model is None:
            print("no model (configs/model.json and its key in .env): chat is off; call skills directly (/skills)")
            return
        messages = ([{"role": "system", "content": system_prompt()}]
                    + self.history[-2 * int(limits.get("talk_history_turns")):]
                    + [{"role": "user", "content": text}])
        t0 = self.now()
        try:
            call = self.model.complete(messages, tools=self.model_tools())
            args = json.loads(call.arguments or "{}")
        except KeyboardInterrupt:
            raise
        except Exception as e:                  # the network, the provider, a bad reply: say so and go on
            self.counts["errors"] += 1
            print(f"model error after {self.now() - t0:.1f} s: {type(e).__name__}: {e}")
            self.speak("Sorry, I could not think of an answer just now.")
            return
        print(f"  ({call.name} after {self.now() - t0:.1f} s)")
        if call.name == "move" and self.mode == "standing" and int(args.get("move_id", -1)) in self.menu:
            self.run_skill("move", str(int(args["move_id"])), heard=text, say=str(args.get("say") or ""), chosen_by="model")
            return
        if call.name == "walk" and self.mode == "standing":
            amounts = {k: _clamp(args.get(k), limits.get(cap)) for k, cap in
                       (("turn_deg", "talk_turn_max_deg"), ("forward_m", "talk_walk_max_m"), ("side_m", "talk_side_max_m"))}
            if any(amounts.values()):
                self.run_skill("walk", "", heard=text, say=str(args.get("say") or ""), chosen_by="model", walk=amounts)
                return
        routines, tasks = self.offered()
        if call.name == "play_routine" and args.get("name") in routines:
            self.run_skill("routine", args["name"], heard=text, say=str(args.get("say") or ""), chosen_by="model")
            return
        if call.name == "run_task" and args.get("name") in tasks:
            self.run_skill("task", args["name"], heard=text, say=str(args.get("say") or ""), chosen_by="model")
            return
        if call.name == "say" and str(args.get("text", "")).strip():
            self._remember(text, args["text"])
            self.speak(str(args["text"]))
            return
        self.counts["ignored"] += 1
        print(f"  (ignored: {args.get('reason') or call.name})")

    # -- running a skill ------------------------------------------------------------------------------
    def _busy(self, what: str, action: Callable[[], object]) -> object:
        self.busy.set()
        self.running = what
        try:
            return action()
        finally:
            self.busy.clear()
            self.running = ""

    def _stopped(self, what: str) -> None:
        self.counts["stopped"] += 1
        print(f"stopped {what}: the robot is back to neutral")

    def run_skill(self, kind: str, name: str, *, heard: str = "", say: str = "", chosen_by: str = "user",
                  walk: Optional[dict] = None) -> None:
        """One skill, the one way: checked against the mode, confirmed when the model chose a routine
        or task, run, and its result said."""
        heard = heard or f"/{kind} {name}".strip()
        why = self.why_not(kind, name)
        if why is not None:
            self.counts["refused"] += 1
            print(f"refused: {why}")
            self._remember(heard, f"[{kind} {name}: not possible now: {why}]")
            if chosen_by == "model":
                self.speak("Sorry, I cannot do that right now.")
            return
        if kind == "move":
            mid = self.resolve_move(name)
            m = self.menu[mid]
            self._remember(heard, f"[played move {mid}: {m['name']}]")
            if say:
                self.speak(say)
            self.counts["moves"] += 1
            move = {"kind": "mimic", "id": mid, "label": m["name"], "duration": m["duration"]}
            if self._busy(f"move {m['name']} (Esc stops)", lambda: self.moves.play([move])) == 130:
                self._stopped("the move")
            self._refresh_fsm()
            return
        if kind == "walk":
            amounts = walk or {}
            what = ", ".join(f"{k} {v:+g}" for k, v in amounts.items() if v)
            self._remember(heard, f"[walked: {what}]")
            if say:
                self.speak(say)
            self.counts["walks"] += 1
            if self._busy(f"walk {what} (Esc stops)", lambda: self.moves.walk(**amounts)) == 130:
                self._stopped("the walk")
            self._refresh_fsm()
            return
        self._run_on_robot(kind, name, heard, say, chosen_by)

    def _run_on_robot(self, kind: str, name: str, heard: str, say: str, chosen_by: str) -> None:
        if chosen_by == "model":
            what = (f"the {name} routine ({self.routines[name].notes})" if kind == "routine"
                    else f"the task {name}: {self.tasks[name].instruction}")
            self.speak(say or (f"I will play my {name} routine." if kind == "routine"
                               else f"I will {self.tasks[name].instruction}."))
            if not self.ui.confirm(f"Start {what}? It takes over the arms ({self.mode} mode)."):
                self._remember(heard, f"[{kind} {name}: not approved at the keyboard]")
                self.speak("Okay, I will not do that.")
                return
        elif say:
            self.speak(say)
        if self.mode == "standing":
            fsm = self.moves.fsm()
            if fsm not in sdk.STANDING_FSMS:         # after a built-in move it stands in 802: back to walk mode
                print(f"the robot is in FSM {sdk.fsm_name(fsm)}; walk mode first")
                self.moves.switch(int(limits.get("walk_fsm")))
        self.counts["routines" if kind == "routine" else "tasks"] += 1
        settings = {"microphone": self.mic, "camera_shared": self.camera}
        try:
            result = self._busy(f"{kind} {name} (Esc stops)", lambda: self.runner(kind, name, self.mode, settings))
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
        finally:
            self._refresh_fsm()
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

    def _refresh_fsm(self) -> None:
        try:
            code, fsm = self.moves._call(self.moves.loco, "GetFsmId")
            self.fsm = int(fsm) if code == 0 else None
        except Exception:
            self.fsm = None

    # -- mode and settings ----------------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Change what may run and how. Never moves the robot."""
        if mode not in MODES:
            print(f"mode is standing or gantry, not {mode!r}")
            return
        if mode == self.mode:
            print(f"already in {mode} mode")
            return
        if mode == "standing":
            fsm = self.moves.fsm()
            if fsm not in sdk.STANDING_FSMS | PLAY_FSMS:
                print(f"refused: the robot reports FSM {sdk.fsm_name(fsm)}, not standing; stand it up with the remote "
                      "(or g1 moves fsm walk), then /mode standing")
                return
            question = "Is the robot standing on its own feet, area clear, spotter and remote ready?"
        else:
            question = "Is the robot hanging in the gantry (feet just touching or off the floor)?"
        if not self.ui.confirm(question):
            print("mode unchanged")
            return
        self.mode = mode
        print(f"mode: {mode}" + ("" if mode == "standing" else
                                 " (no built-ins or base motion; routines and tasks run with the gantry bring-up and damp at the end)"))
        self._remember(f"/mode {mode}", f"[now in {mode} mode]")

    def set_mic(self, on: bool) -> None:
        if self.mic is None:
            print("no microphone: the robot's speech recognition did not subscribe")
            return
        self.mic_on = on
        if not on:
            self._seq = self.mic.seq                # nothing said so far is answered
        print(f"mic {'on: what the robot hears is handled like chat' if on else 'off: speech is ignored'}")

    def set_camera(self, on: bool) -> None:
        if on == (self.camera is not None):
            print(f"camera already {'on' if on else 'off'}")
            return
        if not on:
            self._stop_camera()
            print("camera off")
            return
        if self.camera_factory is None:
            print("no camera here")
            return
        try:
            self.camera = self.camera_factory(self.mode)
            if self.viewer_factory is not None:
                self.viewer = self.viewer_factory(self.camera)
        except (SystemExit, Exception) as e:
            self._stop_camera()
            print(f"camera: could not open it: {e}")
            return
        url = self.viewer.url if self.viewer is not None else "(no view)"
        print(f"camera on: {url}  (from a laptop: ssh -L {getattr(self.viewer, 'port', '')}:127.0.0.1:"
              f"{getattr(self.viewer, 'port', '')} unitree@<the Jetson>); tasks use it too")

    def _stop_camera(self) -> None:
        if self.viewer is not None:
            self.viewer.stop()
            self.viewer = None
        if self.camera is not None:
            self.camera.stop()
            self.camera = None

    def set_lidar(self, on: bool) -> None:
        from g1.envs.base import shield_sigint
        if on == (self.lidar is not None):
            print(f"lidar already {'on' if on else 'off'}")
            return
        if not on:
            with shield_sigint("ending mapping and saving the map; this finishes on its own"):
                self.lidar.stop()
            self.lidar = None
            return
        if self.lidar_factory is None:
            print("no lidar here")
            return
        session = self.lidar_factory()
        try:
            session.start()
        except KeyboardInterrupt:
            print("lidar: not started (stopped)")
            return
        except Exception as e:
            print(f"lidar: {e}")
            return
        self.lidar = session

    # -- the loop --------------------------------------------------------------------------------------
    def handle(self, cmd: Command) -> None:
        """One command, on the main thread."""
        n, a = cmd.name, cmd.args
        if n == "chat":
            self.chat(cmd.text, "typed")
        elif n == "heard":
            print(f"\nheard: {cmd.text}")
            self.chat(cmd.text, "heard")
        elif n in ("routine", "task"):
            self.run_skill(n, a[0] if a else "")
        elif n == "move":
            self.run_skill("move", " ".join(a))
        elif n == "walk":
            amounts = {"forward_m": float(a[0]) if a else 0.0, "side_m": float(a[1]) if len(a) > 1 else 0.0,
                       "turn_deg": float(a[2]) if len(a) > 2 else 0.0}
            if self.why_not("walk") is None:
                amounts = {k: _clamp(v, limits.get(cap)) for (k, v), cap in
                           zip(amounts.items(), ("talk_walk_max_m", "talk_side_max_m", "talk_turn_max_deg"))}
            self.run_skill("walk", "", walk=amounts)
        elif n == "say":
            self.speak(cmd.text)
        elif n == "mode":
            self.set_mode(a[0] if a else "")
        elif n in ("mic", "camera", "lidar"):
            getattr(self, f"set_{n}")(bool(a) and a[0] == "on")
        elif n == "status":
            if self.status_probe is not None:
                self.status_probe()
            self._refresh_fsm()
            print(self.status())
        else:
            print(f"unknown command {n!r}")

    def run(self) -> int:
        """Events and speech, one at a time, until quit. A stop (Esc, Ctrl-C) with nothing running is
        a no-op; quitting ends any running skill first (through SIGINT) and then the session."""
        self._refresh_fsm()
        try:
            while not self.quit.is_set():
                try:
                    try:
                        cmd = self.events.get(timeout=limits.get("talk_poll_s"))
                    except queue.Empty:
                        said = self.heard()
                        cmd = Command("heard", text=said) if said else None
                    if cmd is not None:
                        self.handle(cmd)
                except KeyboardInterrupt:           # a stop between skills: nothing to stop
                    pass
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        """Lidar mapping ended and saved, the camera closed, the counts printed."""
        from g1.envs.base import shield_sigint
        with shield_sigint("closing the session; this finishes on its own"):
            if self.lidar is not None:
                self.lidar.stop()
                self.lidar = None
            self._stop_camera()
        print("session: " + ", ".join(f"{k} {v}" for k, v in self.counts.items())
              + (f"; microphone: {self.mic.counts}" if self.mic is not None else ""))


def robot_runner(*, ask: bool = False) -> Callable[..., dict]:
    """Runs a routine or task on the robot through ``cli.run_once`` in the session's mode, with its
    microphone and camera, kept unreviewed (no questions mid-session)."""
    def run(kind: str, name: str, mode: str, settings: dict) -> dict:
        from g1 import routines
        from g1.agent.task import find, load_task
        from g1.cli import run_once
        spec = routines.run_spec(name) if kind == "routine" else load_task(find(name))
        code, folder = run_once(spec, sim=False, mode=mode, ask=ask, settings=settings)
        return result_of(code, folder)
    return run


def result_of(code: int, folder) -> dict:
    """What the session says after a run: ``ok``, ``outcome``, ``stopped`` and the model's done summary."""
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
