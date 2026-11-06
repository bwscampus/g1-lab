"""``g1 start``'s engine: the robot in one session, driven by what it hears and what you type.

For each thing said (``rt/audio_msg``, the robot's speech recognition) or typed as chat, the model
(``configs/model.json``) makes exactly one choice:

* ``say``: an answer, spoken with ``TtsMaker``; no action.
* ``move``: a **built-in** stored dance or gesture (ids 5xx/6xx), played by ``g1 moves``'s player;
* ``play_routine``: a **routine** (``routines/``); ``run_task``: a **task** (``tasks/``).
* ``remember``: a fact kept in MEMORY.md (or one forgotten), with a line said.
* ``ignore``: not meant for it.

Who the robot is comes from ``knowledge/SOUL.md`` (people write it) and what it knows from
``knowledge/MEMORY.md`` (the robot writes it), both read every turn after the rules in ``MECHANICS``;
each exchange goes into the day's log, ``knowledge/sessions/DATE.md``, and when the session ends the
model distills it into MEMORY.md (``g1/knowledge.py``).

It only chooses, never writes a routine or task, and a routine or task it chooses is confirmed y/N
first. You can also call any skill yourself (``g1/console.py``: ``/routine``, ``/task``, ``/move``,
``/say``), which runs without asking. Everything runs here, on the main thread, one event
at a time (``events``), so Ctrl-C, Esc and quitting all stop a skill the one way the safe returns are
shielded on: SIGINT.

The **mode** is who commands the legs. ``standing``: we do, through LocoClient (built-in moves; the
routines and tasks whose ``modes`` list standing run in ``--mode standing``). ``gantry``: never (no
built-ins, no base motion; only the routines and tasks listing gantry, the arms through arm_sdk while
Unitree's controller holds the legs). A mode never changes the FSM; it **blocks** FSMs (``MODE_FSMS``):
gantry allows only 0, 1, 4 and 200, standing every FSM but 0 and 1. ``/fsm ID`` switches the FSM (a
person decides) and is refused for one the mode blocks ("switch modes first"); ``/mode`` is refused
while the robot is in an FSM the new mode blocks; a skill is refused in one. The one switch the session
makes itself is the start's bring-up, in both modes: 0 -> 1 -> 4 -> 200 from wherever the robot is on
that way (after a y/N). Quitting leaves the FSM as it is. The settings:
the **mic** (speech handled like chat, or ignored), the **speaker** (replies said out loud, or only
printed; tasks run here follow it too), the **camera** (opened once and served at
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
from g1.envs.moves import FSM_WARN, PLAY_FSMS, Moves, gloss

# The moves the model may play: the gestures and dances (mimic ids 5xx, 6xx). Never the get-up and
# lie-down motions (4xx) or the stance (101), whatever the robot lists.
TALK_MOVES = range(500, 700)
MODES = ("standing", "gantry")


GANTRY_FSMS = frozenset({0, 1, 4, 200})        # hanging: limp, locked stand, main operation
STANDING_BLOCKED = frozenset({0, 1})            # on its feet: never zero torque or damp (it would fall)
BRING_UP = (0, 1, 4, 200)                       # the start's bring-up, both modes
MAIN_FSM = 200                                  # main operation: arm_sdk works; routines and tasks run (and stay) here


def fsm_allowed(mode: str, fsm) -> bool:
    """Whether ``mode`` lets the robot be in ``fsm`` (gantry: 0, 1, 4, 200; standing: all but 0, 1)."""
    return fsm in GANTRY_FSMS if mode == "gantry" else fsm not in STANDING_BLOCKED


def allowed_text(mode: str) -> str:
    return "0, 1, 4 and 200" if mode == "gantry" else "any FSM but 0 and 1"


# The session's rules: how to answer and what is safe. Who the robot is, where, and for whom is SOUL.md
# (knowledge/, written by people); what it remembers is MEMORY.md (written by the robot). These stay in
# code so no edit of either file can remove them.
MECHANICS = """You are a Unitree G1 humanoid robot (about 1.3 m tall, made by Unitree Robotics), running on your own
onboard computer. People talk to you. Each turn you get what was just said near you (by voice, as your speech
recognition heard it, or typed by the person at your keyboard; speech can be misheard: answer the most likely
meaning). Answer with exactly one tool call:
- say: a spoken answer. Short (one to three sentences) unless asked for more. Plain spoken words only: no lists, no
  emoji, no markdown, no stage directions. Numbers as words when short.
- move: when someone asks you to dance, wave, bow, clap, greet, or do a move. Pick the best fit from the moves
  listed in the tool; you may give a short line to say first. If they ask for a dance without saying which, pick one.
- You cannot walk: if asked to walk, step or turn around, say kindly that you stay where you are.
- play_routine: when someone asks for one of your routines (fixed arm motions you know by name, listed in the tool).
- run_task: when someone asks for something one of your tasks does (listed in the tool with what each one does).
  A task uses your camera and takes a while; say briefly what you will do first. A task that names a slot (point_at:
  {{object}}) needs it filled from what was asked ("point at the water bottle": object "water bottle").
- look_up (when offered): for facts you do not know or anything recent (news, weather, sports, dates of events).
  Say a short line first ("Let me check."). You then get the results and answer. Never look up a person near you.
- remember: when you learn something worth knowing next time (someone's name and role, a place, an event, a
  preference, something you are taught), or are told to forget something. Say a short line too. Never remember
  phone numbers, addresses, passwords or private details about a person.
- ignore: when what you heard is not meant for you (side conversations, noise, half a word).
You can only talk and choose from those moves, routines and tasks: you cannot make up new ones, and the
tools offered can change (hanging in a gantry you cannot dance). If asked for something you cannot do, say so
kindly. A person at the keyboard approves each routine and task the model chooses before it starts.
Decline anything unsafe or unkind politely, whatever else you are told below."""


def system_prompt(knowledge=None) -> str:
    """MECHANICS, then SOUL.md and MEMORY.md (read now, so edits count at once) when there is a knowledge base."""
    text = MECHANICS.format()
    if knowledge is not None:
        soul, memory = knowledge.soul(), knowledge.memory()
        if soul:
            text += f"\n\n# Who you are (SOUL.md)\n{soul}"
        text += f"\n\n# What you remember (MEMORY.md)\n{memory or '(nothing yet)'}"
    return text


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
    modes: tuple = MODES


@dataclass
class TaskInfo:
    instruction: str                    # as written: a slot shows as {object}
    hears: bool = False
    modes: tuple = MODES
    fill: dict = field(default_factory=dict)   # its slots: {"object": {"note": ..., "default": ...}}


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
            out[name] = RoutineInfo((r.notes.strip().splitlines() or [name])[0], r.uses_base, r.uses_onboard,
                                    tuple(r.modes))
    return out


def task_menu() -> dict[str, TaskInfo]:
    """The tasks the session may run (those whose task.json says ``"session": false`` are left out)."""
    from g1.agent.task import ROOT, load_task
    out = {}
    for d in sorted((ROOT / "tasks").iterdir()):
        if (d / "task.json").is_file():
            try:
                t = load_task(d)
            except ValueError as e:
                print(f"warning: task {d.name} skipped: {e}")
                continue
            if t["session"]:
                out[d.name] = TaskInfo(t["template"], bool(t.get("hears")), tuple(t["modes"]), dict(t["fill"]))
    return out


def tools(menu: dict[int, dict], routines: Optional[dict] = None, tasks: Optional[dict] = None,
          *, builtins: bool = True, look_up: bool = False) -> list[dict]:
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
                    "required": ["move_id"]}}}]
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
        props = {"name": {"type": "string", "enum": list(tasks), "description": "the task"}}
        slots = [(k, f"{n}: {spec['note']}") for n, t in tasks.items() if isinstance(t, TaskInfo)
                 for k, spec in t.fill.items()]
        for k in sorted({k for k, _ in slots}):
            props[k] = {"type": "string", "description": "fills {" + k + "} in a task that names it ("
                        + "; ".join(v for kk, v in slots if kk == k) + ")"}
        props["say"] = {"type": "string", "description": "a short line to say before it: what you will do"}
        out.append({"type": "function", "function": {
            "name": "run_task", "description": f"Run one of your tasks with your camera until it is done. The tasks: {listing}.",
            "parameters": {"type": "object", "properties": props, "required": ["name"]}}})
    out.append({"type": "function", "function": {
        "name": "remember", "description": "Keep a fact in your long-term memory (MEMORY.md), or forget one.",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string", "description": "the fact to keep, one short sentence (or empty when forgetting)"},
            "forget": {"type": "string", "description": "optional: a memory to remove, copied exactly as it appears"},
            "say": {"type": "string", "description": "a short line to say, e.g. that you will remember it"}},
            "required": ["fact"]}}})
    if look_up:
        out.append({"type": "function", "function": {
            "name": "look_up", "description": "Search the web for a fact you do not know or something recent; "
                                              "you get the results next and then answer.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "a short web search query"},
                "say": {"type": "string", "description": "a short line to say while you look, e.g. Let me check."}},
                "required": ["query"]}}})
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


def speech_pieces(text: str, size: int) -> list[str]:
    """``text`` as pieces of at most ``size`` characters, split after sentence ends (else at a word,
    else hard), nothing dropped."""
    sentences = [s for s in re.split(r"(?<=[.!?;:])\s+", text.strip()) if s]
    pieces: list[str] = []
    for s in sentences:
        while len(s) > size:                         # one long sentence: at the last space that fits
            cut = s.rfind(" ", 0, size + 1)
            cut = cut if cut > 0 else size
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        if pieces and len(pieces[-1]) + 1 + len(s) <= size:
            pieces[-1] += " " + s
        elif s:
            pieces.append(s)
    return pieces


@dataclass
class Command:
    """One thing to do, from the console (typed) or the session itself."""
    name: str                      # chat | routine | task | move | say | fsm | mode | mic | camera | lidar | status
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
                 lidar_factory: Optional[Callable[[], Any]] = None, status_probe: Optional[Callable[[], None]] = None, knowledge=None) -> None:
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
        self.mic_on = False                         # off at the start: /mic on handles what it hears like chat
        self.knowledge = knowledge                  # g1/knowledge.py: SOUL.md, MEMORY.md, the daily log (None: none)
        self._source = "typed"                      # what the exchange being handled came from (the log)
        self.speaker_on = True                      # off: replies printed only, never sent to TtsMaker
        self.camera = None
        self.viewer = None
        self.lidar = None
        self.history: list[dict] = []           # user / assistant text turns, newest last
        self.counts = {"heard": 0, "typed": 0, "said": 0, "moves": 0, "routines": 0, "tasks": 0,
                       "stopped": 0, "refused": 0, "ignored": 0, "remembered": 0, "looked_up": 0, "errors": 0}
        self._seq = mic.seq if mic is not None else 0

    # -- what may run ------------------------------------------------------------------------------
    def why_not(self, kind: str, name: str = "") -> Optional[str]:
        """Why ``kind name`` may not run now (None: it may)."""
        if kind == "move":
            if self.mode != "standing":
                return "built-in moves need standing mode: /mode standing"
            if kind == "move" and name and self.resolve_move(name) is None:
                return f"no built-in move {name!r} (/skills lists them)"
        if kind == "routine":
            if name not in self.routines:
                return f"no routine {name!r} (/skills lists them)"
        if kind == "task":
            if name not in self.tasks:
                return f"no task {name!r} (/skills lists them)"
            if self.tasks[name].hears and not self.mic_on:
                return f"task {name} listens: /mic on first"
        if kind in ("routine", "task"):
            modes = (self.routines if kind == "routine" else self.tasks)[name].modes
            if self.mode not in modes:
                return f"{kind} {name} runs in {' and '.join(modes)} mode only: /mode {modes[0]}"
            if self.runner is None:
                return "no runner for routines and tasks"
        return self.fsm_refusal(kind)

    def fsm_refusal(self, kind: str) -> Optional[str]:
        """Why no ``kind`` may start from the FSM the robot reports (None: it may)."""
        if self.fsm is None:
            return "the robot's FSM is unknown (no reply to GetFsmId): /status"
        here = sdk.fsm_name(self.fsm)
        if not fsm_allowed(self.mode, self.fsm):
            return f"the robot is in FSM {here}, which {self.mode} mode blocks ({allowed_text(self.mode)}): /fsm first"
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

    def can_look_up(self) -> bool:
        """The chat model may search the web: it has ``search`` and is served by OpenRouter."""
        return callable(getattr(self.model, "search", None)) and getattr(self.model, "provider", "") == "openrouter"

    def model_tools(self, *, look_up: bool = True) -> list[dict]:
        routines, tasks = self.offered()
        return tools(self.menu, routines, tasks, builtins=self.mode == "standing",
                     look_up=look_up and self.can_look_up())

    def skills(self) -> list[str]:
        """``/skills``: the skills this mode (and the mic) allows now, grouped; nothing else. Typing
        one it leaves out still says why it is refused."""
        groups = {
            "built-ins": [f"/move {mid}  {m['name']} ({m['duration']:.0f} s)" for mid, m in self.menu.items()
                          if self.why_not("move", str(mid)) is None],
            "routines": [f"/routine {n}  {r.notes}" for n, r in self.routines.items() if self.why_not("routine", n) is None],
            "tasks": [f"/task {n}  {t.instruction}" for n, t in self.tasks.items() if self.why_not("task", n) is None],
        }
        lines = [f"mode: {self.mode}"]
        for title, items in groups.items():
            lines.append(f"{title}:" + ("" if items else f" (none in {self.mode} mode)"))
            lines += [f"  · {item}" for item in items]
        return lines

    # -- the status line -----------------------------------------------------------------------------
    def status(self) -> str:
        parts = [self.mode]
        if self.fsm is not None:
            parts.append(f"FSM {self.fsm}" + ("" if fsm_allowed(self.mode, self.fsm) else f" (blocked in {self.mode})"))
        parts.append(f"mic {'on' if self.mic_on else 'off'}")
        parts.append(f"speaker {'on' if self.speaker_on else 'off'}")
        parts.append(f"camera {self.viewer.url if self.viewer is not None else 'off'}")
        lidar_url = getattr(self.lidar, "url", None) if self.lidar is not None else None
        parts.append(f"lidar {lidar_url or 'off'}")
        parts.append(f"running {self.running}" if self.running else "idle")
        return " · ".join(parts)

    # -- speaking ------------------------------------------------------------------------------------
    def speak(self, text: str) -> None:
        """Say all of ``text``: printed whole, spoken in pieces of at most ``say_max_chars`` split at
        sentence ends (one TtsMaker call each, the next after the speaker is quiet), never cut off."""
        text = " ".join(text.split())
        if not text:
            return
        print(f"robot: {text}")
        if self.audio is None or not self.speaker_on:
            return
        self.counts["said"] += 1
        for piece in speech_pieces(text, int(limits.get("say_max_chars"))):
            code, _ = sdk.call(self.audio, "TtsMaker", piece, int(limits.get("tts_speaker_id")),
                               log=self.moves.log, verbose=self.verbose, label="AudioClient")
            if code != 0:
                print(f"warning: TtsMaker failed ({sdk.explain(code)}); voice interaction on in the app?")
                return
            # The robot speaks on its own; wait out the words, then until its speaker reports quiet.
            self.sleep(max(limits.get("say_min_s"), len(piece) / limits.get("speech_chars_per_s")))
            t0 = self.now()
            while self.mic is not None and self.mic.speaking and self.now() - t0 < limits.get("talk_speech_timeout_s"):
                self.sleep(limits.get("talk_poll_s"))

    def _remember(self, heard: str, what: str) -> None:
        """One exchange: kept for the next turns (in memory) and in the day's log (knowledge/sessions/)."""
        self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": what}]
        self._log(heard, what)

    def _log(self, heard: str, what: str) -> None:
        if self.knowledge is not None:
            try:
                self.knowledge.log(self._source, heard, what)
            except OSError as e:
                print(f"warning: the session log was not written ({e})")

    # -- memory ----------------------------------------------------------------------------------------
    def _learn(self, heard: str, fact: str, forget: str, say: str) -> None:
        """The model's ``remember``: keep a fact, forget one, or both; then say its line."""
        if self.knowledge is None:
            print("  (remember: no knowledge base in this session)")
            self._remember(heard, "[could not remember: no memory]")
            self.speak(say or "Sorry, I cannot keep memories right now.")
            return
        done = []
        if forget.strip():
            if self.knowledge.forget(forget):
                print(f"memory: - {forget.strip()}")
                done.append(f"forgot: {forget.strip()}")
            else:
                print(f"memory: nothing matched {forget.strip()!r}")
        if fact.strip():
            line = self.knowledge.remember(fact)
            if line:
                self.counts["remembered"] += 1
                print(f"memory: + {line}")
                done.append(f"remembered: {fact.strip()}")
            else:
                print(f"memory: already known: {fact.strip()}")
        self._remember(heard, f"[{'; '.join(done) or 'memory unchanged'}]" + (f" {say}" if say else ""))
        if say:
            self.speak(say)

    def memory_lines(self) -> list[str]:
        """``/memory``: where it is, how big, and the newest bullets."""
        if self.knowledge is None:
            return ["no knowledge base in this session"]
        k = self.knowledge
        bullets = k.memory_bullets()
        size = len(k.memory_path.read_text(encoding="utf-8")) if k.memory_path.exists() else 0
        lines = [f"{k.memory_path}: {len(bullets)} memories, {size} of {limits.get('memory_max_chars'):g} chars"
                 + (" (over: the oldest are left out of the prompt until the next distill)" if k.over() else "")]
        shown = int(limits.get("memory_show_bullets"))
        lines += [f"  - {b}" for b in bullets[-shown:]] or ["  (nothing yet)"]
        return lines

    def soul_lines(self) -> list[str]:
        """``/soul``: where it is and how it starts."""
        if self.knowledge is None:
            return ["no knowledge base in this session"]
        soul = self.knowledge.soul()
        shown = int(limits.get("memory_show_bullets"))
        return [f"{self.knowledge.soul_path} (edit it in an editor; read afresh every turn)"] + \
            [f"  {line}" for line in soul.splitlines()[:shown]] + (["  ..."] if len(soul.splitlines()) > shown else [])

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
        self._refresh_fsm()                          # what may run (the tools offered) depends on it
        messages = ([{"role": "system", "content": system_prompt(self.knowledge)}]
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
        self._act(text, call.name, args, messages)

    def _act(self, text: str, name: str, args: dict, messages: Optional[list] = None) -> None:
        """Run the model's one choice ``name(args)`` for ``text``. ``messages``: the request it answered
        (a look_up asks again from them; None: no second look-up)."""
        if name == "look_up" and messages is not None and self.can_look_up() and str(args.get("query") or "").strip():
            self._look_up(text, str(args["query"]).strip(), str(args.get("say") or ""), messages)
            return
        if name == "move" and self.mode == "standing" and int(args.get("move_id", -1)) in self.menu:
            self.run_skill("move", str(int(args["move_id"])), heard=text, say=str(args.get("say") or ""), chosen_by="model")
            return
        routines, tasks = self.offered()
        if name == "play_routine" and args.get("name") in routines:
            self.run_skill("routine", args["name"], heard=text, say=str(args.get("say") or ""), chosen_by="model")
            return
        if name == "run_task" and args.get("name") in tasks:
            fill = {k: str(args.get(k) or "") for k in tasks[args["name"]].fill}
            self.run_skill("task", args["name"], heard=text, say=str(args.get("say") or ""), chosen_by="model", fill=fill)
            return
        if name == "remember":
            self._learn(text, str(args.get("fact") or ""), str(args.get("forget") or ""), str(args.get("say") or ""))
            return
        if name == "say" and str(args.get("text", "")).strip():
            self._remember(text, args["text"])
            self.speak(str(args["text"]))
            return
        self.counts["ignored"] += 1
        print(f"  (ignored: {args.get('reason') or name})")
        self._log(text, f"ignored ({args.get('reason') or name})")

    def _look_up(self, text: str, query: str, say: str, messages: list) -> None:
        """Say the line, search the web (bounded by look_up_timeout_s), then ask the model again with the
        results, without look_up, and run its choice."""
        if say:
            self.speak(say)
        self.counts["looked_up"] += 1
        timeout = limits.get("look_up_timeout_s")
        box: dict = {}

        def search():
            try:
                box["text"] = self.model.search(query, timeout=timeout)
            except Exception as e:                   # the network, the provider
                box["error"] = e
        t = threading.Thread(target=search, name="look-up", daemon=True)
        t.start()
        t.join(timeout)
        found = None if t.is_alive() else box.get("text")
        if not found:
            why = "no reply in time" if t.is_alive() else (box.get("error") or "nothing found")
            print(f"  look_up {query!r}: {why}")
            self._remember(text, f"[looked up: {query}: no answer]")
            self.speak("Sorry, I could not look that up right now.")
            return
        print(f"  look_up {query!r}: {found}")
        again = messages + [{"role": "assistant", "content": f"[looked up: {query}]"},
                            {"role": "user", "content": f"Web search results for {query!r}: {found}\n"
                                                        f"Now answer what was said: {text}"}]
        t0 = self.now()
        try:
            call = self.model.complete(again, tools=self.model_tools(look_up=False))
            args = json.loads(call.arguments or "{}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.counts["errors"] += 1
            print(f"model error after {self.now() - t0:.1f} s: {type(e).__name__}: {e}")
            self.speak("Sorry, I could not think of an answer just now.")
            return
        print(f"  ({call.name} after {self.now() - t0:.1f} s)")
        self._act(text, call.name, args)

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
                  fill: Optional[dict] = None) -> None:
        """One skill, the one way: checked against the mode, confirmed when the model chose a routine
        or task, run, and its result said. ``fill``: a task's slots (what to point at); each must be given."""
        heard = heard or (f"/{kind} {name} " + " ".join((fill or {}).values())).strip()
        self._refresh_fsm()
        why = self.why_not(kind, name)
        if why is None and kind == "task":
            from g1.agent.task import clean_fill
            fill = {k: clean_fill((fill or {}).get(k)) for k in self.tasks[name].fill}
            missing = [k for k, v in fill.items() if not v]
            if missing:
                why = (f"task {name} needs its {', '.join(missing)} (" + "; ".join(
                    self.tasks[name].fill[k]["note"] for k in missing) + f"): /task {name} WHAT")
        if why is not None:
            self.counts["refused"] += 1
            print(f"refused: {why}")
            self._remember(heard, f"[{kind} {name}: not possible now: {why}]")
            if chosen_by == "model":
                self.speak("Sorry, I cannot do that right now." if "needs its" not in why
                           else "What should I point at? Please tell me again." if "object" in why
                           else "Sorry, I need a bit more to go on. Please tell me again.")
            return
        if kind == "move":
            mid = self.resolve_move(name)
            m = self.menu[mid]
            self._remember(heard, f"[played move {mid}: {m['name']}]")
            if say:
                self.speak(say)
            self.counts["moves"] += 1
            move = {"kind": "mimic", "id": mid, "label": m["name"], "duration": m["duration"]}
            caller = self.fsm
            if self.fsm not in PLAY_FSMS and not self._switch_to(MAIN_FSM, "a stored move plays from standing"):
                self.speak("Sorry, I cannot do that right now.")
                return
            try:
                if self._busy(f"move {m['name']} (Esc stops)", lambda: self.moves.play([move])) == 130:
                    self._stopped("the move")
            finally:
                self._return_to(caller)                  # a stored move ends in 802: back where it was called from
            return
        self._run_on_robot(kind, name, heard, say, chosen_by, fill or {})

    def _run_on_robot(self, kind: str, name: str, heard: str, say: str, chosen_by: str, fill: dict) -> None:
        label = name + (f" ({', '.join(fill.values())})" if fill else "")
        instruction = ""
        if kind == "task":
            from g1.agent.task import fill_instruction
            instruction = fill_instruction(self.tasks[name].instruction, self.tasks[name].fill, fill)[0]
        if chosen_by == "model":
            what = (f"the {name} routine ({self.routines[name].notes})" if kind == "routine"
                    else f"the task {name}: {instruction}")
            self.speak(say or (f"I will play my {name} routine." if kind == "routine"
                               else f"I will {instruction.split(';')[0]}."))
            if not self.ui.confirm(f"Start {what}? It takes over the arms ({self.mode} mode)."):
                self._remember(heard, f"[{kind} {name}: not approved at the keyboard]")
                self.speak("Okay, I will not do that.")
                return
        elif say:
            self.speak(say)
        if self.fsm != MAIN_FSM and not self._switch_to(MAIN_FSM, f"a {kind} runs in main operation"):
            self._remember(heard, f"[{kind} {label}: could not run: not in FSM {MAIN_FSM}]")
            self.speak(f"Sorry, I could not do the {name} {kind}.")
            return
        self.counts["routines" if kind == "routine" else "tasks"] += 1
        settings = {"microphone": self.mic, "camera_shared": self.camera, "speaker": self.speaker_on}
        run = ((lambda: self.runner(kind, name, self.mode, settings, fill=fill)) if fill
               else (lambda: self.runner(kind, name, self.mode, settings)))
        try:
            result = self._busy(f"{kind} {label} (Esc stops)", run)
        except KeyboardInterrupt:
            self._stopped(f"{kind} {label}")
            self._remember(heard, f"[{kind} {label}: stopped]")
            return
        except (SystemExit, Exception) as e:        # refused or failed: the env's teardown has already run
            self.counts["errors"] += 1
            print(f"{kind} {label} failed: {e}")
            self._remember(heard, f"[{kind} {label}: could not run]")
            self.speak(f"Sorry, I could not do the {name} {kind}.")
            return
        finally:
            self._refresh_fsm()
        if result.get("stopped"):
            self._stopped(f"{kind} {label}")
        outcome = "stopped" if result.get("stopped") else (result.get("outcome") or "unknown")
        self._remember(heard, f"[{kind} {label}: {outcome}]")
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

    def _switch_to(self, target: int, why: str = "") -> bool:
        """``SetFsmId(target)`` now and wait (``fsm_enter_timeout_s``) for the robot to report it; prints
        what happened. True when it is there. Never asks: callers check the mode and ask first."""
        self._refresh_fsm()
        if self.fsm == target:
            return True
        before = sdk.fsm_name(self.fsm) if self.fsm is not None else "unknown"
        print(f"FSM {before} -> {sdk.fsm_name(target)}: SetFsmId({target})" + (f" ({why})" if why else ""))
        code, _ = self.moves._call(self.moves.loco, "SetFsmId", target)
        if code != 0:
            print(f"refused: SetFsmId({target}) -> {sdk.explain(code)}")
            self._refresh_fsm()
            return False
        t0 = self.now()
        while True:
            self._refresh_fsm()
            if self.fsm == target or self.now() - t0 > limits.get("fsm_enter_timeout_s"):
                break
            self.sleep(limits.get("fsm_poll_s"))
        now = sdk.fsm_name(self.fsm) if self.fsm is not None else "unknown"
        print(f"FSM {now} after {self.now() - t0:.1f} s" + ("" if self.fsm == target else
              f": SetFsmId returned 0 but the robot did not enter {sdk.fsm_name(target)}"))
        return self.fsm == target

    def _return_to(self, caller) -> None:
        """After a built-in: back to the FSM it was called from (when it left it). Never raises."""
        try:
            self._refresh_fsm()
            if caller is not None and self.fsm != caller and not self._switch_to(caller, "back where the built-in started"):
                print(f"warning: still in FSM {self.fsm}, not {caller}: /fsm {caller}")
        except Exception as e:
            print(f"warning: could not return to FSM {caller}: {e}")

    def start(self) -> bool:
        """The session's first step: a person confirms the mode (hanging in the gantry, or on its own feet
        with the area clear), then the bring-up 0 -> 1 -> 4 -> 200 from wherever the robot is on that way
        (both modes; elsewhere it is left as it is). False: not started."""
        self._refresh_fsm()
        if self.mode == "gantry":
            question = "Is the robot hanging in the gantry (feet just touching or off the floor)?"
        else:
            question = ("Is the robot standing on its own feet, area clear, spotter and remote ready? Built-ins "
                        "(dances, gestures) play when asked; routines and tasks the model chooses are asked each time.")
        up = self.fsm in BRING_UP and self.fsm != BRING_UP[-1]
        if up:
            path = " -> ".join(sdk.fsm_name(f) for f in BRING_UP[BRING_UP.index(self.fsm):])
            question += f" Then the bring-up: FSM {path} (locked stand: feet just touching)."
        if not self.ui.confirm(question):
            return False
        if up:
            try:
                self.moves.switch(BRING_UP[-1])
            except SystemExit as e:
                print(f"bring-up stopped: {e}")
            self._refresh_fsm()
        here = sdk.fsm_name(self.fsm) if self.fsm is not None else "unknown"
        print(f"FSM {here}" + ("" if fsm_allowed(self.mode, self.fsm) else
                               f": {self.mode} mode blocks it ({allowed_text(self.mode)}); /fsm or /mode first"))
        return True

    def set_fsm(self, target: int) -> None:
        """``/fsm ID``: try ``SetFsmId(ID)`` now, outside the allowed order (to find out what the robot
        takes), and report the FSM it reaches. A y/N first for the states that can drop or seat it."""
        if not fsm_allowed(self.mode, target):
            other = "standing" if self.mode == "gantry" else "gantry"
            print(f"refused: {self.mode} mode blocks FSM {sdk.fsm_name(target)} ({self.mode} allows "
                  f"{allowed_text(self.mode)}); switch modes first: /mode {other}")
            return
        warning = {0: "zero torque: every motor goes off; a standing robot falls. Only hanging in the gantry",
                   **FSM_WARN}.get(target)
        self._refresh_fsm()
        before = sdk.fsm_name(self.fsm) if self.fsm is not None else "unknown"
        if warning and not self.ui.confirm(f"Switch FSM {before} -> {sdk.fsm_name(target)}? {warning}"):
            print("not switched")
            return
        if self.fsm == target:
            print(f"already in FSM {before}")
            return
        self._switch_to(target)
        now = sdk.fsm_name(self.fsm) if self.fsm is not None else "unknown"
        self._remember(f"/fsm {target}", f"[FSM {now}]")

    # -- mode and settings ----------------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Change what may run and which FSMs are allowed, after a y/N. Never changes the FSM: refused
        while the robot is in one the new mode blocks."""
        if mode not in MODES:
            print(f"mode is standing or gantry, not {mode!r}")
            return
        if mode == self.mode:
            print(f"already in {mode} mode")
            return
        self._refresh_fsm()
        if self.fsm is None or not fsm_allowed(mode, self.fsm):
            here = sdk.fsm_name(self.fsm) if self.fsm is not None else "unknown"
            print(f"refused: the robot is in FSM {here}, which {mode} mode blocks ({mode} allows {allowed_text(mode)}); "
                  "/fsm to one it allows first (or the remote)")
            return
        question = ("Is the robot standing on its own feet, area clear, spotter and remote ready?" if mode == "standing"
                    else "Is the robot hanging in the gantry (feet just touching or off the floor)?")
        if not self.ui.confirm(question):
            print("mode unchanged")
            return
        self.mode = mode
        print(f"mode: {mode}, FSMs allowed: {allowed_text(mode)}" + (
            " (we command the legs: built-in moves)" if mode == "standing" else
            " (the legs are never commanded: no built-ins or base motion)"))
        self._remember(f"/mode {mode}", f"[now in {mode} mode]")

    def set_mic(self, on: bool) -> None:
        if self.mic is None:
            print("no microphone: the robot's speech recognition did not subscribe")
            return
        self.mic_on = on
        if not on:
            self._seq = self.mic.seq                # nothing said so far is answered
        print(f"mic {'on: what the robot hears is handled like chat' if on else 'off: speech is ignored'}")

    def set_speaker(self, on: bool) -> None:
        self.speaker_on = on
        print("speaker on: replies are said out loud" if on else "speaker off: replies only here (tasks' say too)")

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
        self._source = {"chat": "typed", "heard": "voice"}.get(n, "command")
        if n == "chat":
            self.chat(cmd.text, "typed")
        elif n == "heard":
            print(f"\nheard: {cmd.text}")
            self.chat(cmd.text, "heard")
        elif n == "routine":
            self.run_skill(n, a[0] if a else "")
        elif n == "task":
            name = a[0] if a else ""
            slots = self.tasks[name].fill if name in self.tasks else {}
            self.run_skill(n, name, fill={k: cmd.text for k in slots} if slots else None)
        elif n == "move":
            self.run_skill("move", " ".join(a))
        elif n == "say":
            self.speak(cmd.text)
        elif n == "mode":
            self.set_mode(a[0] if a else "")
        elif n == "fsm":
            self.set_fsm(int(a[0]))
        elif n in ("mic", "speaker", "camera", "lidar"):
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

    def _distill(self) -> None:
        """At the end, once the robot is at neutral and the lidar saved: what this session taught goes
        into MEMORY.md. Esc or Ctrl-C skips it."""
        if self.knowledge is None or self.model is None or not self.knowledge.session_lines:
            return
        print(f"memory: distilling this session (at most {limits.get('memory_distill_timeout_s'):g} s; Esc skips)")
        try:
            self._busy("saving memory (Esc skips)", lambda: self.knowledge.distill(self.model))
        except KeyboardInterrupt:
            print("memory: skipped")

    def shutdown(self) -> None:
        """Lidar mapping ended and saved, the camera closed (the FSM left as it is), this session distilled
        into memory, the counts printed."""
        from g1.envs.base import shield_sigint
        with shield_sigint("closing the session; this finishes on its own"):
            if self.lidar is not None:
                self.lidar.stop()
                self.lidar = None
            self._stop_camera()
        self._refresh_fsm()
        print(f"left in FSM {sdk.fsm_name(self.fsm) if self.fsm is not None else 'unknown'} (quitting changes no FSM)")
        self._distill()
        print("session: " + ", ".join(f"{k} {v}" for k, v in self.counts.items())
              + (f"; microphone: {self.mic.counts}" if self.mic is not None else ""))


def robot_runner(*, ask: bool = False) -> Callable[..., dict]:
    """Runs a routine or task on the robot through ``cli.run_once`` in the session's mode, with its
    microphone and camera, kept unreviewed (no questions mid-session)."""
    def run(kind: str, name: str, mode: str, settings: dict, fill: Optional[dict] = None) -> dict:
        from g1 import routines
        from g1.agent.task import find, load_task
        from g1.cli import run_once
        spec = routines.run_spec(name) if kind == "routine" else load_task(find(name), fill=fill)
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
