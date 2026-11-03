"""g1 talk: a conversation demo. The robot stands ready, listens, and answers what it is asked,
or plays one of Unitree's stored moves when asked for one. Nothing else.

    g1 talk                     the robot only (UNITREE_IFACE), voice interaction on in the Unitree app

Each turn: what was said (the robot's own speech recognition on ``rt/audio_msg``, through
``g1/microphone.py``) goes to the model (``configs/model.json``), which answers with exactly
one call: ``say`` (spoken with the robot's text-to-speech, ``TtsMaker``), ``move`` (a stored
move from the allow-list below, played by ``g1 moves``'s player) or ``ignore`` (chatter not
meant for it). It never runs a task, never takes the arms (no arm_sdk), never walks and never
switches the FSM after the start: its only effects are speech and the allow-listed moves.

At the start it brings the robot to standing (``g1 moves fsm walk``'s order) if it is not,
and asks once before any move may play. Ctrl-C ends it; during a move, the move is stopped first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Callable, Optional

from g1.core import limits
from g1.envs import sdk
from g1.envs.moves import FSM_TARGETS, PLAY_FSMS, Moves, connect, gloss

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
- ignore: when what you heard is not meant for you (side conversations, noise, half a word).
You can only talk and play those moves. You cannot walk, see, pick things up or change anything else; if asked,
say so kindly. Keep everything school-appropriate and decline anything unsafe or unkind politely.
You are about 1.3 m tall and made by Unitree Robotics; you run on your own onboard computer."""


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


def tools(menu: dict[int, dict]) -> list[dict]:
    listing = "; ".join(f"{mid} {m['name']} ({m['duration']:.0f} s)" for mid, m in menu.items())
    return [
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
            "name": "ignore", "description": "Say nothing: what was heard was not meant for you.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "description": "a few words, for the log"}},
                "required": ["reason"]}}},
    ]


class Talk:
    """The conversation loop over a Microphone, the robot's AudioClient, a ``Moves`` player and a
    model client (``complete(messages, tools=...) -> ToolCall``). The clock is injectable for tests."""

    def __init__(self, mic, audio, moves: Moves, model, menu: dict[int, dict], *, verbose: bool = False,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.monotonic):
        self.mic, self.audio, self.moves, self.model, self.menu = mic, audio, moves, model, menu
        self.verbose, self.sleep, self.now = verbose, sleep, now
        self.tools = tools(menu)
        self.history: list[dict] = []           # user / assistant text turns, newest last
        self.counts = {"heard": 0, "said": 0, "moves": 0, "ignored": 0, "errors": 0}

    # -- hearing ---------------------------------------------------------------------------------
    def listen(self, seq: int) -> tuple[str, int]:
        """Block until someone has said something and then been quiet for ``talk_quiet_s``;
        everything said since ``seq``, joined, and the new seq."""
        while True:
            heard = self.mic.since(seq)
            if heard and self.now() - heard[-1].stamp >= limits.get("talk_quiet_s"):
                return " ".join(h.text for h in heard), heard[-1].seq
            self.sleep(limits.get("talk_poll_s"))

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

    def answer(self, heard: str) -> Optional[int]:
        """One turn: ask the model about ``heard`` and do what it chose. A Ctrl-C during a move
        returns 130 (the move was stopped); otherwise None."""
        messages = ([{"role": "system", "content": SYSTEM}]
                    + self.history[-2 * int(limits.get("talk_history_turns")):]
                    + [{"role": "user", "content": heard}])
        t0 = self.now()
        try:
            call = self.model.complete(messages, tools=self.tools)
            args = json.loads(call.arguments or "{}")
        except Exception as e:                  # the network, the provider, a bad reply: say so and go on
            self.counts["errors"] += 1
            print(f"model error after {self.now() - t0:.1f} s: {type(e).__name__}: {e}")
            self.speak("Sorry, I could not think of an answer just now.")
            return None
        print(f"  ({call.name} after {self.now() - t0:.1f} s)")
        if call.name == "move" and int(args.get("move_id", -1)) in self.menu:
            mid = int(args["move_id"])
            self.history += [{"role": "user", "content": heard},
                             {"role": "assistant", "content": f"[played move {mid}: {self.menu[mid]['name']}]"}]
            if args.get("say"):
                self.speak(str(args["say"]))
            self.counts["moves"] += 1
            inv_move = {"kind": "mimic", "id": mid, "label": self.menu[mid]["name"],
                        "duration": self.menu[mid]["duration"]}
            code = self.moves.play([inv_move])
            return 130 if code == 130 else None
        if call.name == "say" and str(args.get("text", "")).strip():
            self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": args["text"]}]
            self.speak(str(args["text"]))
            return None
        self.counts["ignored"] += 1
        print(f"  (ignored: {args.get('reason') or call.name})")
        return None

    def run(self) -> int:
        print(f"listening (say something to the robot; Ctrl-C ends). {len(self.menu)} moves allowed.")
        seq = self.mic.seq
        try:
            while True:
                heard, seq = self.listen(seq)
                self.counts["heard"] += 1
                print(f"\nheard: {heard}")
                if self.answer(heard) == 130:
                    return 130
                seq = self.mic.seq                # nothing said while it spoke or moved is answered
        except KeyboardInterrupt:
            print("\nending talk")
            return 0
        finally:
            print("talk: " + ", ".join(f"{k} {v}" for k, v in self.counts.items())
                  + f"; microphone: {self.mic.counts}")


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
    moves, link = connect(iface, verbose)
    audio = link.audio()
    audio.SetTimeout(limits.get("rpc_timeout_s"))
    audio.Init()
    mic = subscribe(link, time.monotonic)
    if mic is None:
        raise SystemExit("no microphone: cannot listen")
    menu = talk_menu(moves.inventory())
    if not menu:
        raise SystemExit("the robot lists none of the allowed moves (5xx, 6xx); g1 moves list")
    print(f"model: {model.model} on {model.provider}")
    fsm = moves.fsm()
    if fsm not in PLAY_FSMS:
        print(f"the robot is in FSM {sdk.fsm_name(fsm)}; standing it up for talk mode")
        moves.switch(FSM_TARGETS["walk"])
    else:
        print(f"ready: FSM {sdk.fsm_name(fsm)}")
    if not moves.confirm("talk mode plays moves when asked, without asking again: area clear, spotter and remote ready?"):
        raise SystemExit("not started")
    moves.confirm = lambda question: True
    return Talk(mic, audio, moves, model, menu, verbose=verbose).run()
