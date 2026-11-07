"""The robot's knowledge base, as an AI agent keeps one: who it is, what it remembers, what happened.

    knowledge/
      SOUL.md            who the robot is: identity, personality, how it talks, where it is, what it
                         declines. Written by people, never by the robot. Copied from SOUL.example.md
                         (tracked) on the first g1 start.
      MEMORY.md          what it has learned: dated bullets the robot writes itself (the model's
                         ``remember`` tool, and the distill at the end of a session); people may edit it.
      sessions/DATE.md   that day's log, one line per exchange (what was heard or typed, what it did).

All three are per robot and git-ignored. Only ``g1 start``'s session reads them (``g1/session.py``):
the system prompt is the session's rules (in code, never editable away) + SOUL.md + MEMORY.md, read
afresh each turn, so a hand edit or the robot's own write counts at once. Tasks never see them.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional

from g1.core import limits

ROOT = limits.ROOT / "knowledge"
MEMORY_TITLE = "# Memory\n\nWhat I have learned, one dated bullet each (newest last).\n"

# The distill's one tool: the model returns what to add to and remove from MEMORY.md.
UPDATE_MEMORY = {"type": "function", "function": {
    "name": "update_memory",
    "description": "Change your long-term memory after a conversation: facts to add, bullets to remove.",
    "parameters": {"type": "object", "properties": {
        "add": {"type": "array", "items": {"type": "string"},
                "description": "new facts worth keeping, one short sentence each (not already in memory)"},
        "remove": {"type": "array", "items": {"type": "string"},
                   "description": "existing bullets to remove (wrong, outdated, or merged into an added one), copied exactly"}},
        "required": ["add", "remove"]}}}

DISTILL = """You keep the long-term memory of a Unitree G1 humanoid robot. Below are its memory (MEMORY.md) and the log of
the conversation it just had. Call update_memory once:
- add: facts from the conversation worth knowing next time (people's names and roles they told you, places, events,
  preferences, things you were taught), one short sentence each. Nothing already in memory, nothing trivial, and
  never phone numbers, addresses, passwords or private details about a person.
- remove: bullets now wrong or outdated, or that your added facts replace, copied exactly as they appear.
{condense}Both lists may be empty."""


def bullet_text(line: str) -> str:
    """``"- fact (2026-11-06)"`` -> ``"fact (2026-11-06)"``; not a bullet -> ``""``."""
    line = line.strip()
    return line[2:].strip() if line.startswith("- ") else ""


class Knowledge:
    """One robot's knowledge folder. ``now`` gives the local time (tests pass their own)."""

    def __init__(self, root: Path = ROOT, *, now: Callable[[], dt.datetime] = dt.datetime.now) -> None:
        self.root, self.now = Path(root), now
        self.soul_path, self.memory_path = self.root / "SOUL.md", self.root / "MEMORY.md"
        self.sessions = self.root / "sessions"
        self._lock = threading.Lock()
        self.started = self.now()
        self.session_lines: list[str] = []           # this session's log lines (the distill's input)
        self.saved = 0                               # session_lines[:saved] already distilled (/save)

    # -- the files ----------------------------------------------------------------------------------
    def ensure(self) -> list[str]:
        """Create what is missing: SOUL.md from the example, an empty MEMORY.md. Returns notes."""
        notes = []
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.soul_path.exists():
            example = self.root / "SOUL.example.md"
            if not example.exists():
                example = ROOT / "SOUL.example.md"
            if example.exists():
                shutil.copyfile(example, self.soul_path)
                notes.append(f"knowledge: created {self.soul_path} from SOUL.example.md; edit it to say who the robot is")
        if not self.memory_path.exists():
            self.memory_path.write_text(MEMORY_TITLE, encoding="utf-8")
            notes.append(f"knowledge: created {self.memory_path} (the robot's memory)")
        return notes

    def soul(self) -> str:
        """SOUL.md, at most ``soul_max_chars`` (the start: it is read top down)."""
        text = _read(self.soul_path).strip()
        cap = int(limits.get("soul_max_chars"))
        return text if len(text) <= cap else text[:cap] + "\n[SOUL.md is longer; the rest was left out]"

    def memory_bullets(self) -> list[str]:
        return [b for b in (bullet_text(line) for line in _read(self.memory_path).splitlines()) if b]

    def memory(self) -> str:
        """MEMORY.md for the prompt: the newest bullets within ``memory_max_chars``."""
        bullets, cap = self.memory_bullets(), int(limits.get("memory_max_chars"))
        kept, size = [], 0
        for b in reversed(bullets):
            size += len(b) + 3
            if size > cap:
                break
            kept.append(b)
        kept.reverse()
        text = "\n".join(f"- {b}" for b in kept)
        if len(kept) < len(bullets):
            text = f"[{len(bullets) - len(kept)} older memories left out: memory is over its size]\n" + text
        return text

    def over(self) -> bool:
        return len(_read(self.memory_path)) > limits.get("memory_max_chars")

    # -- writing memory -----------------------------------------------------------------------------
    def remember(self, fact: str) -> Optional[str]:
        """Append ``- fact (date)``; returns the bullet, or None for an empty or repeated fact."""
        fact = " ".join(fact.split()).rstrip()
        if not fact or any(b.split(" (")[0].lower() == fact.lower() or b.lower() == fact.lower()
                           for b in self.memory_bullets()):
            return None
        line = f"{fact} ({self.now():%Y-%m-%d})"
        with self._lock:
            text = _read(self.memory_path) or MEMORY_TITLE
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_path.write_text(text.rstrip("\n") + f"\n- {line}\n", encoding="utf-8")
        return line

    def forget(self, bullet: str) -> bool:
        """Remove the bullet whose text is ``bullet`` (with or without its date); False if none."""
        want = " ".join(bullet.removeprefix("- ").split()).lower()
        if not want:
            return False
        with self._lock:
            lines = _read(self.memory_path).splitlines()
            keep = [line for line in lines
                    if not (bullet_text(line) and (bullet_text(line).lower() == want
                                                   or bullet_text(line).split(" (")[0].lower() == want))]
            if len(keep) == len(lines):
                return False
            self.memory_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
        return True

    # -- the daily log --------------------------------------------------------------------------------
    def log(self, source: str, heard: str, what: str) -> None:
        """One exchange: ``- 14:03 [voice] can you dance -> move 604 mechanical dance``."""
        now = self.now()
        line = f"- {now:%H:%M} [{source}] {' '.join(heard.split())} -> {' '.join(what.split())}"
        with self._lock:
            self.sessions.mkdir(parents=True, exist_ok=True)
            path = self.sessions / f"{now:%Y-%m-%d}.md"
            new = not path.exists()
            with path.open("a", encoding="utf-8") as f:
                if new:
                    f.write(f"# {now:%Y-%m-%d}\n\n")
                if not self.session_lines:
                    f.write(f"\n## session {self.started:%H:%M}\n")
                f.write(line + "\n")
            self.session_lines.append(line)

    # -- the distill ----------------------------------------------------------------------------------
    def unsaved(self) -> list[str]:
        """This session's log lines not yet distilled into MEMORY.md (since the last /save)."""
        return self.session_lines[self.saved:]

    def distill_messages(self) -> list[dict]:
        log = "\n".join(self.unsaved())
        cap = int(limits.get("memory_distill_log_chars"))
        if len(log) > cap:
            log = "[earlier lines left out]\n" + log[-cap:]
        condense = ("- memory is over its size: also merge and remove bullets so it gets shorter.\n"
                    if self.over() else "")
        memory = "\n".join(f"- {b}" for b in self.memory_bullets()) or "(empty)"
        return [{"role": "system", "content": DISTILL.format(condense=condense)},
                {"role": "user", "content": f"MEMORY.md:\n{memory}\n\nThis conversation's log:\n{log}"}]

    def apply(self, add: list, remove: list) -> tuple[int, int]:
        removed = sum(1 for b in remove or [] if isinstance(b, str) and self.forget(b))
        added = sum(1 for f in add or [] if isinstance(f, str) and self.remember(f))
        return added, removed

    def distill(self, model, *, timeout: Optional[float] = None) -> Optional[tuple[int, int]]:
        """Ask the model once what was said since the last save taught, and apply it (``/save``). None when
        skipped (nothing new, no model, no reply in time, a bad reply). The call runs on a daemon thread so
        the wait is bounded and a KeyboardInterrupt (Esc, Ctrl-C) ends it at once."""
        if model is None or not self.unsaved():
            return None
        upto = len(self.session_lines)
        timeout = limits.get("memory_distill_timeout_s") if timeout is None else timeout
        box: dict = {}

        def ask():
            try:
                call = model.complete(self.distill_messages(), tools=[UPDATE_MEMORY])
                box["args"] = json.loads(call.arguments or "{}") if call.name == "update_memory" else None
            except Exception as e:                   # the network, the provider, a bad reply
                box["error"] = e
        t = threading.Thread(target=ask, name="memory-distill", daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            print(f"memory: no reply within {timeout:g} s; nothing distilled")
            return None
        if "error" in box or not box.get("args"):
            print(f"memory: nothing distilled ({box.get('error') or 'no update_memory call'})")
            return None
        added, removed = self.apply(box["args"].get("add") or [], box["args"].get("remove") or [])
        self.saved = upto
        print(f"memory: +{added} -{removed} ({self.memory_path})")
        return added, removed


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
