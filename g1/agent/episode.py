"""Episode records: what is read and saved at every decision step.

runs/<YYYYmmdd-HHMMSS>_<env>_<instruction-slug>_<outcome>/
    episode.json        the run's one meta file (see EpisodeWriter)
    step_0001.json      the record below
    step_0001.png       the camera frame, LOSSLESS (the exact RGB matrix reloads)
    events.jsonl, protocol.json, usage.jsonl, replies.jsonl
                        the GPT-Policy-style run trace (see EpisodeWriter)
    trace.npz           every tick, commanded and measured (g1/envs/trace.py, written by the env)

Nothing else is saved: a new file is a deliberate change (tests/agent/test_agent.py checks the set).

A step record holds the joint angles at start and end, the frame it decided
on, the model's raw reply and parsed decision, the tool that ran and how it
ended, and the base pose before and after. ``load_episode`` gives the records
back with the RGB array attached — the dataset a learned controller could be trained on.
"""
from __future__ import annotations

import dataclasses
import json
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from g1.core import images

OUTCOMES = ("completed", "running", "done", "give_up", "checked", "rejected", "no_frame")


@dataclass
class StepRecord:
    step: int
    t: dict                     # run_start/end, wall_start/end, clock_start/end, think_wall
    q_start: list
    q_end: list
    cmd_start: list
    cmd_end: list
    frame: dict                 # seq, stamp, age, image (relative path), shape
    base_pose: dict             # cmd_start/end, env_start/end (or None)
    decision: dict              # name, arguments (incl. note), raw, latency, notes, model, attempt
    tool: dict                  # name, args, duration, chunk, chunks, needs_base
    outcome: dict               # status, duration
    image: Optional[np.ndarray] = field(default=None, repr=False, compare=False)   # filled by the loader

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("image", None)
        return d

    @classmethod
    def from_json(cls, d: dict, root: Optional[Path] = None) -> "StepRecord":
        rec = cls(**{k: v for k, v in d.items() if k != "image"})
        if root is not None and rec.frame.get("image"):
            rec.image = load_png(root / rec.frame["image"])
        return rec


def save_png(path: Path, image_rgb: np.ndarray) -> None:
    """Lossless: the exact RGB matrix reads back (``images.py``)."""
    images.write_png(path, image_rgb)


def load_png(path: Path) -> np.ndarray:
    return images.read_rgb(path)


def slug(text: str, n: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "run"


def _git_head() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=2, cwd=Path(__file__).resolve().parents[2]).stdout.strip() or None
    except Exception:
        return None


OUTCOME_OF = {              # runtime status -> outcome suffix of the run directory (theirs)
    "completed": "success", "failed": "failed", "give_up": "give_up", "budget_exhausted": "give_up",
    "interrupted": "interrupted", "unreviewed": "unreviewed",
}


class EpisodeWriter:
    """The run recorder: step records (this repo's training data) plus the
    GPT-Policy file set —

        episode.json      run metadata, the result and verdict, state, error, elapsed_s and
                          the usage totals; rewritten after every step and on close
        events.jsonl      append-only: every observation, decision, timing, result,
                          error, retry, terminal, verdict, with wall-clock at_s (the
                          conversation is its observation/model_decision/execution_result events)
        protocol.json     the system prompt, tool schemas and output schema
        usage.jsonl       one line per model call
        replies.jsonl     one line per model call: everything the server sent back, as sent

    The measured joint state is the env's trace.npz, not repeated here. PNG
    encoding (40-80 ms at 720p) runs on a writer thread so the control loop
    never pays for it; events are small and go straight to disk in order.
    ``episode.json`` is rewritten after every step so a crash never leaves an
    unreadable run. ``close(status, human)`` decides
    the outcome like their ``RunRecorder.close`` and renames the directory
    ``<name>_<outcome>``."""

    def __init__(self, root: Path | str, *, env: str, instruction: str, model: str, tools: list[str],
                 allow_base: bool = True, threaded: bool = True, extra: Optional[dict] = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{stamp}_{env}_{slug(instruction)}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.meta: dict[str, Any] = {"instruction": instruction, "env": env, "model": model, "tools": tools,
                                     "allow_base": allow_base, "started": datetime.now().isoformat(timespec="seconds"),
                                     "ended": None, "result": None, "steps": 0, "git": _git_head(), **(extra or {})}
        self.threaded = threaded
        self.started_at = time.time()
        self.usage_calls: list[dict] = []
        self.model_status: Optional[str] = None      # completed | give_up, from the terminal event
        self.human_outcome: Optional[str] = None
        self.closed = False
        self._events = (self.dir / "events.jsonl").open("x", encoding="utf-8")
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=False) if threaded else None
        if self._thread is not None:
            self._thread.start()
        self._write_meta()
        self.event("run_started", dict(self.meta))

    # -- the event stream --------------------------------------------------------
    def event(self, kind: str, payload: Optional[dict] = None) -> dict:
        values = dict(payload or {})
        if kind == "terminal":
            self.model_status = "give_up" if values.get("name") == "give_up" else "completed"
        elif kind == "human_evaluation":
            if values.get("outcome") not in ("success", "failed"):
                raise ValueError("human outcome must be success or failed")
            self.human_outcome = values["outcome"]
        ev = {"at_s": time.time(), "event": kind, **values}
        self._events.write(json.dumps(ev, ensure_ascii=False, default=_plain) + "\n")
        self._events.flush()
        return ev

    def protocol(self, record: dict) -> None:
        """The system prompt, tool schemas and output schema, once (``protocol.json``)."""
        _save(self.dir / "protocol.json", record)

    def usage(self, call: dict) -> None:
        """One model call (their usage ledger; no pricing is shipped)."""
        entry = {**call, "call": len(self.usage_calls) + 1}
        chunks = entry.pop("chunks", None)
        if chunks is not None:
            with (self.dir / "replies.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({k: entry.get(k) for k in ("call", "step", "request_id", "status")} | {"chunks": chunks},
                                   ensure_ascii=False, default=_plain) + "\n")
        entry.setdefault("estimated_cost_usd", None)
        entry.setdefault("cost_unavailable_reason", "model_price_unknown")
        self.usage_calls.append(entry)
        with (self.dir / "usage.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=_plain) + "\n")

    def usage_summary(self) -> dict:
        calls = self.usage_calls
        tokens: dict[str, int] = {}
        missing = 0
        for c in calls:
            u = c.get("usage") or {}
            if not u:
                missing += 1
            for k, v in u.items():
                if isinstance(v, int) and not isinstance(v, bool):
                    tokens[k] = tokens.get(k, 0) + v
        return {"calls": len(calls), "failed_calls": sum(c.get("status") != "completed" for c in calls),
                "elapsed_s": round(sum(float(c.get("elapsed_s", 0.0)) for c in calls), 6),
                "tokens": tokens, "usage_missing_calls": missing,
                "estimated_cost_usd": None, "cost_status": "unknown",
                "by_model": sorted({c.get("model") for c in calls if c.get("model")})}

    # -- step records ---------------------------------------------------------------
    def write_step(self, rec: StepRecord, image: Optional[np.ndarray]) -> None:
        self.meta["steps"] = max(self.meta["steps"], rec.step)
        self._submit(("step", rec.to_json(), None if image is None else np.array(image, copy=True)))

    def finish(self, result: str, **extra) -> None:
        """The model-side result (``episode.json``); the verdict comes in ``close``."""
        self.meta.update(result=result, ended=datetime.now().isoformat(timespec="seconds"), **extra)
        self._submit(("meta",))

    def close(self, status: str = "completed", human: Optional[str] = None,
              error: Optional[str] = None, join: float = 10.0) -> Path:
        """Stop the writer, settle the outcome and rename the directory.

        ``status`` is the runtime status (completed | give_up | budget_exhausted
        | failed | interrupted). A human verdict wins; a model conclusion
        without one is ``unreviewed``; otherwise the runtime status stands."""
        if self.closed:
            return self.dir
        self.closed = True
        if human is not None and self.human_outcome is None:
            self.event("human_evaluation", {"outcome": human, "source": "terminal"})
        if self.human_outcome is not None:
            task_status = "completed" if self.human_outcome == "success" else "failed"
        elif self.model_status or status in ("completed", "give_up", "budget_exhausted"):
            task_status = "unreviewed"
        else:
            task_status = status
        outcome = OUTCOME_OF.get(task_status, task_status)
        verdict = {"task_status": task_status, "outcome": outcome,
                   "model_outcome": OUTCOME_OF.get(self.model_status or "", None),
                   "human_outcome": self.human_outcome,
                   "outcome_source": "human" if self.human_outcome else
                                     "unreviewed" if task_status == "unreviewed" else "runtime"}
        self.meta.update(verdict)
        self.meta.setdefault("ended", datetime.now().isoformat(timespec="seconds"))
        self.event("run_finished", {"status": status, **verdict, "error": error})
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=join)
            self._thread = None
        self.meta.update(state=status, error=error, elapsed_s=time.time() - self.started_at,
                         usage=self.usage_summary())
        self._write_meta()
        self._events.close()
        dest = self.dir.with_name(f"{self.dir.name}_{outcome}")
        if dest.exists():
            raise FileExistsError(f"run directory already exists: {dest}")
        self.dir.rename(dest)
        self.dir = dest
        return dest

    # -- internals ---------------------------------------------------------
    def _submit(self, job: tuple) -> None:
        if self.threaded and self._thread is not None:
            self._q.put(job)
        else:
            self._do(job)

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            self._do(job)

    def _do(self, job: tuple) -> None:
        if job[0] == "meta":
            self._write_meta()
        else:
            self._write(job[1], job[2])

    def _write(self, d: dict, image: Optional[np.ndarray]) -> None:
        name = f"step_{d['step']:04d}"
        if image is not None:
            save_png(self.dir / f"{name}.png", image)
            d["frame"]["image"] = f"{name}.png"
        (self.dir / f"{name}.json").write_text(json.dumps(d, indent=1))
        self._write_meta()

    def _write_meta(self) -> None:
        (self.dir / "episode.json").write_text(json.dumps(self.meta, indent=1))


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _save(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=_plain) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_episode(directory: Path | str, images: bool = True) -> tuple[dict, list[StepRecord]]:
    root = Path(directory)
    meta = json.loads((root / "episode.json").read_text())
    steps = [StepRecord.from_json(json.loads(p.read_text()), root if images else None)
             for p in sorted(root.glob("step_*.json"))]
    return meta, steps
