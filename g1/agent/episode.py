"""Run records: what a run saves, each fact once.

runs/<YYYYmmdd-HHMMSS>_<env>_<instruction-slug>_<outcome>/
    summary.json      which run this was and what happened: scan this first
    config.json       how it was set up: the task, env, mode, model, tools, system prompt,
                      output schema, every limit, the code snapshot (written once, at the start)
    step_0001.json    one query to the model: the frame, the query as sent, the response,
                      the action it chose and that action's measured result
    step_0001.png     the frame sent, LOSSLESS (the exact RGB matrix reloads)
    trace.npz         every tick, commanded and measured (g1/envs/trace.py, written by the env)

Nothing else is saved: a new file is a deliberate change (tests/agent/test_agent.py checks the set).
A step file's ``tick_start``/``tick_end`` (and its action's) are rows of trace.npz, one row per
control tick, so a frame and its decision line up with the measured joints; ``t_start``/``t_end``
are the same instants in run seconds (trace.npz ``t`` is the link's clock). ``load_episode`` gives the steps back with
the RGB array attached.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from g1.core import images

OUTCOME_OF = {              # runtime status -> outcome suffix of the run directory (theirs)
    "completed": "success", "failed": "failed", "give_up": "give_up", "budget_exhausted": "give_up",
    "interrupted": "interrupted", "unreviewed": "unreviewed",
}


def save_png(path: Path, image_rgb: np.ndarray) -> None:
    """Lossless: the exact RGB matrix reads back (``images.py``)."""
    images.write_png(path, image_rgb)


def load_png(path: Path) -> np.ndarray:
    return images.read_rgb(path)


def fresh_dir(parent: Path, stamp: str, rest: str) -> Path:
    """``parent/<stamp>_<rest>``, made now; ``<stamp>-2_<rest>`` and on when runs of one task start in
    the same second (g1 bench runs several at once)."""
    parent.mkdir(parents=True, exist_ok=True)
    k = 1
    while True:
        d = parent / f"{stamp}{'' if k == 1 else f'-{k}'}_{rest}"
        try:
            d.mkdir(exist_ok=False)
            return d
        except FileExistsError:
            k += 1


def slug(text: str, n: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "run"


def new_summary(run: str, *, task: str, instruction: str, env: str, mode: Optional[str], model: Optional[str],
                max_decisions: Optional[int] = None) -> dict[str, Any]:
    """The summary.json fields, in the order they read (a tools run fills fewer)."""
    return {"run": run, "task": task, "instruction": instruction, "env": env, "mode": mode, "model": model,
            "started": datetime.now().isoformat(timespec="seconds"), "ended": None, "duration_s": None,
            "outcome": None, "outcome_source": None, "model_outcome": None, "end_reason": None, "error": None,
            "decisions": {"used": 0, "max": max_decisions}, "steps": 0, "tools_used": [], "closing": None,
            "usage": {"calls": 0, "failed_calls": 0, "tokens": {}, "model_s": 0.0},
            "health": None, "interrupt": None, "sdk_outside_steps": []}


class EpisodeWriter:
    """The run recorder. ``config.json`` at the start; one ``step_NNNN.json`` per model
    query, written when the query is sent and rewritten as its response, action and
    result arrive (so a crash keeps everything up to that point); ``summary.json``
    rewritten after every step and settled by ``close``, which names the outcome like
    GPT-Policy's ``RunRecorder.close`` and renames the directory ``<name>_<outcome>``.
    PNG encoding (40-80 ms at 720p) runs on a writer thread so the control loop never
    pays for it."""

    def __init__(self, root: Path | str, *, env: str, instruction: str, model: str, config: dict,
                 task: str = "", mode: Optional[str] = None, max_decisions: Optional[int] = None,
                 threaded: bool = True) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = fresh_dir(Path(root), stamp, f"{env}_{slug(instruction)}")
        self.started_at = time.time()
        self.summary = new_summary(self.dir.name, task=task, instruction=instruction, env=env, mode=mode,
                                   model=model, max_decisions=max_decisions)
        self.closed = False
        self.threaded = threaded
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=False) if threaded else None
        if self._thread is not None:
            self._thread.start()
        _save(self.dir / "config.json", config)
        self._submit(("summary",))

    # -- what the agent reports ----------------------------------------------------
    def step(self, rec: dict, image: Optional[np.ndarray] = None) -> None:
        """Write (or rewrite) one step file; ``image`` once, with the first write."""
        self.summary["steps"] = max(self.summary["steps"], rec["step"])
        self._submit(("step", json.loads(json.dumps(rec, default=_plain)),
                      None if image is None else np.array(image, copy=True)))

    def model_call(self, call: dict) -> None:
        """One model call's totals into the summary (the call itself is in its step file)."""
        u = self.summary["usage"]
        u["calls"] += 1
        u["failed_calls"] += call.get("status") != "completed"
        u["model_s"] = round(u["model_s"] + float(call.get("elapsed_s") or 0.0), 6)
        usage = call.get("usage") or {}
        for k, v in usage.items():
            if isinstance(v, int) and not isinstance(v, bool):
                u["tokens"][k] = u["tokens"].get(k, 0) + v
        # inside the totals: the reasoning share of completion_tokens, the cached share of prompt_tokens
        for k, part in (("reasoning_tokens", "completion_tokens_details"), ("cached_tokens", "prompt_tokens_details")):
            v = (usage.get(part) or {}).get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                u["tokens"][k] = u["tokens"].get(k, 0) + v
        cost = usage.get("cost")                  # what the provider charged, when it says (OpenRouter, USD)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            u["cost_usd"] = round(u.get("cost_usd", 0.0) + float(cost), 8)

    def update(self, **values) -> None:
        self.summary.update(values)
        self._submit(("summary",))

    def close(self, status: str = "completed", human: Optional[str] = None, error: Optional[str] = None,
              end_reason: Optional[str] = None, join: float = 10.0) -> Path:
        """Stop the writer, settle the outcome and rename the directory.

        ``status`` is the runtime status (completed | give_up | budget_exhausted | failed |
        interrupted). A human verdict wins; a model conclusion without one is ``unreviewed``;
        otherwise the runtime status stands."""
        if self.closed:
            return self.dir
        self.closed = True
        model = {"completed": "success", "give_up": "give_up"}.get(status)
        if human is not None:
            outcome, source = human, "human"
        elif model is not None or status == "budget_exhausted":
            outcome, source = "unreviewed", "unreviewed"
        else:
            outcome, source = OUTCOME_OF.get(status, status), "runtime"
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=join)
            self._thread = None
        dest = self.dir.with_name(f"{self.dir.name}_{outcome}")
        if dest.exists():
            raise FileExistsError(f"run directory already exists: {dest}")
        self.dir.rename(dest)
        self.dir = dest
        self.summary.update(run=dest.name, outcome=outcome, outcome_source=source, model_outcome=model,
                            end_reason=end_reason or self.summary["end_reason"], error=error,
                            ended=datetime.now().isoformat(timespec="seconds"),
                            duration_s=round(time.time() - self.started_at, 3))
        self._write_summary()
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
        if job[0] == "step":
            d, image = job[1], job[2]
            name = f"step_{d['step']:04d}"
            if image is not None:
                save_png(self.dir / f"{name}.png", image)
            _save(self.dir / f"{name}.json", d)
        self._write_summary()

    def _write_summary(self) -> None:
        _save(self.dir / "summary.json", self.summary)


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


def save_json(path: Path, value: Any) -> None:
    """Atomic JSON write (the run's summary.json/config.json outside the writer: tools runs, verdicts)."""
    _save(path, value)


def load_episode(directory: Path | str, images: bool = True) -> tuple[dict, list[dict]]:
    """``(summary, steps)``; each step dict gains ``image`` (the RGB array) when asked and present."""
    root = Path(directory)
    summary = json.loads((root / "summary.json").read_text())
    steps = []
    for p in sorted(root.glob("step_*.json")):
        d = json.loads(p.read_text())
        png = p.with_suffix(".png")
        d["image"] = load_png(png) if images and png.exists() else None
        steps.append(d)
    return summary, steps
