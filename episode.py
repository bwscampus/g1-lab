"""Episode records: what is read and saved at every decision step.

runs/<YYYYmmdd-HHMMSS>_<env>_<goal-slug>/
    episode.json        goal, env, model, skills, result, step count (rewritten every step)
    step_0001.json      the record below
    step_0001.png       the camera frame, LOSSLESS (the exact RGB matrix reloads)

A step record holds the joint angles at start and end, the frame it decided
on, the model's raw reply and parsed decision, the skill that ran and how it
ended, and the base pose before and after. ``load_episode`` gives the records
back with the RGB array attached — the dataset a policy can be learned from.

    python -m episode runs/<dir>      # step table + the equivalent --policy chain
"""
from __future__ import annotations

import argparse
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

OUTCOMES = ("completed", "cutoff", "done", "timeout", "no_decision", "no_frame")


@dataclass
class StepRecord:
    step: int
    t: dict                     # policy_start/end, wall_start/end, clock_start/end, think_wall
    q_start: list
    q_end: list
    cmd_start: list
    cmd_end: list
    frame: dict                 # seq, stamp, age, image (relative path), shape
    base_pose: dict             # cmd_start/end, env_start/end (or None)
    decision: dict              # scene, path_clear, found, action, args, reason, raw, latency, model, notes, retries
    skill: dict                 # name, args, max_duration, kind, needs_base
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
    """Lossless. OpenCV wants BGR, so the conversion happens here and nowhere else."""
    import cv2
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR),
                     [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"could not write {path}")


def load_png(path: Path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def slug(text: str, n: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "run"


def _git_head() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=2, cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        return None


class EpisodeWriter:
    """Writes step records; PNG encoding (40-80 ms at 720p) runs on a writer
    thread so the control loop never pays for it. ``episode.json`` is rewritten
    after every step so a crash never leaves an unreadable run."""

    def __init__(self, root: Path | str, *, env: str, goal: str, model: str, skills: list[str],
                 allow_base: bool = True, threaded: bool = True, extra: Optional[dict] = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{stamp}_{env}_{slug(goal)}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.meta: dict[str, Any] = {"goal": goal, "env": env, "model": model, "skills": skills,
                                     "allow_base": allow_base, "started": datetime.now().isoformat(timespec="seconds"),
                                     "ended": None, "result": None, "steps": 0, "git": _git_head(), **(extra or {})}
        self.threaded = threaded
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=False) if threaded else None
        if self._thread is not None:
            self._thread.start()
        self._write_meta()

    def write_step(self, rec: StepRecord, image: Optional[np.ndarray]) -> None:
        self.meta["steps"] = max(self.meta["steps"], rec.step)
        job = (rec.to_json(), None if image is None else np.array(image, copy=True))
        if self.threaded:
            self._q.put(job)
        else:
            self._write(*job)

    def finish(self, result: str, **extra) -> None:
        self.meta.update(result=result, ended=datetime.now().isoformat(timespec="seconds"), **extra)
        if self.threaded:
            self._q.put(("meta",))
        else:
            self._write_meta()

    def close(self, join: float = 10.0) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=join)
            self._thread = None
        self._write_meta()

    # -- internals ---------------------------------------------------------
    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            if job[0] == "meta":
                self._write_meta()
            else:
                self._write(*job)

    def _write(self, d: dict, image: Optional[np.ndarray]) -> None:
        name = f"step_{d['step']:04d}"
        if image is not None:
            save_png(self.dir / f"{name}.png", image)
            d["frame"]["image"] = f"{name}.png"
        (self.dir / f"{name}.json").write_text(json.dumps(d, indent=1))
        self._write_meta()

    def _write_meta(self) -> None:
        (self.dir / "episode.json").write_text(json.dumps(self.meta, indent=1))


def load_episode(directory: Path | str, images: bool = True) -> tuple[dict, list[StepRecord]]:
    root = Path(directory)
    meta = json.loads((root / "episode.json").read_text())
    steps = [StepRecord.from_json(json.loads(p.read_text()), root if images else None)
             for p in sorted(root.glob("step_*.json"))]
    return meta, steps


def chain_of(steps: list[StepRecord]) -> str:
    """The ``--policy`` chain string that replays the skills that ran."""
    items = []
    for s in steps:
        if s.outcome.get("status") not in ("completed", "cutoff"):
            continue
        args = ":".join(f"{k}={v}" for k, v in s.skill["args"].items())
        items.append(s.skill["name"] + (":" + args if args else ""))
    return ",".join(items)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m episode", description="inspect a recorded run")
    p.add_argument("dir")
    args = p.parse_args(argv)
    meta, steps = load_episode(args.dir, images=False)
    print(f"{meta['goal']!r} on {meta['env']} with {meta['model']}: {len(steps)} step(s), result {meta['result']}")
    for s in steps:
        d = s.decision
        print(f"  #{s.step:>3} {s.skill['name']}({s.skill['args']}) -> {s.outcome['status']:<11} "
              f"{d.get('latency', 0):.1f}s  {d.get('scene', '')[:60]}")
    print("replay:  --policy " + (chain_of(steps) or "<nothing completed>"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
