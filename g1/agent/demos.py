"""In-context demos: successful past runs of a task, shown to the model when the task runs again.

    tasks/<name>/demos/<run>/
        demo.json       the steps as the model lived them: each tool call (with its note) and the
                        result the model saw next; which env, who judged it a success
        step_NNNN.png   the camera frames those steps decided on

A person promotes a run (``promote``): only a success scored by the twin or labelled by a human
qualifies. A demo carries the model's own words and what it was shown, never host truth: the
measured base pose and its error, the twin's evaluation and criteria, the task's success, and the
done summary and hindsight are not copied. ``load_demos`` picks them for a run (same env only: the
twin's room looks nothing like the real one), and ``render`` turns them into the message blocks the
decider puts at the head of every request (``g1/agent/decider.py``).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

ELIGIBLE = ("twin", "human")               # who may have judged a run a success
DROPPED_ARGS = ("summary", "hindsight")    # the done tool's closing words: written for people
# Never in a demo, at any depth: the host's measurements and the twin's truth.
HOST_ONLY = frozenset({"base_measured_pose", "base_measured_source", "base_error", "base_target_pose", "base_pose",
                       "motion_progress", "twin", "evaluation", "criteria", "measured", "success", "health"})
RESULT_CHARS = 240                         # a demo's result line, cut


class DemoError(ValueError):
    """A run that cannot become a demo."""


def scrub(value: Any) -> Any:
    """``value`` without any host-only key, at any depth."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in HOST_ONLY}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def _observation(step: dict) -> dict:
    """The observation the model was sent at this step: the last user message's JSON block."""
    for msg in reversed(step.get("query") or []):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            text = block.get("text", "") if block.get("type") == "text" else ""
            if text.startswith("{"):
                try:
                    return json.loads(text)
                except ValueError:
                    continue
        return {}
    return {}


def promote(run_dir: str | Path) -> Path:
    """Make a successful run a demo of its task: ``tasks/<task>/demos/<run>/``. Refuses a run that
    is not a success judged by the twin or a person, or that is not under its task's ``runs/``."""
    run = Path(run_dir)
    try:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DemoError(f"{run}: no readable summary.json ({e})") from None
    if summary.get("outcome") != "success" or summary.get("outcome_source") not in ELIGIBLE:
        raise DemoError(f"{run.name}: a demo is a success judged by the twin or a person; this run is "
                        f"{summary.get('outcome')} ({summary.get('outcome_source')})")
    if run.parent.name != "runs" or run.parent.parent.name != summary.get("task"):
        raise DemoError(f"{run}: not under tasks/{summary.get('task')}/runs/")
    if not summary.get("instruction"):
        raise DemoError(f"{run.name}: not a task run (no instruction)")
    files = sorted(run.glob("step_*.json"))
    raw = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    steps, images = [], []
    for i, step in enumerate(raw):
        decision = (step.get("response") or {}).get("decision")
        if not decision or not decision.get("name"):
            continue                                        # a reply that chose nothing: nothing to imitate
        args = {k: v for k, v in (decision.get("arguments") or {}).items() if k not in DROPPED_ARGS}
        later = _observation(raw[i + 1]) if i + 1 < len(raw) else {}
        seen = later.get("previous_result")
        if seen is None and (step.get("result") or {}).get("status") in ("done", "give_up"):
            seen = {"status": step["result"]["status"]}
        image = (step.get("frame") or {}).get("image")
        if image and not (run / image).is_file():
            image = None
        steps.append({"turn": (step.get("decision") if step.get("decision") is not None else i),
                      "tool": decision["name"], "arguments": scrub(args), "result": scrub(seen), "image": image})
        if image:
            images.append(image)
    if not steps:
        raise DemoError(f"{run.name}: no decisions to show")
    dest = run.parent.parent / "demos" / run.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in images:
        shutil.copy2(run / name, dest / name)
    demo = {"run": run.name, "task": summary["task"], "instruction": summary["instruction"], "env": summary.get("env"),
            "outcome_source": summary["outcome_source"], "decisions": len(steps), "steps": steps}
    (dest / "demo.json").write_text(json.dumps(demo, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def load_demos(task: dict, env_name: str, k: int) -> list[dict]:
    """Up to ``k`` demos of ``task`` recorded in ``env_name`` (sim or robot), the fewest decisions
    first, then the newest. Each comes with ``dir`` (where its images are)."""
    if k <= 0:
        return []
    root = Path(task["dir"]) / "demos"
    found = []
    for path in sorted(root.glob("*/demo.json")) if root.is_dir() else []:
        try:
            demo = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"warning: demo {path.parent.name} unreadable, skipped")
            continue
        if demo.get("env") == env_name:
            found.append({**demo, "dir": path.parent})
    found.sort(key=lambda d: d["run"], reverse=True)          # the newest first, then
    found.sort(key=lambda d: d["decisions"])                   # the fewest decisions first (stable)
    return found[:k]


def _line(step: dict) -> str:
    args = {k: v for k, v in step["arguments"].items() if k != "note"}
    note = step["arguments"].get("note", "")
    result = json.dumps(step["result"], separators=(",", ":"), ensure_ascii=False) if step["result"] is not None else "-"
    if len(result) > RESULT_CHARS:
        result = result[:RESULT_CHARS] + "…"
    return (f"turn {step['turn']}: {step['tool']}({json.dumps(args, separators=(',', ':'), ensure_ascii=False)})"
            f"{' — ' + note if note else ''}; result: {result}")


def _spread(n: int, k: int) -> list[int]:
    """``k`` indices spread evenly over ``range(n)``, the last always included."""
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)} if k > 1 else {n - 1})


def render(demos: list[dict], *, images_max: int, image_width: int,
           read_image=None, image_part=None) -> tuple[list[dict], str]:
    """The demos as message content blocks (text, and up to ``images_max`` images a demo), and the
    same as plain text for config.json. ``read_image`` and ``image_part`` are the image I/O (Pillow
    and the vlm JPEG encoder by default)."""
    if not demos:
        return [], ""
    if read_image is None:
        from g1.core.images import read_rgb as read_image
    if image_part is None:
        from g1.vlm import image_part
    header = ("Earlier successful runs of this task, as you lived them (examples: things may have moved since, "
              "so look before you act):")
    blocks: list[dict] = [{"type": "text", "text": header}]
    text = [header]
    for n, demo in enumerate(demos, 1):
        title = f"Example {n}: {demo['decisions']} decisions, a success."     # where it ran is never said
        blocks.append({"type": "text", "text": title})
        text.append(title)
        with_images = [i for i, s in enumerate(demo["steps"]) if s.get("image")]
        shown = {with_images[i] for i in _spread(len(with_images), images_max)}
        for i, step in enumerate(demo["steps"]):
            if i in shown:
                label = f"Camera image you decided on at turn {step['turn']}:"
                blocks.append({"type": "text", "text": label})
                blocks.append(image_part(read_image(Path(demo["dir"]) / step["image"]), image_width))
                text.append(f"{label} [{step['image']}]")
            line = _line(step)
            blocks.append({"type": "text", "text": line})
            text.append(line)
    closing = "Now the current run:"
    blocks.append({"type": "text", "text": closing})
    text.append(closing)
    return blocks, "\n".join(text)
