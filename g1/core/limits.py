"""Every tunable limit and default, from one file: ``configs/limits.json``.

Nothing else in the repo holds a limit. Each entry there has a ``value``, a
``unit``, a ``source`` (where the number came from) and a ``note`` (what it
bounds). Code reads a value with ``limits.get("name")``; a tool's parameters
refer to one with ``limit("name")`` and its prompt text with ``{limit:name}``.

    g1 limits                        # the table, guesses first
    g1 limits --source guess         # only the numbers nobody has checked
    G1_LIMITS=my_limits.json g1 ...  # another file for one run

A missing or misspelt name fails at import, naming it: a limit is never
silently defaulted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]          # the repo
PATH = ROOT / "configs" / "limits.json"
SOURCES = ("guess", "gpt-policy", "repo", "measured", "user", "robot")     # least to most trusted

# Every name the code reads. The file must define exactly these.
NAMES = (
    "control_dt_s", "record_step_s",
    "base_vx_max", "base_vy_max", "base_vyaw_max", "walk_speed", "side_speed", "turn_rate", "min_segment_s",
    "joint_margin_rad", "command_vel_max", "max_violations",
    "move_dx_max_m", "move_dy_max_m", "move_dyaw_max_deg", "look_yaw_max_deg",
    "arm_path_max_waypoints", "arm_path_seconds_min", "arm_path_seconds_max", "arm_path_seconds_default",
    "hold_seconds_max", "wave_hand_seconds", "shake_hand_seconds", "gesture_seconds_max", "gesture_ramp_s",
    "say_max_chars", "speech_chars_per_s", "say_min_s", "say_pause_max_s", "tts_speaker_id",
    "bookend_ramp_s", "return_to_stand_s", "return_ramp_s", "return_released_s",
    "max_decisions", "step_timeout_s", "max_time_s",
    "settle_min_s", "settle_pos_tol_rad", "settle_vel_tol", "settle_samples", "settle_timeout_s",
    "frame_max_age_s", "frame_timeout_s", "max_failures",
    "max_tokens", "request_timeout_s", "image_width_px", "jpeg_quality", "live_image_window",
    "retry_max", "retry_delay_base_s", "retry_delay_cap_s", "recovery_timeout_s",
    "states_hz", "demo_frames", "demo_frames_max",
    "video_fps", "video_candidates_per_window", "video_window_s", "video_keyframes_per_window",
    "video_candidate_width_px", "video_keyframe_width_px", "video_max_duration_s",
)


def load(path: Path | str | None = None) -> dict[str, dict]:
    p = Path(path) if path else PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"limits file not found: {p}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"limits file {p}: {e}") from None
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("limits"), dict):
        raise ValueError(f"limits file {p}: needs version 1 and a limits object")
    limits = data["limits"]
    missing = [n for n in NAMES if n not in limits]
    unknown = [n for n in limits if n not in NAMES]
    if missing:
        raise ValueError(f"limits file {p}: missing {missing}")
    if unknown:
        raise ValueError(f"limits file {p}: unknown {unknown} (a typo? the code reads {len(NAMES)} names, see limits.NAMES)")
    for name, entry in limits.items():
        if not isinstance(entry, dict):
            raise ValueError(f"limits file {p}: {name} must be an object")
        v = entry.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"limits file {p}: {name}.value must be a number, got {v!r}")
        if v < 0:
            raise ValueError(f"limits file {p}: {name}.value must not be negative")
        if entry.get("source") not in SOURCES:
            raise ValueError(f"limits file {p}: {name}.source must be one of {SOURCES}")
        for key in ("unit", "note"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"limits file {p}: {name} needs a {key}")
    _check_consistency(limits, p)
    return limits


def _check_consistency(limits: dict, p: Path) -> None:
    """The relations the numbers must keep, so an edit cannot make them contradict."""
    v = {k: e["value"] for k, e in limits.items()}
    rules = [("walk_speed", "base_vx_max"), ("side_speed", "base_vy_max"), ("turn_rate", "base_vyaw_max"),
             ("arm_path_seconds_min", "arm_path_seconds_default"), ("arm_path_seconds_default", "arm_path_seconds_max"),
             ("demo_frames", "demo_frames_max"), ("video_keyframes_per_window", "video_candidates_per_window"),
             ("wave_hand_seconds", "gesture_seconds_max"), ("shake_hand_seconds", "gesture_seconds_max"),
             ("settle_min_s", "settle_timeout_s"), ("retry_delay_base_s", "retry_delay_cap_s")]
    for low, high in rules:
        if v[low] > v[high]:
            raise ValueError(f"limits file {p}: {low} ({v[low]}) must not exceed {high} ({v[high]})")
    for name in ("control_dt_s", "walk_speed", "side_speed", "turn_rate", "max_decisions", "max_tokens",
                 "settle_samples", "states_hz", "image_width_px", "say_max_chars", "speech_chars_per_s"):
        if v[name] <= 0:
            raise ValueError(f"limits file {p}: {name} must be positive")


LIMITS: dict[str, dict] = load(os.environ.get("G1_LIMITS"))


def get(name: str) -> Any:
    """The value of one limit."""
    return LIMITS[name]["value"]


def use(path: Path | str | None) -> None:
    """Swap the limits for this process. Values already read into module
    constants at import are not re-read: set $G1_LIMITS before starting instead."""
    new = load(path)
    LIMITS.clear()
    LIMITS.update(new)


def describe(source: Optional[str] = None) -> str:
    rows = [(n, e) for n, e in LIMITS.items() if source is None or e["source"] == source]
    rows.sort(key=lambda r: (SOURCES.index(r[1]["source"]), NAMES.index(r[0])))
    width = max((len(n) for n, _ in rows), default=4)
    lines = [f"{'limit':<{width}}  {'value':>8}  {'unit':<6}  {'source':<10}  note"]
    for n, e in rows:
        lines.append(f"{n:<{width}}  {e['value']:>8g}  {e['unit']:<6}  {e['source']:<10}  {e['note']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="g1 limits", description="every tunable limit, and where it came from")
    p.add_argument("--source", choices=SOURCES, default=None, help="only limits from this source (guess: never checked)")
    args = p.parse_args(argv)
    print(f"{os.environ.get('G1_LIMITS') or PATH}\n")
    print(describe(args.source))
    counts = {s: sum(e["source"] == s for e in LIMITS.values()) for s in SOURCES}
    print("\n" + ", ".join(f"{c} {s}" for s, c in counts.items() if c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
