"""Every tunable limit and default of the robot and the run, from one file: ``configs/limits.json``.

Nothing else in the repo holds a limit, and .env holds none: it is for what is
private to one machine (the token, the robot's address and key). Which model is
asked, and how, is ``configs/model.json`` (``g1/vlm.py``). Each entry has a
``value``, a ``unit``, a ``source`` (where the number came from) and a ``note`` (what
it bounds). A setting that is a word, not a number, has unit
``text`` and, when only some words are allowed, ``choices``. ``g1 limits`` prints
them all as one table. Code reads a value with ``limits.get("name")``; a tool's parameters
refer to one with ``limit("name")`` and its prompt text with ``{limit:name}``.

    G1_LIMITS=my_limits.json g1 ...  # another file for one run

A missing or misspelt name fails at import, naming it: a limit is never
silently defaulted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]          # the repo
PATH = ROOT / "configs" / "limits.json"
SOURCES = ("guess", "gpt-policy", "repo", "unitree", "measured", "user", "robot")     # least to most trusted

# Every name the code reads. The file must define exactly these.
NAMES = (
    "control_dt_s", "chain_pause_s", "camera_timeout_s", "camera_poll_hz", "view_port", "view_width_px", "view_jpeg_quality", "view_fps", "png_compress_level",
    "base_vx_max", "base_vy_max", "base_vyaw_max", "walk_speed", "side_speed", "turn_rate", "min_segment_s",
    "joint_margin_rad", "command_vel_max", "arm_sdk_kp", "arm_sdk_kd", "max_violations", "joint_moved_rad", "max_printed_violations",
    "move_dx_max_m", "move_dy_max_m", "move_dyaw_max_deg", "walk_forward_min_m", "look_yaw_max_deg", "look_seconds_min", "look_seconds_max", "look_seconds_default",
    "point_down_deg_max", "point_down_deg_default", "point_sag_deg", "point_rise_s_min", "point_rise_s_max", "point_rise_s_default", "point_hold_s_min", "point_hold_s_max", "point_hold_s_default",
    "arm_path_max_waypoints", "arm_path_seconds_min", "arm_path_seconds_max", "arm_path_seconds_default", "tpose_hold_s_min", "tpose_hold_s_default", "tpose_rise_s_min", "tpose_rise_s_max", "tpose_rise_s_default", "sixseven_reps_max", "sixseven_reps_default", "sixseven_swing_s_min", "sixseven_swing_s_max", "sixseven_swing_s_default", "sixseven_settle_s_min", "sixseven_settle_s_max", "sixseven_settle_s_default", "sixseven_hold_s_min", "sixseven_hold_s_max", "sixseven_hold_s_default", "sixseven_lower_s",
    "hold_seconds_max", "hold_seconds_min", "hold_seconds_default", "wave_hand_seconds", "shake_hand_seconds", "gesture_seconds_max", "gesture_seconds_min", "gesture_ramp_s", "sit_seconds", "sit_seconds_min", "sit_seconds_max", "sit_line_up_m", "sit_back_m",
    "say_max_chars", "speech_chars_per_s", "say_min_s", "say_pause_max_s", "tts_speaker_id",
    "bookend_ramp_s", "bookend_ramp_max_s", "return_to_stand_s", "return_ramp_s", "return_released_s", "rpc_timeout_s", "lowstate_timeout_s", "lowstate_settle_s", "lowstate_poll_s", "release_s", "base_send_period_s", "fsm_enter_timeout_s", "fsm_poll_s", "gantry_damp_s", "gantry_lock_stand_s", "restore_fsm_s", "sit_stand_timeout_s", "status_probe_s", "health_period_s",
    "twin_physics_dt_s", "twin_fsm_delay_s", "twin_lock_stand_s", "twin_mask_arms_out_rad", "twin_sit_s", "twin_seated_fall_z_m", "twin_sit_kp", "twin_sit_kd", "twin_damp_kd", "twin_fall_z_m",
    "twin_fall_tilt_deg", "twin_band_kp", "twin_band_kd", "twin_waist_kp", "twin_waist_kd", "twin_waist_hold_s", "twin_arm_kp",
    "twin_arm_kd", "twin_vel_feedback", "twin_vel_integral", "twin_walk_blend_s",
    "twin_domain", "twin_lowstate_hz", "twin_camera_fps", "twin_camera_height_px", "twin_camera_width_px", "twin_settle_s", "twin_start_z_m", "twin_rope_slack_m", "twin_rope_lateral", "twin_walk_floor_margin", "twin_vel_integral_max", "twin_vel_filter", "twin_gesture_edge", "twin_wave_amp_rad", "twin_wave_hz", "twin_shake_amp_rad", "twin_shake_hz", "twin_volume", "twin_start_timeout_s", "twin_status_period_s", "twin_summary_timeout_s", "twin_truth_period_s", "sim_viewer_hold_s",
    "max_time_s",
    "settle_min_s", "settle_pos_tol_rad", "settle_vel_tol", "settle_samples", "settle_timeout_s",
    "frame_max_age_s", "frame_timeout_s", "max_failures", "overrun_fraction_max", "success_margin_deg", "bench_run_mb", "bench_parallel_max",
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
        if entry.get("unit") == "text":
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"limits file {p}: {name}.value must be a word (unit text), got {v!r}")
            if "choices" in entry and v not in entry["choices"]:
                raise ValueError(f"limits file {p}: {name}.value must be one of {entry['choices']}, got {v!r}")
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"limits file {p}: {name}.value must be a number, got {v!r}")
        elif v < 0:
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
             ("wave_hand_seconds", "gesture_seconds_max"), ("shake_hand_seconds", "gesture_seconds_max"),
             ("settle_min_s", "settle_timeout_s"),
             ("gesture_seconds_min", "wave_hand_seconds"), ("gesture_seconds_min", "shake_hand_seconds"),
             ("sit_seconds_min", "sit_seconds"), ("sit_seconds", "sit_seconds_max"),
             ("bookend_ramp_s", "bookend_ramp_max_s"), ("walk_forward_min_m", "move_dx_max_m"),
             ("tpose_hold_s_default", "hold_seconds_max"),
             ("point_down_deg_default", "point_down_deg_max")]
    # every tool argument with a range: min <= default <= max
    for stem in ("look_seconds", "hold_seconds", "point_rise_s", "point_hold_s", "tpose_hold_s", "tpose_rise_s", "sixseven_swing_s",
                 "sixseven_settle_s", "sixseven_hold_s"):
        lo, hi = f"{stem}_min", f"{stem}_max" if f"{stem}_max" in v else "hold_seconds_max"
        rules += [(lo, f"{stem}_default"), (f"{stem}_default", hi)]
    rules.append(("sixseven_reps_default", "sixseven_reps_max"))
    for low, high in rules:
        if v[low] > v[high]:
            raise ValueError(f"limits file {p}: {low} ({v[low]}) must not exceed {high} ({v[high]})")
    for name in ("control_dt_s", "walk_speed", "side_speed", "turn_rate",
                 "settle_samples", "say_max_chars", "speech_chars_per_s",
                 "twin_physics_dt_s", "twin_lowstate_hz", "twin_camera_fps"):
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


def table(limits: dict[str, dict] | None = None, width: int | None = None, title: str = "") -> str:
    """Every limit as one table: name, value, unit, source, note (wrapped to the terminal)."""
    import shutil
    import textwrap
    rows = [(n, f"{e['value']:g}" if isinstance(e["value"], (int, float)) else str(e["value"]),
             e["unit"], e["source"], e["note"]) for n, e in (LIMITS if limits is None else limits).items()]
    head = ("name", "value", "unit", "source", "note")
    w = [max(len(r[i]) for r in rows + [head]) for i in range(4)]
    width = width or shutil.get_terminal_size((160, 24)).columns
    note_w = max(30, width - sum(w) - 8)
    line = lambda c: f"{c[0]:<{w[0]}}  {c[1]:>{w[1]}}  {c[2]:<{w[2]}}  {c[3]:<{w[3]}}  {c[4]}"
    out = [line(head), line(tuple("-" * n for n in w) + ("-" * min(note_w, 40),))]
    pad = " " * (sum(w) + 8)
    for r in rows:
        note = textwrap.wrap(r[4], note_w) or [""]
        out.append(line(r[:4] + (note[0],)))
        out.extend(pad + more for more in note[1:])
    guesses = sum(r[3] == "guess" for r in rows)
    out.append(f"\n{len(rows)} limits{title} from {os.environ.get('G1_LIMITS') or PATH}; {guesses} are guesses "
               f"(source 'guess': review these first). Edit the file to change one.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """``g1 limits [SOURCE]``: print the table, or only the rows from one source (e.g. guess)."""
    argv = list(argv or [])
    if argv and argv[0] in ("-h", "--help"):
        print(f"usage: g1 limits [{'|'.join(SOURCES)}]\n\n{main.__doc__}")
        return 0
    rows = LIMITS
    if argv:
        if argv[0] not in SOURCES:
            print(f"g1 limits: unknown source {argv[0]!r}; choose from {', '.join(SOURCES)}")
            return 2
        rows = {n: e for n, e in LIMITS.items() if e["source"] == argv[0]}
    print(table(rows, title=f" with source {argv[0]!r}" if argv else ""))
    return 0
