"""How close is the twin to the robot? Run the same task on both and compare.

Every ``g1 run`` writes each control tick to ``trace.npz`` in its run directory:
what was commanded (q, the arm_sdk weight, kp, kd, the base velocity) and what
LowState measured (q, dq, tau_est, the IMU), plus, on the in-process twin, its
true base pose. ``g1 trace compare`` lines two runs up tick by tick (the same
chain gives the same ticks) and reports, per commanded joint, how each tracked
its command and how far the two runs differ.

    g1 run tpose --mode gantry          # the robot
    g1 run tpose --sim --mode gantry    # the twin
    g1 trace compare tasks/tpose/runs/<robot run> tasks/tpose/runs/<twin run>
    g1 trace show tasks/tpose/runs/<run>

  tracking RMS  root-mean-square of measured - commanded, after the weight reached 1
  sag           mean of measured - commanded while the command holds still (gravity)
  lag           the delay that best lines the measured angle up with the command
  A-B RMS       root-mean-square of the difference between the two runs' measured angles
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from g1.core.config import CONTROL_DT, JOINT_NAMES, NUM_JOINTS


class Trace:
    FIELDS = ("t", "q_cmd", "weight", "kp", "kd", "base_cmd", "q", "dq", "tau", "quat", "base_true")

    def __init__(self, path: Path, env_name: str) -> None:
        self.path = Path(path)
        self.env = env_name
        self.rows: dict[str, list] = {k: [] for k in self.FIELDS}

    def add(self, t: float, action, q, dq, tau, quat, base_true) -> None:
        cmd = np.full(NUM_JOINTS, np.nan)
        cmd[list(action.joints)] = np.asarray(action.q)[list(action.joints)]
        r = self.rows
        r["t"].append(t)
        r["q_cmd"].append(cmd)
        r["weight"].append(float(action.weight))
        r["kp"].append(float(action.kp))
        r["kd"].append(float(action.kd))
        r["base_cmd"].append(np.asarray(action.base if action.base is not None else (0.0, 0.0, 0.0), dtype=float))
        nan = np.full(NUM_JOINTS, np.nan)
        r["q"].append(np.asarray(q, dtype=float))
        r["dq"].append(nan if dq is None else np.asarray(dq, dtype=float))
        r["tau"].append(nan if tau is None else np.asarray(tau, dtype=float))
        r["quat"].append(np.full(4, np.nan) if quat is None else np.asarray(quat, dtype=float))
        r["base_true"].append(np.full(3, np.nan) if base_true is None else np.asarray(base_true, dtype=float))

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, env=self.env, **{k: np.asarray(v) for k, v in self.rows.items()})
        print(f"trace: {len(self.rows['t'])} ticks -> {self.path}")
        return self.path


def load(path) -> dict:
    """A trace.npz, or a run directory holding one."""
    path = Path(path)
    if path.is_dir():
        path = path / "trace.npz"
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# -- the numbers ----------------------------------------------------------------------------------
def joint_stats(tr: dict, j: int) -> dict | None:
    cmd, q, w = tr["q_cmd"][:, j], tr["q"][:, j], tr["weight"]
    owned = np.isfinite(cmd) & (w >= 0.999)
    if owned.sum() < 10:
        return None
    err = (q - cmd)[owned]
    still = owned & np.r_[False, np.abs(np.diff(cmd)) < 1e-4]
    c, m = cmd[owned] - cmd[owned].mean(), q[owned] - q[owned].mean()
    lag = 0
    if c.std() > 1e-3:
        lags = range(0, min(25, len(c) // 4))
        lag = max(lags, key=lambda k: float(np.dot(c[:len(c) - k], m[k:])))
    return {"rms": float(np.sqrt(np.mean(err ** 2))), "sag": float(np.mean((q - cmd)[still])) if still.any() else math.nan,
            "max": float(np.max(np.abs(err))), "lag_s": lag * CONTROL_DT}


def compare(a: dict, b: dict) -> str:
    n = min(len(a["t"]), len(b["t"]))
    lines = [f"A: {a['env']} ({len(a['t'])} ticks)   B: {b['env']} ({len(b['t'])} ticks)   compared: {n} ticks"]
    if len(a["t"]) != len(b["t"]):
        lines.append("warning: different lengths: were they the same chain?")
    a = {k: v[:n] if np.ndim(v) else v for k, v in a.items()}
    b = {k: v[:n] if np.ndim(v) else v for k, v in b.items()}
    if not np.allclose(np.nan_to_num(a["q_cmd"]), np.nan_to_num(b["q_cmd"]), atol=0.05):
        lines.append("warning: the commands differ by more than 0.05 rad somewhere: the runs started from "
                     "different poses or were different chains")
    lines.append(f"{'joint':<22} {'track RMS A':>11} {'B':>7} {'sag A':>8} {'B':>7} {'lag A':>6} {'B':>6} {'A-B RMS':>8}")
    for j in range(NUM_JOINTS):
        sa, sb = joint_stats(a, j), joint_stats(b, j)
        if sa is None or sb is None:
            continue
        diff = float(np.sqrt(np.nanmean((a["q"][:, j] - b["q"][:, j]) ** 2)))
        lines.append(f"{JOINT_NAMES[j]:<22} {sa['rms']:11.4f} {sb['rms']:7.4f} {sa['sag']:+8.4f} {sb['sag']:+7.4f} "
                     f"{sa['lag_s']:6.2f} {sb['lag_s']:6.2f} {diff:8.4f}")
    for name, tr in (("A", a), ("B", b)):
        if np.isfinite(tr["base_true"]).all():
            x, y, yaw = tr["base_true"][-1]
            lines.append(f"base {name} (truth): ended at ({x:+.2f}, {y:+.2f}) m yaw {math.degrees(yaw):+.0f} deg")
    lines.append("rad and s. Tune the twin (configs/limits.json twin_*, the model's armature/friction) until "
                 "A-B RMS is small, then mark those limits measured.")
    return "\n".join(lines)


def show(tr: dict) -> str:
    lines = [f"{tr['env']}: {len(tr['t'])} ticks, {tr['t'][-1] - tr['t'][0]:.2f} s"]
    for j in range(NUM_JOINTS):
        s = joint_stats(tr, j)
        if s is not None:
            lines.append(f"  {JOINT_NAMES[j]:<22} tracking RMS {s['rms']:.4f}  max {s['max']:.4f}  sag {s['sag']:+.4f}  "
                         f"lag {s['lag_s']:.2f} s")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="g1 trace", description="compare a twin run with a robot run (see --trace)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="two runs of the same task (run directories or trace.npz), tick by tick")
    c.add_argument("a")
    c.add_argument("b")
    s = sub.add_parser("show", help="one run's tracking")
    s.add_argument("path")
    args = p.parse_args(argv)
    if args.cmd == "compare":
        print(compare(load(args.a), load(args.b)))
    else:
        print(show(load(args.path)))
    return 0
