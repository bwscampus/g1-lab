"""g1 moves: Unitree's own stored moves on the robot, and the FSM switches they need.

    g1 moves [list]             read-only: the FSM and every stored move (mimic motions, arm actions, taught)
    g1 moves check MOVE ...     read-only: is each installed, how long it takes, what it needs
    g1 moves fsm TARGET         switch the FSM along its allowed order (damp, stand, main, walk, sit, up)
    g1 moves play MOVE ...      play stored moves one after another (asks first)
    g1 moves walk|side|turn N   walk N m (negative: back), sidestep N m (positive: left), turn N deg (positive: left)

A MOVE is a mimic motion by id or name (``516``, ``bow``; FSM ``550<id>``, whole body),
``arm:<id or name>`` (the arm service's preset actions: ``arm:20``, ``arm:high_five``) or
``custom:<name>`` (an action taught in the Unitree app: ``custom:Waist_Drum_Dance``). Names
match with or without the ``vq_`` prefix, ignoring case.

The robot is the only source of what is installed (``GetMimicMotion``, ``GetActionList``): the
twin has none of it, so this command talks to the robot only (``UNITREE_IFACE``). Every call
goes through ``sdk.call``; a refusal is printed with its code and stops the command. Ctrl-C
while a move plays stops it (back to the FSM it started from; an arm action is released)
and that stop runs to completion.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from typing import Any, Callable, Optional

from g1.core import limits
from g1.envs import sdk
from g1.envs.base import shield_sigint
from g1.envs.link import DdsLink, dds_domain, is_loopback

MIMIC_FSM = 550                 # SetFsmId(550<motion id>) plays a stored motion (unitree_sdk2 GetMimicMotion)
ARM_RELEASE = 99                # ExecuteAction(99) releases an arm action that holds its last keyframe

# The FSM targets by name, and the order the onboard controller takes them in (Unitree's G1
# docs and the SDK's helpers; the repo's gantry bring-up is 1 -> 4 -> 200). Damp (1) drops a
# standing robot, so from standing it is only ever a final step, never a way through.
# Walking is 501 on the 29-DoF G1 (3-DoF waist) and 500 on the 1-DoF-waist one: ours takes
# SetFsmId(500) with code 0 and stays in 4 (2026-11-03).
FSM_TARGETS = {"damp": 1, "stand": 4, "main": 200, "walk": 501, "sit": 3, "up": 801}
FSM_NEXT = {0: (1,), 1: (4,), 4: (200, 501, 500, 1), 200: (501, 500, 3, 1), 500: (200, 3, 1), 501: (200, 3, 1),
            3: (801,), 801: (200, 501, 500), 802: (501, 500, 200)}
FSM_WARN = {1: "damp: every joint goes limp; a standing robot falls. Only with it hanging in the gantry",
            4: "locked stand from damp: hang it in the gantry (feet just touching), or have it already standing",
            3: "sit: straight down where it stands; a chair must be right behind its heels",
            801: "sit to stand: from sitting, clear space in front"}
# Moves play only from standing under the onboard controller. A stored move runs in 550 and
# ends in 802 (our G1, 2026-11-03: 501 -> 550 -> 802), from which the next one plays.
PLAY_FSMS = frozenset({200, 500, 501, 801, 802})
WALK_FSMS = frozenset({200, 500, 501})          # walk in these; from 802 (after a stored move) it is untested, so switch first

# English for the stored motions' pinyin names (the robot lists them as below).
GLOSS = {"lie_down": "lie down", "yingbin": "welcome the guests (long)", "chayao": "hands on hips", "feiwen": "flying kiss",
         "fuxiongqueren": "hand on chest, agree", "jingli": "salute", "naopigu": "scratch bottom",
         "shuangshoudazhaohu": "greet with both hands", "shuangshouhuanying": "welcome with both hands",
         "shuangshoupaijian": "pat shoulders", "shuangshouwulian": "cover face", "shuangshouyouzhi": "both hands point right",
         "shuangshouzuozhi": "both hands point left", "sikao": "thinking", "toudingbixin": "heart over head",
         "youqiecai": "chop right", "zuoqiecai": "chop left", "youshoudazhaohu": "greet with right hand",
         "zuoshouzhaohu": "greet with left hand", "youshouqianzhi": "right hand points ahead",
         "zuoshouzhixiang": "left hand points", "ziwojieshao": "self-introduction", "qishen": "get up",
         "POPPING": "popping dance", "JIXIEWU": "mechanical dance", "NEWBALEI": "dance",
         "WOYAOFANGUOXUESHAN": "dance ('crossing the snow mountain')", "Skeleton0": "dance (mocap take)"}


def gloss(name: str) -> str:
    return next((g for k, g in GLOSS.items() if k in name), "")


def _key(name: str) -> str:
    name = name.strip().lower()
    return name[3:] if name.startswith("vq_") else name


# -- what is installed ------------------------------------------------------------------------

def inventory(loco, arm, log: list, verbose: bool = False) -> dict:
    """Ask the robot what it has. ``{"mimic": {id: {name, duration}}, "arm": [{id, name, ...}],
    "custom": [{name, time}], "codes": {...}}``; a query that fails leaves its part empty."""
    out: dict = {"mimic": {}, "arm": [], "custom": [], "codes": {}}
    code, data = sdk.call(loco, "GetMimicMotion", log=log, verbose=verbose, label="LocoClient")
    out["codes"]["mimic"] = code
    if code == 0 and isinstance(data, dict):
        out["mimic"] = {int(k): v for k, v in (data.get("motion") or {}).items()}
    code, data = sdk.call(arm, "GetActionList", log=log, verbose=verbose, label="ArmActionClient")
    out["codes"]["arm"] = code
    if code == 0 and data:
        parts = data if isinstance(data, list) and data and isinstance(data[0], list) else [data]
        for part in parts:
            for a in part:
                (out["arm"] if "id" in a else out["custom"]).append(a)
    return out


def resolve(move: str, inv: dict) -> Optional[dict]:
    """A MOVE -> ``{"kind", "id"|"name", "label", "duration"}``, or None when it is not installed."""
    kind, _, rest = move.partition(":") if ":" in move else ("mimic", "", move)
    rest = rest.strip()
    if kind == "mimic":
        for mid, m in inv["mimic"].items():
            if rest == str(mid) or _key(rest) == _key(m.get("name", "")):
                return {"kind": "mimic", "id": mid, "label": m.get("name", ""), "duration": float(m.get("duration", 0))}
    elif kind == "arm":
        for a in inv["arm"]:
            if rest == str(a["id"]) or _key(rest) == _key(a.get("name", "")):
                return {"kind": "arm", "id": int(a["id"]), "label": a.get("name", ""), "duration": None,
                        "fsm": a.get("fsm"), "mode_machine": a.get("mode_machine")}
    elif kind == "custom":
        for a in inv["custom"]:
            if _key(rest) == _key(a.get("name", "")):
                return {"kind": "custom", "name": a["name"], "label": a["name"], "duration": float(a.get("time", 0))}
    return None


def describe(m: dict) -> str:
    if m["kind"] == "mimic":
        what = f"{m['id']:>4}  SetFsmId({MIMIC_FSM}{m['id']:03d})"
    else:
        what = f"arm:{m['id']}" if m["kind"] == "arm" else f"custom:{m['name']}"
    dur = f"{m['duration']:5.1f} s" if m.get("duration") else "     ? s"
    extra = [gloss(m["label"])]
    if m.get("fsm"):
        extra.append(f"only in FSM {'/'.join(map(str, m['fsm']))}")
    if m.get("mode_machine"):
        extra.append(f"only on mode_machine {'/'.join(map(str, m['mode_machine']))}")
    return f"  {what:<22} {dur}  {m['label']}" + "".join(f"  ({e})" for e in extra if e)


def list_lines(inv: dict, mode_machine: Optional[int] = None) -> list[str]:
    lines = [f"mimic motions (whole body; play: g1 moves play <id or name>), code {sdk.explain(inv['codes']['mimic'])}:"]
    lines += [describe(resolve(str(mid), inv)) for mid in sorted(inv["mimic"])] or ["  none"]
    lines.append(f"arm actions (upper body; play: g1 moves play arm:<id or name>), code {sdk.explain(inv['codes']['arm'])}"
                 + (f"; this robot is mode_machine {mode_machine}" if mode_machine is not None else "") + ":")
    lines += [describe(resolve(f"arm:{a['id']}", inv)) for a in inv["arm"] if a["id"] != ARM_RELEASE] or ["  none"]
    lines.append("taught in the app (play: g1 moves play custom:<name>):")
    lines += [describe(resolve(f"custom:{a['name']}", inv)) for a in inv["custom"]] or ["  none"]
    return lines


# -- the FSM, in order --------------------------------------------------------------------------

def fsm_path(start: int, target: int) -> Optional[list[int]]:
    """The shortest allowed sequence of FSMs from ``start`` to ``target`` (excluding start).
    From a standing FSM damp is only ever the last step (it drops the robot); from zero torque
    it is the way up (0 -> 1 -> 4). None when there is no allowed way."""
    if start == target:
        return []
    prev: dict = {start: None}
    queue = deque([start])
    while queue:
        fsm = queue.popleft()
        for nxt in FSM_NEXT.get(fsm, ()):
            if nxt in prev or (nxt == 1 and fsm != 0 and nxt != target):
                continue
            prev[nxt] = fsm
            if nxt == target:
                path = [nxt]
                while prev[path[0]] != start:
                    path.insert(0, prev[path[0]])
                return path
            queue.append(nxt)
    return None


class Moves:
    """The robot's stored moves and FSM, over one LocoClient and one arm-action client.
    The clock is injectable for tests."""

    def __init__(self, loco, arm, *, verbose: bool = False, sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.monotonic, confirm: Callable[[str], bool] = None):
        self.loco, self.arm, self.verbose = loco, arm, verbose
        self.sleep, self.now = sleep, now
        self.confirm = confirm or _ask
        self.log: list = []

    def _call(self, client, name: str, *args) -> tuple[int, Any]:
        label = "ArmActionClient" if client is self.arm else "LocoClient"
        return sdk.call(client, name, *args, log=self.log, verbose=self.verbose, label=label)

    def _require(self, code: int, what: str) -> None:
        if code != 0:
            hint = ("no reply from the robot: check the link (g1 status)" if sdk.is_transport(code) else
                    "the robot refused: the remote or the app may own it, or this state does not allow it")
            raise SystemExit(f"{what} failed with {sdk.explain(code)}; {hint}")

    def fsm(self) -> int:
        code, fsm = self._call(self.loco, "GetFsmId")
        self._require(code, "GetFsmId")
        return int(fsm)

    def inventory(self) -> dict:
        return inventory(self.loco, self.arm, self.log, self.verbose)

    # -- switching -------------------------------------------------------------------------------
    def switch(self, target: int) -> None:
        start = self.fsm()
        path = fsm_path(start, target)
        if path is None:
            raise SystemExit(f"no allowed way from FSM {sdk.fsm_name(start)} to {sdk.fsm_name(target)} "
                             f"(order: {_order()}); use the remote")
        if not path:
            print(f"already in FSM {sdk.fsm_name(target)}")
            return
        print(f"FSM {sdk.fsm_name(start)} -> " + " -> ".join(sdk.fsm_name(f) for f in path))
        for f in path:
            if f in FSM_WARN:
                print(f"  {FSM_WARN[f]}")
        if not self.confirm("switch the FSM?"):
            raise SystemExit("not switched")
        for f in path:
            self._enter(f)

    def _enter(self, target: int) -> None:
        hold = {1: limits.get("gantry_damp_s"), 4: limits.get("gantry_lock_stand_s"),
                3: limits.get("sit_seconds")}.get(target, 0.0)
        timeout = max(hold, limits.get("sit_stand_timeout_s") if target == 801 else limits.get("fsm_enter_timeout_s"))
        self._require(self._call(self.loco, "SetFsmId", target)[0], f"SetFsmId({target})")
        t0 = self.now()
        while (fsm := self.fsm()) != target:
            if self.now() - t0 > timeout:
                raise SystemExit(f"the robot did not report FSM {target} within {timeout:g} s (reports "
                                 f"{sdk.fsm_name(fsm)}); check the remote and the app")
            self.sleep(limits.get("fsm_poll_s"))
        self.sleep(max(0.0, hold - (self.now() - t0)))       # the motion goes on after the FSM is reported
        print(f"  FSM {sdk.fsm_name(target)} after {self.now() - t0:.1f} s")

    # -- playing -----------------------------------------------------------------------------------
    def play(self, moves: list[dict]) -> int:
        start = self.fsm()
        if start not in PLAY_FSMS:
            raise SystemExit(f"stored moves play from standing (FSM {', '.join(map(str, sorted(PLAY_FSMS)))}); "
                             f"the robot reports {sdk.fsm_name(start)}: g1 moves fsm walk (or the remote) first")
        for m in moves:
            if m.get("fsm") and start not in m["fsm"]:
                raise SystemExit(f"arm:{m['id']} ({m['label']}) runs only in FSM {m['fsm']}; the robot is in "
                                 f"{sdk.fsm_name(start)} (g1 moves fsm walk)")
        total = sum(m["duration"] or limits.get("arm_action_s") for m in moves)
        print(f"from FSM {sdk.fsm_name(start)}, about {total:.0f} s:")
        for m in moves:
            print(describe(m))
        if not self.confirm("clear the area around it, spotter and remote ready. Play?"):
            raise SystemExit("not played")
        for i, m in enumerate(moves):
            if i:
                self.sleep(limits.get("chain_pause_s"))
            try:
                self._play_one(m, start)
            except KeyboardInterrupt:
                with shield_sigint("stopping the move; this finishes on its own"):
                    self._stop(m, start)
                print("stopped")
                return 130
        end = self.fsm()
        print(f"done; FSM {sdk.fsm_name(end)}" + ("" if end == start else f" (started in {sdk.fsm_name(start)})"))
        return 0

    def _play_one(self, m: dict, start: int) -> None:
        print(f"play {m['label']}" + (f" ({gloss(m['label'])})" if gloss(m["label"]) else ""))
        if m["kind"] == "mimic":
            fsm_id = MIMIC_FSM * 1000 + m["id"]
            self._require(self._call(self.loco, "SetFsmId", fsm_id)[0], f"SetFsmId({fsm_id})")
        elif m["kind"] == "arm":
            self._require(self._call(self.arm, "ExecuteAction", m["id"])[0], f"ExecuteAction({m['id']})")
        else:
            self._require(self._call(self.arm, "ExecuteCustomAction", m["name"])[0], f"ExecuteCustomAction({m['name']!r})")
        wait = (m["duration"] or limits.get("arm_action_s")) + limits.get("moves_settle_s")
        t0, seen = self.now(), None
        while True:
            fsm = self.fsm()
            if fsm != seen:
                print(f"  {self.now() - t0:5.1f} s  FSM {sdk.fsm_name(fsm)}")
                seen = fsm
            elapsed = self.now() - t0
            if m["kind"] == "mimic":               # it is over when the robot leaves 550; some run past their duration
                if fsm != MIMIC_FSM and elapsed > limits.get("fsm_poll_s") * 2:
                    break
                if elapsed > wait + limits.get("moves_overrun_s"):
                    break
            elif elapsed >= wait:
                break
            self.sleep(limits.get("fsm_poll_s"))
        if m["kind"] == "mimic" and seen != MIMIC_FSM:
            self.sleep(limits.get("moves_settle_s"))
        if m["kind"] == "arm":                       # some hold their last keyframe (up to 20 s)
            self._call(self.arm, "ExecuteAction", ARM_RELEASE)
        if m["kind"] == "mimic" and seen not in PLAY_FSMS:
            print(f"  warning: after the motion the robot reports {sdk.fsm_name(seen)}, not a standing FSM")

    def _stop(self, m: dict, start: int) -> None:
        if m["kind"] == "mimic":
            code, _ = self._call(self.loco, "SetFsmId", start)
            print(f"stop: SetFsmId({start}) -> {sdk.explain(code)}")
        elif m["kind"] == "arm":
            code, _ = self._call(self.arm, "ExecuteAction", ARM_RELEASE)
            print(f"stop: release the arms -> {sdk.explain(code)}")
        else:
            code, _ = self._call(self.arm, "StopCustomAction")
            print(f"stop: StopCustomAction -> {sdk.explain(code)}")

    # -- walking -----------------------------------------------------------------------------------
    def walk(self, forward_m: float = 0.0, side_m: float = 0.0, turn_deg: float = 0.0) -> int:
        """Turn, then walk forward (negative: back), then sidestep (positive: left), at the repo's
        speeds (walk_speed, side_speed, turn_rate), each within the base's hard ceilings. The
        velocity (SetVelocity with a 1 s duration: the robot's own dead-man) is re-sent every
        base_send_period_s and zeroed at the end, so the robot stops by itself within 1 s if this
        process dies. From after a stored move (802) it switches to walk mode (501) first.
        Ctrl-C stops it at once. 0, or 130 when stopped."""
        legs = [((0.0, 0.0, 1.0), math.radians(turn_deg), min(limits.get("turn_rate"), limits.get("base_vyaw_max"))),
                ((1.0, 0.0, 0.0), forward_m, min(limits.get("walk_speed"), limits.get("base_vx_max"))),
                ((0.0, 1.0, 0.0), side_m, min(limits.get("side_speed"), limits.get("base_vy_max")))]
        legs = [(axis, amount, speed) for axis, amount, speed in legs if abs(amount) > 1e-6]
        if not legs:
            return 0
        fsm = self.fsm()
        if fsm not in WALK_FSMS:
            print(f"walking needs walk mode; the robot is in FSM {sdk.fsm_name(fsm)}")
            self.switch(FSM_TARGETS["walk"])
        try:
            for axis, amount, speed in legs:
                vel = [a * speed * (1 if amount > 0 else -1) for a in axis]
                seconds = abs(amount) / speed
                what = (f"turn {math.degrees(amount):+.0f} deg" if axis[2] else
                        f"{'forward' if axis[0] else 'sideways'} {amount:+.2f} m")
                print(f"walk: {what} ({seconds:.1f} s)")
                t0 = self.now()
                while self.now() - t0 < seconds:
                    self._require(self._call(self.loco, "SetVelocity", *vel, 1.0)[0], "SetVelocity")
                    self.sleep(min(limits.get("base_send_period_s"), max(0.0, seconds - (self.now() - t0))))
                self._call(self.loco, "SetVelocity", 0.0, 0.0, 0.0, 1.0)
                self.sleep(limits.get("moves_settle_s"))
        except KeyboardInterrupt:
            with shield_sigint("stopping; this finishes on its own"):
                code, _ = self._call(self.loco, "SetVelocity", 0.0, 0.0, 0.0, 1.0)
                print(f"stop: SetVelocity(0) -> {sdk.explain(code)}")
            return 130
        return 0


def _order() -> str:
    return "; ".join(f"{a} -> {'/'.join(map(str, b))}" for a, b in FSM_NEXT.items())


def _ask(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _mode_machine(link, wait_s: float) -> Optional[int]:
    """The robot's model variant, from one LowState (arm actions list the variants they run on)."""
    seen: dict = {}
    sub = link.lowstate(lambda msg: seen.setdefault("mm", int(msg.mode_machine)))
    t0 = time.monotonic()
    while "mm" not in seen and time.monotonic() - t0 < wait_s:
        time.sleep(limits.get("lowstate_poll_s"))
    del sub
    return seen.get("mm")


def connect(iface: str, verbose: bool = False, link=None) -> tuple[Moves, Any]:
    link = link or DdsLink()
    link.init(dds_domain(iface), iface)
    loco, arm = link.loco(), link.arm_action()
    for client in (loco, arm):
        client.SetTimeout(limits.get("rpc_timeout_s"))
        client.Init()
    return Moves(link.mimic(loco), arm, verbose=verbose), link


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="g1 moves", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", nargs="?", default="list", choices=("list", "check", "fsm", "play", "walk", "side", "turn"))
    p.add_argument("args", nargs="*", help="check/play: MOVEs; fsm: one of " + ", ".join(FSM_TARGETS)
                   + "; walk: metres (negative: back); side: metres (positive: left); turn: degrees (positive: left)")
    a = p.parse_args(argv)
    iface = os.environ.get("UNITREE_IFACE")
    if not iface:
        p.error("set UNITREE_IFACE=<iface or IP> in .env")
    if is_loopback(iface):
        p.error("g1 moves is the robot's stored moves; the twin has none (UNITREE_IFACE is loopback)")
    if a.action in ("check", "play") and not a.args:
        p.error(f"{a.action}: name at least one MOVE (g1 moves list shows them)")
    if a.action in ("walk", "side", "turn"):
        try:
            amount = float(a.args[0]) if len(a.args) == 1 else None
        except ValueError:
            amount = None
        if amount is None:
            p.error(f"{a.action}: one number (walk 0.5, side -0.3, turn 90)")
    if a.action == "fsm" and (len(a.args) != 1 or a.args[0] not in FSM_TARGETS):
        p.error("fsm: one of " + ", ".join(f"{k} ({v})" for k, v in FSM_TARGETS.items()))
    try:
        moves, link = connect(iface, os.environ.get("G1_VERBOSE", "") not in ("", "0"))
    except ImportError as e:
        p.error(f"unitree_sdk2py is not installed ({e})")
    if a.action == "fsm":
        moves.switch(FSM_TARGETS[a.args[0]])
        return 0
    if a.action in ("walk", "side", "turn"):
        if not moves.confirm(f"{a.action} {amount:g}{' deg' if a.action == 'turn' else ' m'}: clear path, spotter and remote ready?"):
            raise SystemExit("not walked")
        return moves.walk(**{{"walk": "forward_m", "side": "side_m", "turn": "turn_deg"}[a.action]: amount})
    fsm = moves.fsm()
    inv = moves.inventory()
    if a.action == "list":
        print(f"FSM {sdk.fsm_name(fsm)}" + ("" if fsm in PLAY_FSMS else " (moves play from standing: g1 moves fsm walk)"))
        print("\n".join(list_lines(inv, _mode_machine(link, limits.get("lowstate_timeout_s")))))
        return 0
    found = {m: resolve(m, inv) for m in a.args}
    missing = [m for m, r in found.items() if r is None]
    if a.action == "check":
        for m, r in found.items():
            print(describe(r) if r else f"  {m:<22} not installed")
        return 1 if missing else 0
    if missing:
        raise SystemExit(f"not installed on this robot: {', '.join(missing)} (g1 moves list)")
    return moves.play(list(found.values()))
