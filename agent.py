"""The decision step, as a Policy.

    observe   the newest frame and the joint angles, standing still
    decide    ask the decider (a background request; the loop never waits)
    act       run the chosen skill to its end
    record    write the step (joint angles, the frame, the decision, the outcome)

A skill may be longer than one step: while it runs, a record is closed every
``STEP_MAX`` seconds, capturing the frame and the joint angles at that moment
and carrying the same decision with its chunk number. The motion is not
interrupted — one model call covers the whole skill — so a 12 s ``sixseven``
becomes four recorded steps and the camera is still read every 3 s.

States: takeover -> settle -> snapshot -> think -> act -> settle -> ... -> handback.
Every sub-policy is seeded from the agent's own last *commanded* q, never the
measured one (the same rule as Routine and Selector). ``search`` drives it with
the vision model; ``replay`` rebuilds a saved run as a plain Routine and needs
neither camera nor model.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from config import CONTROL_DT, UPPER_BODY, joint_index
from decider import Context, Decider, Decision, HFDecider
from episode import EpisodeWriter, StepRecord, chain_of, load_episode
from policy import EPS, Action, Obs, Policy, SegmentPolicy
from routines import build_policy
from skills import STEP_MAX, Handback, Skill, Takeover, menu, skill_policy, skill_segments

WAIST_YAW = joint_index("waist_yaw")


class Agent(Policy):
    joints = sorted(UPPER_BODY)
    uses_camera = True

    def __init__(self, goal: str, decider: Decider, skills: Sequence[type[Skill]], *,
                 recorder: Optional[EpisodeWriter] = None, max_steps: int = 30,
                 step_timeout: float = 30.0, settle: float = 0.5, frame_max_age: float = 0.5,
                 frame_timeout: float = 2.0, max_retries: int = 2, max_failures: int = 3,
                 history: int = 5, name: str = "search") -> None:
        self.goal = goal
        self.decider = decider
        self.skills = {s.name: s for s in skills}
        self.recorder = recorder
        self.max_steps = max_steps
        self.step_timeout = step_timeout          # wall seconds: the model runs on wall time
        self.settle = settle
        self.frame_max_age = frame_max_age
        self.frame_timeout = frame_timeout
        self.max_retries = max_retries
        self.max_failures = max_failures
        self.history_n = history
        self.name = name
        self.result: Optional[str] = None
        self.steps: list[StepRecord] = []

    # -- lifecycle -------------------------------------------------------------
    def reset(self, obs: Obs) -> None:
        self._cmd = obs.q.copy()
        self.result = None
        self.steps = []
        self._step = 1
        self._failures = 0
        self._retries = 0
        self._base_cmd = np.zeros(3)             # dead-reckoned (x, y, yaw) from commanded velocity
        self._history: list[dict] = []
        self._snap: Optional[dict] = None
        self._seq0: Optional[int] = None
        self._chunk = 1
        self._chunks = 1
        self._chunk_t0 = 0.0
        self._sub: Optional[SegmentPolicy] = None
        self._state = ""
        self._t0 = 0.0
        self._closed = False
        self._enter_sub("takeover", 0.0, Takeover())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.decider.stop()
        if self.recorder is not None:
            if self.result is None:
                self.recorder.finish("aborted")
            self.recorder.close()

    # -- state helpers ---------------------------------------------------------------
    def _enter(self, state: str, t: float) -> None:
        self._state, self._t0 = state, t

    def _enter_sub(self, state: str, t: float, part) -> None:
        self._enter(state, t)
        self._sub = SegmentPolicy(skill_segments(part), joints=self.joints, name=self.name)
        self._sub.reset(Obs(self._cmd))

    def _hold(self) -> Action:
        return self.action(self._cmd.copy(), weight=1.0, base=None)

    def _emit(self, a: Action) -> Action:
        self._cmd = a.q.copy()
        if a.base is not None:
            vx, vy, vyaw = a.base
            x, y, yaw = self._base_cmd
            yaw += vyaw * CONTROL_DT
            x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * CONTROL_DT
            y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * CONTROL_DT
            self._base_cmd = np.array([x, y, yaw])
        return a

    def _say(self, msg: str) -> None:
        print(f"[{self.name}] {msg}")

    def _context(self, obs: Obs, frame, note: str = "") -> Context:
        x, y, yaw = self._base_cmd
        return Context(self.goal, self._step, self.max_steps, frame, obs.q.copy(),
                       math.degrees(float(self._cmd[WAIST_YAW])), (float(x), float(y), math.degrees(yaw)),
                       list(self._history[-self.history_n:]), list(self.skills.values()), note)

    def _to_settle(self, t: float, obs: Obs) -> Action:
        self._enter("settle", t)
        self._seq0 = obs.frame.seq if obs.frame is not None else None
        return self._hold()

    def _to_handback(self, t: float, result: str) -> Action:
        self.result = result
        self._say(f"result: {result}")
        self._enter_sub("handback", t, Handback())
        return self._hold()

    # -- the record ---------------------------------------------------------------------
    def _snapshot(self, t: float, obs: Obs, frame=None, think_wall=None) -> dict:
        f = frame if frame is not None else obs.frame
        return {"t": t, "wall": time.time(),
                "clock": None if f is None else f.stamp + (obs.frame_age if f is obs.frame else 0.0),
                "q": obs.q.copy(), "cmd": self._cmd.copy(),
                "frame_seq": None if f is None else f.seq,
                "frame_stamp": None if f is None else f.stamp,
                "frame_age": None if f is None else obs.frame_age,
                "image": None if f is None else f.image,
                "base_cmd": [float(v) for v in self._base_cmd],
                "base_env": None if obs.base_pose is None else list(obs.base_pose),
                "think_wall": think_wall}

    def _record(self, status: str, t: float, obs: Obs, decision: Optional[Decision],
                skill: Optional[Skill], duration: float) -> None:
        snap = self._snap or {}
        env_pose = obs.base_pose
        rec = StepRecord(
            step=self._step,
            t={"policy_start": snap.get("t"), "policy_end": t, "wall_start": snap.get("wall"), "wall_end": time.time(),
               "clock_start": snap.get("clock"), "clock_end": None if obs.frame is None else obs.frame.stamp + obs.frame_age,
               "think_wall": snap.get("think_wall")},
            q_start=[float(v) for v in snap.get("q", obs.q)], q_end=[float(v) for v in obs.q],
            cmd_start=[float(v) for v in snap.get("cmd", self._cmd)], cmd_end=[float(v) for v in self._cmd],
            frame={"seq": snap.get("frame_seq"), "stamp": snap.get("frame_stamp"), "age": snap.get("frame_age"),
                   "image": None, "shape": None if snap.get("image") is None else list(snap["image"].shape)},
            base_pose={"cmd_start": snap.get("base_cmd"), "cmd_end": [float(v) for v in self._base_cmd],
                       "env_start": snap.get("base_env"), "env_end": None if env_pose is None else list(env_pose)},
            decision=None if decision is None else {**decision.to_json(), "model": getattr(self.decider, "model", ""),
                                                    "retries": self._retries},
            skill=None if skill is None else {"name": skill.name, "args": dict(skill.args),
                                              "duration": skill.duration, "chunk": self._chunk,
                                              "chunks": self._chunks, "needs_base": skill.needs_base},
            outcome={"status": status, "duration": duration},
        )
        self.steps.append(rec)
        if decision is not None and status in ("completed", "done"):
            self._history.append({"step": self._step, "action": decision.action, "args": decision.args,
                                  "outcome": status, "scene": decision.scene})
        if self.recorder is not None:
            self.recorder.write_step(rec, snap.get("image"))

    def _close_chunk(self, status: str, t: float, obs: Obs) -> None:
        self._record(status, t, obs, self._decision, self._skill, t - self._chunk_t0)
        self._step += 1
        if status == "running":
            self._chunk += 1
            self._chunk_t0 = t
            self._snap = self._snapshot(t, obs)     # the next chunk starts here

    def _fail(self, status: str, t: float, obs: Obs) -> Action:
        self._say(f"step {self._step}: {status}")
        self._record(status, t, obs, None, None, t - (self._snap or {}).get("t", t))
        self._failures += 1
        self._retries = 0
        self._step += 1
        if self._failures >= self.max_failures or self._step > self.max_steps:
            return self._to_handback(t, "error" if self._failures >= self.max_failures else "max_steps")
        return self._to_settle(t, obs)

    # -- the loop ----------------------------------------------------------------------
    def step(self, t: float, obs: Obs) -> Optional[Action]:
        st = self._state
        if st in ("takeover", "handback"):
            a = self._sub.step(t - self._t0, obs)
            if a is not None:
                return self._emit(a)
            if st == "handback":
                if self.recorder is not None:
                    self.recorder.finish(self.result or "aborted", steps=len(self.steps))
                return None
            return self._to_settle(t, obs)

        if st == "settle":
            if t - self._t0 >= self.settle - EPS:
                self._enter("snapshot", t)
            return self._hold()

        if st == "snapshot":
            f = obs.frame
            if f is not None and f.seq != self._seq0 and obs.frame_age <= self.frame_max_age:
                self._snap = self._snapshot(t, obs, frame=f)
                self._ask(obs, f, "")
                self._enter("think", t)
            elif t - self._t0 > self.frame_timeout:
                self._say("no fresh camera frame")
                return self._fail("no_frame", t, obs)
            return self._hold()

        if st == "think":
            d = self.decider.latest()
            if d is not None and d.step == self._step and d.frame_seq == self._snap["frame_seq"]:
                self._snap["think_wall"] = time.monotonic() - self._wall0
                self._say(f"step {self._step}: {d.scene} | path {'clear' if d.path_clear else 'blocked'} "
                          f"-> {d.action}({d.args}) — {d.reason}")
                skill = self.skills[d.action](**d.args)
                self._chunk, self._chunks = 1, 1
                if skill.terminal:
                    self._record("done", t, obs, d, skill, 0.0)
                    return self._to_handback(t, "found" if d.args.get("found") else "not_found")
                self._decision, self._skill = d, skill
                self._chunks = max(1, math.ceil(skill.duration / STEP_MAX - 1e-9))
                self._chunk_t0 = t
                self._enter("act", t)
                self._sub = skill_policy(skill, self.joints, self.name)
                self._sub.reset(Obs(self._cmd))
                return self.step(t, obs)
            if not self.decider.pending:
                if self._retries < self.max_retries:
                    self._retries += 1
                    note = f"your previous reply was rejected: {self.decider.last_error}"
                    self._say(f"step {self._step}: retry {self._retries} ({self.decider.last_error})")
                    self._ask(obs, self._snap_frame(), note)
                    return self._hold()
                return self._fail("no_decision", t, obs)
            if time.monotonic() - self._wall0 > self.step_timeout:
                return self._fail("timeout", t, obs)
            return self._hold()

        if st == "act":
            a = self._sub.step(t - self._t0, obs)
            if a is not None:
                if t - self._chunk_t0 >= STEP_MAX - EPS:
                    # close this chunk and start the next; the motion itself is unbroken
                    self._close_chunk("running", t, obs)
                    if self._step > self.max_steps:
                        return self._to_handback(t, "max_steps")
                return self._emit(a)
            self._close_chunk("completed", t, obs)
            self._failures = 0
            self._retries = 0
            if self._step > self.max_steps:
                return self._to_handback(t, "max_steps")
            return self._to_settle(t, obs)

        raise RuntimeError(f"unknown state {st!r}")

    def _snap_frame(self):
        from camera import Frame
        s = self._snap
        return Frame(s["image"], s["frame_stamp"], s["frame_seq"])

    def _ask(self, obs: Obs, frame, note: str) -> None:
        self._wall0 = time.monotonic()
        if not self.decider.request(self._context(obs, frame, note)):
            self._say("decider busy; will retry")


# --------------------------------------------------------------------------
# CLI builders
# --------------------------------------------------------------------------

def build_search(args, can_walk: bool) -> Agent:
    if not getattr(args, "goal", None):
        raise ValueError("search needs --goal, e.g. --goal \"find the mug\"")
    echo = (lambda s: print(s, end="", flush=True)) if getattr(args, "vision_echo", False) else None
    decider = HFDecider.from_env(getattr(args, "vision_model", None), on_text=echo)
    skills = menu(can_walk)
    recorder = None
    if not getattr(args, "no_log", False):
        recorder = EpisodeWriter(getattr(args, "log", "runs"), env=args.env, goal=args.goal, model=decider.model,
                                 skills=[s.name for s in skills], allow_base=can_walk,
                                 extra={"max_steps": args.max_steps, "step_timeout": args.step_timeout})
        print(f"recording to {recorder.dir}")
    return Agent(args.goal, decider, skills, recorder=recorder, max_steps=args.max_steps,
                 step_timeout=args.step_timeout)


def build_replay(args, can_walk: bool) -> Policy:
    if not getattr(args, "episode", None):
        raise ValueError("replay needs --episode runs/<dir>")
    meta, steps = load_episode(args.episode, images=False)
    chain = chain_of(steps)
    if not chain:
        raise ValueError(f"{args.episode}: no completed skills to replay")
    policy = build_policy(chain, pause=getattr(args, "pause", 1.0))
    policy.name = f"replay:{Path(args.episode).name}"
    print(f"replaying {len(steps)} recorded step(s) of {meta.get('goal')!r}: {chain}")
    return policy


AGENTS = {"search": build_search, "replay": build_replay}
