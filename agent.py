"""The decision step, as a Policy — GPT-Policy's loop on our executor.

    observe   a fresh frame and the measured joint state, standing still
    decide    ask the decider (a background request; the loop never waits)
    act       run the chosen skill to its end, then wait until the joints
              have measurably settled
    feed back what happened (residuals, base error, the settle report) as
              ``previous_result`` in the next observation; a rejected
              selection is fed back the same way and costs a decision
    record    every observation, decision, timing, result and error

A skill may be longer than one step: while it runs, a record is closed every
``STEP_MAX`` seconds, capturing the frame and the joint angles at that moment
and carrying the same decision with its chunk number. The motion is not
interrupted — one model call covers the whole skill — so a 12 s ``sixseven``
becomes four recorded steps and the camera is still read every 3 s.

States: takeover -> settle -> snapshot -> think -> act -> settle -> ... ->
handback, plus ``wait`` for the backoff before a model retry (each retry
re-observes; no robot action is ever replayed). ``done`` / ``give_up`` end
the run with the model's conclusion; the *label* is the human's, asked at
close. Every sub-policy is seeded from the agent's own last *commanded* q,
never the measured one (the same rule as Routine and Selector). ``search``
drives it with the vision model; ``replay`` rebuilds a saved run as a plain
Routine and needs neither camera nor model.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from config import CONTROL_DT, JOINT_NAMES, UPPER_BODY, joint_index
from decider import AgentContext, AgentTurn, Decider, Decision, VLMDecider, ProtocolError, build_context, observation
from envs.monitor import JointMonitor
from episode import EpisodeWriter, StepRecord, chain_of, load_episode
from vlm import Overloaded, QuotaExceeded
from policy import EPS, Action, Obs, Policy, SegmentPolicy
from routines import build_policy
from skills import STEP_MAX, Check, Handback, Skill, Takeover, menu, skill_policy, skill_segments, use_catalog

WAIST_YAW = joint_index("waist_yaw")

# their MODEL_RETRY_DELAYS_S / MODEL_RECOVERY_TIMEOUT_S
RETRY_DELAYS = tuple(min(2.0 * (2 ** k), 8.0) for k in range(20))
RECOVERY_TIMEOUT = 300.0


def dry_run(skill: Skill, cmd: np.ndarray, joints: Sequence[int]) -> dict:
    """``check``: plan a skill from the commanded pose through a JointMonitor
    without physics — joint limits, command speed, base limits, duration and
    the base displacement it would produce. Moves nothing."""
    policy = SegmentPolicy(skill_segments(skill), joints=list(joints), name="check")
    policy.reset(Obs(cmd))
    mon = JointMonitor(strict=False)
    mon.reset(hold=cmd)
    n = 0
    while (a := policy.step(n * CONTROL_DT, Obs(cmd))) is not None:
        mon.observe(a.q, a)
        n += 1
    x, y, yaw = mon.base_pose()
    moved = [JOINT_NAMES[j] for j in mon.rows() if np.isfinite(mon.cmd_min[j]) and mon.cmd_max[j] - mon.cmd_min[j] > 5e-3]
    return {"status": "ok" if not mon.violations else "rejected",
            "skill": skill.name, "arguments": {k: v for k, v in skill.args.items() if k != "note"},
            "duration_s": n * CONTROL_DT, "base_delta": [x, y, math.degrees(yaw)],
            "peak_command_vel_rad_s": float(mon.peak_vel.max()), "joints_moved": moved,
            "violations": [str(v) for v in mon.violations[:12]], "n_violations": len(mon.violations)}


class Agent(Policy):
    joints = sorted(UPPER_BODY)
    uses_camera = True

    def __init__(self, goal: str, decider: Decider, skills: Sequence[type[Skill]], *,
                 recorder: Optional[EpisodeWriter] = None, max_decisions: int = 30,
                 step_timeout: float = 60.0, settle_min: float = 0.5, settle_tol: float = 0.03,
                 settle_vel_tol: float = 0.05, settle_samples: int = 10, settle_timeout: float = 3.0,
                 frame_max_age: float = 0.5, frame_timeout: float = 2.0, max_failures: int = 3,
                 can_walk: bool = True, has_loco: bool = False, safety_notes: Sequence[str] = (), content: Sequence = (),
                 verdict: Optional[Callable[[], Optional[str]]] = None, name: str = "search",
                 retry_delays: Sequence[float] = RETRY_DELAYS, recovery_timeout: float = RECOVERY_TIMEOUT) -> None:
        self.goal = goal
        self.decider = decider
        self.skills = {s.name: s for s in skills}
        self.recorder = recorder
        self.max_decisions = max_decisions
        self.step_timeout = step_timeout          # wall seconds: the model runs on wall time
        self.settle_min = settle_min              # so StopMove has landed before a frame is taken
        self.settle_tol = settle_tol
        self.settle_vel_tol = settle_vel_tol
        self.settle_samples = settle_samples
        self.settle_timeout = settle_timeout
        self.frame_max_age = frame_max_age
        self.frame_timeout = frame_timeout
        self.max_failures = max_failures
        self.can_walk = can_walk
        self.has_loco = has_loco
        self.safety_notes = list(safety_notes)
        self.content = tuple(content)
        self.verdict = verdict
        self.name = name
        self.retry_delays = tuple(retry_delays)
        self.recovery_timeout = recovery_timeout
        self.context: AgentContext = build_context(list(skills), can_walk=can_walk, max_decisions=max_decisions,
                                                   safety_notes=self.safety_notes)
        self.result: Optional[str] = None         # runtime status: completed | give_up | budget_exhausted | failed
        self.error: Optional[str] = None
        self.steps: list[StepRecord] = []
        self._closed = False
        self._finalized = False

    # -- lifecycle -------------------------------------------------------------
    def reset(self, obs: Obs) -> None:
        self._cmd = obs.q.copy()
        self._prev_q = obs.q.copy()
        self.result = None
        self.error = None
        self.steps = []
        self._step = 1                            # record counter (3 s chunks, rejections, terminals)
        self._env_step = 0                        # decision index: their env_step
        self._failures = 0
        self._attempt = 0
        self._deadline: Optional[float] = None
        self._request_id = 0
        self._previous: Optional[dict] = None     # the next observation's previous_result
        self._pending: Optional[dict] = None      # a finished skill awaiting its settle report
        self._base_cmd = np.zeros(3)              # dead-reckoned (x, y, yaw) from commanded velocity
        self._snap: Optional[dict] = None
        self._seq0: Optional[int] = None
        self._chunk = 1
        self._chunks = 1
        self._chunk_t0 = 0.0
        self._sub: Optional[SegmentPolicy] = None
        self._state = ""
        self._t0 = 0.0
        self._turn: Optional[AgentTurn] = None
        self._wall0 = 0.0
        self._wait_until = 0.0
        self.decider.start(self.context)
        self._event("protocol", self.context.record())
        self._enter_sub("takeover", 0.0, Takeover())

    def close(self) -> None:
        """Stop the model, ask the human, settle the record. Also the Ctrl-C path."""
        if self._closed:
            return
        self._closed = True
        self.decider.stop()
        status = self.result or "interrupted"
        self._event("execution_finished", {"status": status, "error": self.error})
        human = None
        if self.recorder is not None and self.verdict is not None:
            try:
                human = self.verdict()
            except (EOFError, KeyboardInterrupt):
                human = None
            if human is None:
                self._event("human_evaluation_skipped", {"reason": "no_terminal_or_cancelled"})
        if self.recorder is not None:
            if self.recorder.meta.get("result") is None:
                self.recorder.finish(status, steps=len(self.steps))
            final = self.recorder.close(status, human=human, error=self.error)
            self._say(f"recorded to {final}")

    # -- state helpers ---------------------------------------------------------------
    def _enter(self, state: str, t: float) -> None:
        self._state, self._t0 = state, t

    def _enter_sub(self, state: str, t: float, part: Skill) -> None:
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

    def _event(self, kind: str, payload: Optional[dict] = None) -> None:
        if self.recorder is not None:
            self.recorder.event(kind, payload)

    def _pose_deg(self, xyyaw) -> list[float]:
        x, y, yaw = xyyaw
        return [float(x), float(y), math.degrees(float(yaw))]

    def _to_settle(self, t: float, obs: Obs) -> Action:
        self._enter("settle", t)
        self._seq0 = obs.frame.seq if obs.frame is not None else None
        self._settled = 0
        self._settle_err = math.inf
        self._settle_vel = math.inf
        return self._hold()

    def _to_handback(self, t: float, result: str, error: Optional[str] = None) -> Action:
        self.result = result
        self.error = error
        self._say(f"result: {result}" + (f" ({error})" if error else ""))
        if self.recorder is not None:
            self.recorder.finish(result, steps=len(self.steps), decisions=self._env_step)
        self._event("return_home", {"step": self._env_step, "trigger": result})
        self._enter_sub("handback", t, Handback())
        return self._hold()

    def _fail(self, t: float, error: str) -> Action:
        self._event("run_error", {"step": self._env_step, "error": error})
        return self._to_handback(t, "failed", error)

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

    def _record(self, status: str, t: float, obs: Obs, decision: Optional[dict],
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
            decision=decision,
            skill=None if skill is None else {"name": skill.name, "args": dict(skill.args),
                                              "duration": skill.duration, "chunk": self._chunk,
                                              "chunks": self._chunks, "needs_base": skill.needs_base},
            outcome={"status": status, "duration": duration},
        )
        self.steps.append(rec)
        if self.recorder is not None:
            self.recorder.write_step(rec, snap.get("image"))

    def _decision_json(self, d: Decision) -> dict:
        return {**d.to_json(), "model": getattr(self.decider, "model", ""), "attempt": self._attempt,
                "env_step": self._env_step}

    def _close_chunk(self, status: str, t: float, obs: Obs) -> None:
        self._record(status, t, obs, self._decision_json(self._decision), self._skill, t - self._chunk_t0)
        self._step += 1
        if status == "running":
            self._chunk += 1
            self._chunk_t0 = t
            self._snap = self._snapshot(t, obs)     # the next chunk starts here

    # -- the observation -----------------------------------------------------------------
    def _state_payload(self, obs: Obs) -> dict:
        return {"joint_pos": obs.q, "joint_vel": obs.qd, "joint_torque": obs.tau,
                "waist_yaw_deg": math.degrees(float(self._cmd[WAIST_YAW])),
                "base_pose_cmd": self._pose_deg(self._base_cmd),
                "base_pose_env": None if obs.base_pose is None else self._pose_deg(obs.base_pose)}

    def _observe(self, t: float, obs: Obs, frame) -> None:
        """Build the turn from this snapshot and ask the decider."""
        t0 = time.perf_counter()
        images_meta, images = [], {}
        if frame is not None:
            h, w = frame.image.shape[:2]
            images_meta = [{"name": "head", "width": int(w), "height": int(h),
                            "captured_age_s": float(obs.frame_age)}]
            images = {"head": frame.image}
        extra = {"env_step": self._env_step, "decisions_left": self.max_decisions - self._env_step,
                 "can_walk": self.can_walk}
        if self._attempt:
            extra["attempt"] = self._attempt
        text = observation(self.goal, self._state_payload(obs), images_meta, extra, self._previous)
        self._request_id += 1
        self._turn = AgentTurn(text, images, self.content if self._env_step == 0 and not self._attempt else (),
                               request_id=self._request_id, step=self._env_step,
                               frame_seq=None if frame is None else frame.seq,
                               frame_stamp=None if frame is None else frame.stamp)
        self._observe_prep = time.perf_counter() - t0
        self._event("observation", {"step": self._env_step, "input_json": text, "state": self._state_payload(obs),
                                    "images": [{**m, "path": f"step_{self._step:04d}.png"} for m in images_meta],
                                    **({"attempt": self._attempt} if self._attempt else {}),
                                    **({"content": _content_records(self._turn.content)} if self._turn.content else {})})
        self._wall0 = time.monotonic()
        if not self.decider.request(self._turn):
            self._say("decider busy; will retry")

    # -- feedback -------------------------------------------------------------------------
    def _settle_report(self) -> dict:
        return {"settled": self._settled >= self.settle_samples, "observed_s": self._settle_elapsed,
                "consecutive_samples": self._settled, "required_samples": self.settle_samples,
                "max_position_error_rad": self._settle_err, "max_velocity_rad_s": self._settle_vel,
                "position_tolerance_rad": self.settle_tol, "velocity_tolerance_rad_s": self.settle_vel_tol}

    def _feedback(self, obs: Obs) -> None:
        """The finished skill's execution result: full in the record, compact for the model."""
        p = self._pending
        self._pending = None
        residual = [abs(float(obs.q[j] - self._cmd[j])) if j in self.joints else None for j in range(len(obs.q))]
        measured = obs.base_pose if obs.base_pose is not None else self._base_cmd
        target = self._pose_deg(self._base_cmd)
        meas = self._pose_deg(measured)
        start_cmd = p["base_cmd_start"]
        start_meas = p["base_env_start"] if p["base_env_start"] is not None else start_cmd
        settle = self._settle_report()
        fb = {"joint_residual_rad": residual,
              "max_joint_residual_rad": max((r for r in residual if r is not None), default=0.0),
              "base_target_pose": target, "base_measured_pose": meas,
              "base_measured_source": "env" if obs.base_pose is not None else "cmd",
              "base_error": [a - b for a, b in zip(target, meas)],
              "motion_progress": {"requested_delta": [a - b for a, b in zip(target, self._pose_deg(start_cmd))],
                                  "achieved_delta": [a - b for a, b in zip(meas, self._pose_deg(start_meas))],
                                  "remaining_delta": [a - b for a, b in zip(target, meas)]},
              "settle": settle}
        result = {"status": "completed", "duration_s": p["duration"], "execution_feedback": fb}
        self._event("execution_result", {"step": p["env_step"], "name": p["name"], "result": result})
        # their compact model view: no motion_progress, no settle bookkeeping once settled
        view = {k: v for k, v in fb.items() if k != "motion_progress"}
        if settle["settled"]:
            view["settle"] = {k: v for k, v in settle.items()
                              if k not in ("consecutive_samples", "required_samples",
                                           "position_tolerance_rad", "velocity_tolerance_rad_s")}
        self._previous = {"tool": p["name"], "result": {**result, "execution_feedback": view}}

    def _reject(self, t: float, obs: Obs, name: Optional[str], error: str, raw: str = "",
                arguments: Optional[dict] = None, **details) -> Action:
        """A selection that ran nothing: feedback for the next turn, one decision spent."""
        self._say(f"decision {self._env_step}: {error}")
        err = {"tool": name, "error": error, **details}
        self._event("tool_error", {"step": self._env_step, **err})
        self._previous = err
        self._record("rejected", t, obs, {"name": name, "arguments": arguments or {}, "raw": raw, "error": error,
                                          "model": getattr(self.decider, "model", ""), "env_step": self._env_step},
                     None, 0.0)
        self._step += 1
        return self._spent(t, obs)

    def _spent(self, t: float, obs: Obs) -> Action:
        """One decision consumed; on to the next observation, or out of budget."""
        self._env_step += 1
        self._attempt = 0
        self._deadline = None
        if self._env_step >= self.max_decisions:
            self._event("budget_exhausted", {"decisions_used": self._env_step, "max_decisions": self.max_decisions})
            return self._to_handback(t, "budget_exhausted")
        return self._to_settle(t, obs)

    # -- the loop ----------------------------------------------------------------------
    def step(self, t: float, obs: Obs) -> Optional[Action]:
        if self.recorder is not None:
            self.recorder.state(t, obs.q, self._cmd, obs.qd, self._base_cmd)
        st = self._state
        if st in ("takeover", "handback"):
            a = self._sub.step(t - self._t0, obs)
            if a is not None:
                return self._emit(a)
            if st == "handback":
                return None
            return self._to_settle(t, obs)

        if st == "settle":
            err = float(np.max(np.abs(obs.q[self.joints] - self._cmd[self.joints])))
            if obs.qd is not None:
                vel = float(np.max(np.abs(obs.qd[self.joints])))
            else:
                vel = float(np.max(np.abs(obs.q[self.joints] - self._prev_q[self.joints])) / CONTROL_DT)
            self._prev_q = obs.q.copy()
            self._settle_err, self._settle_vel = err, vel
            self._settled = self._settled + 1 if (err <= self.settle_tol and vel <= self.settle_vel_tol) else 0
            elapsed = t - self._t0
            self._settle_elapsed = elapsed
            done = self._settled >= self.settle_samples and elapsed >= self.settle_min - EPS
            if done or elapsed >= self.settle_timeout - EPS:
                if self._pending is not None:
                    self._feedback(obs)
                self._enter("snapshot", t)
            return self._hold()

        if st == "snapshot":
            f = obs.frame
            if f is not None and f.seq != self._seq0 and obs.frame_age <= self.frame_max_age:
                self._failures = 0
                self._snap = self._snapshot(t, obs, frame=f)
                self._observe(t, obs, f)
                self._enter("think", t)
            elif t - self._t0 > self.frame_timeout:
                # their state_unavailable: observe anyway, say so, count the failures
                self._failures += 1
                self._say(f"no fresh camera frame ({self._failures}/{self.max_failures})")
                err = {"tool": "observe", "error": f"frame_unavailable: no fresh frame within {self.frame_timeout}s",
                       "recovering": True}
                self._event("state_observation_error", {"step": self._env_step, **err})
                if self._failures >= self.max_failures:
                    self._record("no_frame", t, obs, None, None, t - self._t0)
                    self._step += 1
                    return self._fail(t, "no_frame")
                self._previous = err
                self._snap = self._snapshot(t, obs, frame=None)
                self._observe(t, obs, None)
                self._enter("think", t)
            return self._hold()

        if st == "think":
            d = self.decider.latest()
            if d is not None and d.request_id == self._request_id:
                self._snap["think_wall"] = time.monotonic() - self._wall0
                self._timing("completed")
                return self._decide(t, obs, d)
            if not self.decider.pending:
                self._timing("failed")
                return self._model_error(t, obs, self.decider.last_error)
            if time.monotonic() - self._wall0 > self.step_timeout:
                self._timing("timeout")
                return self._fail(t, f"model timeout after {self.step_timeout}s")
            return self._hold()

        if st == "wait":
            if time.monotonic() >= self._wait_until:
                self._seq0 = obs.frame.seq if obs.frame is not None else None
                self._enter("snapshot", t)
            return self._hold()

        if st == "act":
            a = self._sub.step(t - self._t0, obs)
            if a is not None:
                if t - self._chunk_t0 >= STEP_MAX - EPS:
                    # close this chunk and start the next; the motion itself is unbroken
                    self._close_chunk("running", t, obs)
                return self._emit(a)
            self._close_chunk("completed", t, obs)
            self._event("tool_timing", {"step": self._env_step, "name": self._skill.name,
                                        "execute_s": t - self._act_t0, "status": "completed"})
            self._pending = {"name": self._skill.name, "env_step": self._env_step, "duration": t - self._act_t0,
                             "base_cmd_start": self._act_base_cmd, "base_env_start": self._act_base_env}
            return self._spent(t, obs)

        raise RuntimeError(f"unknown state {st!r}")

    def _timing(self, status: str) -> None:
        self._event("decision_timing", {"step": self._env_step, "observation_prepare_s": self._observe_prep,
                                        "agent_decide_s": time.monotonic() - self._wall0, "status": status,
                                        **({"attempt": self._attempt} if self._attempt else {})})
        if self.recorder is not None and self.decider.last_call is not None:
            self.recorder.usage(self.decider.last_call)

    def _model_error(self, t: float, obs: Obs, e: Optional[BaseException]) -> Action:
        if isinstance(e, ProtocolError):
            return self._reject(t, obs, None, f"invalid_selection: {e}", raw=e.raw)
        if isinstance(e, Overloaded):
            if self._deadline is None:
                self._deadline = time.monotonic() + self.recovery_timeout
            if self._attempt >= len(self.retry_delays) or time.monotonic() >= self._deadline:
                reason = "retry_limit" if self._attempt >= len(self.retry_delays) else "recovery_deadline"
                self._event("model_retry_exhausted", {"step": self._env_step, "retries": self._attempt,
                                                      "reason": reason, "error": str(e)})
                return self._fail(t, f"model overloaded: {reason}")
            delay = self.retry_delays[self._attempt] + random.uniform(0, 0.5)
            self._attempt += 1
            self._event("model_retry", {"step": self._env_step, "retry": self._attempt,
                                        "max_retries": len(self.retry_delays), "delay_s": delay,
                                        "recovery_timeout_s": self.recovery_timeout, "error": str(e)})
            self._say(f"model overloaded; retry {self._attempt} in {delay:.1f}s (no action is replayed)")
            self._wait_until = time.monotonic() + delay
            self._enter("wait", t)
            return self._hold()
        if isinstance(e, QuotaExceeded):
            return self._fail(t, f"model quota exhausted: {e}")
        return self._fail(t, f"model error: {e}")

    def _decide(self, t: float, obs: Obs, d: Decision) -> Action:
        self._event("model_decision", {"step": self._env_step, "decision": d.wire if d.wire is not None
                                       else {"name": d.name, "arguments": d.arguments},
                                       "latency_s": d.latency, "notes": d.notes})
        args = {k: v for k, v in d.arguments.items() if k != "note"}
        self._say(f"decision {self._env_step}: {d.name}({args}) — {d.note}")
        try:
            skill = self.skills[d.name](**d.arguments)
        except (KeyError, ValueError) as e:
            return self._reject(t, obs, d.name, f"tool_rejected: {e}", raw=d.raw, arguments=d.arguments)
        self._chunk, self._chunks = 1, 1
        if skill.terminal:
            self._event("terminal", {"step": self._env_step, "name": d.name, "arguments": d.arguments})
            self._record("done" if d.name == "done" else d.name, t, obs, self._decision_json(d), skill, 0.0)
            self._step += 1
            self._env_step += 1
            return self._to_handback(t, "give_up" if d.name == "give_up" else "completed")
        if isinstance(skill, Check):
            try:
                verdict = dry_run(skill.target(), self._cmd, self.joints)
            except ValueError as e:
                return self._reject(t, obs, d.name, f"tool_rejected: {e}", raw=d.raw, arguments=d.arguments)
            self._event("execution_result", {"step": self._env_step, "name": d.name, "result": verdict})
            self._previous = {"tool": d.name, "result": verdict}
            self._record("checked", t, obs, self._decision_json(d), skill, 0.0)
            self._step += 1
            return self._spent(t, obs)
        if skill.needs_base and not self.can_walk:
            return self._reject(t, obs, d.name, "tool_rejected: the base cannot be driven in this run",
                                raw=d.raw, arguments=d.arguments)
        if skill.needs_loco and not self.has_loco:
            return self._reject(t, obs, d.name, "tool_rejected: no onboard controller for gestures in this env",
                                raw=d.raw, arguments=d.arguments)
        # their IK check before submission: plan it from the commanded pose, reject on any violation
        verdict = dry_run(skill, self._cmd, self.joints)
        if verdict["status"] != "ok":
            return self._reject(t, obs, d.name, "motion_not_executed", raw=d.raw, arguments=d.arguments,
                                violations=verdict["violations"], n_violations=verdict["n_violations"],
                                requested_motion={k: v for k, v in d.arguments.items() if k != "note"})
        self._decision, self._skill = d, skill
        self._chunks = max(1, math.ceil(skill.duration / STEP_MAX - 1e-9))
        self._chunk_t0 = t
        self._act_t0 = t
        self._act_base_cmd = self._base_cmd.copy()
        self._act_base_env = None if obs.base_pose is None else tuple(obs.base_pose)
        self._enter("act", t)
        self._sub = skill_policy(skill, self.joints, self.name)
        self._sub.reset(Obs(self._cmd))
        return self.step(t, obs)


def _content_records(parts) -> list:
    from demo import content_records
    return content_records(parts)


# --------------------------------------------------------------------------
# CLI builders
# --------------------------------------------------------------------------

def ask_verdict() -> Optional[str]:
    """Their terminal prompt: success or failed, nothing else; EOF/Ctrl-C skips."""
    import sys
    if not sys.stdin.isatty():
        return None
    while True:
        answer = input("Task result [s success / f failed]: ").strip().lower()
        if answer in ("s", "success"):
            return "success"
        if answer in ("f", "failed", "fail"):
            return "failed"


def build_search(args, can_walk: bool, has_loco: bool = False) -> Agent:
    from demo import DEFAULT_FRAMES, MAX_FRAMES, VideoPart, build_request, build_selector, prepare, save_input
    if not getattr(args, "goal", None) and not getattr(args, "input_json", None):
        raise ValueError("search needs --goal, e.g. --goal \"find the mug\" (or --input-json)")
    request = build_request(getattr(args, "goal", None), manifest=getattr(args, "input_json", None),
                            demo=getattr(args, "demo", None), mode=getattr(args, "demo_mode", None),
                            refs=getattr(args, "ref", None) or [])
    args.goal = request.instruction
    if getattr(args, "skills", None):
        use_catalog(args.skills)
    echo = (lambda s: print(s, end="", flush=True)) if getattr(args, "vision_echo", False) else None
    decider = VLMDecider.from_env(getattr(args, "vision_model", None), provider=getattr(args, "vision_provider", None),
                                  on_text=echo,
                                 live_image_window=getattr(args, "live_image_window", 8),
                                 fresh_turns=getattr(args, "fresh_turns", False))
    skills = menu(can_walk, has_loco)
    from scene import safety_notes
    notes = safety_notes(getattr(args, "scene", None)) + list(getattr(args, "safety_note", None) or [])
    recorder = None
    if not getattr(args, "no_log", False):
        recorder = EpisodeWriter(getattr(args, "log", "runs"), env=args.env, goal=args.goal, model=decider.model,
                                 skills=[s.name for s in skills], allow_base=can_walk,
                                 extra={"max_decisions": args.max_decisions, "step_timeout": args.step_timeout,
                                        "live_image_window": decider.live_image_window,
                                        "fresh_turns": decider.fresh_turns, "safety_notes": notes,
                                        "request": request.record()})
        print(f"recording to {recorder.dir}")
    # demonstrations and reference images: compiled before any device opens, archived with the run
    content = ()
    if request.content:
        import tempfile
        frames = max(1, min(MAX_FRAMES, getattr(args, "demo_frames", DEFAULT_FRAMES)))
        selector = None
        if any(isinstance(p, VideoPart) for p in request.content):
            selector = build_selector(getattr(args, "demo_select", "auto"), frames, getattr(args, "vision_model", None),
                                      getattr(args, "vision_provider", None))
        where = recorder.dir / "input" if recorder is not None else Path(tempfile.mkdtemp(prefix="g1-demo-"))
        prepared, reports = prepare(request, where, selector=selector, max_frames=frames)
        content = prepared.content
        for r in reports:
            print(f"demonstration {r['source']}: {r['keyframes']} keyframe(s), {r['mode']}; {r['summary'][:160]}")
        if recorder is not None:
            save_input(prepared, where)
            recorder.event("input_manifest", {**prepared.record(), "reports": reports})
            if selector is not None:
                for call in getattr(selector, "calls", []):
                    recorder.usage(call)
    verdict = None if getattr(args, "no_verdict", False) else ask_verdict
    return Agent(args.goal, decider, skills, recorder=recorder, max_decisions=args.max_decisions,
                 step_timeout=args.step_timeout, can_walk=can_walk, has_loco=has_loco, safety_notes=notes,
                 verdict=verdict, content=content)


def build_replay(args, can_walk: bool, has_loco: bool = False) -> Policy:
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
