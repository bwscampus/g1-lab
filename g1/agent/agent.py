"""The decision step — GPT-Policy's loop on our executor.

    observe   a fresh frame and the measured joint state, standing still
    decide    ask the decider (a background request; the loop never waits)
    act       run the chosen tool to its end, then wait until the joints
              have measurably settled
    feed back what happened (residuals, base error, the settle report) as
              ``previous_result`` in the next observation; a rejected
              selection is fed back the same way and costs a decision
    record    one step file per query to the model: the frame, the query as sent,
              the response, the action and its measured result (``episode.py``)

A step is one query, so a retry after an overload is a step of its own and a
12 s ``sixseven`` is one step; what the joints did tick by tick is trace.npz,
rows ``tick_start`` to ``tick_end`` (one row per control tick, ``t / dt``).

States: takeover -> settle -> snapshot -> think -> act -> settle -> ... ->
handback, plus ``wait`` for the backoff before a model retry (each retry
re-observes; no robot action is ever replayed). ``done`` / ``give_up`` end
the run with the model's conclusion; the *label* is the human's, asked at
close. Every tool is seeded from the agent's own last *commanded* q, never
the measured one (the same rule as a Chain). ``build_agent`` makes one for a
task with an instruction.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from g1.agent.decider import (AgentContext, AgentTurn, Decider, Decision, ProtocolError, VLMDecider, build_context,
                              observation)
from g1.agent.episode import EpisodeWriter
from g1.core import limits
from g1.core.action import EPS, Action, Obs, Runnable
from g1.core.images import sent_size
from g1.core.config import CONTROL_DT, LOCO_FSM, UPPER_BODY, joint_index
from g1.tools import Tool, menu, prefixed
from g1.tools.control import Handback, Takeover, dry_run
from g1.vlm import IMAGE_WIDTH, Overloaded, QuotaExceeded

WAIST_YAW = joint_index("waist_yaw")

# their MODEL_RETRY_DELAYS_S / MODEL_RECOVERY_TIMEOUT_S; the numbers are in configs/limits.json
RETRY_DELAYS = tuple(min(limits.get("retry_delay_base_s") * (2 ** k), limits.get("retry_delay_cap_s"))
                     for k in range(int(limits.get("retry_max"))))
RECOVERY_TIMEOUT = limits.get("recovery_timeout_s")


class Agent(Runnable):
    joints = sorted(UPPER_BODY)
    uses_camera = True

    def __init__(self, instruction: str, decider: Decider, tools: Sequence[type[Tool]], *,
                 recorder: Optional[EpisodeWriter] = None,
                 max_decisions: int = limits.get("max_decisions"),
                 step_timeout: float = limits.get("step_timeout_s"),
                 settle_min: float = limits.get("settle_min_s"),
                 settle_tol: float = limits.get("settle_pos_tol_rad"),
                 settle_vel_tol: float = limits.get("settle_vel_tol"),
                 settle_samples: int = limits.get("settle_samples"),
                 settle_timeout: float = limits.get("settle_timeout_s"),
                 frame_max_age: float = limits.get("frame_max_age_s"),
                 frame_timeout: float = limits.get("frame_timeout_s"),
                 max_failures: int = limits.get("max_failures"),
                 can_walk: bool = True, has_loco: bool = False, safety_notes: Sequence[str] = (),
                 verdict: Optional[Callable[[], Optional[str]]] = None, name: str = "search",
                 retry_delays: Sequence[float] = RETRY_DELAYS, recovery_timeout: float = RECOVERY_TIMEOUT) -> None:
        self.instruction = instruction
        self.decider = decider
        self.tools = {t.name: t for t in tools}
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
        self.verdict = verdict
        self.name = name
        self.retry_delays = tuple(retry_delays)
        self.recovery_timeout = recovery_timeout
        self.context: AgentContext = build_context(list(tools), can_walk=can_walk, max_decisions=max_decisions,
                                                   safety_notes=self.safety_notes)
        self.result: Optional[str] = None         # runtime status: completed | give_up | budget_exhausted | failed
        self.error: Optional[str] = None
        self.end_reason: Optional[str] = None     # why it ended, from a fixed set (summary.json)
        self.steps: list[dict] = []               # the step records, as written
        self._closed = False
        self._finalized = False

    # -- lifecycle -------------------------------------------------------------
    def reset(self, obs: Obs) -> None:
        self._cmd = obs.q.copy()
        self._prev_q = obs.q.copy()
        self.result = None
        self.error = None
        self.end_reason = None
        self.steps = []
        self._rec: Optional[dict] = None          # the step being recorded (the latest query)
        self._calls: list[dict] = []
        self._interrupt: Optional[dict] = None
        self._t = 0.0
        self._env_step = 0                        # decision index: their env_step
        self._failures = 0
        self._attempt = 0
        self._deadline: Optional[float] = None
        self._request_id = 0
        self._view: Optional[dict] = None     # what the robot measured when the latest image was taken
        self._previous: Optional[dict] = None     # the next observation's previous_result
        self._pending: Optional[dict] = None      # a finished tool awaiting its settle report
        self._base_cmd = np.zeros(3)              # dead-reckoned (x, y, yaw) from commanded velocity
        self._seq0: Optional[int] = None
        self._sub: Optional[Tool] = None
        self._state = ""
        self._t0 = 0.0
        self._turn: Optional[AgentTurn] = None
        self._wall0 = 0.0
        self._wait_until = 0.0
        self._seated = False                      # a sit finished: nothing may walk or call onboard again
        self.decider.start(self.context)
        self._enter_sub("takeover", 0.0, Takeover())

    def on_interrupt(self, reason: str, detail: str = "") -> None:
        """``reason``: ctrl_c | max_time | error (``cli._return_to_safety``)."""
        self.error = self.error or (detail or reason)
        self.end_reason = {"ctrl_c": "interrupted"}.get(reason, reason)
        self._interrupt = {"trigger": reason, "detail": detail, "decision": self._env_step, "safe_return": None}
        if self.recorder is not None:
            self.recorder.update(interrupt=self._interrupt, end_reason=self.end_reason)

    def on_returned(self, outcome: str) -> None:
        if self._interrupt is not None:
            self._interrupt = {**self._interrupt, "safe_return": outcome}
            if self.recorder is not None:
                self.recorder.update(interrupt=self._interrupt)

    def close(self) -> None:
        """Stop the model, ask the human, settle the record. Also the Ctrl-C path."""
        if self._closed:
            return
        self._closed = True
        self.decider.stop()
        status = self.result or "interrupted"
        human = None
        if self.recorder is not None and self.verdict is not None:
            try:
                human = self.verdict()
            except (EOFError, KeyboardInterrupt):
                human = None
        if self.recorder is not None:
            self._record_sdk()
            final = self.recorder.close(status, human=human, error=self.error,
                                        end_reason=self.end_reason or "interrupted")
            calls = self._calls
            if calls:
                secs = sorted(float(c.get("elapsed_s", 0.0)) for c in calls)
                reasoning = [((c.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                             for c in calls]
                self._say(f"model: {len(calls)} calls, thinking {secs[len(secs) // 2]:.1f} s median / "
                          f"{secs[-1]:.1f} s max, reasoning tokens up to {max(reasoning)}")
            self._say(f"recorded to {final}")

    # -- state helpers ---------------------------------------------------------------
    def _enter(self, state: str, t: float) -> None:
        self._state, self._t0 = state, t

    def _enter_sub(self, state: str, t: float, part: Tool) -> None:
        self._enter(state, t)
        self._sub = Tool.of(prefixed(part), joints=self.joints, name=self.name, allows_start=True)
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

    def attach(self, env) -> None:
        """The env whose SDK calls the record should carry (set by the runner)."""
        self._env = env
        self._sdk_seen = 0

    def _record_sdk(self) -> None:
        """New SDK calls go to the step they happened in (its action and settle); before the
        first query and during the hand-back, to summary.json."""
        env = getattr(self, "_env", None)
        log = getattr(env, "sdk_log", None)
        if not log or self.recorder is None:
            return
        new = [dict(e) for e in log[self._sdk_seen:]]
        self._sdk_seen = len(log)
        if not new:
            return
        if self._rec is not None and self._state not in ("takeover", "handback"):
            self._rec["sdk_calls"].extend(new)
            self._write()
        else:
            self.recorder.update(sdk_outside_steps=self.recorder.summary["sdk_outside_steps"] + new)

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

    def _to_handback(self, t: float, result: str, error: Optional[str] = None,
                     reason: Optional[str] = None) -> Action:
        self.result = result
        self.error = error
        self.end_reason = reason or {"completed": "done"}.get(result, result)
        if self._rec is not None and self._rec["t_end"] is None:
            self._finish(t)                      # a step that ends the run (budget out after its action)
        self._say(f"result: {result}" + (f" ({error})" if error else ""))
        if self.recorder is not None:
            self.recorder.update(end_reason=self.end_reason, error=error)
        self._enter_sub("handback", t, Handback())
        return self._hold()

    def _fail(self, t: float, error: str, reason: str) -> Action:
        """``reason``: no_frame | step_timeout | quota | model_overloaded | model_error."""
        return self._to_handback(t, "failed", error, reason)

    # -- the record ---------------------------------------------------------------------
    def _write(self, image=None) -> None:
        if self.recorder is not None and self._rec is not None:
            self.recorder.step(self._rec, image)

    def _open(self, t: float, obs: Obs, frame, sent: Optional[tuple]) -> None:
        """A new step: one query, numbered as its request (step_NNNN is request NNNN)."""
        self._rec = {
            "step": self._request_id, "decision": self._env_step, "attempt": self._attempt,
            "t_start": t, "t_end": None, "tick_start": self._tick(t), "tick_end": None,
            "wall_start": time.time(), "wall_end": None,
            "frame": None if frame is None else {
                "image": f"step_{self._request_id:04d}.png", "seq": frame.seq, "stamp": frame.stamp,
                "age_s": float(obs.frame_age), "shape": list(frame.image.shape),
                "sent": {"width": sent[0], "height": sent[1], "jpeg_quality": limits.get("jpeg_quality")}},
            "frame_error": None if frame is not None else (self._previous or {}).get("error"),
            "query": None, "response": None, "action": None, "result": None, "sdk_calls": []}
        self.steps.append(self._rec)
        self._write(None if frame is None else frame.image)

    def _finish(self, t: float, result: Optional[dict] = None) -> None:
        """The step's end, with its result: what the host reports about it, in full."""
        if result is not None:
            self._rec["result"] = result
        self._rec.update(t_end=t, tick_end=self._tick(t), wall_end=time.time())
        self._write()

    def _tick(self, t: float) -> int:
        """Run time as a tick: the row of trace.npz written for it (one row per control tick)."""
        return int(round(t / self.dt))

    def _decision_used(self, tool_name: Optional[str]) -> None:
        if self.recorder is not None:
            s = self.recorder.summary
            self.recorder.update(decisions={**s["decisions"], "used": self._env_step + 1},
                                 tools_used=s["tools_used"] + [tool_name or "(invalid)"])

    # -- the observation -----------------------------------------------------------------
    def _state_payload(self, obs: Obs) -> dict:
        """What the robot knows about itself: measured joints and its own commands. Never the
        simulator's truth (obs.base_pose), which only the record keeps."""
        return {"joint_pos": obs.q, "joint_vel": obs.qd, "joint_torque": obs.tau,
                "waist_yaw_deg": math.degrees(float(self._cmd[WAIST_YAW])),
                "base_pose_cmd": self._pose_deg(self._base_cmd)}

    def _observe(self, t: float, obs: Obs, frame) -> None:
        """Build the turn from this snapshot and ask the decider."""
        t0 = time.perf_counter()
        images_meta, images, w, h = [], {}, 0, 0
        self._view = {"frame_size": None, "q": obs.q.copy(), "imu_quat": obs.imu_quat}   # as measured now
        if frame is not None:
            # the size the model sees: pixels it names (locate_point) are in this image
            w, h = sent_size(frame.image.shape[1], frame.image.shape[0], getattr(self.decider, "max_width", IMAGE_WIDTH))
            self._view["frame_size"] = (w, h)
            images_meta = [{"name": "head", "width": int(w), "height": int(h),
                            "captured_age_s": float(obs.frame_age)}]
            images = {"head": frame.image}
        extra = {"env_step": self._env_step, "decisions_left": self.max_decisions - self._env_step,
                 "can_walk": self.can_walk}
        if self._attempt:
            extra["attempt"] = self._attempt
        text = observation(self.instruction, self._state_payload(obs), images_meta, extra, self._previous)
        self._request_id += 1
        self._turn = AgentTurn(text, images, request_id=self._request_id, step=self._env_step,
                               frame_seq=None if frame is None else frame.seq,
                               frame_stamp=None if frame is None else frame.stamp)
        self._observe_prep = time.perf_counter() - t0
        self._open(t, obs, frame, None if frame is None else (int(w), int(h)))
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
        """The finished tool's execution result: full in the record, compact for the model."""
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
        self._finish(self._t, result)
        # their compact model view: no motion_progress, no settle bookkeeping once settled, and only what
        # the robot can measure: the base as commanded, never the simulator's truth
        view = {k: v for k, v in fb.items()
                if k not in ("motion_progress", "base_measured_pose", "base_measured_source", "base_error")}
        view["base_commanded_delta"] = fb["motion_progress"]["requested_delta"]
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
        self._previous = err
        self._finish(t, {"status": "rejected", **err})
        self._decision_used(name)
        return self._spent(t, obs)

    def _spent(self, t: float, obs: Obs) -> Action:
        """One decision consumed; on to the next observation, or out of budget."""
        self._env_step += 1
        self._attempt = 0
        self._deadline = None
        if self._env_step >= self.max_decisions:
            return self._to_handback(t, "budget_exhausted")
        return self._to_settle(t, obs)

    # -- the loop ----------------------------------------------------------------------
    def step(self, t: float, obs: Obs) -> Optional[Action]:
        self._t = t
        self._record_sdk()
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
                self._observe(t, obs, f)
                self._enter("think", t)
            elif t - self._t0 > self.frame_timeout:
                # their state_unavailable: observe anyway, say so, count the failures
                self._failures += 1
                self._say(f"no fresh camera frame ({self._failures}/{self.max_failures})")
                err = {"tool": "observe", "error": f"frame_unavailable: no fresh frame within {self.frame_timeout}s",
                       "recovering": True}
                if self._failures >= self.max_failures:
                    return self._fail(t, "no_frame", "no_frame")
                self._previous = err
                self._observe(t, obs, None)
                self._enter("think", t)
            return self._hold()

        if st == "think":
            d = self.decider.latest()
            if d is not None and d.request_id == self._request_id:
                self._response("completed")
                return self._decide(t, obs, d)
            if not self.decider.pending:
                self._response("failed")
                return self._model_error(t, obs, self.decider.last_error)
            if time.monotonic() - self._wall0 > self.step_timeout:
                self._response("timeout")
                self._finish(t, {"status": "timeout"})
                return self._fail(t, f"model timeout after {self.step_timeout}s", "step_timeout")
            return self._hold()

        if st == "wait":
            if time.monotonic() >= self._wait_until:
                self._seq0 = obs.frame.seq if obs.frame is not None else None
                self._enter("snapshot", t)
            return self._hold()

        if st == "act":
            a = self._sub.step(t - self._t0, obs)
            if a is not None:
                return self._emit(a)
            self._seated = self._seated or self._tool.command in LOCO_FSM
            self._rec["action"].update(t_end=t, tick_end=self._tick(t), duration_s=t - self._act_t0,
                                       q_end=[float(v) for v in obs.q], cmd_end=[float(v) for v in self._cmd],
                                       base_cmd_end=[float(v) for v in self._base_cmd])
            self._write()
            self._decision_used(self._tool.name)
            self._pending = {"name": self._tool.name, "env_step": self._env_step, "duration": t - self._act_t0,
                             "base_cmd_start": self._act_base_cmd, "base_env_start": self._act_base_env}
            return self._spent(t, obs)

        raise RuntimeError(f"unknown state {st!r}")

    def _response(self, status: str) -> None:
        """The query as sent and everything that came back for it, into the step."""
        call = self.decider.last_call
        # on a timeout the request is still out and last_call is the one before it
        call = dict(call) if call is not None and call.get("request_id") == self._request_id else None
        self._rec["query"] = self.decider.queries.pop(self._request_id, None)
        self._rec["response"] = {"status": status, "observation_prepare_s": self._observe_prep,
                                 "decide_s": time.monotonic() - self._wall0,
                                 **({k: v for k, v in call.items() if k != "request_id"} if call else {})}
        if call is not None:
            self._calls.append(call)
            if self.recorder is not None:
                self.recorder.model_call(call)
        self._write()

    def _model_error(self, t: float, obs: Obs, e: Optional[BaseException]) -> Action:
        self._rec["response"]["error"] = f"{type(e).__name__}: {e}" if e is not None else "no reply"
        if isinstance(e, ProtocolError):
            self._rec["response"]["text"] = e.raw
            return self._reject(t, obs, None, f"invalid_selection: {e}", raw=e.raw)
        if isinstance(e, Overloaded):
            if self._deadline is None:
                self._deadline = time.monotonic() + self.recovery_timeout
            if self._attempt >= len(self.retry_delays) or time.monotonic() >= self._deadline:
                reason = "retry_limit" if self._attempt >= len(self.retry_delays) else "recovery_deadline"
                self._finish(t, {"status": "retry_exhausted", "retries": self._attempt, "reason": reason})
                return self._fail(t, f"model overloaded: {reason}", "model_overloaded")
            delay = self.retry_delays[self._attempt] + random.uniform(0, limits.get("retry_jitter_s"))
            self._attempt += 1
            self._finish(t, {"status": "retry", "retry": self._attempt, "max_retries": len(self.retry_delays),
                             "delay_s": delay, "recovery_timeout_s": self.recovery_timeout})
            self._say(f"model overloaded; retry {self._attempt} in {delay:.1f}s (no action is replayed)")
            self._wait_until = time.monotonic() + delay
            self._enter("wait", t)
            return self._hold()
        if isinstance(e, QuotaExceeded):
            self._finish(t)
            return self._fail(t, f"model quota exhausted: {e}", "quota")
        self._finish(t)
        return self._fail(t, f"model error: {e}", "model_error")

    def _decide(self, t: float, obs: Obs, d: Decision) -> Action:
        self._rec["response"]["decision"] = {"name": d.name, "arguments": d.arguments, "notes": list(d.notes)}
        self._rec["response"]["text"] = d.raw
        args = {k: v for k, v in d.arguments.items() if k != "note"}
        self._say(f"decision {self._env_step}: {d.name}({args}) — {d.note}")
        try:
            tool = self.tools[d.name](**d.arguments)
        except (KeyError, ValueError) as e:
            return self._reject(t, obs, d.name, f"tool_rejected: {e}", raw=d.raw, arguments=d.arguments)
        if tool.kind == "terminal":
            self._finish(t, {"status": d.name})
            self._decision_used(d.name)
            if self.recorder is not None:
                self.recorder.update(closing={k: v for k, v in d.arguments.items() if k != "note"})
            self._env_step += 1
            return self._to_handback(t, "give_up" if d.name == "give_up" else "completed")
        if tool.kind == "query":
            try:
                verdict = tool.query(self._cmd, self.joints, view=self._view)
            except ValueError as e:
                return self._reject(t, obs, d.name, f"tool_rejected: {e}", raw=d.raw, arguments=d.arguments)
            self._previous = {"tool": d.name, "result": verdict}
            self._finish(t, verdict)
            self._decision_used(d.name)
            return self._spent(t, obs)
        if tool.needs_base and not self.can_walk:
            return self._reject(t, obs, d.name, "tool_rejected: the base cannot be driven in this run",
                                raw=d.raw, arguments=d.arguments)
        if tool.needs_loco and not self.has_loco:
            return self._reject(t, obs, d.name, "tool_rejected: no onboard controller for gestures in this env",
                                raw=d.raw, arguments=d.arguments)
        if self._seated and (tool.needs_base or tool.needs_loco):
            return self._reject(t, obs, d.name, "tool_rejected: the robot is seated; it cannot walk, gesture or "
                                "sit again (end with done)", raw=d.raw, arguments=d.arguments)
        # their IK check before submission: plan it from the commanded pose, reject on any violation
        verdict = dry_run(tool, self._cmd, self.joints)
        if verdict["status"] != "ok":
            return self._reject(t, obs, d.name, "motion_not_executed", raw=d.raw, arguments=d.arguments,
                                violations=verdict["violations"], n_violations=verdict["n_violations"],
                                requested_motion={k: v for k, v in d.arguments.items() if k != "note"})
        self._decision, self._tool = d, tool
        self._rec["action"] = {"tool": tool.name, "args": dict(tool.args), "dry_run": verdict,
                               "planned_s": tool.duration, "needs_base": tool.needs_base,
                               "t_start": t, "t_end": None, "tick_start": self._tick(t), "tick_end": None,
                               "duration_s": None,
                               "q_start": [float(v) for v in obs.q], "cmd_start": [float(v) for v in self._cmd],
                               "base_cmd_start": [float(v) for v in self._base_cmd]}
        self._write()
        self._act_t0 = t
        self._act_base_cmd = self._base_cmd.copy()
        self._act_base_env = None if obs.base_pose is None else tuple(obs.base_pose)
        self._enter("act", t)
        self._sub = Tool.of(prefixed(tool), joints=self.joints, name=self.name)
        self._sub.reset(Obs(self._cmd))
        return self.step(t, obs)


# --------------------------------------------------------------------------
# CLI builders
# --------------------------------------------------------------------------

def ask_save() -> bool:
    """After a robot run: keep it? Without a terminal it is kept (unreviewed); EOF/Ctrl-C keeps it."""
    import sys
    if not sys.stdin.isatty():
        return True
    while True:
        try:
            answer = input("Save this run? [y / n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return True
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def ask_verdict() -> Optional[str]:
    """Their terminal prompt: success or failed, nothing else; EOF/Ctrl-C skips."""
    import sys
    if not sys.stdin.isatty():
        return None
    while True:
        try:
            answer = input("Task result [s success / f failed]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if answer in ("s", "success"):
            return "success"
        if answer in ("f", "failed", "fail"):
            return "failed"


def build_agent(task: dict, *, env_name: str, can_walk: bool, has_loco: bool, runs: Path | str,
                mode: Optional[str] = None, config: Optional[dict] = None) -> Agent:
    """The agent for a task with an instruction: the vision model (limits vlm_provider, vlm_model / openrouter_model), the menu
    this env allows, the scene's and the task's safety notes, recorded under ``runs``; ``config``
    is the run's setup (``task.run_config``), completed here with the model, tools and prompt.
    It asks nothing at the end: ``g1 run`` asks, on the robot only (save it? then the verdict)."""
    decider = VLMDecider.from_env()
    tools = menu(can_walk, has_loco)
    notes = list(task["safety_notes"])          # rules only: the room's layout is never told
    agent = Agent(task["instruction"], decider, tools, max_decisions=task["max_decisions"],
                  step_timeout=limits.get("step_timeout_s"), can_walk=can_walk, has_loco=has_loco,
                  safety_notes=notes, verdict=None)
    config = {**(config or {}), "model": decider.settings(), **agent.context.record()}
    agent.recorder = EpisodeWriter(runs, env=env_name, instruction=task["instruction"], model=decider.model,
                                   config=config, task=task["name"], mode=mode, max_decisions=task["max_decisions"])
    print(f"recording to {agent.recorder.dir}")
    return agent
