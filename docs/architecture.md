# Architecture

## Vocabulary (GPT-Policy's)

| word | meaning here |
|---|---|
| **tool** | what the model picks, one per turn, and the host executes: a fixed motion (or a query) computed from its arguments, checked, executed, settled and reported. One class: name, prompt, `params`, `segments()`. Ends by construction. Visible to the model by default; chains in a task's `tools`. |
| **instruction** | the goal text the model is given (`task.json`'s `instruction`). |
| **task** | a folder with an instruction (or a fixed `tools` chain) and its context (scene, safety notes, budget); no code. What `g1 run` runs. |
| **observation** | the one JSON object a turn sends: instruction, images, measured state, extra, `previous_result`. |
| **decision** | the model's reply: `{"name": tool, "arguments": {...}}`, validated against the tool's schema. |
| **execution result** | what the host reports after a tool: residuals, the commanded base displacement, a settle report — or why nothing ran. |
| **verdict** | the success/failed label after the run: a human's on the robot, the twin's on `--sim` when the task has `success` criteria (`outcome_source` says which). `done` is the model's conclusion, not the label; `false_done` marks a `done` the twin did not confirm. |

Retired words: skill, policy, primitive, routine, goal. Nothing in the repo is called a policy.

## The loop

Everything the envs run is a `Runnable` (`g1/core/action.py`): `reset(obs)` once, then
`step(t, obs) -> Action | None` at 50 Hz (`CONTROL_DT`). Three things are runnable:

* a **Tool** — plays its `segments()` from the pose it was seeded with
* a **Chain** of tools (`g1/tools/__init__.py`) — the takeover bookend, the tools with a
  pause between, the handback bookend, as one continuous command stream. The bookends only
  blend the arm_sdk weight (0→1 holding the pose the arms are observed in; 1→0 holding the
  pose the last tool ended in): **nothing moves the arms to a neutral pose**, at the start,
  at the end, or as a side effect of another tool. A tool touches only the joints it is
  about and leaves them where it ends; one that needs a starting pose makes it its own
  first segment (`tpose` raises the arms itself).
* the **Agent** (`g1/agent/agent.py`) — the decision loop below, which runs one tool at a time

```
with env: obs = env.observe(env.reset()); program.reset(obs)
          loop: action = program.step(t, obs); obs = env.observe(env.step(action))
env.report()
```

`Action` is the full 29-vector `q` (only `joints` entries matter), the arm_sdk `weight`
(1 = the program owns the joints, 0 = the onboard controller does), an optional base
velocity `(vx, vy, vyaw)` and an optional one-shot onboard `command`. `Obs` is the measured
`q`, the latest camera `frame` (latest-only: the same `seq` until a newer one lands) with
its `frame_age` on the env's clock, the env's base pose estimate and, where measured, joint
velocity and torque.

## The decision step (GPT-Policy's, on this executor)

```
takeover -> settle -> snapshot -> think -> act -> settle -> snapshot -> ... -> handback
```

* **settle**: hold until the measured joints are still (0.03 rad from the command, 0.05
  rad/s, ten consecutive ticks, at least 0.5 s so `StopMove` has landed, 3 s timeout) and
  report it.
* **snapshot**: a fresh frame (younger than `frame_max_age_s`; after `frame_timeout_s` the
  model is observed without one and told so; `max_failures` of those end the run).
* **think**: the observation JSON + the image go to the decider on a background thread;
  the loop holds still and never waits. An overloaded model is retried with backoff,
  re-observing every time; a quota error or `step_timeout_s` fails the run.
* **act**: the decision (the model's one tool call) is parsed (jsonschema against the tool's
  parameters, numbers clamped with a note), the tool is built, **dry-run through the joint
  monitor from the commanded pose** (a limit, speed or base violation comes back as
  `motion_not_executed` and moves nothing), then played to its end. A long tool is still one
  query and one step file; what the joints did tick by tick is `trace.npz`.
* **feedback**: `previous_result` on the next observation — per-joint residual, target vs
  commanded base displacement, the settle report — or the error (`invalid_selection`,
  `tool_rejected`, `motion_not_executed` with the violations). Errors cost a decision; there
  is no re-ask loop. `done` / `give_up` end the run; `check` answers without moving.

Every tool the agent plays is seeded from the agent's own last **commanded** q, never the
measured one: on the robot the measured pose lags the command by gravity sag, and seeding
from it would jump at every boundary. A Chain composes at the segment level for the same
reason.

## The menu

Offered to the model (`menu(allow_base, has_loco)` hides what an env cannot do; `--mode gantry` hides walking):

```
move(dx_m, dy_m, dyaw_deg)      one base displacement -> LocoClient.Move (translate, then turn)   ≙ move_to
arm_path(waypoints)             joint-space waypoints over waist + arms -> arm_sdk               ≙ move_eef_chunk
look(yaw_deg, seconds)          turn the waist (the camera and a raised arm turn with it)
point(down_deg, rise_s, hold_s) the right arm straight out where the torso faces, left UP: the next image shows it
lower_arm(seconds)              the right arm back down (a run that ends pointing lowers it before the hand-back)
hold(seconds)                   stand still
check(tool, arguments)          dry-run without moving                                          ≙ check_path
wave_hand / shake_hand          the onboard controller's own gestures (the twin plays stand-ins) ≙ set_gripper
say(text, pause_s)              the onboard text-to-speech; the twin prints it                   ours, not in GPT-Policy
done(summary, hindsight) / give_up(reason, hindsight)
```

Presets (`visible = False`: tools chains only): `walk_forward`, `turn`,
`tpose`, `sixseven`; bookends `takeover`, `handback`. An onboard call a tool may make is
allow-listed by service: `config.LOCO_METHODS` (`WaveHand`, `ShakeHand`, `Sit`) and
`config.AUDIO_METHODS` (`TtsMaker`); the robot env routes by name and refuses the rest.
`Sit` is the one FSM switch (`config.LOCO_FSM`: sent as `SetFsmId(3)`; the teardown stands
up with 801). Nothing else on the SDK — other FSMs, damp, torque, squat, stand height, raw
audio playback, volume, the LED strip — is reachable from a reply. `locate_point(x_px, y_px)`
is GPT-Policy's pixel query for one monocular camera: the pixel's direction (from the torso, the
waist yaw that centres it, and the angle below horizontal), carried through the measured waist
joints and the IMU's pelvis tilt, never a distance.

## Envs

There is one env class, `RobotEnv` (`g1/envs/robot.py`), and two things it can talk to.
Every SDK object, sleep and clock read comes from its link (`g1/envs/link.py`).

**The robot** (`g1 run TASK --mode ...`): only high-level control. The camera connects over
WebRTC before any FSM change; `LocoClient` FSM transitions (`--mode gantry`: Damp → FSM 4 →
FSM 200 → run → release → Damp; `--mode standing`: remember the FSM → FSM 200 → run → release
→ restore); then arm targets on `rt/arm_sdk`; base velocity through `LocoClient.Move` in
standing mode only (a 1 s dead-man re-sent from a 10 Hz thread, `StopMove` before the arms
release); the monitor runs report-only. Never call the SDK from `step`.

**The twin** (`g1 run TASK --sim`): `TwinProcess` (`g1/twin/main.py`) starts the twin as its
own process for the task's scene, on the loopback interface and a private DDS domain
(`g1/twin/bridge.py`: SDK RPC servers for sport, voice, motion_switcher and the twin's own
`Summary`, `rt/lowstate` at 500 Hz, `rt/arm_sdk`; the head camera in shared memory), runs the
same `RobotEnv` against it in real time, and stops it. The twin is unitree_mujoco's 29-DoF G1
(free base, torque motors at the real limits, a PD loop every 2 ms physics step) behind a
stand-in for the onboard controller (`g1/twin/onboard.py`): the FSM (0, 1, 4, 200, 500), the
arm_sdk blend of target, kp and kd, and NVIDIA's GR00T whole-body controller for the legs and
waist (a Balance network at zero command, a Walk network otherwise; it sees all 29 joints, so
it accounts for the arms), its velocity command corrected by the twin's own velocity estimate.
FSM 4 eases into the stand pose and holds it with Balance. Move is a 1 s
dead-man, gestures are keyframed, a fall damps it, and `--mode gantry` hangs it from a rope.
Nothing is pinned or slid.

With `--sim` the run is also **checked**: the `JointMonitor` (`g1/envs/monitor.py`) records
every tick the measured angle of every joint and the commanded target, and flags a measured or
commanded angle outside the limits minus `joint_margin_rad` (the legs, which the onboard
controller owns, at the hardware limits), a blended-command speed over `command_vel_max`, a
weight outside [0, 1] and a base velocity over `BASE_VEL_MAX`; the run fails on any violation
or a fall (the twin's `Summary`) and aborts after `max_violations`.

The tests use `SimEnv` (`g1/envs/sim.py`), the same `RobotEnv` with the link swapped for an
in-process loopback that steps the twin in lockstep on sim time: about 30x real time headless,
and two runs are bit-identical. Every run writes `trace.npz`; `g1 trace compare` reports how far
the twin's joints are from the robot's on the same task, and the twin's `twin_*` limits are
what to tune until they agree.

Where the twin knowingly departs from the robot: its walker is not Unitree's onboard controller
(GR00T stands at a 0.75 m pelvis height, the robot's controller its own); the gestures are
stand-ins; video comes over shared memory, not WebRTC. Like the robot's controller, it holds the
waist steady while it walks (the walker takes it from arm_sdk over `twin_walk_blend_s` and hands
it back over `twin_waist_hold_s` after a stop) and leaves it to arm_sdk's gains while standing, so
the waist at kp 60 sags and wobbles in sim as it would on the robot.

Why GR00T: on the twin it stood 7x calmer than the previous walker (unitree_rl_gym's) after a
move, stood still within about a second of a stop instead of three, took a 120 N sideways push
the old one fell to, and landed a 3 m walk within 4 cm. Its weights are under the NVIDIA Open
Model License; `g1 twin fetch` downloads them, they are never committed.

**Ctrl-C, the task's `max_time_s` and any exception** run `safe_return` first: from the last commanded
pose to STAND over 3 s, the weight from where it was to 0 over 2 s, base stopped, at 50 Hz
with no gap, SIGINT ignored throughout and through the robot's hand-over. There is
deliberately no second-Ctrl-C escape.

## The record

`tasks/<name>/runs/<ts>_<env>_<instruction>_<outcome>/` (`<env>` is `sim` or `robot`, from `--sim`),
four kinds of file and nothing else (`g1/agent/episode.py`):

* `summary.json`: which run and what happened, on one screen: task, instruction, env, mode, model,
  times; outcome, its source, the model's own outcome, `end_reason` (done, give_up,
  budget_exhausted, step_timeout, quota, model_overloaded, model_error, no_frame, interrupted,
  max_time, error; a tools run: completed, checks_failed), error; decisions used of max, the tool
  sequence, the model's closing words; usage totals; health (the joint check and violations, tick
  overruns, SDK failures, on the twin whether it fell, its true pose and torque-limit time); an
  interruption and its safe return; SDK calls made outside any step (bring-up).
* `config.json`: how the run was set up: the task as loaded, env, mode, camera, the model settings,
  the tools on offer (name, prompt, parameter schema), the system prompt, every
  limit, versions, and the code: `{head, snapshot, ref}`. A dirty tree is saved as a commit on HEAD
  kept by `refs/g1-runs/<sha>` (`task.code_snapshot`, a temporary index: nothing else moves), so
  `git checkout <snapshot>` gives the code that ran. Never a `.env` value.
* `step_NNNN.json` + `.png`: one per query to the model (a retry is its own step). The frame sent
  (lossless; the model got it resized to JPEG, size and quality recorded), the query exactly as sent
  without the system prompt (history turns included, images named by their step file), the response
  (status, timing, usage, every chunk the server streamed, reasoning included, refusals such as 429;
  the parsed decision or the error), the action (tool, arguments, dry-run verdict, start and end
  time, joints and commands at both ends, the commanded base), its result in full after the settle,
  and the SDK calls it made. `tick_start`/`tick_end` are rows of `trace.npz` (one per control tick;
  `t_start`/`t_end` the same instants in run seconds), so a frame lines up with the measured joints.
* `trace.npz`: every tick at 50 Hz, commanded and measured (on the twin, its true base pose too: each
  tick in-process; with `--sim`, the twin's samples from its Summary interpolated onto the ticks, which
  both processes stamp on the monotonic clock).

`summary.json`'s `health` also says whether the run's timing can be trusted: `overrun_fraction` (ticks
more than one tick late), `max_tick_gap_s`, `load_avg`, and `timing: degraded` above the limit
`overrun_fraction_max`. A loaded machine makes the arms lag their commands, so a degraded run says
little about the task.

The verdict wins the outcome. On `--sim` a task with `success` criteria is scored by the twin from its
truth (`g1/twin/core.py` `Evaluator`: the arm's ray to a target, how far the feet moved, forbidden
contacts, a fall) and labelled from it (`outcome_source: twin`); the model never sees the criteria. On
the robot it is the human's. A model conclusion with neither is `unreviewed`. A tools run
has `summary.json` (its checks decide), `config.json` and `trace.npz`. Each run adds a line to the
task's `results.jsonl` (the summary's comparable fields). A new file is a deliberate change: the
tests assert the exact set.

## Conventions

* Frames are RGB everywhere (`image[row, col, channel]`); image I/O goes through
  `g1/core/images.py` (Pillow). OpenCV is not used.
* Joint indices are DDS order (`LowCmd_.motor_cmd`), also the twin model's motor order;
  `g1/core/config.py` holds the table and `tests/core/test_config.py` checks it against the
  model. arm_sdk commands only `UPPER_BODY` (waist + arms).
* The vision model runs only when the agent asks: one request per decision, on a
  background thread, results latest-only.
