# Architecture

## Vocabulary (GPT-Policy's)

| word | meaning here |
|---|---|
| **tool** | what the model picks, one per turn, and the host executes: a fixed motion (or a query) computed from its arguments, checked, executed, settled and reported. One class: name, prompt, `params`, `segments()`. Ends by construction. Visible to the model by default; chains on the CLI. |
| **instruction** | the goal text the model is given (`--instruction`, `task.json`'s `instruction`). |
| **task** | a folder with an instruction and its context (scene, safety notes, demonstration, budget); no code. |
| **observation** | the one JSON object a turn sends: instruction, images, measured state, extra, `previous_result`. |
| **decision** | the model's reply: `{"name": tool, "arguments": {...}}`, validated against the tool's schema. |
| **execution result** | what the host reports after a tool: residuals, base error, a settle report — or why nothing ran. |
| **verdict** | the human's success/failed label after the run. `done` is the model's conclusion, not the label. |

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
* **act**: the decision is parsed (whole JSON object, jsonschema against the tool's
  parameters, numbers clamped with a note), the tool is built, **dry-run through the joint
  monitor from the commanded pose** (a limit, speed or base violation comes back as
  `motion_not_executed` and moves nothing), then played to its end. A record is closed every
  3 s while it runs (`STEP_MAX`), so a long tool is one decision and several records.
* **feedback**: `previous_result` on the next observation — per-joint residual, target vs
  measured base pose, the settle report — or the error (`invalid_selection`,
  `tool_rejected`, `motion_not_executed` with the violations). Errors cost a decision; there
  is no re-ask loop. `done` / `give_up` end the run; `check` answers without moving.

Every tool the agent plays is seeded from the agent's own last **commanded** q, never the
measured one: on the robot the measured pose lags the command by gravity sag, and seeding
from it would jump at every boundary. A Chain composes at the segment level for the same
reason.

## The menu

`g1 tools`. Offered to the model (`menu(allow_base, has_loco)` hides what an env cannot do):

```
move(dx_m, dy_m, dyaw_deg)      one base displacement -> LocoClient.Move (translate, then turn)   ≙ move_to
arm_path(waypoints)             joint-space waypoints over waist + arms -> arm_sdk               ≙ move_eef_chunk
hold(seconds)                   stand still
check(tool, arguments)          dry-run without moving                                          ≙ check_path
wave_hand / shake_hand          the onboard controller's own gestures (robot only)               ≙ set_gripper
say(text, pause_s)              the onboard text-to-speech; sim prints it                        ours, not in GPT-Policy
done(summary, hindsight) / give_up(reason, hindsight)
```

Presets (`visible = False`: CLI chains and replay only): `walk_forward`, `turn`, `look`,
`tpose`, `sixseven`; bookends `takeover`, `handback`. An onboard call a tool may make is
allow-listed by service: `config.LOCO_METHODS` (`WaveHand`, `ShakeHand`) and
`config.AUDIO_METHODS` (`TtsMaker`); the robot env routes by name and refuses the rest.
Nothing else on the SDK — FSM, damp, torque, sit, squat, stand height, raw audio playback,
volume, the LED strip — is reachable from a reply. There is no `locate_point`:
the head camera is monocular, there is no metric pixel query to offer honestly.

## Envs

`sim` (MuJoCo, Menagerie `unitree_g1`) runs the program **and checks it**: the
`JointMonitor` (`g1/envs/monitor.py`) records every tick the measured angle of every joint
and the commanded target, and flags a measured or commanded angle outside the limits minus
`--margin`, a blended-command speed over `--max-vel`, a weight outside [0, 1] and a base
velocity over `BASE_VEL_MAX`; the run fails on any violation and aborts after
`--max-violations`. `--headless` is the fast pre-flight; the viewer needs `mjpython` on
macOS. The pelvis is pinned and a base velocity slides it (legs hold the stand pose: a
rehearsal of the base commands, not gait). `--scene room` and `--sim-objects` build a
real-looking room; `--camera-dir` replays recorded frames instead of rendering.

`robot` uses only high-level control: the camera connects over WebRTC before any FSM
change; `LocoClient` FSM transitions (`--mode gantry`: Damp → FSM 4 → FSM 200 → run →
release → Damp; `--mode standing`: remember the FSM → FSM 200 → run → release → restore); then
arm targets on `rt/arm_sdk`; base velocity through `LocoClient.Move` behind `--walk` (a 1 s
dead-man re-sent from a 10 Hz thread, `StopMove` before the arms release); the monitor runs
report-only. Never call the SDK from `step`.

**Ctrl-C, `--max-time` and any exception** run `safe_return` first: from the last commanded
pose to STAND over 3 s, the weight from where it was to 0 over 2 s, base stopped, at 50 Hz
with no gap, SIGINT ignored throughout and through the robot's hand-over. There is
deliberately no second-Ctrl-C escape.

## The record

`runs/<ts>_<env>_<instruction>_<outcome>/`: `episode.json`, `step_NNNN.json` + `.png` (the
frame losslessly), and the run trace — `config.json`, `events.jsonl` (append-only, every
observation with the verbatim `input_json`, decision, timing, result, error, retry,
terminal, verdict), `transcript.json`, `protocol.json` (the system prompt and schemas),
`states.jsonl` (20 Hz), `usage.jsonl`/`usage.json`, `status.json`. The human verdict wins
the outcome; a model conclusion without one is `unreviewed`. `g1 episode DIR` prints the
step table and the `--tools` chain that replays it; `g1 run --tools replay --episode DIR`
runs it with no camera and no model.

## Conventions

* Frames are RGB everywhere (`image[row, col, channel]`); image I/O goes through
  `g1/core/images.py` (Pillow). OpenCV is not used.
* Joint indices are DDS order (`LowCmd_.motor_cmd`), also the Menagerie actuator order;
  `g1/core/config.py` holds the table and `tests/core/test_config.py` checks it against the
  model. arm_sdk commands only `UPPER_BODY` (waist + arms).
* The vision model runs only when the agent asks: one request per decision, on a
  background thread, results latest-only.
