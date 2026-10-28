# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Unitree G1 (29-DoF) driven one **tool** at a time by a vision-language model, in
GPT-Policy's shape (arXiv:2609.19138): the model picks a tool from a menu and fills in its
arguments; the host plans a fixed motion from them, checks it, runs it, waits for the joints
to settle and reports. The same tools chain without a model. `g1 run TASK` runs a task on the
robot (`RobotEnv`, live via `unitree_sdk2py`); `--sim` runs the *same* `RobotEnv` code over DDS
against the **twin** (`g1/twin/`): unitree_mujoco's 29-DoF G1 with a free base and torque
motors, behind a stand-in onboard controller that answers LocoClient/AudioClient, blends
arm_sdk, stands and walks with NVIDIA's GR00T whole-body controller (Balance and Walk networks,
onnxruntime). Nothing is pinned or slid.
Run it with `--sim` before the robot for anything new.

## Vocabulary (GPT-Policy's; use these words in code, CLI, docs)

- **tool** — one class in `g1/tools/`: `name`, `prompt`, `params` (JSON-schema fragments), `segments()`.
  A fixed motion (or a query) computed from the arguments; ends by construction; **never watches
  the camera while it runs** — the model is the closed loop. Visible to the model by default.
- **instruction** — the goal text. **task** — a folder (`tasks/<name>/task.json`) with an instruction (or a
  fixed `tools` chain) and its context, no code; it is the only thing `g1 run` runs. **observation / decision / execution result / verdict** — as in the paper.
- Retired, do not reintroduce: *skill*, *policy*, *primitive*, *routine*, *goal*, a catalog file, a
  tool that servos on vision, a per-tool timeout.

## Commands

The whole CLI. Everything else is in a file: the task's `task.json`, `configs/limits.json`, `.env`.
Do not add flags; add a task.json key or a limit. `.env` is private to one machine (secrets and the
robot's address): never put an editable setting there.

```
pip install -e ".[sim,dev]"          # unitree_sdk2py (needed for --sim too) / unitree_webrtc_connect come from their repos
g1 twin fetch                        # once: the twin's model (unitree_mujoco), walker (NVIDIA GR00T) and room into assets/

g1 run TASK --sim [--mode gantry]    # the twin in the MuJoCo window (always; mjpython): starts it (TwinProcess),
                                     # runs RobotEnv over DDS on lo0, stops it. Headless only in tests: G1_TWIN_HEADLESS=1
g1 run TASK --mode standing|gantry [--view]   # the robot; --mode is required. standing may walk, gantry never does;
                                     # --view: its head camera at http://127.0.0.1:8765 (live only)
g1 status                            # read-only: link ok / robot refuses / no link (UNITREE_IFACE)
g1 twin fetch|status
g1 trace compare RUN_A RUN_B | show RUN   # two run dirs of the same task: tracking, sag, lag, A-B difference
g1 limits [SOURCE]                   # every limit and default in configs/limits.json as one table

TASK = a folder under tasks/ (tpose, find_the_mug) or a path. .env: UNITREE_IFACE, UNITREE_ROBOT_IP,
UNITREE_AES_128_KEY, UNITREE_DOMAIN, HF_TOKEN, VLM_BASE_URL; G1_VERBOSE=1 prints every SDK call.
The model (`vlm_model`), `reasoning_effort` and `max_tokens` are limits; VLM_MODEL etc. in .env are refused.

pytest                               # all tests, on the in-process twin (~2 min)
pytest -m dds                        # also: g1 run --sim end to end over DDS (needs unitree_sdk2py)
pytest tests/tools/test_tools.py::test_every_tool_pair_is_continuous
```

## Layout

```
g1/cli.py          run | status | twin | trace; run_settings() (every setting an env reads), run_task(), run(), safe_return
g1/core/           config (joint table, STAND_Q, UPPER_BODY, LOCO_METHODS, BASE_VEL_MAX), limits (configs/limits.json),
                   action (Action, Obs, Segment, Runnable, ease, EPS), poses, images (Pillow, RGB), worker
g1/tools/          __init__: discovery (TOOLS), menu(), the three renderings, parse_tool/parse_chain, Chain
                   base: the Tool contract + the segment player + validate_args + num/integer/text/flag/limit
                   move (move, walk_forward, turn) · arms (arm_path, look, point, tpose, sixseven) · gestures (wave_hand,
                   shake_hand) · posture (sit: SetFsmId(3)) · speech (say: AudioClient.TtsMaker) · control (hold, check + dry_run, done,
                   give_up, takeover, handback)
g1/envs/           base (Env, EnvAbort, shield_sigint, verbose, sdk_log), monitor (JointMonitor), scene (the room),
                   robot (RobotEnv, ArmSdk, BaseCommander, Health, probe/status_main), sdk (RPC_CODES, explain, call, FSM_NAMES),
                   link (DdsLink: the SDK seam), sim (SimEnv = RobotEnv on the in-process twin: the tests' env), trace
g1/twin/           assets (g1 twin fetch), model (unitree_mujoco G1 + head camera), walker (GR00T: Balance + Walk),
                   onboard (the stand-in onboard controller: FSM, arm_sdk blend, stand/walk, gestures, falls),
                   services (sport/voice/motion_switcher/twin handlers), core (Twin), loopback (in-process link, lockstep),
                   bridge (DDS servers + rt/lowstate + rt/arm_sdk), shm (camera frames), main (TwinProcess, serve, fetch)
g1/camera.py       Frame, Camera (+subscribe), DirCamera, WebRTCCamera, ShmCamera (the twin's), Viewer (--view)
g1/vlm.py          VLMClient (Hugging Face), load_dotenv
g1/agent/          agent (Agent, build_agent), decider (AgentContext/AgentTurn/Decision, observation(), instructions(),
                   VLMDecider), episode (EpisodeWriter, load_episode), task (find, load_task, run_config, code_snapshot, record_result)
tasks/             tpose/ (a tools chain), find_the_mug/, point_at_the_laptop/, find_and_point_at_the_laptop/, sit_in_the_chair/ (instructions); runs and results.jsonl inside are git-ignored
docs/              architecture, writing-a-tool, writing-a-task, running-on-the-robot, vision-model
tests/             mirrors g1/ (core, tools, envs, agent, twin) + test_camera, test_cli, test_vlm; doubles.py (RedBallDecider,
                   red_blob, sim_env(**settings), scripted_sim); conftest keeps tests away from the real .env
```

Absolute imports (`from g1.core.config import ...`); tests are a package (`tests/__init__.py`), pytest gets the
root via `pythonpath`. `limits.ROOT` is the repo root (configs/, tasks/, assets/, .env all resolve from it).

## Rules that are easy to break

- **Tool metadata lives in the class, nowhere else.** Prompt + `params` + code in one file; the menu is
  generated by import (`g1/tools/__init__.py:discover`). Limits are referenced (`limit("name")`,
  `{limit:name}`) and resolved when the schema is read, never copied. Every limit is in
  `configs/limits.json` with a unit, a note and a source; a name the file lacks fails at import.
  No tunable number is a literal in code, tool ranges and defaults included (`test_limits.py` guards
  the robot and twin timing); a word setting has unit `text` (and `choices`). Plumbing (thread joins,
  queue lengths, seqlock retries) and robot facts (joint table, poses, SDK ids) stay in code.
- **Parameters are numbers with ranges** (`num(lo, hi, default)`), unit-suffixed (`_s`, `_m`, `_deg`,
  `_rad`, or `seconds`), never enums for a continuous quantity; a property without a default is
  required; `note` is added automatically to motion and query tools (required of the model, "" for code).
  `tests/tools/test_tools.py::test_parameters_are_numbers_with_ranges_not_switches` enforces this.
- **A tool never overrides `step`** and never reads the camera. The player in `Tool` interpolates
  `segments()`. Query tools implement `query(cmd, joints)`; terminal tools end the run.
- **Seed from the last commanded q, never the measured one** when composing (Chain composes segments;
  the Agent seeds each tool from its own `_cmd`). On the robot the measured pose lags by gravity sag.
- **Never ignore an SDK return code.** Every LocoClient/AudioClient call goes through `envs/sdk.call`
  (`RobotEnv._call`): timed, code normalised and named (`sdk.explain`: 3102/3104 = the DDS link, other codes =
  the robot refused), appended to `env.sdk_log` (the Agent writes each as an `sdk_call` event). In `setup` a
  failure raises `SystemExit` with a hint **before** arm_sdk is touched (`_require`); in teardown it is printed,
  never raised. `--mode standing` refuses a robot not in `sdk.STANDING_FSMS` and restores only such an FSM.
  `g1 status` is the read-only probe; `G1_VERBOSE=1` prints every call and a per-second health line.
- **No pose reset.** The bookends only blend the weight (`takeover`: 0→1 holding the observed pose;
  `handback`: 1→0 holding the last pose); walking tools use `{}` (base only); only `arm_path` and the arm
  presets move the arms, and a tool that needs a starting pose makes it its own first segment. No
  countdown; the bring-up says "entering control mode (FSM 200)" and skips the transition when already
  there. The Ctrl-C safe return is the one deliberate exception (it ends at STAND).
- **Every movement is dry-run before it executes** (`g1/tools/control.py:dry_run` through a
  `JointMonitor` from the commanded pose); a violation is `motion_not_executed`, spends a decision,
  moves nothing. **Errors are feedback**: no re-ask loop.
- **Ctrl-C runs the safe return to completion** (`cli.safe_return`: last commanded pose → STAND 3 s,
  weight → 0 over 2 s, base stopped, SIGINT shielded). Never add a second-Ctrl-C release or a keypress
  escape. Never call the SDK from `step`. Walking is `LocoClient.Move` in `--mode standing` only; no
  low-level leg control. Onboard calls are allow-listed per service — `config.LOCO_METHODS`
  (`WaveHand`, `ShakeHand`; `needs_loco`; and `Sit`, the one FSM, `config.LOCO_FSM`: `sit`, sent as
  `SetFsmId(3)`, standing mode only, the teardown stands up with 801) and `config.AUDIO_METHODS` (`TtsMaker`; `say`) — routed by name
  in `RobotEnv._command`; a failed speech warns and continues, a failed gesture raises.
  `PlayStream`/`SetVolume`/`LedControl` are never reachable from a reply.
- **Frames are RGB everywhere**; image I/O only through `g1/core/images.py` (Pillow). OpenCV is not
  used and must not be imported (its bundled ffmpeg collides with PyAV's, which decodes the robot's video).
- **The view never touches the tick.** `--view` (live runs) serves the env's camera slot (`Env.source()`)
  from its own threads; `cli.run` starts it after `env.reset()` and stops it inside `with env`. The robot
  takes one WebRTC client, so the live picture must come out of our process.
- **Hugging Face only** for the model (`HF_TOKEN`, `VLM_BASE_URL`, limit `vlm_model`); no other providers,
  no Anthropic SDK. Fakes are test doubles only (`tests/doubles.py`); `search` always uses the real model.
- **Sim is the robot code.** `--sim` runs `RobotEnv` itself over DDS against the twin process; the tests'
  `SimEnv` subclasses `RobotEnv` and only swaps the link (`envs/link.py`). Never give sim its own path for
  something the robot does. Robot code takes every SDK object, sleep and clock read from `self.link`. The
  twin is where sim-only behaviour lives (`g1/twin/onboard.py`), and every number it adds is a `twin_*`
  limit. The walker's trained constants are in `g1/twin/walker.py:Trained` (fixed by training, not
  tunable). Call it the *walker* or the *gait network* (never a policy). It drives legs and waist in FSM 4
  (Balance only) and 200/500; `Onboard.tick` corrects its command with the twin's velocity estimate
  (`twin_vel_feedback`, `twin_vel_integral`), only while walking, never pushing a walk into Balance.
- The twin blends arm_sdk as the robot does: target, kp and kd all `(1-w)*onboard + w*arm_sdk`, so the arms
  sag under gravity at kp 60 in sim too, and a joint left out of the message (kp 0 at weight 1) goes limp.
  While it walks, the walker holds the waist (yaw, roll, pitch) whatever arm_sdk sends, as the real G1's
  controller does (its waist is not locked); standing, arm_sdk's gains apply as sent, so at kp 60 / kd 1.5
  the waist sags under the torso and is underdamped (it can take more than the agent's 3 s settle after a walk).
  The report says when the hold applied. One deliberate departure: the strict sim monitor
  holds the legs to the hardware limits without the joint margin (they belong to the onboard controller).
- The twin serves only on a loopback interface and refuses DDS domain 0 (the robot's); `--sim` gives it a
  private domain and reads its camera from shared memory. With `--sim` the monitor is strict and the twin's
  `Summary` (its `twin` service) fails the run on a fall.
- Joint indexing is DDS order (= the twin model's motor order); index 29 is the arm_sdk weight slot.

- **The model knows only what the robot knows**: its tools, the camera, the task's objective and rules,
  its own joints (measured), IMU and commands, and its own earlier turns. Never the room's layout (no
  furniture, wall or object positions or sizes) and never the twin's truth (true base pose, contacts), in sim
  or on the robot: `Obs.base_pose` reaches only the records. Task `safety_notes` are behaviour rules; world
  facts go in `notes` (never sent). `tests/agent/test_what_the_model_knows.py` guards it.

## The decision step (`g1/agent/agent.py`)

takeover → settle (0.03 rad, 0.05 rad/s, 10 ticks, ≥ 0.5 s, 3 s timeout) → snapshot (a fresh frame) →
think (background `Decider.request`; the loop holds) → act (the whole tool, one step file) → settle → feedback (`previous_result`: residuals, commanded base displacement, settle report | the
error) → … `done`/`give_up` end it; `check` answers via `dry_run`, `locate_point` with a pixel's direction;
turns older than `history_full_turns` are sent as one host-written line each; overload retried with backoff re-observing;
quota or `step_timeout` fails the run; `max_decisions` ends as `budget_exhausted`; `close()` renames the
run directory (`_success`/`_failed`/`_unreviewed`/`_interrupted`). The agent asks nothing: after a robot run
`cli.run_task` asks "Save this run?" (no: the folder is deleted) and then the verdict (`task.label_run`);
`--sim` runs ask nothing.
Records: `tasks/<name>/runs/<ts>_<env>_<instruction>_<outcome>/` (`<env>` = `sim`|`robot`, from `--sim`) with exactly
`summary.json` (which run, what happened), `config.json` (the setup: task, tools, prompt, limits, model, code snapshot),
`step_NNNN.json/.png` (one per model query: frame, query and response as sent/received, action, result) and `trace.npz`
(a tools run: no steps). Each fact is stored once; a new file needs an argument; the tests assert the set. Every run's
code is restorable: `config.json` `code.snapshot` (a commit kept by `refs/g1-runs/<sha>` when the tree was dirty).
The observation JSON and the system prompt are in `g1/agent/decider.py`; `Decision.parse` validates a
whole JSON object against the tool's schema and clamps numbers with a note.
