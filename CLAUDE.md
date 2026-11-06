# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Unitree G1 (29-DoF) driven one **tool** at a time by a vision-language model, in
GPT-Policy's shape (arXiv:2609.19138): the model picks a tool from a menu and fills in its
arguments; the host plans a fixed motion from them, checks it, runs it, waits for the joints
to settle and reports. `g1 start` is the one interactive session: the robot listens, you type (plain text is chat,
`/` commands call skills and settings), and it chooses what to do from a fixed menu (a built-in, a
routine, a task, or just an answer). `g1 run TASK` runs a task on the
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
- **instruction** — the goal text. **task** — a folder (`tasks/<name>/task.json`) with an instruction and its
  context, no code: the model works through it one tool at a time (and may be shown **demos**: successful past
  runs a person promoted, `tasks/<name>/demos/`). **observation / decision / execution result / verdict** — as in the paper.
- **routine** — our reference motion (`routines/<name>/routine.json`): a timeline of tracks (upper body via
  arm_sdk, base via LocoClient, onboard gestures; full body kept but not played), start to finish, no model.
- **built-in** — a motion that comes with the robot and runs on Unitree's controller (the stored dances and
  gestures, walking): `g1 moves`, talk's `move` and `walk`.
- The robot's menu is built-ins, routines and tasks; talk chooses among them and never authors one.
- Retired, do not reintroduce: *skill*, *policy* (until a trained closed-loop controller exists),
  *primitive*, *goal*, a catalog file, a tool that servos on vision, a per-tool timeout.

## Commands

The whole CLI. Everything else is in a file: the task's `task.json`, `configs/limits.json`, `configs/model.json`, `.env`.
Do not add flags; add a task.json key or a limit. `.env` is private to one machine (secrets and the
robot's address): never put an editable setting there.

```
pip install -e ".[sim,dev]"          # unitree_sdk2py (needed for --sim too) / unitree_webrtc_connect come from their repos
g1 twin fetch                        # once: the twin's model (unitree_mujoco), walker (NVIDIA GR00T) and room into assets/

g1 run TASK --sim [--mode gantry]    # the twin in the MuJoCo window (always; mjpython): starts it (TwinProcess),
                                     # runs RobotEnv over DDS on lo0, stops it. Headless only in tests (G1_TWIN_HEADLESS=1) and g1 bench
g1 bench SUITE                       # suites/<name>/suite.json: its tasks on the twin, headless, repeat times,
                                     # scored by the twin (each task's success), one table + report.json (with the
                                     # input/output tokens and cost); runs several at once, each in a fresh process
                                     # with its own twin and DDS domain: as many as memory (bench_run_mb) and the
                                     # performance cores allow, at most bench_parallel_max
g1 run TASK --mode standing|gantry [--view]   # the robot; --mode is required. standing may walk, gantry never does;
                                     # --view: its head camera at http://127.0.0.1:8765 (live only)
g1 status                            # read-only: link ok / robot refuses / no link (UNITREE_IFACE)
g1 lidar                             # the head lidar via Unitree SLAM: checks lidar_driver + unitree_slam are on
                                     # (read-only), maps until Ctrl-C with a live 3D view at http://127.0.0.1:8766,
                                     # then ends mapping and saves runs/lidar/<time>/map.pcd (robot only)
g1 moves [list] | check MOVE...      # read-only: Unitree's moves stored on the robot (mimic motions = FSM 550<id>,
                                     # arm:<id|name> arm actions, custom:<name> taught in the app), installed or not
g1 moves play MOVE... | fsm TARGET   # the robot only, asks first: play stored moves in order from standing; switch the
                                     # FSM along its order (damp, stand 4, main 200, walk 501, run 801, sit 3; never via damp);
                                     # a move runs in 550 and ends in 802, where the next plays
g1 moves walk|side|turn N            # the robot only, asks first: a short walk (SetVelocity's 1 s dead-man, re-sent) in limit walk_fsm (walk mode, 501)
g1 start --mode standing|gantry      # the robot only: one interactive session (Claude-Code-style terminal). It listens
                                     # (rt/audio_msg) and you type: plain text is chat (as if said); the model answers or
                                     # chooses a built-in (5xx/6xx move; standing only; it never walks), a routine or a task
                                     # (model-chosen ones confirmed y/N). /routine /task /move /say call one directly;
                                     # /skills /mode /fsm /mic /speaker /camera /lidar /memory /soul /status /help /quit. Esc stops what runs, to
                                     # neutral; Ctrl-C clears the line (stops a running skill); Ctrl-C twice exits. Modes never change
                                     # the FSM, they block FSMs (gantry: only 0 1 4 200; standing: not 0 1); /fsm ID switches; the
                                     # start brings it up 0 -> 1 -> 4 -> 802 standing, -> 200 gantry (y/N); quitting changes no FSM
g1 twin fetch|status
g1 trace compare RUN_A RUN_B | show RUN   # two run dirs of the same task: tracking, sag, lag, A-B difference
g1 limits [SOURCE]                   # every limit and default in configs/limits.json as one table, + the model in use

TASK = a folder under tasks/ (find_the_mug) or a path; a routine under routines/ (tpose) plays the same way. .env: UNITREE_IFACE, UNITREE_ROBOT_IP,
UNITREE_AES_128_KEY, UNITREE_DOMAIN, HF_TOKEN, VLM_BASE_URL, OPENROUTER_API_KEY; G1_VERBOSE=1 prints every SDK call.
Which model is asked, and how, is `configs/model.json` (`provider`, `models`, `reasoning_effort`, `max_tokens`,
`temperature`, and `agent`: the decision budget, timeouts, retries, image size, history windows). It may be tuned
(benchmark with `g1 bench`); `configs/limits.json` holds the robot's and the run's hard limits, not to be changed
to make a task pass. VLM_MODEL etc. in .env are refused.

pytest                               # all tests, on the in-process twin (~2 min)
pytest -m dds                        # also: g1 run --sim end to end over DDS (needs unitree_sdk2py)
pytest tests/tools/test_tools.py::test_every_tool_pair_is_continuous
```

## Layout

```
g1/cli.py          run | bench | status | twin | trace; run_settings() (every setting an env reads), run_task()/run_once(),
                   twin_verdict(), run(), safe_return
g1/bench.py        g1 bench: load_suite, run_suite, totals (task | execution | infrastructure), table
g1/core/           config (joint table, STAND_Q, UPPER_BODY, LOCO_METHODS, BASE_VEL_MAX, the URDF's waist/arm chains),
                   limits (configs/limits.json), aim (pixel+distance -> a point, the arm's line, the aim solve: point_to),
                   action (Action, Obs, Segment, Runnable, ease, EPS), poses, images (Pillow, RGB), worker
g1/tools/          __init__: discovery (TOOLS), menu(), the three renderings, parse_tool/parse_chain, Chain (internal)
                   base: the Tool contract + the segment player + validate_args + num/integer/text/flag/limit
                   move (move, walk_forward, turn) · arms (arm_path, look, point, point_to, lower_arm) · routines (play_routine,
                   RoutineTool, routine_program: dry-run, then Chain) · gestures (wave_hand,
                   shake_hand) · posture (sit: SetFsmId(3)) · speech (say: AudioClient.TtsMaker) · control (hold, listen, check +
                   dry_run, done, give_up, takeover, handback)
g1/envs/           base (Env, EnvAbort, shield_sigint, verbose, sdk_log), monitor (JointMonitor), scene (the room),
                   robot (RobotEnv, ArmSdk, BaseCommander, Health, probe/status_main), moves (g1 moves), sdk (RPC_CODES, explain, call, FSM_NAMES),
                   link (DdsLink: the SDK seam), sim (SimEnv = RobotEnv on the in-process twin: the tests' env), trace
g1/twin/           assets (g1 twin fetch), model (unitree_mujoco G1 + head camera), walker (GR00T: Balance + Walk),
                   onboard (the stand-in onboard controller: FSM, arm_sdk blend, stand/walk, gestures, falls),
                   services (sport/voice/motion_switcher/twin handlers), core (Twin), loopback (in-process link, lockstep),
                   bridge (DDS servers + rt/lowstate + rt/arm_sdk + rt/audio_msg), shm (camera frames), main (TwinProcess, serve, fetch)
g1/camera.py       Frame, Camera (+subscribe), DirCamera, WebRTCCamera (from a laptop), DirectCamera (on the robot's
                   Jetson: the video hub's multicast, hardware decode), ShmCamera (the twin's), Viewer (--view)
g1/lidar.py        Scan (points N x 4 in m + the 4x4 pose), points_from_cloud2, MapAccumulator, write_pcd,
                   SlamLidar (SLAM's points + odom from a child process), save_map, lidar_main (g1 lidar);
                   lidar_view.py: LidarViewer (the live 3D view: a canvas page, state.json, scan.bin, map.bin)
g1/microphone.py   Microphone (rt/audio_msg: the robot's speech recognition, an utterance log), heard_view; sounds in g1/core/sounds.py
g1/vlm.py          VLMClient (Hugging Face or OpenRouter), load_dotenv
g1/session.py      g1 start's engine (Session: mode, mic/speaker/camera/lidar, the events queue, chat -> the model's one choice,
                   run_skill: the one executor; robot_runner: cli.run_once in the session's mode, its mic and camera)
g1/knowledge.py    g1 start's knowledge base (knowledge/: SOUL.md by people, MEMORY.md by the robot, sessions/DATE.md;
                   the prompt's soul and memory, remember/forget, the daily log, the end-of-session distill)
g1/console.py      g1 start's terminal (prompt_toolkit in its own thread: input box, status line, / completion;
                   parse, Console: Esc/Ctrl-C/y-N/quit, start_main)
g1/routines.py     routines: SCHEMA, load_routine, routine_names, compile_segments (the timeline as Segments), run_spec
g1/agent/          agent (Agent, build_agent), decider (AgentContext/AgentTurn/Decision, observation(), instructions(),
                   VLMDecider), episode (EpisodeWriter, load_episode), task (find, load_task, run_config, code_snapshot, record_result),
                   demos (promote, load_demos, render: in-context demos, host truth never copied)
suites/            stationary/ (point_at_the_laptop x 10), point_on_request/ (point_at_what_you_hear x 5); runs/ inside are git-ignored
tasks/             point_at/ (point at {object}, g1 start fills it; keeps pointing until told to stop), and not in g1 start ("session": false):
                   point_at_the_laptop/, find_the_mug/, find_and_point_at_the_laptop/, sit_in_the_chair/, point_at_what_you_hear/
                   (spoken requests, hears); runs and results.jsonl inside are git-ignored, demos/ is not
routines/          tpose/, sixseven/ (routine.json: upper-body keyframes); runs/ inside are git-ignored
docs/              architecture, writing-a-tool, writing-a-task, writing-a-routine, running-on-the-robot, vision-model
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
  there. The Ctrl-C safe return is one deliberate exception (it ends at STAND); the other is `point`:
  it leaves the right arm up so the next image shows the aim, `lower_arm` lowers it, walking and gestures are
  refused while it is up, and a run (agent or chain) that ends with it up lowers it before the hand-back.
  While it is up the settle checks only the speed of the arm and the waist's pitch and roll (they sag under it).
- **Every movement is dry-run before it executes** (`g1/tools/control.py:dry_run` through a
  `JointMonitor` from the commanded pose); a violation is `motion_not_executed`, spends a decision,
  moves nothing. **Errors are feedback**: no re-ask loop.
- **Ctrl-C runs the safe return to completion** (`cli.safe_return`: last commanded pose → STAND 3 s,
  weight → 0 over 2 s, base stopped, SIGINT shielded). Never add a second-Ctrl-C release or a keypress
  that escapes a safe return. In `g1 start`, **Esc stops** the running built-in, routine or task to its
  neutral and the session goes on; Ctrl-C clears the line or, with a skill running, stops it like Esc; **Ctrl-C
  twice** on an empty line exits (so do /quit, SIGTERM, SIGHUP), stopping anything running first. Every stop
  is SIGINT to the process (the UI thread sends it; skills run on the main thread), so every stop takes the
  one shielded path and nothing cuts a return short. The remote's
  L2+B (damp) is the emergency stop. Never call the SDK from `step`. Walking is `LocoClient.Move` in `--mode standing` only; no
  low-level leg control. Onboard calls are allow-listed per service — `config.LOCO_METHODS`
  (`WaveHand`, `ShakeHand`; `needs_loco`; and `Sit`, the one FSM, `config.LOCO_FSM`: `sit`, sent as
  `SetFsmId(3)`, standing mode only, the teardown stands up with 801) and `config.AUDIO_METHODS` (`TtsMaker`; `say`) — routed by name
  in `RobotEnv._command`; a failed speech warns and continues, a failed gesture raises.
  `PlayStream`/`SetVolume`/`LedControl` are never reachable from a reply. Built-ins (Unitree's stored
  moves) play only from `g1 moves` (a person at the keyboard, any move) and talk (the model, ids 5xx/6xx
  only, standing mode); only `g1 moves`, the session's `/fsm`, its start's bring-up and a built-in move (and back) change the FSM outside a run.
  The session runs a routine or task only through `cli.run_once` (never arm_sdk, the FSM or a tool itself;
  `tests/test_session.py` guards it); one the model chooses is confirmed y/N, one you type is not.
  `g1 lidar` makes only `config.SLAM_METHODS` (start mapping 1801, end mapping 1802, close SLAM 1901) and
  switches only `lidar.SERVICES` (`lidar_driver`, `unitree_slam`): on when off, and off again at the end
  only those it switched on (the robot is left as found). It never navigates.
- **The mode is who commands the legs.** `gantry`: we never do (no Move, Sit, stored move or walk mode; the arms
  through arm_sdk in FSM 200, where Unitree's controller holds the legs; hanging). `standing`: we do, through
  LocoClient only (no low-level leg control until a trained controller exists). Every task and routine declares
  `"modes"` in its json (required; anything that walks or sits lists only standing; a routine with a `base` track
  may not list gantry), and `g1 run --mode X` and the session refuse the others before anything connects. A run changes
  no FSM: it runs and ends in the FSM it finds (`sdk.RUN_FSMS`: 4, 200, 500, 501, 801, 802; it warns outside 200,
  where taking the arms is verified); only a gantry robot found limp gets Damp → 4 → 200 first, and only a sit is
  stood up from. In `g1 start` **a mode never changes the
  FSM; it blocks FSMs** (`session.GANTRY_FSMS` 0, 1, 4, 200; `STANDING_BLOCKED` 0, 1). A person switches with
  `/fsm ID` (`SetFsmId` now, refused for an FSM the mode blocks: "switch modes first"; y/N for 0, 1, 3, 4, 801);
  `/mode` (y/N) is refused while the robot is in an FSM the new mode blocks; a skill is refused from a blocked FSM.
  The start's bring-up (`session.BRING_UP`, from wherever the robot is on that way, after a y/N) ends in 802 standing
  (stored moves play from it) and 200 gantry (arm_sdk). The menus (`/skills`, completion, the model's tools) list what
  the mode allows whatever the FSM (`why_not(..., fsm=False)`); the FSM is checked when a skill runs. A **built-in move** switches to 200 first when not in a playing FSM and goes back to the FSM it was called from (`Session._return_to`); a **routine or task** runs in the
  FSM the robot is in (no switch). Quitting changes no FSM.
- **Routines compile to segments and are dry-run first.** A routine's timeline (upper body, base windows,
  onboard gestures at keyframe times; a smooth segment is never split) becomes the `Segment`s every tool plays
  (`ease`: smooth for keyframes, linear between dense samples), dry-run whole before the takeover. Full-body
  routines are refused until a tracking controller exists; tasks may play upper-body routines (`play_routine`).
- **Demos carry the model's own words, never host truth.** `promote` copies each step's tool call, its note and
  the result the model was shown next; never the measured base pose or error, the twin's evaluation, the
  task's success, or the done summary/hindsight. Demos are same-env only and never say where they ran;
  `tests/agent/test_what_the_model_knows.py` checks every request body.
- **Point clouds are decoded off the run's process.** PointCloud2 arrives as ~0.5 MB a scan; `SlamLidar`
  subscribes in a spawned child and hands numpy bytes back, as the camera's decoder is a child on the Jetson.
- **The camera on the Jetson is direct.** On the robot's Jetson `RobotEnv` reads the video hub's RTP multicast
  with a GStreamer child process (`DirectCamera`, chosen when `on_the_jetson()`); a laptop uses WebRTC. Never
  decode video in the run's own Python process on the Jetson: next to rt/lowstate it starves for the GIL.
- **Frames are RGB everywhere**; image I/O only through `g1/core/images.py` (Pillow). OpenCV is not
  used and must not be imported (its bundled ffmpeg collides with PyAV's, which decodes the robot's video).
- **The view never touches the tick.** `--view` (live runs) serves the env's camera slot (`Env.source()`)
  from its own threads; `cli.run` starts it after `env.reset()` and stops it inside `with env`. The robot
  takes one WebRTC client, so the live picture must come out of our process.
- **A task may take a slot.** task.json `fill` declares each `{name}` its instruction names (a note and the
  default `g1 run` uses, which the twin's `success` is scored on); `load_task(d, fill=...)` fills and cleans them
  (`task_fill_max_chars`). In g1 start the model fills them through `run_task`'s arguments (`/task point_at water
  bottle` when typed); a task with an unfilled slot is refused and asked again. `"session": false` keeps a task out
  of g1 start (not yet tried on the robot); `g1 run` and `g1 bench` still run it.
- **Only g1 start's chat searches the web.** Its `look_up(query)` tool runs `VLMClient.search` (OpenRouter's web
  plugin, `web.max_results` in configs/model.json, bounded by `look_up_timeout_s`), then asks the model again with
  the results and without `look_up`. A task's model never searches: it knows only what the robot senses.
- **Hugging Face or OpenRouter** for the model, switched by `provider` in `configs/model.json` (`HF_TOKEN`,
  `VLM_BASE_URL` | `OPENROUTER_API_KEY`; a model id for each in `models`; `g1/vlm.py:PROVIDERS`, `load_model`); no other providers,
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

- **The twin judges sim runs.** A task's `success` (task.json) is scored by `g1/twin/core.py:Evaluator` from
  the simulator's truth every tick (the arm's ray to a target, the feet's drift, forbidden contacts, a fall),
  sent back in the twin's Summary, and labels the `--sim` run (`cli.twin_verdict`, `outcome_source: twin`,
  `false_done`). Feet, not pelvis: the pelvis twists over planted feet. Never sent to the model.
- **The model knows only what the robot knows**: its tools, the camera, the task's objective and rules,
  its own joints (measured), IMU and commands, and its own earlier turns. Never the room's layout (no
  furniture, wall or object positions or sizes) and never the twin's truth (true base pose, contacts), in sim
  or on the robot: `Obs.base_pose` reaches only the records. Task `safety_notes` are behaviour rules; world
  facts go in `notes` (never sent). `tests/agent/test_what_the_model_knows.py` guards it.
- **The knowledge base is the session's only.** `knowledge/SOUL.md` (who the robot is; people write it,
  the robot never does), `knowledge/MEMORY.md` (what it learned; the robot writes it freely: the model's
  `remember` tool and the distill at the end of a session) and `knowledge/sessions/` are per robot and
  git-ignored (`SOUL.example.md` and the README are tracked). Only `g1 start` reads them, after its rules
  (`session.MECHANICS`: the tools and the safety lines, in code so no file can remove them); tasks and
  `g1/agent/` never import `g1.knowledge` (`tests/test_knowledge.py` guards it). Persona and setting
  (the school, the audience) belong in SOUL.md, never in code.
- **A line typed while a hearing task runs in g1 start is heard.** `Microphone.typed` logs it as an utterance
  (`Session.tell_task`), so "stop" works typed as well as said; afterwards nothing said to the task is chat. A task that
  hears runs with the mic off too (`/mic` governs chat only).
- **Hearing is a sense like the camera.** A task with `hears` subscribes `rt/audio_msg` (the robot's own
  speech recognition; voice interaction on in the app) through the link (`g1/microphone.py`), and each
  observation carries `heard`: what was said since the last one, other languages and the robot's own speech
  (`play_state`) dropped. `listen` is a hold the Agent ends when something is heard (it checks between ticks,
  as the settle does; the tool reads nothing). In sim the twin speaks a task's `voices` on the same topic
  after the takeover; the model hears each line when said, never the script. If the built-in assistant gets
  in the way, read the raw microphone (multicast 239.168.123.161:5555) and do our own speech-to-text.

## The decision step (`g1/agent/agent.py`)

takeover → settle (0.03 rad, 0.05 rad/s, 10 ticks, ≥ 0.5 s, 3 s timeout) → snapshot (a fresh frame) →
think (background `Decider.request`; the loop holds) → act (the whole tool, one step file) → settle → feedback (`previous_result`: residuals, commanded base displacement, settle report | the
error) → … `done`/`give_up` end it; `check` answers via `dry_run`, `locate_point` with a pixel's direction; `point_to` (a pixel and a distance) is
bound to the turn's image and measured joints/IMU and aimed by `g1/core/aim.py`, and its result's `aim` is the
measured miss (the first can miss by the pelvis twist the raised arm causes; the second lands);
`listen` ends early on speech (a task that `hears` adds `heard` to every observation);
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
The observation JSON and the system prompt are in `g1/agent/decider.py`. The model answers with one
**tool call** (OpenAI `tools`, `tool_choice: required`): the only reply format, so only models and hosts
that support tool calls are used. `Decision.parse` validates its arguments against the tool's schema and
clamps numbers with a note; no call, or several, is an invalid selection.
