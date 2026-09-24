# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Movement routines for the Unitree G1 (29-DoF). Every routine runs through the same three
environments, selected with `--env`: `sim` (MuJoCo, and the checks that gate a run) and
`robot` (live via `unitree_sdk2py`). Run sim before the robot for anything new; the stages are
not chained automatically. There is no separate check env — every sim run is checked, and
`--env sim --headless` is the fast windowless pre-flight.

## Commands

```
pip install -e ".[sim,dev]"          # unitree_sdk2py is not on PyPI; install from its repo for --env robot
pip install -e ".[camera]"           # aiortc/av/opencv; unitree_webrtc_connect comes from its repo too

python   run.py --env sim --policy tpose --headless        # fast, windowless, fully checked
python   run.py --env sim --policy tpose,turn:45 --headless   # ad hoc chain of skills (--pause between)
python   run.py --env sim --policy demo --headless         # registered routine
mjpython run.py --env sim   --policy demo         # macOS: the viewer only works under mjpython
python   run.py --env sim   --policy tpose --headless
python   run.py --env robot --policy tpose --iface <iface_or_ip> --mode gantry|standing   # --mode is required

python   run.py --env sim --policy look --camera-noise --headless        # fuzz a camera policy
python   run.py --env sim --policy wave_on_red --camera-dir frames/ --headless   # replay recorded frames
mjpython run.py --env sim   --policy look --sim-target 1.0,0.5,0.6      # red sphere for the head camera to find
python   run.py --env robot --policy look --iface <iface> --mode standing --camera-ip <ip>  # + $UNITREE_AES_128_KEY
python -m camera --ip <ip>           # head camera smoke test: fps and frame gaps, no robot control

python   run.py --env sim --policy describe --headless            # vision policy (needs $HF_TOKEN)
HF_TOKEN=hf_... python -m perception head.png                     # one real model request: streams text, prints the Percept
HF_TOKEN=hf_... mjpython run.py --env sim --policy describe --sim-obstacle 1.2,0,0.225 --vision api --vision-echo
mjpython run.py --env sim   --policy goto_red --sim-target 1.5,0.3,0.6   # walk-to-target loop: base slides to the ball
python   run.py --env robot --policy goto_red --iface <iface> --mode standing --camera-ip <ip> --walk  # real walking

python   run.py --env sim --policy walk_forward:0.5,turn:45,tpose --headless  # a preset: skills chain
python -m scene fetch                                                        # room assets (once, ~35 MB, git-ignored)
HF_TOKEN=hf_... mjpython run.py --env sim --scene room --policy search --goal "find the mug" \
        --sim-objects mug@1.5,1.2 --camera-size 720x1280 --realtime 1 --max-time 600   # ask the model each decision
python   run.py ... --policy search --max-decisions 20 --live-image-window 8 --skills my_catalog.json --no-verdict
python   run.py ... --policy search --goal "find the mug" --demo runs/<earlier run>   # demonstration on turn 0
python   run.py --env sim --policy 'move:1:0.3:-45,arm_path:waypoints=[{"joints":{"left_elbow":-0.4}}]' --headless
python -m demo prepare --goal "find the mug" --demo walk_to_mug.mp4 out/ && python -m demo show out/demo.json
python   run.py --env sim --policy replay --episode runs/<dir> --headless    # a saved run, no camera or model
python -m episode runs/<dir>            # step table + the --policy chain that replays it
HF_TOKEN=hf_... python -m decider runs/<dir>/step_0003.png --goal "find the mug"   # one real decision from a frame

g1 --list                            # `g1` == `python run.py`; --env/--policy fall back to $G1_ENV/$G1_POLICY
g1 --help                            # shows every env's flags (--margin, --max-vel, --free-base, --mode, ...)

pytest                               # all tests; sim test runs headless
pytest tests/test_monitor.py::test_out_of_bounds_fails_in_sim
```

Modules live flat at the repo root with absolute imports (`from config import ...`,
`from envs.base import ...`). Pytest gets the root via `pythonpath` in `pyproject.toml`.

## Architecture

The whole system is one loop in `run.py:run()`:

```
with env: obs = env.observe(env.reset()); policy.reset(obs)
          loop: action = policy.step(t, obs); obs = env.observe(env.step(action))
env.report()
```

- **Policy** (`policy.py`): commands a set of joint indices and returns an `Action` per 20 ms
  tick (`CONTROL_DT`), or `None` when done. It never knows which env it is in. It sees an `Obs`:
  `q` plus the latest camera `frame` (or `None`) and `frame_age` in seconds on the env's clock
  (`inf` without a frame). `step` must never block on a frame; on the robot the whole tick is
  20 ms, so heavy per-frame work belongs on another thread with `step` reading its latest
  result. `Policy.uses_camera` tells the runner whether to open one (`--camera auto|on|off`). `Action.q` is
  always the full 29-vector; only `Action.joints` entries are meaningful. `Action.weight` is the
  arm_sdk blend (1 = policy owns the joints, 0 = onboard controller does). `SegmentPolicy` turns
  a list of pose-to-pose `Segment`s into a policy; goal dicts merge onto the previous pose,
  `"start"` as a goal means the pose observed at reset, and a goal key outside the policy's
  `joints` raises at reset.
- **Colour convention: frames are RGB everywhere**, `image[row, col, channel]` with row 0 at
  the top. WebRTC is decoded with `to_ndarray(format="rgb24")`, the sim renderer is RGB, and
  every detector indexes `r, g, b`. OpenCV thinks in BGR, so it only ever sees data through an
  explicit `cv2.cvtColor` at the boundary (`encode_jpeg`, `DirCamera`, `python -m perception`);
  never pass a frame to `cv2.imwrite`/`imshow` raw, and never request `bgr24`. Verified on a
  live robot frame: `rgb[..., ::-1] == bgr` exactly, and the correctly converted PNG shows the
  blue floor tape as blue.
- **Camera** (`camera.py`): every source is a latest-only slot (`Frame`: RGB uint8 `image`,
  `stamp`, `seq`); frames never queue behind a slow tick, and a policy keys per-frame work on
  `seq` because the same frame is seen every tick until a newer one lands. Sources: `WebRTCCamera`
  (robot, `unitree_webrtc_connect` on a daemon asyncio thread, stamped `time.monotonic()`),
  `DirCamera` / `NoiseCamera` (`--camera-dir` / `--camera-noise`, driven by `poll(now)` on the
  sim clock, replacing the render), and the sim's own offscreen render of a `head` camera. The vision model runs **only
  on request**: `Perceiver` is a `worker.Worker` (latest-only result, `request()` ignored while
  one is in flight or within `--vision-interval`); `ReactivePolicy`/`Selector` ask through
  `VisionQuery` at `vision_refresh` (2 s); `--vision auto|off|api`. Measured on the G1: the
  WebRTC stream is 1280x720 H.264 at ~15 fps (`(720, 1280, 3)` uint8, ~67 ms between
  frames); the three `aiortc ... failed to decode, skipping package` warnings at stream start
  are the packets before the first keyframe and are silenced in `WebRTCCamera.start`. `Env.frame()` / `Env.clock()`
  feed `Env.observe`. `config.HEAD_CAMERA_*` hold the D435 mount pose (URDF `d435_joint`) and FOV.
- **Camera policies**: `ReactivePolicy` (`policy.py`) is the closed-loop base: implement
  `track(t, obs) -> Pose`, called once per new fresh frame; it adds the Takeover/Handback phases
  and a safety envelope (clip to limits minus `margin`, rate-limit to `max_vel` from the last
  *commanded* q, hold when the frame is older than `stale_after`). The defaults sit inside
  the monitor's `--margin`/`--max-vel`, which matters because a run can only validate the frames
  it is shown. `Selector` (`routines.py`) is the trigger form: idle at STAND until a frame predicate
  fires, then run one registered skill; it seeds each sub-policy from its own last commanded q
  (the chaining rule below); its rules take the whole `Obs` and re-run on a new frame *or* a new
  percept. Examples in `vision.py` (`look`, red-blob waist tracking; `describe`, vision-model
  narration) and `routines.POLICIES` (`wave_on_red`, `wave_on_person`). `build_policy` resolves
  routine, then policy, then a chain of skills.
- **Policy vs Skill.** *Policy* is the executor contract (`reset/step` at 50 Hz; the only thing
  an env runs — the executor builds nothing else). *Skill* (`skills.py`) is the one building
  block: `segments()` -> pose segments, with arguments bound and validated at construction
  (`Turn(angle_deg=45)`). **Everything a model reads is in `configs/skills.json`** (the layout of
  GPT-Policy's `tools.json`): per skill the name, the implementing class (`"skill":
  "skills.Turn"`), `enabled` / `terminal` / `internal` / `needs_base`, `description`, `prompt`
  and `parameters` (JSON schema; `{"$schema": "note"}` fragments, `{"$template": "skill_name"}`
  for the menu-dependent enum). `load_catalog` validates the file and binds that metadata onto
  the classes; `SKILLS`, `menu()`, `parse_skill()` come from it, `--skills FILE` swaps it.
  Three renderings: `prompt_catalog` (bullets), `function_schemas` (OpenAI-style), `output_schema`
  (what one reply must match; `strict=True` makes every property required/nullable for the
  provider's structured output). Every movement skill has a required `note` — required of the
  model (the jsonschema check), defaulted to "" for code and the CLI. `Skill -> segments -> SegmentPolicy` via `skill_policy`, or several at
  once via `Routine`. "Walk forward" and "lift the arm" are the same format because a `Segment`
  may hold a `base` velocity for its duration as easily as a pose; locomotion durations round up
  to whole ticks so `distance = v*t` is exact. A skill may be **any length** — the agent records
  a step every `STEP_MAX` (3 s) while one runs — so `STEP_MAX` is an observation cadence, not a
  cap. Every skill except `hold`/`look` ends at STAND (boundaries continuous by construction; a
  test chains every pair). Bookends (`Takeover`, `Handback`) are `internal` skills: never in a
  menu, never CLI-chainable. Skills chain from the CLI (`walk_forward:0.5,turn:45,tpose`);
  `menu(allow_base)` hides base skills where the env cannot walk and the runner refuses such
  chains up front.
- **Two ways to drive the executor.** Either ask the vision model what to do next (`--policy
  search --goal …`) or run a preset: a skill chain, a registered routine, or `--policy replay
  --episode runs/<dir>` (rebuilds a saved run as a Routine; no camera, no model). Learning is
  deferred; its seam is the `Decider` interface and the step records.
- **The decision step** (`agent.py`, `Agent(Policy)`) is GPT-Policy's loop on this executor:
  *settle* (hold until measured joint velocity and command error are under tolerance —
  0.05 rad/s, 0.03 rad, N consecutive ticks, 3 s timeout, at least 0.5 s so `StopMove` has
  landed — and report it) -> *snapshot* (a fresh frame) -> *think* (a background
  `Decider.request`; the loop never waits) -> *act* (the whole skill; a record every 3 s) ->
  settle again -> the next observation carries `previous_result`. Sub-policies are seeded from
  the agent's last commanded q. **Errors are feedback**: a reply that fails the schema, names an
  unknown skill, or picks a skill the env refuses becomes `previous_result = {"tool", "error"}`
  and costs a decision; there is no re-ask loop. An overloaded model is retried with their
  backoff (`min(2·2^k, 8) s`, 20 tries, 300 s deadline), **re-observing every time**; a quota
  error or a `step_timeout` fails the run. `env_step` counts decisions (`--max-decisions`,
  ends as `budget_exhausted`); `check` dry-runs a skill through a `JointMonitor` from the
  commanded pose (`dry_run`) and its verdict flows back like any result; `done`/`give_up` end
  the run with the model's conclusion. `close()` stops the decider, asks the human verdict
  (`verdict` callable; `--no-verdict`), then closes the recorder — also on Ctrl-C. No
  pipelining: a request issued before the skill ends would decide on a stale image.
- **Decider** (`decider.py`), their `AgentSession`: `start(AgentContext)` once (the system
  prompt with conventions, `scene.safety_notes`, the rules, the catalog as bullets and JSON;
  the function schemas; the output schema), then `decide(AgentTurn)` per decision. The
  observation is one JSON object (`observation()`: instruction, images, state with
  joint_pos/vel/torque + waist yaw + base_pose_cmd/env, extra, previous_result; floats rounded
  to 1e-6, compact separators). `Decision.parse` = a whole JSON object (a complete fence is
  fine, a fragment in prose is not) -> menu name -> `validate_args` (coerce, clamp with notes)
  -> jsonschema Draft 2020-12 against the skill's parameters; failures raise `ProtocolError`
  with the raw text. `HFDecider` keeps one conversation per run (system once; per turn the
  observation text + image, then its own reply verbatim); only the last `live_image_window`
  (8) observations keep their image, text is never pruned, demo content on turn 0 never is;
  `fresh_turns` is the stateless mode. It asks the router for `json_schema` output and steps
  down to `json_object`, then none, on a 400. **Fakes are test doubles only**
  (`tests/doubles.py`): `search` always uses the real model; no `--decider fake`.
- **The tools map onto the robot's controls** (the analogue of their `ToolExecutor`). The model
  picks a catalog entry; its class plans segments; the env is the only thing that touches the
  SDK: `Action.q` → `arm_sdk` joint targets, `Action.base` → `LocoClient.Move` via
  `BaseCommander`, `Action.command` → one onboard `LocoClient` call (`Segment.command`, emitted
  on the segment's first tick; allow-listed in `skills.LOCO_METHODS` = `WaveHand`, `ShakeHand`;
  sim aborts on it, the catalog hides such skills where `env.has_loco` is false). The model's
  menu is general primitives: `move(dx_m, dy_m, dyaw_deg)` (translate, then turn, so the
  dead-reckoned end pose is exactly the request), `arm_path(waypoints=[{joints: {name: rad},
  seconds}])` (joint-space waypoints over the 17 arm_sdk joints; the prompt carries the joint
  table and sign conventions; names and limits are checked at construction and by the schema's
  `arm_joints` template), `hold`, `check`, `wave_hand`/`shake_hand` (robot only: weight 1→0,
  the onboard gesture, 0→1), `done`, `give_up`. `walk_forward`, `turn`, `look`, `tpose`,
  `sixseven` stay as CLI presets (`"offer": false`: chainable and replayable, never offered).
  **Every movement is dry-run before it executes** (`dry_run` through a `JointMonitor` from the
  commanded pose — their IK check before submission): a limit, speed or base violation is fed
  back as `motion_not_executed` with the violations, spends a decision, and moves nothing.
  FSM, damp, torque, sit, squat and stand height are unreachable from a model reply by
  construction. JSON arguments work on the CLI:
  `--policy 'move:1:0.3:-45,arm_path:waypoints=[{"joints":{"waist_yaw":0.5}}]'`.
- **Demonstrations** (`demo.py`), their context compiler: a request is an instruction plus
  content parts (`TextPart`, `ImagePart(path, label, detail)`, `VideoPart(path, label, detail,
  mode)`) from `--demo` / `--ref` / `--input-json` (`load_manifest` enforces their rules).
  `prepare` replaces every video part in place with text + images before the session starts:
  a recorded run (`compile_run`: the step PNGs are the keyframes, thinned to `--demo-frames`
  keeping first and last; `video+action` adds the decided skill, 1 Hz joint samples with `=`
  for unchanged values from `states.jsonl`, and the base pose), a video file (`compile_video`:
  ffprobe/ffmpeg via argument arrays, 2 fps candidates, ≤ 24 per 30 s window, `ModelSelector`
  = one HF call per window with a `select_video_frames` schema + a review call across windows,
  or `UniformSelector`; cached by content hash under `runs/.cache/video` with an atomic
  publish), or a `demo.json` bundle (`write_bundle`/`load_bundle`, relative image paths). The
  parts go at the head of the turn-0 user message (`AgentTurn.content`; `HISTORICAL` preamble +
  a mode sentence, per keyframe a JSON label line and the image, a summary) and are never
  pruned; `input/input.json` archives the exact request (`save_input`), the `input_manifest`
  event and the turn-0 `observation` event record it. `video+action` on a bare video is an
  error; ffmpeg is required only for video files.
- **Records** (`episode.py`): `runs/<ts>_<env>_<goal>_<outcome>/` with `episode.json` +
  `step_NNNN.json` + `step_NNNN.png` (the frame losslessly; `load_episode` returns the exact
  RGB array, never JPEG) plus their run trace: `config.json`, `events.jsonl` (append-only,
  every event with `at_s`: observation with the verbatim `input_json`, decision_timing,
  model_decision, tool_timing, execution_result (the full feedback incl. `motion_progress`,
  which the model view omits), tool_error, model_retry, terminal, return_home,
  execution_finished, human_evaluation, run_finished), `transcript.json`, `protocol.json`,
  `states.jsonl` (20 Hz), `usage.jsonl`/`usage.json` (tokens per call; no pricing),
  `status.json`. `close(status, human)` settles the outcome their way — the human's answer
  wins, a model conclusion without one is `unreviewed`, else the runtime status — and renames
  the directory. PNGs and states go through a writer thread; events are written inline, in
  order. `runs/` is git-ignored.
- **Room scene** (`scene.py`): `--scene room` adds a textured floor and walls (Poly Haven, CC0),
  a table and chairs, a doorway, lights, and `--sim-objects name@x,y[,z]` places real YCB
  meshes (CC-BY 4.0) or a primitives pencil, so the real model sees a real-looking scene.
  `python -m scene fetch` downloads into the git-ignored `assets/` with `ATTRIBUTION.md`.
  MuJoCo reads PNG textures only (JPGs are converted). The floor slab's top is z = 0 (the
  Menagerie plane is dropped 2 cm) so objects authored base-down at z = 0 rest on it. Furniture is
  deliberately un-red so the pixel red-blob path is not fooled. `--camera-size 720x1280` renders
  at the robot's resolution (raises the offscreen framebuffer).
- **Targets** (`targets.py`): a policy names what it is looking for. `Target.locate(obs)` is
  pure and returns a `Sighting` — location relative to the camera (`bearing` +right,
  `elevation` +up, in rad) plus the normalised image box (`x, y, width, height`, `area`),
  `frame_seq`/`stamp`, and `distance_m` **only when a detector reported it**: targets never
  estimate distance. `reached(sighting)` is each target's own arrival criterion (`RedDot`: it
  looms past `reach_fraction` or sinks below `reach_elevation`, because the head camera looks
  47° down and a floor-level dot leaves the bottom of the frame ~0.75 m out). `update(obs, t)`
  is the stateful tracker (locates once per new frame/percept seq, keeps `last`, `age(t)` counts
  the input's own age). `uses_vision` says whether locating needs the VLM (`Labeled`, `Salient`,
  `Doorway`) or pixels (`RedDot`). `Doorway` is a stub with `can_reach = False`: `Face` accepts
  it, `GoTo` refuses it at construction — never register a walker with no stop condition.
  `seen(target)` builds a Selector predicate.
- **Behaviours** (`behaviors.py`): `Face(target)` turns the waist toward the target (`look`,
  `describe` are thin subclasses). `GoTo(target)` is the top-level walk-to-it loop: takeover →
  seek (turn to face, then step forward at `v_fwd`; hold when lost; `reached` confirmed twice →
  `finish()`) → return → handback. It steers the base only and keeps the waist at 0. Both log
  the commanded yaw per frame seq and apply a sighting's bearing relative to the yaw *at
  capture*, since sightings (and especially VLM percepts) land after their frame.
- **Base velocity** (`Action.base = (vx, vy, vyaw)` or `None`; `config.BASE_VEL_MAX`):
  `ReactivePolicy.drive(t, obs)` supplies it every tracking tick and it is `None` in every other
  phase, so the base always stops before the return-to-stand. The monitor flags `base_*` violations
  and integrates a kinematic pose for the report; sim slides the pinned pelvis (legs hold the
  stand pose — a rehearsal of the loop, not gait; `--free-base` aborts on a base command); the
  robot needs `--walk`, which starts a `BaseCommander` thread (latest-only slot, `Move` re-sent
  at 10 Hz, `StopMove` when the command clears and again before the arms are released). `Move`
  is a blocking RPC and a 1 s dead-man on the robot, and it works in FSM 200, so walking adds no
  FSM transition. Never call the SDK from `step`.
- **Perception** (`perception.py`): a `Perceiver` turns frames into `Percept`s (one-sentence
  `summary`, `objects` with normalised box `x,y,width,height`, `distance_m`, and a `bearing`
  computed locally from `x` via `vision.bearing`, plus `path_clear`) and publishes them
  latest-only from a daemon worker, one request in flight, at most one per `min_interval`.
  `Env.observe` offers each frame and attaches `obs.percept` with `obs.percept_age` = env-clock
  age of the *frame described* (so the model's latency is included). `HFPerceiver` is the only
  network backend: Hugging Face Inference Providers (`https://router.huggingface.co/v1`,
  `$HF_TOKEN`, model `$G1_VISION_MODEL` / `--vision-model`, default `perception.DEFAULT_MODEL`),
  stdlib `urllib`, streamed SSE, `response_format: json_object` dropped automatically on a 400.
  A rule-based perceiver double runs inline in `tests/doubles.py` so the tests are
  deterministic; `--vision auto` picks it for policies with `uses_vision`, `api` is always
  explicit. The runner starts the perceiver before the env and stops it after, so it never gates
  arm release. `ReactivePolicy.fresh(obs)` is the hook that keys `track` on `percept.seq`
  instead of `frame.seq`; `Describe` also remembers the commanded waist yaw per frame seq and
  aims relative to the yaw *at capture*, because the percept lands seconds after its frame.
  Check and headless sim outrun a real model: use the fake there, or `--realtime 1` in sim.
- **Routine**: one `SegmentPolicy` = `Takeover + s1 + Hold(pause) + s2 + ... + Handback`, labels
  prefixed with the skill name. Both bookends go to `poses.STAND`, the Menagerie `stand` keyframe
  arm pose (= `config.STAND_Q`), so the sim takeover is a pure weight ramp with no visible motion.
  Composition is at the segment level on purpose: chaining Policy objects would restart each one
  from the env's *measured* q, which on the robot lags the command by gravity sag and produces a
  boundary jump no gate can see. If a non-segment policy ever needs chaining, seed it from the
  previous policy's last commanded q, never measured q.
- **Env** (`envs/base.py`): `setup/reset/step/teardown/report`, plus a classmethod `add_args`
  that registers env-specific CLI flags on the shared parser. Raise `EnvAbort` to stop early
  while still getting `report()` called. Registries are plain dicts: `envs.ENVS`,
  `skills.SKILLS` (building blocks, chainable from the CLI), `routines.ROUTINES` (named
  compositions), `routines.POLICIES` and `agent.AGENTS`. `run.build_policy` resolves `--policy`:
  routine, policy, then a comma-separated chain of skills. A new skill must be added to
  `skills.SKILLS` to be runnable.
- **The monitor** (`envs/monitor.py`) is the check that used to be its own env: per tick it
  records the **measured** angle of every joint and the commanded target, and flags a measured
  or commanded angle outside `JOINT_LO/HI ± --margin`, a blended-command speed over `--max-vel`,
  a weight outside [0, 1] and a base velocity over `config.BASE_VEL_MAX`; it integrates the base
  pose kinematically. `report()` prints the table (measured min/max, commanded min/max, peak
  velocity, a `!` on any joint that left its bounds) and fails the run. Sim runs it strictly and
  aborts after `--max-violations`; the robot runs it **report-only** over measured angles
  (teardown already releases the arms, so stopping mid-run is its own risk). Measured angles are
  the point: physics, contact or gravity can put a joint where no command asked.
- **Blend semantics are emulated**: sim computes `cmd = (1-w)*hold + w*target` so what you see
  matches what arm_sdk does on the robot. `hold` is the Menagerie `stand` keyframe
  (`config.STAND_Q`) in sim and the live pose on the robot.
- **Joint indexing** (`config.py`): DDS order of `LowCmd_.motor_cmd`, which is also the
  Menagerie `unitree_g1` actuator order. `tests/test_config.py` asserts the hardcoded limit table
  matches the model. Index 29 is the arm_sdk weight slot on the robot, not a joint.
- **Robot env** uses only high-level control (the camera, when a policy uses it, connects over
  WebRTC *before* any FSM transition so a bad link fails before the robot is touched, and is
  closed last in `teardown`; the robot takes one WebRTC client, so disconnect the Unitree app;
  `report()` counts tick overruns): `LocoClient` FSM transitions then targets
  published on `rt/arm_sdk`. `--mode gantry` does Damp -> FSM 4 -> FSM 200, runs, releases the
  arms, Damps. `--mode standing` records the current FSM (no check), goes to FSM 200, runs,
  releases the arms and returns to the recorded FSM (never damps). It refuses any joint outside
  `UPPER_BODY` (waist + arms) and always releases the arms in `teardown`, including on Ctrl-C.
  Walking is high-level only (`LocoClient.Move` behind `--walk`); do not add low-level leg
  control through this path.
- **Sim env** pins the pelvis by overwriting the free-joint state each substep;
  `--free-base` disables that. The scene path is resolved from `$G1_MJCF`, then the
  `mujoco-menagerie` pip package, then `~/Robotics/mujoco_menagerie`, and loaded through
  `MjSpec` (mujoco >= 3.2) to add the `head` camera on `torso_link` and, with `--sim-target
  x,y,z`, a red sphere and, with `--sim-obstacle x,y,z`, a grey chair-sized box (kept neutral so
  the red-blob fake is not fooled). The camera renders every `--camera-every` ticks (default 3)
  via `mujoco.Renderer`, which works alongside the passive viewer under `mjpython`. `--realtime`
  defaults to 1.0 with the viewer and to unpaced when headless.
