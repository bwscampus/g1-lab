# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Movement routines for the Unitree G1 (29-DoF). Every routine runs through the same three
environments, selected with `--env` on the run command: `check` (bounds/velocity sanity
check, no hardware), `sim` (MuJoCo viewer), `robot` (live via `unitree_sdk2py`). Run them in
that order for any new motion or routine. The stages are not chained automatically.

## Commands

```
pip install -e ".[sim,dev]"          # unitree_sdk2py is not on PyPI; install from its repo for --env robot
pip install -e ".[camera]"           # aiortc/av/opencv; unitree_webrtc_connect comes from its repo too

python   run.py --env check --policy tpose                 # single motion
python   run.py --env check --policy tpose,sixseven        # ad hoc chain (--pause between)
python   run.py --env check --policy demo                  # registered routine
mjpython run.py --env sim   --policy demo         # macOS: the viewer only works under mjpython
python   run.py --env sim   --policy tpose --headless
python   run.py --env robot --policy tpose --iface <iface_or_ip> --mode gantry|standing   # --mode is required

python   run.py --env check --policy look --camera-noise          # camera policy fuzzed with random frames
python   run.py --env check --policy wave_on_red --camera-dir frames/   # replay recorded frames (sorted by name)
mjpython run.py --env sim   --policy look --sim-target 1.0,0.5,0.6      # red sphere for the head camera to find
python   run.py --env robot --policy look --iface <iface> --mode standing --camera-ip <ip>  # + $UNITREE_AES_128_KEY
python -m camera --ip <ip>           # head camera smoke test: fps and frame gaps, no robot control

python   run.py --env check --policy describe --camera-noise      # vision policy with the offline fake perceiver
HF_TOKEN=hf_... python -m perception head.png                     # one real model request: streams text, prints the Percept
HF_TOKEN=hf_... mjpython run.py --env sim --policy describe --sim-obstacle 1.2,0,0.225 --vision api --vision-echo
mjpython run.py --env sim   --policy goto_red --sim-target 1.5,0.3,0.6   # walk-to-target loop: base slides to the ball
python   run.py --env robot --policy goto_red --iface <iface> --mode standing --camera-ip <ip> --walk  # real walking

g1 --list                            # `g1` == `python run.py`; --env/--policy fall back to $G1_ENV/$G1_POLICY
g1 --help                            # shows every env's flags (--margin, --max-vel, --free-base, --mode, ...)

pytest                               # all tests; sim test runs headless
pytest tests/test_check.py::test_out_of_bounds_fails
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
  `DirCamera` / `NoiseCamera` (check, driven by `poll(now)` on the check clock), and the sim's
  own offscreen render of a `head` camera (stamped with sim time). Measured on the G1: the
  WebRTC stream is 1280x720 H.264 at ~15 fps (`(720, 1280, 3)` uint8, ~67 ms between
  frames); the three `aiortc ... failed to decode, skipping package` warnings at stream start
  are the packets before the first keyframe and are silenced in `WebRTCCamera.start`. `Env.frame()` / `Env.clock()`
  feed `Env.observe`. `config.HEAD_CAMERA_*` hold the D435 mount pose (URDF `d435_joint`) and FOV.
- **Camera policies**: `ReactivePolicy` (`policy.py`) is the closed-loop base: implement
  `track(t, obs) -> Pose`, called once per new fresh frame; it adds the Takeover/Handback phases
  and a safety envelope (clip to limits minus `margin`, rate-limit to `max_vel` from the last
  *commanded* q, hold when the frame is older than `stale_after`). The defaults sit inside
  check's `--margin`/`--max-vel`, which matters because check can only validate the frames it is
  shown. `Selector` (`routines.py`) is the trigger form: idle at STAND until a frame predicate
  fires, then run one registered motion; it seeds each sub-policy from its own last commanded q
  (the chaining rule below); its rules take the whole `Obs` and re-run on a new frame *or* a new
  percept. Examples in `vision.py` (`look`, red-blob waist tracking; `describe`, vision-model
  narration) and `routines.POLICIES` (`wave_on_red`, `wave_on_person`). `build_policy` resolves
  routine, then policy, then motions.
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
  phase, so the base always stops before the return-to-stand. check flags `base_*` violations
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
  `FakePerceiver` runs inline (no thread) off `vision.red_blob` so check and tests are
  deterministic; `--vision auto` picks it for policies with `uses_vision`, `api` is always
  explicit. The runner starts the perceiver before the env and stops it after, so it never gates
  arm release. `ReactivePolicy.fresh(obs)` is the hook that keys `track` on `percept.seq`
  instead of `frame.seq`; `Describe` also remembers the commanded waist yaw per frame seq and
  aims relative to the yaw *at capture*, because the percept lands seconds after its frame.
  Check and headless sim outrun a real model: use the fake there, or `--realtime 1` in sim.
- **Motion vs Routine**: a `Motion` (`motions/`) is a factory for segments with no bookends;
  contract: no `"start"` goal, first segment sets its full entry pose, may end anywhere. A
  `Routine` (`routines.py`) is one `SegmentPolicy` = `Takeover + m1 + Hold(pause) + m2 + ... +
  Handback` (bookends in `motions/bookends.py`), labels prefixed with the motion name. Both
  bookends go to `motions.poses.STAND`, the Menagerie `stand` keyframe arm pose (=
  `config.STAND_Q`), so the sim takeover is a pure weight ramp with no visible motion.
  Composition is at the segment level on purpose: chaining Policy objects would restart each one
  from the env's *measured* q, which on the robot lags the command by gravity sag and produces a
  boundary jump the check stage cannot see. If a non-segment policy ever needs chaining, seed it
  from the previous policy's last commanded q, never measured q.
- **Env** (`envs/base.py`): `setup/reset/step/teardown/report`, plus a classmethod `add_args`
  that registers env-specific CLI flags on the shared parser. Raise `EnvAbort` to stop early
  while still getting `report()` called. Registries are plain dicts: `envs.ENVS`,
  `motions.MOTIONS` (building blocks, chainable from the CLI) and `routines.ROUTINES` (named
  compositions). `run.build_policy` resolves `--policy`: routine name first, else comma-separated
  motions. A new motion must be added to `motions/__init__.py` to be runnable.
- **Blend semantics are emulated everywhere**: `check` and `sim` both compute
  `cmd = (1-w)*hold + w*target` so what you see in sim matches what arm_sdk does on the robot.
  `hold` is the Menagerie `stand` keyframe (`config.STAND_Q`) in check/sim and the live pose on
  the robot.
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
