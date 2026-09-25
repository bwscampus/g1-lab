# g1-lab

Monorepo for Unitree G1 movement routines. Every policy goes through the same
two stages, chosen with one flag on the run command:

| stage   | `--env` | what it does |
|---------|---------|--------------|
| 1 sim   | `sim`   | runs the policy on the Menagerie `unitree_g1` model in MuJoCo **and checks it**: every joint's measured angle, every commanded target, command speed, arm_sdk weight and base velocity are gated, and the run fails on a violation. `--headless` is the fast windowless pre-flight; without it you get the viewer. |
| 2 robot | `robot` | deploys live through `unitree_sdk2py`: high-level bring-up with `LocoClient`, then targets on `rt/arm_sdk`; the same joint monitor reports (but never aborts) afterwards. |

```
python   run.py --env sim   --policy tpose --headless   # checked, no window, ~1 s
mjpython run.py --env sim   --policy tpose              # macOS needs mjpython for the viewer
python   run.py --env robot --policy tpose --iface eth0 --mode gantry
python   run.py --env robot --policy sixseven --iface eth0 --mode standing
```

After `pip install -e .`, `g1` is a shortcut for `python run.py`. `--env` falls back to
`$G1_ENV` and `--policy` to `$G1_POLICY`, so `G1_ENV=sim g1 -p tpose` also works.
`g1 --list` prints the registered envs, routines, policies, agents and skills;
`g1 --help` shows every env's flags.

## Setup

Goal: from a fresh machine to a checked sim run. Python 3.10+ on
macOS or Linux; the robot stage needs more and is covered separately.

**1. Python environment.** Any venv or conda env works; this repo was developed
on Python 3.10 in conda.

```
conda create -n g1 python=3.10 -y && conda activate g1     # or: python3 -m venv .venv && source .venv/bin/activate
```

**2. Install the repo.** The `sim` extra pulls in MuJoCo (which also provides
`mjpython` on macOS); `dev` adds pytest and OpenCV (used by `--camera-dir`).

```
git clone <this repo> g1-lab && cd g1-lab
pip install -e ".[sim,dev]"
```

**3. Get the G1 model.** The sim loads the Menagerie `unitree_g1` scene from a
local clone. (The `mujoco-menagerie` package on PyPI ships no model files, so
it does not help here.)

```
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git ~/Robotics/mujoco_menagerie
```

`~/Robotics/mujoco_menagerie/unitree_g1/scene.xml` is where the sim looks by
default. To keep the clone elsewhere, point at the scene file instead:

```
export G1_MJCF=/path/to/mujoco_menagerie/unitree_g1/scene.xml
```

**4. Check it works.**

```
g1 --list                                        # envs, routines, policies, agents, skills
g1 --env sim --policy tpose --headless           # checked run: prints the joint table and PASS
mjpython run.py --env sim --policy demo               # the same, in the viewer (macOS: mjpython; Linux: python)
pytest                                           # ~100 tests, all through headless sim (~100 s)
```

The camera policies need nothing extra: sim renders the head camera itself, or
feeds random / replayed frames with `--camera-noise` / `--camera-dir`.

```
python   run.py --env sim   --policy look --camera-noise --headless
mjpython run.py --env sim   --policy look --sim-target 1.0,0.5,0.6
mjpython run.py --env sim   --policy wave_on_red --sim-target 1.0,0.5,0.6
```

Common problems:

* `On macOS the viewer needs mjpython` — use `mjpython` for `--env sim` without
  `--headless`. It is installed with the `mujoco` package into the same env.
* `No G1 MJCF found` — step 3 is missing, or `G1_MJCF` points at the wrong file.
* `No module named pytest` / `cv2` — the `dev` extra was not installed.

The robot stage (`--env robot`) additionally needs `unitree_sdk2py` and, for
camera policies, `unitree_webrtc_connect` plus `pip install -e ".[camera]"`;
neither is on PyPI and neither is needed for sim.

## Layout

```
config.py           29-DoF joint table (DDS order), limits, groups, stand pose
policy.py           Policy / Action / Obs interface, SegmentPolicy (scripted), ReactivePolicy (camera)
camera.py           Frame sources: WebRTCCamera (robot), DirCamera / NoiseCamera (sim replay); `python -m camera`
vision.py           pure detectors and image geometry: red_blob, bearing, elevation
targets.py          Target / Sighting: what a policy looks for (RedDot, Labeled, Salient, Doorway stub)
behaviors.py        Face(target) turns the waist toward it; GoTo(target) walks to it
skills.py           Skill, the one building block (segments() only); the catalog loader and its three
                    model-facing renderings; SKILLS, menu(), parse_skill()
configs/skills.json the skill catalog: name, class, flags, description, prompt, JSON-schema parameters
                    (walk_forward, turn, look, hold, tpose, sixseven, check, done, give_up + bookends)
poses.py            shared pose dicts: STAND (baseline), ARMS_UP, SIXSEVEN
agent.py            the decision step as a Policy: search (ask the model, feed back, record) and replay
decider.py          the model I/O contract (observation JSON, system prompt, output schema) and the
                    persistent HF session; python -m decider tries one frame
episode.py          the run recorder: step records (JSON + lossless PNG) plus events.jsonl, transcript,
                    protocol, states, usage, status under runs/; python -m episode inspects a run
demo.py             demonstrations on turn 0: a recorded run, a video (ffmpeg + model-picked keyframes)
                    or a goal image, compiled into a portable demo.json; python -m demo prepare/show
scene.py            the sim room: textures, furniture, real object meshes; python -m scene fetch
perception.py       Percept / Perceiver: describe a frame with the vision model, on request only
vlm.py, worker.py   the vision-model client for any OpenAI-compatible API (providers, env, urllib, SSE);
                    background worker with a latest-only result
run.py              CLI and the single run loop shared by all envs
envs/
  base.py           Env interface: setup / reset / step / teardown / report
  monitor.py        JointMonitor: measured + commanded joint bounds, speed, weight, base limits
  sim.py            stage 1 (MuJoCo + the monitor; --headless = no window)
  robot.py          stage 2 (ArmSdk publisher + LocoClient bring-up; monitor report-only)
routines.py         Routine = Takeover + skills (+ pauses) + Handback; Selector (camera-triggered
                    skill); ROUTINES and POLICIES registries
tests/              pytest; everything runs through headless sim
```

## Skills and routines

A **skill** is the one building block — the same format for "lift the arm" and
"walk forward". The class holds only the motion, `segments()` returning pose
segments (a walk is a segment holding a base velocity); everything a model reads
lives in the catalog, `configs/skills.json`: the name, the class, whether it is
enabled / terminal / internal / needs the base, a one-line `description`, the
longer `prompt`, and `parameters` as a JSON schema. Arguments are bound and
validated at construction, defaults fill in, and the first segment should set
the skill's full entry pose so it works after any other skill. It must not use
the `"start"` goal (reserved for the takeover bookend).

```python
# skills.py (or any importable module)
class Nod(Skill):
    def segments(self):
        out = [Segment(STAND, 2.0, label="to stand")]
        for _ in range(self.reps):
            out += [Segment({18: 0.5, 25: 0.5}, 0.8, label="elbows up"), Segment(STAND, 0.8)]
        return tuple(out)
```

```json
{"name": "nod", "skill": "skills.Nod", "enabled": true, "needs_base": false,
 "description": "Dip the elbows reps times.",
 "prompt": "Dip both elbows reps times (1-5, default 2). A gesture toward a person; not a search move.",
 "parameters": {"type": "object",
                "properties": {"reps": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
                               "note": {"$schema": "note"}},
                "required": ["note"], "additionalProperties": false}}
```

Add the entry to `configs/skills.json` and it is runnable on its own
(`--policy nod`), with arguments (`--policy nod:3` or `nod:reps=3`), chained
(`--policy tpose,nod:3`), and offered to the model in `search` (`"offer": false`
keeps it a CLI preset; `"needs_loco": true` marks a robot-only onboard gesture). Every movement
skill takes a `note` (the model's evidence and intent; required of the model,
empty from code and the CLI). `--skills FILE` swaps the whole catalog — prompts
and ranges — without touching code; the loader validates it. A **routine** wraps skills with the
bookends exactly once: takeover, skill, pause, skill, ..., handback. The
baseline both bookends go to is `STAND`, the Menagerie `stand` keyframe's
relaxed hanging-arm pose, which is also what sim starts from. Name a
composition in `routines.py` (`ROUTINES["demo"]`) when it is worth keeping.

Composition happens at the segment level, so every transition between skills is
one continuous command stream and the monitor's velocity limit covers it. A
skill may be any length: the agent records a step every 3 s while one runs.

For anything not expressible as pose segments, subclass `Policy` directly and
implement `reset(obs)` and `step(t, obs)`. `obs.q` is the joint state; see below
for `obs.frame`.

### Camera policies

Every env can hand the policy the latest head-camera frame in `obs.frame`
(`Frame`: RGB uint8 `image`, `stamp`, `seq`) with `obs.frame_age` in seconds on
the env's clock (`inf` when there is no frame). Frames are **latest-only**: they
never queue behind a slow tick, and the same frame (same `seq`) is seen every
tick until a newer one arrives. `step` must not block on a frame; the robot tick
is 20 ms, so heavy per-frame work belongs on another thread. Only policies with
`uses_camera = True` get a camera opened (`--camera auto|on|off` overrides).

Where frames come from:

| env   | source                                                                    |
|-------|---------------------------------------------------------------------------|
| sim   | a `head` camera rendered at the D435 mount every `--camera-every` ticks; `--sim-target x,y,z` adds a red sphere. `--camera-dir DIR` replays image files and `--camera-noise` fuzzes instead of rendering |
| robot | the head camera over WebRTC (`--camera-ip`, default `$UNITREE_ROBOT_IP`), connected before any FSM change; close the Unitree app first |

Two ways to use them:

* **Reactive**: subclass `ReactivePolicy` and implement `track(t, obs) -> pose`,
  called once per new frame. It wraps the result in the takeover/handback
  bookends and a safety envelope (clip to the joint limits minus a margin,
  rate-limit from the last commanded pose, hold when the frame is stale) whose
  defaults sit inside the monitor's gates, since a run can only validate the
  frames it is shown. `look` (`behaviors.py`) turns the waist toward a red blob.
* **Triggers**: `Selector([(predicate, "skill"), ...])` idles at `STAND` until a
  predicate on the frame fires, then runs that registered skill and hands back.
  `wave_on_red` runs `sixseven` when something red is in view.

```
python   run.py --env sim   --policy look --camera-noise --headless
mjpython run.py --env sim   --policy look --sim-target 1.0,0.5,0.6
python -m camera --ip <robot-ip>                       # stream smoke test, no control: prints 1280x720, ~15 fps
python   run.py --env robot --policy look --iface <iface> --mode standing --camera-ip <robot-ip>
```

### Describing the scene with a vision model

The red-blob detector is a stand-in. For real understanding, a `Perceiver`
(`perception.py`) sends frames to a vision-language model on a background
thread and publishes the latest `Percept`: a one-sentence `summary` ("There is
a chair in front of you"), the `objects` in view with a normalised box, a rough
`distance_m` and a `bearing` (rad, computed locally from the box and the camera
FOV), and `path_clear`. Policies read it as `obs.percept`; `obs.percept_age` is
the age of the frame it describes, so it includes the model's latency, and a
policy holds when it grows stale. The control loop never waits on the model.

Backend: **any OpenAI-compatible chat-completions API** (`vlm.py`), chosen by
three variables plus the provider's own key:

```
VLM_PROVIDER   huggingface (default) | openai | openrouter | groq | together | deepinfra |
               mistral | xai | gemini | ollama | custom
VLM_MODEL      the model id to query (required, except huggingface has a default)
VLM_BASE_URL   overrides the provider's endpoint; required for custom (any /v1 server: vLLM, ...)
VLM_API_KEY    overrides the provider's key variable: HF_TOKEN, OPENAI_API_KEY, OPENROUTER_API_KEY,
               GROQ_API_KEY, TOGETHER_API_KEY, DEEPINFRA_API_KEY, MISTRAL_API_KEY, XAI_API_KEY,
               GEMINI_API_KEY; ollama and custom need none
```

Put them in a `.env` in the repo root (`cp .env.example .env`; git-ignored; read
on first use, exported variables win) or export them.
`--vision-provider` / `--vision-model` override the first two per run. With
nothing set, the default is Hugging Face Inference Providers (a fine-grained
token with "Make calls to Inference Providers" as `HF_TOKEN`) and
`vlm.DEFAULT_MODEL`, a Qwen3-VL instruct model verified live on the router;
a suffix such as `Qwen/Qwen3-VL-30B-A3B-Instruct:deepinfra` pins one of its
providers. Structured output (`json_schema`, then `json_object`) and
`stream_options` are requested and stepped down automatically when a server
rejects them.

```
python   run.py --env sim   --policy describe --headless               # needs the provider's key
export HF_TOKEN=hf_...                                              # or e.g.:
export VLM_PROVIDER=openai VLM_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-...
export VLM_PROVIDER=ollama VLM_MODEL=qwen2.5vl                      # a local server, no key
python -m perception head.png                                       # one real request, prints the Percept + latency
mjpython run.py --env sim --policy describe --sim-obstacle 1.2,0,0.225 --vision api --vision-echo
python   run.py --env sim --policy describe --headless --realtime 1 --sim-target 1.0,0.5,0.6 --vision api
```

### Targets and walking

A policy says what it is looking for by naming a `Target` (`targets.py`).
A target finds itself in an observation (`locate`) and returns a `Sighting`:
where it is relative to the camera (`bearing`, `elevation`), its image box and
apparent size, and a `distance_m` only if a detector reported one. Targets never
estimate distance; instead each defines `reached(sighting)` in its own terms.
`RedDot` (pixels) counts as reached when it looms large or drops to the bottom
of the frame; `Labeled` / `Salient` come from the vision model; `Doorway` is a
stub that can be faced but not walked to.

`Face(target)` turns the waist toward the target (`look`, `describe`, `face_door`).
`GoTo(target)` is the walk-to-it loop: turn to face, step forward, hold when the
target is lost, stop when `reached` says so. It drives the base through
`Action.base = (vx, vy, vyaw)`:

* **sim** slides the pinned pelvis kinematically — the legs hold the stand
  pose, so this rehearses the loop and the limits, not the gait — and the
  monitor enforces `BASE_VEL_MAX` (0.3 m/s forward) and reports the path.
* **robot** walks for real with `LocoClient.Move`, only with `--walk`. Move is
  a 1 s dead-man command re-sent from a 10 Hz thread and stopped before the arms
  release; it works in FSM 200, so both `--mode`s apply. Pre-flight: on the
  floor or hoisted with feet touching, ~2 m clear all round, no tether to snag,
  spotter on the remote with L2+B.

**Ctrl-C is safe.** An interrupted run (Ctrl-C, `--max-time`, an error) does not
release the arms where they are: the runner first brings them from the last
commanded pose to the stand pose over 3 s, then fades the arm_sdk weight to 0
over 2 s so the onboard controller takes over smoothly, with the base stopped
from the first tick. Further Ctrl-Cs during that return, and during the robot's
teardown, are ignored — there is no forced release.

```
python   run.py --env sim   --policy goto_red --camera-noise --headless
mjpython run.py --env sim   --policy goto_red --sim-target 1.5,0.3,0.6     # slides ~1 m to the ball, "reached"
python   run.py --env robot --policy goto_red --iface <iface> --mode standing --camera-ip <ip>          # refuses: needs --walk
python   run.py --env robot --policy goto_red --iface <iface> --mode standing --camera-ip <ip> --walk   # walks to a red object
```

### Search: a decision loop over skills

`search` is the top-level behaviour, GPT-Policy's closed loop on this executor:
each **decision** reads a fresh camera frame and the measured joint state
(standing still, once the joints have measurably settled), sends the model one
JSON observation, runs the skill it picks to its end, waits for the joints to
settle again, and feeds back what happened. The robot stands still while the
model thinks; the 50 Hz loop never waits.

The observation is one JSON object — `instruction`, `images` (name, size, age),
`state` (measured `joint_pos` / `joint_vel` / `joint_torque`, waist yaw, the
commanded and measured base pose), `extra` (`env_step`, `decisions_left`,
`can_walk`) and, from the second decision on, `previous_result`: the last
skill's `execution_feedback` (per-joint residual, target vs measured base pose,
a measured settle report) or, when nothing ran, why (`tool_rejected: …`,
`invalid_selection: …`). The reply is one skill selection, `{"name", "arguments"}`,
validated against the catalog's schema; every movement skill carries a `note`
(the evidence and the purpose), `done(summary, hindsight)` and
`give_up(reason, hindsight)` end the run, and `check(skill, arguments)` dry-runs
a skill through the joint monitor without moving. Errors are feedback, not
retries: a rejected or invalid reply costs a decision and the model sees why on
the next turn.

**The menu maps onto the robot's controls.** The model picks a tool and its
arguments; the host plans it, dry-runs it through the joint monitor (a plan
that leaves the limits or moves too fast comes back as `motion_not_executed`
and moves nothing), executes it, waits for the settle and reports. The tools:

```
move(dx_m, dy_m, dyaw_deg)       one relative base displacement -> LocoClient.Move (translate, then turn)
arm_path(waypoints=[...])        joint-space waypoints over waist + arms -> arm_sdk targets
hold(seconds)                    stand still
check(skill, arguments)          dry-run without moving
wave_hand / shake_hand           the onboard controller's own gestures (robot only)
done / give_up                   the model's conclusion (a human assigns the label)
```

The prompt carries the joint table (names, limits, the stand pose) and the
sign conventions. `walk_forward`, `turn`, `look`, `tpose` and `sixseven` remain
as CLI presets (`"offer": false` in the catalog): chainable and replayable,
never offered to the model. Nothing else on the SDK — FSM, damp, torque, sit —
is reachable from a reply. The system prompt (conventions, the scene's hidden obstacles, the
rules, the catalog as bullets and as JSON) is sent once; the conversation is the
model's memory, and only the last `--live-image-window` (8) observations keep
their image (`--fresh-turns` makes every decision a fresh chat). An overloaded
model is retried with backoff, re-observing each time; a decision budget
(`--max-decisions`, 30) ends the run as `budget_exhausted`.

**Skills** are the one format for "walk forward" and "lift the arm" (see
*Skills and routines* above). A skill may be longer than one step: the agent
records a step every 3 s while it runs, so a 3 m walk is one decision and five
records. `--list` shows the menu. Chain them from the CLI, no model needed:

```
python   run.py --env sim   --policy walk_forward:0.5,turn:45,tpose,sixseven:1 --headless
mjpython run.py --env sim   --policy turn:-30,walk_forward:0.6,look:20,hold:1
python   run.py --env sim   --policy 'move:1:0.3:-45,arm_path:waypoints=[{"joints":{"left_elbow":-0.4},"seconds":1.5}]' --headless
```

**With the model**, in a real-looking room (real textures, real object meshes;
fetch once, ~35 MB, git-ignored):

```
python -m scene fetch
export HF_TOKEN=hf_...            # or VLM_PROVIDER=... VLM_MODEL=... and that provider's key
mjpython run.py --env sim --scene room --policy search --goal "find the mug" \
        --sim-objects mug@1.5,1.2 pencil@0.9,-0.4 --camera-size 720x1280 --realtime 1 --max-time 600
```

Objects: `mug`, `marker`, `cracker_box`, `mustard` (YCB scans) and `pencil`
(primitives), placed with `name@x,y` on the floor (the robot starts at the origin
facing +x; the room spans x −2..4, y −3..3, doorway in the +x wall).

**Every step is recorded** under `runs/<timestamp>_<env>_<goal>_<outcome>/`:
`step_NNNN.json` (times, joint angles at start and end, the decision with the
model's raw reply, the skill and its chunk number, the outcome, base pose) and
`step_NNNN.png` — the camera frame stored losslessly, so `load_episode()` gives
back the exact RGB matrix. A long skill produces one record per 3 s chunk
(`running`, then `completed`). That is the dataset a learned policy trains on
later. Beside them, the run trace: `events.jsonl` (every observation with the
exact `input_json`, decision, timing, result, error, retry, verdict),
`transcript.json` (the conversation), `protocol.json` (the system prompt and
schemas), `states.jsonl` (measured joints at 20 Hz), `usage.jsonl` / `usage.json`
(tokens per model call), `config.json` and `status.json`.

**The label is yours.** When the run ends the terminal asks
`Task result [s success / f failed]`; `done` is the model's conclusion, not a
success label. The directory is named by the outcome: `_success` / `_failed`
from your answer, `_unreviewed` when you skip it (Ctrl-C/EOF, or `--no-verdict`),
`_interrupted` / `_failed` for runtime failures; `episode.json` keeps
`model_outcome` and `human_outcome` apart.

**Replay a saved run** — no camera, no model, from the start pose:

```
python -m episode runs/<dir>                                    # step table + the equivalent chain
python run.py --env sim --policy replay --episode runs/<dir>   # or --headless, or --env robot --walk
```

On the robot: `--policy search --goal "find a pencil" --camera-ip <ip> --walk`
(without `--walk` the walking skills are simply not offered to the model), and
`--safety-note "a table 1 m behind the robot"` for what the camera cannot see.
Try one decision on a saved frame first: `python -m decider runs/<dir>/step_0003.png --goal "..."`.

**Demonstrations on turn 0** (GPT-Policy's context compiler). Before the first
observation the model can be shown a previous episode, prefixed `HISTORICAL
DEMONSTRATION` so it is read as reference, never as pending commands:

```
--demo runs/<earlier run>               # its step PNGs are the keyframes; video+action by default:
                                        #   the skill picked on each frame, joint angles at 1 Hz, base pose
--demo walk_to_mug.mp4                  # a phone video: ffmpeg samples 2 fps, the vision model picks
                                        #   <=8 keyframes per 30 s window with a stage and a reason
                                        #   (--demo-select uniform: evenly spaced, no model call)
--demo out/demo.json                    # a bundle compiled once with python -m demo prepare
--ref mug.png                           # a goal photo, labelled; repeatable
--input-json task.json                  # {"instruction", "content": ["text", {"image": p, "label": l},
                                        #                            {"video": p, "mode": "video"}]}
```

`--demo-mode video` sends images only; `video+action` (recorded runs and
bundles only) adds the actions. `--demo-frames` caps the keyframes (12, max
24); demo images are never pruned from the conversation. Video keyframes are
cached by content hash under `runs/.cache/video`. The exact request is archived
in the run as `input/input.json` with the images beside it, so a good run is the
next run's demonstration:

```
python -m demo prepare --goal "find the mug" --demo runs/<good run> out/    # compile once
python -m demo show out/demo.json                                          # what the model gets
mjpython run.py --env sim --scene room --policy search --goal "find the mug" \
        --sim-objects mug@1.5,1.2 --demo out/demo.json --realtime 1 --max-time 600
```

### Vision-model cost and pacing

`--vision auto` (default) uses the offline fake for policies that need vision;
`--vision api` is always explicit because it costs money: roughly 0.5–1.5k
tokens per image, so at the default `--vision-interval 2` a run can make up to
1800 requests an hour. `--vision-interval` and `--max-time` are the knobs.
`describe` turns the waist toward the most salient object and prints each new
summary; `wave_on_person` runs `sixseven` when the model reports a person.
Check and headless sim run faster than realtime, so a real model there
describes frames from well before its answer lands; use the fake, or pace sim
with `--realtime 1`.

### Robot modes

`--mode` is required for `--env robot`; there is no default.

`--mode gantry`: full bring-up and shutdown for a robot hanging in a
gantry. Damp, FSM 4 (locked stand), FSM 200 (main operation), run the policy,
release the arms, Damp.

`--mode standing`: for a robot already standing under its own controller. Records
the current FSM id, goes to FSM 200, runs the policy, releases the arms and
returns the robot to the recorded FSM. It never damps and does not check which
FSM the robot starts in.

Both modes release the arms on exit, including on Ctrl-C. Pre-flight for either:
not in debug mode, clear space around the arms, someone on the remote with L2+B
ready.
