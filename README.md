# g1-lab

Monorepo for Unitree G1 movement routines. Every routine goes through the same
three stages, chosen with one flag on the run command:

| stage   | `--env` | what it does |
|---------|---------|--------------|
| 1 check | `check` | naive sanity check: joint bounds (with margin), command speed, arm_sdk weight range. No hardware, no GUI. |
| 2 sim   | `sim`   | replays the policy in the MuJoCo viewer on the Menagerie `unitree_g1` model. |
| 3 robot | `robot` | deploys live through `unitree_sdk2py`: high-level bring-up with `LocoClient`, then targets on `rt/arm_sdk`. |

```
python   run.py --env check --policy tpose
mjpython run.py --env sim   --policy tpose            # macOS needs mjpython for the viewer
python   run.py --env robot --policy tpose --iface eth0 --mode gantry
python   run.py --env robot --policy sixseven --iface eth0 --mode standing
```

After `pip install -e .`, `g1` is a shortcut for `python run.py`. `--env` falls back to
`$G1_ENV` and `--policy` to `$G1_POLICY`, so `G1_ENV=sim g1 -p tpose` also works.
`g1 --list` prints the registered envs, routines and motions; `g1 --help` shows every
env's flags.

## Setup

Goal: from a fresh machine to the `check` and `sim` stages. Python 3.10+ on
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
g1 --list                                        # envs, routines, policies, motions
g1 --env check --policy tpose                    # stage 1: prints a joint table and PASS
python   run.py --env sim --policy tpose --headless   # stage 2 without a window
mjpython run.py --env sim --policy demo               # stage 2 in the viewer (macOS: mjpython; Linux: python)
pytest                                           # 31 tests, sim ones run headless (~10 s)
```

The camera policies need nothing extra in these two stages: `check` feeds
random or replayed frames and `sim` renders the head camera itself.

```
python   run.py --env check --policy look --camera-noise
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
neither is on PyPI and neither is needed for `check` or `sim`.

## Layout

```
config.py           29-DoF joint table (DDS order), limits, groups, stand pose
policy.py           Policy / Action / Obs interface, SegmentPolicy (scripted), ReactivePolicy (camera)
camera.py           Frame sources: WebRTCCamera (robot), DirCamera / NoiseCamera (check); `python -m camera`
vision.py           pure detectors and image geometry: red_blob, bearing, elevation
targets.py          Target / Sighting: what a policy looks for (RedDot, Labeled, Salient, Doorway stub)
behaviors.py        Face(target) turns the waist toward it; GoTo(target) walks to it
skills.py           Skill: the decision-level unit (walk_forward, turn, look, hold, arms_up, wave, done)
agent.py            the decision step as a Policy: search (ask the model) and replay (a saved run)
decider.py          Context -> Decision via the vision model; python -m decider tries one frame
episode.py          per-step records (JSON + lossless PNG) under runs/; python -m episode inspects them
scene.py            the sim room: textures, furniture, real object meshes; python -m scene fetch
perception.py       Percept / Perceiver: describe a frame with the vision model, on request only
hf.py, worker.py    Hugging Face client (urllib, SSE); background worker with a latest-only result
run.py              CLI and the single run loop shared by all envs
envs/
  base.py           Env interface: setup / reset / step / teardown / report
  check.py          stage 1
  sim.py            stage 2
  robot.py          stage 3 (ArmSdk publisher + LocoClient bring-up)
motions/            reusable building blocks (no takeover/handback), MOTIONS registry
  poses.py          shared pose dicts: STAND (baseline), ARMS_UP, SIXSEVEN
  bookends.py       Takeover, Handback, Hold
  tpose.py          TPose(hold, rise)
  sixseven.py       SixSeven(reps, swing_time, ...)
routines.py         Routine = Takeover + motions (+ pauses) + Handback; Selector (camera-triggered
                    motion); ROUTINES and POLICIES registries
tests/              pytest; the sim test runs headless
```

## Motions and routines

A **motion** is a reusable building block: a class returning a tuple of pose
segments, with no takeover/handback. Its first segment should set its full entry
pose so it works after any other motion; it may end anywhere. It must not use
the `"start"` goal (reserved for the takeover bookend). Parameters go through
`__init__`.

```python
from motions.poses import STAND
from policy import Motion, Segment

class Wave(Motion):
    name = "wave"

    def __init__(self, reps: int = 2):
        self.reps = reps

    def segments(self):
        out = [Segment(STAND, 2.0, label="to stand")]
        for _ in range(self.reps):
            out += [Segment({18: 0.5}, 0.8, label="elbow up"), Segment({18: 1.28}, 0.8)]
        return tuple(out)
```

Register it in `motions/__init__.py`, then it is runnable on its own
(`--policy wave`) or chained (`--policy tpose,wave`). A **routine** wraps motions
with the bookends exactly once: takeover, motion, pause, motion, ..., handback.
The baseline both bookends go to is `STAND`, the Menagerie `stand` keyframe's relaxed
hanging-arm pose, which is also what `check` and `sim` start from.
Name a composition in `routines.py` (`ROUTINES["demo"]`) when it is worth keeping.

Composition happens at the segment level, so every transition between motions is
one continuous command stream and the `check` env's velocity limit covers it.

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
| check | none by default; `--camera-dir DIR` replays image files, `--camera-noise` fuzzes |
| sim   | a `head` camera rendered at the D435 mount every `--camera-every` ticks; `--sim-target x,y,z` adds a red sphere |
| robot | the head camera over WebRTC (`--camera-ip`, default `$UNITREE_ROBOT_IP`), connected before any FSM change; close the Unitree app first |

Two ways to use them:

* **Reactive**: subclass `ReactivePolicy` and implement `track(t, obs) -> pose`,
  called once per new frame. It wraps the result in the takeover/handback
  bookends and a safety envelope (clip to the joint limits minus a margin,
  rate-limit from the last commanded pose, hold when the frame is stale) whose
  defaults sit inside `check`'s gates, since `check` can only validate the
  frames it is shown. `look` (`vision.py`) turns the waist toward a red blob.
* **Triggers**: `Selector([(predicate, "motion"), ...])` idles at `STAND` until a
  predicate on the frame fires, then runs that registered motion and hands back.
  `wave_on_red` runs `sixseven` when something red is in view.

```
python   run.py --env check --policy look --camera-noise
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

Backend: **Hugging Face Inference Providers** only. Get a fine-grained token with
the "Make calls to Inference Providers" permission and export it as `HF_TOKEN`.
The default model is `perception.DEFAULT_MODEL` (a Qwen3-VL instruct model,
verified live on the router); override with `--vision-model` or
`$G1_VISION_MODEL`, and pin a provider with a suffix such as
`Qwen/Qwen3-VL-30B-A3B-Instruct:deepinfra` when you need structured output.

```
python   run.py --env check --policy describe --camera-noise        # offline: fake perceiver
export HF_TOKEN=hf_...
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

* **check** enforces `BASE_VEL_MAX` (0.3 m/s forward) and reports the path.
* **sim** slides the pinned pelvis kinematically — the legs hold the stand
  pose, so this rehearses the loop and the limits, not the gait.
* **robot** walks for real with `LocoClient.Move`, only with `--walk`. Move is
  a 1 s dead-man command re-sent from a 10 Hz thread and stopped before the arms
  release; it works in FSM 200, so both `--mode`s apply. Pre-flight: on the
  floor or hoisted with feet touching, ~2 m clear all round, no tether to snag,
  spotter on the remote with L2+B.

```
python   run.py --env check --policy goto_red --camera-noise
mjpython run.py --env sim   --policy goto_red --sim-target 1.5,0.3,0.6     # slides ~1 m to the ball, "reached"
python   run.py --env robot --policy goto_red --iface <iface> --mode standing --camera-ip <ip>          # refuses: needs --walk
python   run.py --env robot --policy goto_red --iface <iface> --mode standing --camera-ip <ip> --walk   # walks to a red object
```

### Search: a decision loop over skills

`search` is the top-level behaviour: a **step** reads the joint angles and a fresh
camera frame, asks the vision model what to do next (scene, is the path clear,
which skill), runs that skill to its end, and records everything. The robot
stands still while the model thinks; the 50 Hz loop never waits.

**Skills** are the one format for "walk forward" and "lift the arm": a name, a
parameter menu the model chooses from, a 3 s bound, and a build that returns
pose segments (a walk is a segment holding a base velocity). `--list` shows them.
Chain them from the CLI exactly like motions, no model needed:

```
python   run.py --env check --policy walk_forward:0.5,turn:45,arms_up,wave
mjpython run.py --env sim   --policy turn:-30,walk_forward:0.6,look:20,hold:1
```

**With the model**, in a real-looking room (real textures, real object meshes;
fetch once, ~35 MB, git-ignored):

```
python -m scene fetch
export HF_TOKEN=hf_...
mjpython run.py --env sim --scene room --policy search --goal "find the mug" \
        --sim-objects mug@1.5,1.2 pencil@0.9,-0.4 --camera-size 720x1280 --realtime 1 --max-time 600
```

Objects: `mug`, `marker`, `cracker_box`, `mustard` (YCB scans) and `pencil`
(primitives), placed with `name@x,y` on the floor (the robot starts at the origin
facing +x; the room spans x −2..4, y −3..3, doorway in the +x wall).

**Every step is recorded** under `runs/<timestamp>_<env>_<goal>/`: `step_NNNN.json`
(times, joint angles at start and end, the decision with the model's raw reply,
the skill, the outcome, base pose) and `step_NNNN.png` — the camera frame stored
losslessly, so `load_episode()` gives back the exact RGB matrix. That is the
dataset a learned policy trains on later.

**Replay a saved run** — no camera, no model, from the start pose:

```
python -m episode runs/<dir>                                    # step table + the equivalent chain
python run.py --env sim --policy replay --episode runs/<dir>   # or --env check / --env robot --walk
```

On the robot: `--policy search --goal "find a pencil" --camera-ip <ip> --walk`
(without `--walk` the walking skills are simply not offered to the model).
Try one decision on a saved frame first: `python -m decider runs/<dir>/step_0003.png --goal "..."`.

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
