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
skills.py           Skill, the one building block: walk_forward, turn, look, hold, tpose, sixseven, done
                    (+ the internal bookends takeover / handback); SKILLS registry
poses.py            shared pose dicts: STAND (baseline), ARMS_UP, SIXSEVEN
agent.py            the decision step as a Policy: search (ask the model) and replay (a saved run)
decider.py          Context -> Decision via the vision model; python -m decider tries one frame
episode.py          per-step records (JSON + lossless PNG) under runs/; python -m episode inspects them
scene.py            the sim room: textures, furniture, real object meshes; python -m scene fetch
perception.py       Percept / Perceiver: describe a frame with the vision model, on request only
hf.py, worker.py    Hugging Face client (urllib, SSE); background worker with a latest-only result
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
"walk forward". It has a name, a parameter schema (which is also the menu a
vision model chooses from), and `segments()` returning pose segments; a walk is
a segment holding a base velocity. Arguments are bound and validated at
construction, defaults fill in, and the first segment should set the skill's
full entry pose so it works after any other skill. It must not use the
`"start"` goal (reserved for the takeover bookend).

```python
from poses import STAND
from policy import Segment
from skills import Skill

class Nod(Skill):
    name = "nod"
    description = "Dip the elbows twice."
    params = {"type": "object", "required": [],
              "properties": {"reps": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2}}}

    def segments(self):
        out = [Segment(STAND, 2.0, label="to stand")]
        for _ in range(self.reps):
            out += [Segment({18: 0.5, 25: 0.5}, 0.8, label="elbows up"), Segment(STAND, 0.8)]
        return tuple(out)
```

Add it to `skills.SKILLS` and it is runnable on its own (`--policy nod`), with
arguments (`--policy nod:3` or `nod:reps=3`), chained (`--policy tpose,nod:3`),
and offered to the model in `search`. A **routine** wraps skills with the
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

Backend: **Hugging Face Inference Providers** only. Get a fine-grained token with
the "Make calls to Inference Providers" permission and export it as `HF_TOKEN`.
The default model is `perception.DEFAULT_MODEL` (a Qwen3-VL instruct model,
verified live on the router); override with `--vision-model` or
`$G1_VISION_MODEL`, and pin a provider with a suffix such as
`Qwen/Qwen3-VL-30B-A3B-Instruct:deepinfra` when you need structured output.

```
python   run.py --env sim   --policy describe --headless               # needs $HF_TOKEN
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

* **sim** slides the pinned pelvis kinematically — the legs hold the stand
  pose, so this rehearses the loop and the limits, not the gait — and the
  monitor enforces `BASE_VEL_MAX` (0.3 m/s forward) and reports the path.
* **robot** walks for real with `LocoClient.Move`, only with `--walk`. Move is
  a 1 s dead-man command re-sent from a 10 Hz thread and stopped before the arms
  release; it works in FSM 200, so both `--mode`s apply. Pre-flight: on the
  floor or hoisted with feet touching, ~2 m clear all round, no tether to snag,
  spotter on the remote with L2+B.

```
python   run.py --env sim   --policy goto_red --camera-noise --headless
mjpython run.py --env sim   --policy goto_red --sim-target 1.5,0.3,0.6     # slides ~1 m to the ball, "reached"
python   run.py --env robot --policy goto_red --iface <iface> --mode standing --camera-ip <ip>          # refuses: needs --walk
python   run.py --env robot --policy goto_red --iface <iface> --mode standing --camera-ip <ip> --walk   # walks to a red object
```

### Search: a decision loop over skills

`search` is the top-level behaviour: a **step** reads the joint angles and a fresh
camera frame, asks the vision model what to do next (scene, is the path clear,
which skill), runs that skill to its end, and records everything. The robot
stands still while the model thinks; the 50 Hz loop never waits.

**Skills** are the one format for "walk forward" and "lift the arm" (see
*Skills and routines* above). A skill may be longer than one step: the agent
records a step every 3 s while it runs, so a 3 m walk is one decision and five
records. `--list` shows the menu. Chain them from the CLI, no model needed:

```
python   run.py --env sim   --policy walk_forward:0.5,turn:45,tpose,sixseven:1 --headless
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
the skill and its chunk number, the outcome, base pose) and `step_NNNN.png` — the
camera frame stored losslessly, so `load_episode()` gives back the exact RGB
matrix. A long skill produces one record per 3 s chunk (`running`, then
`completed`). That is the dataset a learned policy trains on later.

**Replay a saved run** — no camera, no model, from the start pose:

```
python -m episode runs/<dir>                                    # step table + the equivalent chain
python run.py --env sim --policy replay --episode runs/<dir>   # or --headless, or --env robot --walk
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
