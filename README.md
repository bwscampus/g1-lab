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
vision.py           red-blob detector and the `look` example policy
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
python -m camera --ip <robot-ip>                       # stream smoke test, no control
python   run.py --env robot --policy look --iface <iface> --mode standing --camera-ip <robot-ip>
```

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
