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

```
pip install -e ".[sim,dev]"
```

`unitree_sdk2py` is not on PyPI; install it from
https://github.com/unitreerobotics/unitree_sdk2_python for `--env robot`. The
sim looks for the G1 model in `$G1_MJCF`, then the `mujoco-menagerie` pip
package, then `~/Robotics/mujoco_menagerie/unitree_g1/scene.xml`.

## Layout

```
config.py           29-DoF joint table (DDS order), limits, groups, stand pose
policy.py           Policy / Action interface + SegmentPolicy helper for scripted moves
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
routines.py         Routine = Takeover + motions (+ pauses) + Handback; ROUTINES registry
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
implement `reset(q0)` and `step(t, q)`.

### Robot modes

`--mode` is required for `--env robot`; there is no default.

`--mode gantry`: full bring-up and shutdown for a robot hanging in a
gantry. Damp, FSM 4 (locked stand), FSM 200 (main operation), run the policy,
release the arms, Damp.

`--mode standing`: the robot must already be in FSM 4 (locked stand) or the run
aborts. Goes to FSM 200, runs the policy, releases the arms and returns the
robot to FSM 4. It never damps.

Both modes release the arms on exit, including on Ctrl-C. Pre-flight for either:
not in debug mode, clear space around the arms, someone on the remote with L2+B
ready.
