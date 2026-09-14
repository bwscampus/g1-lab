# g1-lab

Monorepo for Unitree G1 movement policies. Every policy goes through the same
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
```

After `pip install -e .`, `g1` is a shortcut for `python run.py`. `--env` falls back to
`$G1_ENV` and `--policy` to `$G1_POLICY`, so `G1_ENV=sim g1 -p tpose` also works.
`g1 --list` prints the registered envs and policies; `g1 --help` shows every
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
policies/
  tpose.py          neutral -> T-pose -> neutral (neutral -> arms out -> neutral)
tests/              pytest; the sim test runs headless
```

## Writing a policy

A policy commands a set of joints and returns an `Action` per 20 ms tick, or
`None` when done. It never knows which env it is in.

```python
from config import UPPER_BODY
from policy import Segment, SegmentPolicy

class Wave(SegmentPolicy):
    name = "wave"
    joints = UPPER_BODY
    segments = (
        Segment("start", 2.0, weight=lambda a: a),   # ramp arm_sdk weight in while holding
        Segment({18: 0.5}, 1.5),                     # left elbow
        Segment({18: 1.5}, 1.5),
        Segment("start", 2.0, weight=lambda a: 1 - a),
    )
```

Register it in `policies/__init__.py`, then run it through `check`, `sim`,
`robot` in that order. For anything not expressible as pose segments, subclass
`Policy` directly and implement `reset(q0)` and `step(t, q)`.

### Robot modes

`--mode` is required for `--env robot`; there is no default.

`--mode gantry`: full bring-up and shutdown for a robot hanging in a
gantry. Damp, FSM 4 (locked stand), FSM 200 (main operation), run the policy,
release the arms, Damp.

`--mode standing`: the robot must already be in FSM 4 (locked stand) or the run
aborts. Goes to FSM 200, runs the policy, releases the arms and leaves the robot
in FSM 200. It never damps.

Both modes release the arms on exit, including on Ctrl-C. Pre-flight for either:
not in debug mode, clear space around the arms, someone on the remote with L2+B
ready.
