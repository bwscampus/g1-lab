# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Movement policies for the Unitree G1 (29-DoF). Every policy runs through the same three
environments, selected with `--env` on the run command: `check` (bounds/velocity sanity
check, no hardware), `sim` (MuJoCo viewer), `robot` (live via `unitree_sdk2py`). Run them in
that order for any new policy. The stages are not chained automatically.

## Commands

```
pip install -e ".[sim,dev]"          # unitree_sdk2py is not on PyPI; install from its repo for --env robot

python   run.py --env check --policy tpose
mjpython run.py --env sim   --policy tpose        # macOS: the viewer only works under mjpython
python   run.py --env sim   --policy tpose --headless
python   run.py --env robot --policy tpose --iface <iface_or_ip> --mode gantry|standing   # --mode is required

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
with env: q = env.reset(); policy.reset(q); loop: action = policy.step(t, q); q = env.step(action)
env.report()
```

- **Policy** (`policy.py`): commands a set of joint indices and returns an `Action` per 20 ms
  tick (`CONTROL_DT`), or `None` when done. It never knows which env it is in. `Action.q` is
  always the full 29-vector; only `Action.joints` entries are meaningful. `Action.weight` is the
  arm_sdk blend (1 = policy owns the joints, 0 = onboard controller does). `SegmentPolicy` turns
  a list of pose-to-pose `Segment`s into a policy; `"start"` as a goal means the pose observed at
  reset. Scripted motions should ramp weight 0->1 while holding `"start"`, then ramp 1->0 at the
  end so the robot's controller takes the arms back smoothly.
- **Env** (`envs/base.py`): `setup/reset/step/teardown/report`, plus a classmethod `add_args`
  that registers env-specific CLI flags on the shared parser. Raise `EnvAbort` to stop early
  while still getting `report()` called. Registries are plain dicts: `envs.ENVS` and
  `policies.POLICIES`; a new policy must be added to `policies/__init__.py` to be runnable.
- **Blend semantics are emulated everywhere**: `check` and `sim` both compute
  `cmd = (1-w)*hold + w*target` so what you see in sim matches what arm_sdk does on the robot.
  `hold` is the Menagerie `stand` keyframe (`config.STAND_Q`) in check/sim and the live pose on
  the robot.
- **Joint indexing** (`config.py`): DDS order of `LowCmd_.motor_cmd`, which is also the
  Menagerie `unitree_g1` actuator order. `tests/test_config.py` asserts the hardcoded limit table
  matches the model. Index 29 is the arm_sdk weight slot on the robot, not a joint.
- **Robot env** uses only high-level control: `LocoClient` FSM transitions then targets
  published on `rt/arm_sdk`. `--mode gantry` does Damp -> FSM 4 -> FSM 200, runs, releases the
  arms, Damps. `--mode standing` requires the robot already in FSM 4, goes to FSM 200, runs,
  releases the arms and stays in FSM 200 (never damps). It refuses any joint outside
  `UPPER_BODY` (waist + arms) and always releases the arms in `teardown`, including on Ctrl-C.
  Do not add low-level leg control through this path.
- **Sim env** pins the pelvis by overwriting the free-joint state each substep (no model edit);
  `--free-base` disables that. The model is resolved from `$G1_MJCF`, then the
  `mujoco-menagerie` pip package, then `~/Robotics/mujoco_menagerie`.
