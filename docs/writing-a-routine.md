# Writing a routine

A routine is our reference motion: a folder under `routines/` with a `routine.json` (and, for a
recorded motion, a `motion.npz`), no code. It plays start to finish with no model in the loop:
`g1 run NAME` plays it, `g1 talk` can choose it, and a task's model can play an upper-body one as
a single step (`play_routine`).

```
mkdir routines/salute && $EDITOR routines/salute/routine.json
g1 run salute --sim                    # on the twin, fully checked
g1 run salute --mode gantry            # on the robot, hanging (an upper-body routine)
```

A built-in is different: it comes with the robot and runs on Unitree's controller (`g1 moves`).
A task is different too: the model decides each step (`docs/writing-a-task.md`).

## routine.json: a timeline of tracks

```json
{
  "notes": "Right hand to the brow in a salute, held, then down. The first line is what talk's menu shows.",
  "tracks": {
    "upper_body": {"keyframes": [
      {"t": 1.5, "joints": {"right_shoulder_pitch": -1.2, "right_shoulder_roll": -0.3, "right_elbow": -0.6}, "label": "salute"},
      {"t": 3.5, "joints": {"right_shoulder_pitch": -1.2, "right_shoulder_roll": -0.3, "right_elbow": -0.6}, "label": "holding"},
      {"t": 5.0, "joints": {"right_shoulder_pitch": 0.2, "right_shoulder_roll": -0.2, "right_elbow": 1.28}, "label": "down"}
    ]},
    "base": [],
    "onboard": []
  }
}
```

Each track is one level of control the SDK offers:

| track | what it does | how it runs |
|---|---|---|
| `upper_body` | waist and arm joint targets (the joint names in `g1/core/config.py:JOINTS`, within their limits). `keyframes`: `t` is when the pose is **reached**, in seconds from the start; between keyframes the motion is eased; a joint a keyframe leaves out keeps its previous target. Or `samples`: `"motion.npz"` with `t` [N], `q` [N, J] and `joints` [J] names, a recorded or trained motion, reached smoothly at its first sample and then played linearly sample to sample (at most `routine_sample_rate_max_hz`, the 50 Hz control rate). | arm_sdk, the weight blended in and out by the takeover and handback; Unitree's controller keeps the robot balanced. |
| `base` | velocity windows `{"t0", "t1", "vx", "vy", "vyaw"}` (m/s, m/s, rad/s; within `base_vx_max`, `base_vy_max`, `base_vyaw_max`). | LocoClient `Move`. `--mode standing` only: a gantry run refuses it. |
| `onboard` | `{"t", "call": "WaveHand" \| "ShakeHand", "seconds", "args"}`: an onboard gesture at time `t`. The timeline pauses there: the arms are handed to the onboard controller (`gesture_ramp_s`), the gesture plays for `seconds`, the arms are taken back, then the timeline goes on. | LocoClient, allow-listed (`config.LOCO_METHODS`). |
| `full_body` | all 29 joints. Kept in the format so recorded or trained full-body motions have a place; **playing one is refused** until a tracking controller exists (open-loop full-body playback with Unitree's balance off would drop the robot). | not yet. |

Rules the loader checks (`g1/routines.py`, before anything moves):

* With an `upper_body` track, every base `t0`/`t1` and every onboard `t` falls **on a keyframe
  (or sample) time**: a smooth segment is never split. Put a keyframe where a window starts and
  ends (repeat the pose to hold it).
* A gesture never starts inside a base window.
* At most `routine_keyframes_max` keyframes and `routine_duration_max_s` in all.
* Angles within each joint's limits; legs only in `full_body`.

## What happens when it plays

* The routine compiles to the same `Segment`s every tool plays (`compile_segments`), and the
  whole of it is **dry-run** through the joint monitor from the commanded start pose before the
  takeover: a joint limit, a command speed over `command_vel_max` or a base speed over the
  ceiling refuses the routine and nothing moves.
* Then: the takeover (the weight 0 to 1 holding the pose the arms are in), the routine, the right
  arm lowered if it ends pointing, the handback (1 to 0 holding where it ended). Nothing moves the
  arms to a neutral pose that the routine does not say: start from where the arms are, or make
  the stand pose your first keyframe.
* Ctrl-C (in `g1 run` or in talk) runs the safe return: from the last commanded pose to the stand
  pose, the weight to 0, the base stopped.
* Runs land under `routines/NAME/runs/` (git-ignored) with `summary.json`, `config.json` and
  `trace.npz`; `g1 trace compare` compares two of them (the twin's and the robot's).

`routines/tpose` and `routines/sixseven` are the examples: keyframes written out from the presets
they used to be.
