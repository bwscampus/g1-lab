# Writing a tool

A tool is what the model picks. It is one Python class in `g1/tools/`: what the model reads
(`name`, `prompt`, `params`) and what the robot runs (`segments()`) together, so they cannot
drift apart. Drop a file in the directory and it is in the menu on the next run; a bad
definition fails at import with the file named.

```
g1 new tool bow                          # writes g1/tools/bow.py from the template
g1 run --env sim --tools bow --headless  # runs it, fully checked
g1 tools                                 # it is offered to the model now
```

## The contract

```python
from g1.core.action import Segment
from g1.tools.base import Tool, num, integer, text, flag, limit

class Bow(Tool):
    name = "bow"                       # snake_case; the model calls it by this
    prompt = "..."                     # one paragraph: what it does, when to use it, what must be clear
    params = {                         # argument name -> spec
        "angle_deg": num(5.0, 30.0, 15.0, "how far to bend forward, degrees"),
        "down_s": num(0.5, 5.0, 1.5, "seconds to bend down"),
        "hold_s": num(0.0, limit("hold_seconds_max"), 1.0, "seconds to stay bowed"),
        "reps": integer(1, 3, 1, "how many bows"),
    }
    # visible = True      False keeps it out of the model's menu (CLI chains and replay only)
    # needs_base = False  True if it drives the base; hidden where the env cannot walk
    # needs_loco = False  True for an onboard gesture; hidden in sim
    # joints = UPPER_BODY the joints it may command (waist + arms; the legs are never ours)

    def segments(self) -> tuple[Segment, ...]:
        ...
```

Arguments are validated and clamped at construction and become attributes
(`self.angle_deg`). The player (`reset`/`step`) is part of `Tool` and is not overridden: it
eases between the segments at 50 Hz from wherever the previous tool left the joints.

## Parameters: numbers with ranges, not switches

A continuous quantity is a `num(lo, hi, default, description)` — never `speed: slow|fast`.
The caller (the model, the CLI, another student) picks the seconds, metres or degrees, so
one tool serves every situation. The rules, enforced by a test on every tool:

* every number and integer has a `minimum` and a `maximum`; the host clamps and tells the
  model it did
* a number's name carries its unit: `_s`, `_m`, `_deg`, `_rad` (or is `seconds`)
* a property without a `default` is required; give one to everything that has a sensible
  default
* `enum` only for genuinely discrete choices (the menu's tool names in `check`)
* `limit("name")` in a range and `{limit:name}` in the prompt read `configs/limits.json`,
  so text and schema quote the number the host enforces. Add a limit there (with its unit,
  a note and `"source": "guess"` until you have verified it) rather than a literal, when
  anything else should respect the same number
* every motion tool also takes a `note` — the model's evidence and intent, required of the
  model, empty from code and the CLI. Do not declare it.

`text(description)` is a required non-empty string, `flag(default)` a boolean; a plain
JSON-schema dict works for anything else (`arm_path`'s waypoints array).

## Segments: the motion

`segments()` returns a tuple of `Segment(goal, duration, ...)`: go to `goal` (joint index →
radians, merged onto the previous pose, so a segment may set only some joints) over
`duration` seconds with a cosine ease. Optional: `label` (printed once when it starts),
`weight` (a function of progress 0→1 giving the arm_sdk blend; 1 = you own the joints),
`base=(vx, vy, vyaw)` (a base velocity held for the whole segment), `command=("WaveHand",
{...})` (one onboard call on the segment's first tick; only `config.LOCO_METHODS`).

* Touch only the joints your tool is about and leave them where it ends. Nothing else ever
  moves the arms to a neutral pose — not the run's start or end (the bookends only blend
  the arm_sdk weight in place), not walking — so a tool that needs a starting pose makes it
  its own first segment (`tpose` raises the arms itself, `sixseven` settles into palms-up
  first). `g1/core/poses.py` has `STAND` (the Menagerie stand keyframe) if you want it.
* Joints and conventions: `g1 tools --joints` prints every arm_sdk joint with its limits and
  stand value. Verified on the model: elbow 0 is a 90° bend with the forearm forward, ~1.57
  is a straight arm, more negative bends the forearm up; `shoulder_roll` +1.57 (left) /
  −1.57 (right) with `shoulder_pitch` 0 and elbow 1.47 is a T-pose; `wrist_roll` −1.57 (left)
  / +1.57 (right) turns the palms up; `waist_yaw` positive looks LEFT; `waist_pitch`
  positive bends forward (limits ±0.52 rad).
* Base motion: a segment with `base=` moves the robot (`LocoClient.Move` on the robot behind
  `--walk`; the pinned pelvis slides in sim). Use `g1/tools/move.py`'s `ticks()` so the
  duration is whole ticks and `distance = v * t` is exact; keep speeds under
  `walk_speed`/`side_speed`/`turn_rate` in the limits file. Set `needs_base = True`.
* Onboard calls: a segment's `command=(name, kwargs)` is one call on the robot, on the
  segment's first tick, and only names on an allow-list are accepted (`g1/core/config.py`):
  `LOCO_METHODS` (`WaveHand`, `ShakeHand` — the gestures in `g1/tools/gestures.py`: hand the
  arms to the onboard controller, weight 1→0, the call, take them back; `needs_loco = True`,
  sim aborts on them) and `AUDIO_METHODS` (`TtsMaker` — `say` in `g1/tools/speech.py`: hold
  the pose for as long as the text takes; sim prints it). The robot env refuses any other
  name, and a tool declaring one fails at import. FSM, damp, torque, sit, squat, raw audio
  playback and volume are out of reach by construction — the legs belong to the onboard
  controller, so a squat is not a tool this robot can be given through arm_sdk.

## What happens to it

* `g1 run --env sim --tools bow --headless` plays it between the bookends and the joint
  monitor checks every tick: measured and commanded angles inside the limits minus a margin,
  command speed under `command_vel_max`, weight in [0, 1], base velocity under the ceiling.
  A FAIL names the joint, the tick and the value. `mjpython -m g1 run --env sim --tools bow`
  shows it.
* Chained (`--tools bow:20,hold:1,bow:10`), each tool starts from the previous tool's last
  *commanded* pose, so the seam is continuous and the velocity gate covers it.
* Offered to the model, its prompt goes in the system prompt and its schema in the catalog;
  before it executes, the agent dry-runs it through the monitor from the commanded pose (the
  model's `check` does the same on request) and a violation is fed back as
  `motion_not_executed` without moving. While it runs, a record (frame + joints) is closed
  every 3 s.

## Why no tool watches the camera

In GPT-Policy every tool is open-loop with respect to vision: the host plans a fixed motion
from the arguments, executes it, and reports; the model closes the loop by looking again and
deciding again. A tool that tracked a target while running would move part of the reasoning
out of the model and into code, and the run would no longer test the model. So a tool has no
`step` of its own, no camera, no per-tool timeout: it ends because its segments end. If you
want "approach the mug", the model does it with `move` and fresh images, one decision at a
time.

## Query tools

`kind = "query"` tools move nothing: they implement `query(cmd, joints) -> dict` and the
answer becomes `previous_result`. `check` is one (`g1/tools/control.py`). `kind =
"terminal"` tools (`done`, `give_up`) end the run.
