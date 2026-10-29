# Writing a task

A task is a folder under `tasks/` with a `task.json` and no code. It is the only thing
`g1 run` runs: either an **instruction** (the model reads it, the tools do the moving) or a
fixed **tools** chain (no model).

```
mkdir tasks/find_the_marker && $EDITOR tasks/find_the_marker/task.json
g1 run find_the_marker --sim                  # on the twin
g1 run find_the_marker --mode standing        # on the robot
```

## task.json

```json
{
  "instruction": "find the marker and stop in front of it",
  "scene": "room",
  "objects": ["marker@1.2,-0.8"],
  "safety_notes": ["never walk backward without looking back first"],
  "max_decisions": 30,
  "max_time_s": 600,
  "notes": "for people; the model never sees this"
}
```

```json
{"tools": "tpose,look:30,hold:1,look:-30"}
```

| key | meaning |
|---|---|
| `instruction` | what the model is asked to do, one sentence (needs `HF_TOKEN`). |
| `tools` | instead: a fixed chain, `name:args` separated by commas, each tool starting where the last ended. Exactly one of `instruction` and `tools`. |
| `scene`, `objects` | the twin's room (`g1 twin fetch` once) and what to put in it: `name@x,y[,z]` (`mug`, `marker`, `cracker_box`, `mustard`, `pencil`, `chair`, `laptop`; chair and laptop face the start). z defaults to the floor; the table's top is at z = 0.77 (its centre is x 2.2, y −1.5, 1.2 × 0.7 m), so `laptop@1.75,-1.5,0.77` stands on it. The room spans x −2..4, y −3..3. Ignored on the robot. |
| `start` | the twin only: where the robot starts, `[x, y, yaw_deg]` in the room (default the origin facing +x). Never sent to the model. |
| `safety_notes` | rules for the robot's behaviour in this task ("do not walk", "never touch furniture"), added to the system prompt. Never claims about the world (where a chair or wall is): the model knows only what the robot can sense, and such a claim is false wherever the room differs. The same holds for `instruction`: a goal, never where its target is ("find a free-standing chair", not "the chair 1.5 m ahead"). World facts for people go in `notes`; `tests/agent/test_what_the_model_knows.py` fails a task whose text says where something is. |
| `max_decisions`, `max_time_s` | the budget; a rejected reply costs a decision too. `max_time_s` also stops a tools chain (with the safe return). |
| `camera_size` | the twin's head camera, `"HxW"` (default `480x640`; the robot streams `720x1280`). |
| `success` | the twin only: what counts as done, scored from the simulator's truth every tick, and the `--sim` run labelled from it (`outcome_source: twin`). `point_at` (an object, e.g. `laptop`) with `within_deg` and `for_s`: the right arm's ray (shoulder to wrist) within that angle of the object's centre for that long without a break; `feet_drift_max_m`, `feet_yaw_drift_max_deg`: how far the feet may move and turn (walking or sliding; not the pelvis, which twists over planted feet); `no_contact`: geom or body names (or prefixes: `table_top`, `table_leg`, `laptop`) the robot must never touch. A fall always fails. Never sent to the model. |
| `notes` | for people. |

## What a run produces

`tasks/<name>/runs/<ts>_<env>_<...>/` (`<env>`: `sim` or `robot`) and one line in
`tasks/<name>/results.jsonl`. Every run has `summary.json` (which run, how it ended, usage,
health), `config.json` (the task, settings, limits and a restorable code snapshot) and `trace.npz`
(every tick commanded and measured, for `g1 trace`). An instruction run also has one
`step_NNNN.json` + `.png` per query to the model: the frame, the query and response as sent and
received, the action and its result. See `docs/architecture.md`, "The record".

A run on the twin (`--sim`) asks nothing: it is kept, and its outcome is the twin's verdict when the
task has `success`; otherwise the run's checks for a tools chain and `unreviewed` for an instruction.
`summary.json` keeps the model's own conclusion (`model_outcome`) and `false_done` (it said done, the
twin disagreed). To measure a task, run it repeatedly: `g1 bench SUITE` (below). A run on the robot asks `Save this run? [y / n]`
first; no deletes it (no folder, no `results.jsonl` line), yes asks
`Task result [s success / f failed]` and the run is labelled with your answer.

## Measuring a task: `g1 bench`

A suite is `suites/<name>/suite.json`: `{"tasks": ["point_at_the_laptop"], "repeat": 10}`.
`g1 bench <name>` runs every task `repeat` times on the twin, headless, in one process (the same
code, model and limits for every run), and prints one table, also saved as
`suites/<name>/runs/<ts>/report.json` with every run's row:

* the task: success as the twin judged it, false `done`s, decisions, time, the best pointing error,
  how far the feet moved;
* the execution: falls, forbidden contacts, unsettled results, joint violations;
* the infrastructure: failed model calls, runs with degraded timing, runs it spoiled (the model's
  provider gave up, quota, no frame, a loaded machine). Those are not counted in the success rate.

Give a task `success` before benchmarking it; without it its runs stay `unreviewed`.

## Writing a good task

* Say what done looks like ("stop in front of it", "with the mug in the lower third of the
  image"). `done` is the model's conclusion; it will call it when its image matches your
  words.
* Put in `safety_notes` rules, not facts about the room. The head camera looks 48° down (its
  image starts about 19° below horizontal), so the robot cannot see what is behind it or on a
  table more than about 1.5 m away: write rules that make it look ("look back before walking
  backward", "walk closer to inspect things on furniture"), not where things are. In the twin,
  a `start` beside the furniture sets up a task without telling the model anything.
* Run it with `--sim` (the MuJoCo window shows it), then on the robot (`--view` for its
  camera) only after reading `docs/running-on-the-robot.md`. `--mode gantry` never walks.
* The tools the model gets are the same in every task (`g1/tools/`). If the task needs a
  motion the menu lacks, that is a tool to write, not a note to add.
