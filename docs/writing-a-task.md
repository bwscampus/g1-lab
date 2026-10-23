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
| `notes` | for people. |

## What a run produces

`tasks/<name>/runs/<ts>_<env>_<...>/` and one line in `tasks/<name>/results.jsonl`. Every run
has `trace.npz` (every tick commanded and measured, for `g1 trace`) and `status.json`. An
instruction run also has every frame the model decided on (`step_NNNN.png`, lossless), every
decision with the raw reply, the execution results, the conversation, the system prompt and
schemas, the joint state at 20 Hz and token usage.

A run on the twin (`--sim`) asks nothing: it is kept, its outcome the run's checks for a tools
chain and `unreviewed` for an instruction. A run on the robot asks `Save this run? [y / n]`
first; no deletes it (no folder, no `results.jsonl` line), yes asks
`Task result [s success / f failed]` and the run is labelled with your answer.

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
