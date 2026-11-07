# Writing a task

A task is a folder under `tasks/` with a `task.json` and no code: an **instruction** the model
reads, the tools doing the moving, one at a time. `g1 run` runs it, and `g1 start` can choose it (or you, with `/routine` or `/task`).
A fixed motion with no model in it is a **routine** (`docs/writing-a-routine.md`), not a task.

```
mkdir tasks/find_the_marker && $EDITOR tasks/find_the_marker/task.json
g1 run find_the_marker --sim                  # on the twin
g1 run find_the_marker --mode standing        # on the robot
```

## task.json

```json
{
  "instruction": "find the marker and stop in front of it",
  "modes": ["standing"],
  "scene": "room",
  "objects": ["marker@1.2,-0.8"],
  "safety_notes": ["never walk backward without looking back first"],
  "max_decisions": 30,
  "max_time_s": 600,
  "notes": "for people; the model never sees this"
}
```

| key | meaning |
|---|---|
| `instruction` | what the model is asked to do, one sentence (needs `HF_TOKEN`). |
| `fill` | slots the instruction names as `{name}`: `{"object": {"note": "what to point at, as the person named it", "default": "laptop"}}`. g1 start fills them from the request (the model's `run_task` arguments, or `/task point_at water bottle`) and refuses the task without them; `g1 run` uses each `default`, so a `success` is scored on the default. Every `{name}` must be declared and every slot named. |
| `walks` | `false`: the base is never driven in this task, in any mode: no `move`, `walk_forward`, `turn` or `sit` (the legs only balance under Unitree's controller), so it looks around with the waist and works with the arms. Default `true` (walking where the mode allows it). |
| `turns_waist` | `false`: the waist is locked straight and stiff for the whole run: every arm_sdk message holds waist yaw, roll and pitch at 0 with `waist_lock_kp`/`waist_lock_kd`, whatever a tool sends, and `look`, `point_to` and `arm_path` are not offered (the agent refuses them too). It sees and points only the way it faces (`point`). For tasks where turning the torso could tip the robot (`point_at`). Default `true`. |
| `tilts_waist` | `false`: the waist's roll and pitch (tilting the torso side to side, forward and back) are held at 0 for the whole run, and the whole waist is stiff (`waist_lock_kp`/`waist_lock_kd`), in every arm_sdk message. Yaw still turns (`look`, `point_to`). Tilting the torso tipped the robot. Default `true`. `point_at` uses it. |
| `session` | `false` keeps the task out of g1 start (its menu, `/skills`, the model's tools): one not yet tried on the robot. `g1 run` and `g1 bench` still run it. Default `true`. |
| `modes` | required: the `--mode`s it runs in. `["standing"]` when it walks or sits (it commands the legs); `["gantry", "standing"]` when the arms and waist are enough (in gantry the legs are never commanded, so the walking tools are not offered). `g1 run` and `g1 start` refuse any other mode. |
| `scene`, `objects` | the twin's room (`g1 twin fetch` once) and what to put in it: `name@x,y[,z]` (`mug`, `marker`, `cracker_box`, `mustard`, `pencil`, `chair`, `laptop`; chair and laptop face the start). z defaults to the floor; the table's top is at z = 0.77 (its centre is x 2.2, y −1.5, 1.2 × 0.7 m), so `laptop@1.75,-1.5,0.77` stands on it. The room spans x −2..4, y −3..3. Ignored on the robot. |
| `start` | the twin only: where the robot starts, `[x, y, yaw_deg]` in the room (default the origin facing +x). Never sent to the model. |
| `safety_notes` | rules for the robot's behaviour in this task ("do not walk", "never touch furniture"), added to the system prompt. Never claims about the world (where a chair or wall is): the model knows only what the robot can sense, and such a claim is false wherever the room differs. The same holds for `instruction`: a goal, never where its target is ("find a free-standing chair", not "the chair 1.5 m ahead"). World facts for people go in `notes`; `tests/agent/test_what_the_model_knows.py` fails a task whose text says where something is. |
| `max_decisions`, `max_time_s` | the budget; a rejected reply costs a decision too. `max_time_s` stops the run (with the safe return). |
| `camera_size` | the twin's head camera, `"HxW"` (default `480x640`; the robot streams `720x1280`). |
| `hears` | `true`: the run listens. The robot's speech recognition (`rt/audio_msg`, `g1/microphone.py`) reaches the model as `heard` in every observation (what people said since the last one: text and seconds ago, English only by default, `heard_languages`), and the `listen` tool is offered: it holds the pose until someone speaks. What the robot hears while its own speaker plays is dropped. On the robot, voice interaction must be on in the Unitree app (`docs/running-on-the-robot.md`). An instruction task only. |
| `voices` | the twin only: a person speaking to it, `[{"at_s": 5, "text": "Point at the mug."}, ...]`, each line heard on `rt/audio_msg` `at_s` after the program takes over, exactly as the robot's speech recognition publishes it. The model hears each line when it is said; it never sees the script. |
| `success` | the twin only: what counts as done, scored from the simulator's truth every tick, and the `--sim` run labelled from it (`outcome_source: twin`). `point_at` (an object, e.g. `laptop`) with `within_deg` and `for_s`: the right arm's ray (shoulder to wrist) within that angle of the object's centre for that long without a break; `requests` (with `voices`): `[{"voice": 0, "point_at": "mug"}, {"voice": 2, "point_at": null}]`, each met when, between that voice and the next, the arm points at its object as `point_at` does, or for `null` (asked for something that is not there) at none of the scene's objects; `feet_drift_max_m`, `feet_yaw_drift_max_deg`: how far the feet may move and turn (walking or sliding; not the pelvis, which twists over planted feet); `no_contact`: geom or body names (or prefixes: `table_top`, `table_leg`, `laptop`) the robot must never touch. Angle criteria pass up to `success_margin_deg` (1°, `configs/limits.json`) beyond their number, for measurement error: 10 accepts up to 11. A fall always fails. Never sent to the model. |
| `notes` | for people. |

## What a run produces

`tasks/<name>/runs/<ts>_<env>_<...>/` (`<env>`: `sim` or `robot`) and one line in
`tasks/<name>/results.jsonl`. Every run has `summary.json` (which run, how it ended, usage,
health), `config.json` (the task, settings, limits and a restorable code snapshot) and `trace.npz`
(every tick commanded and measured, for `g1 trace`). An instruction run also has one
`step_NNNN.json` + `.png` per query to the model: the frame, the query and response as sent and
received, the action and its result. See `docs/architecture.md`, "The record".

A run on the twin (`--sim`) asks nothing: it is kept, and its outcome is the twin's verdict when the
task has `success`; otherwise `unreviewed`.
`summary.json` keeps the model's own conclusion (`model_outcome`) and `false_done` (it said done, the
twin disagreed). To measure a task, run it repeatedly: `g1 bench SUITE` (below). A run on the robot asks `Save this run? [y / n]`
first; no deletes it (no folder, no `results.jsonl` line), yes asks
`Task result [s success / f failed]` and the run is labelled with your answer.

## Demos: what earlier runs teach the next

A successful run can become a **demo** of its task: the next run's model is shown it, at the head
of every request, as an example of how the task went before (its tool calls, its notes, what it was
shown back, and a few of the frames it decided on). Only a success counts: one the twin scored
(`outcome_source: twin`) or one you labelled. A person chooses; nothing is promoted on its own.

```python
from g1.agent.demos import promote
promote("tasks/point_at_the_laptop/runs/20261030-074440_sim_find-the-laptop-by-turning-your-_success")
# -> tasks/point_at_the_laptop/demos/<run>/ (demo.json + its frames; commit it)
```

* A demo is shown only to runs in the same place it came from (twin demos to twin runs, robot
  demos to robot runs: the twin's room looks nothing like a real one), the fewest decisions first,
  then the newest, at most `agent.demos_max` of them (`configs/model.json`; 0 turns demos off;
  `demo_images_max` and `demo_image_width_px` size them).
* A demo carries the model's own words and what it was shown, never the host's truth: no measured
  base pose, no twin verdict or criteria, no `success`, and not the done summary or hindsight.
* Each run's `config.json` records which demos it was shown (`demos.ids`) and their text; its step
  files show one `{"type": "demos"}` marker where they were.
* Measure a demo's worth with `g1 bench`, `demos_max` 0 against 2, with a varied `start`, so a
  demo of the same room cannot simply be replayed.

## Measuring a task: `g1 bench`

A suite is `suites/<name>/suite.json`: `{"tasks": ["point_at_the_laptop"], "repeat": 10}`.
`g1 bench <name>` runs every task `repeat` times on the twin, headless, in one process (the same
code, model and limits for every run), and prints one table, also saved as
`suites/<name>/runs/<ts>/report.json` with every run's row:

* the task: success as the twin judged it, false `done`s, decisions, time, the best pointing error,
  how far the feet moved;
* the execution: falls, forbidden contacts, unsettled results, joint violations;
* the infrastructure: failed model calls, runs with degraded timing, runs it spoiled (the model's
  provider gave up, quota, no frame, a loaded machine). Those are not counted in the success rate;
* the cost: input tokens (cached part apart), output tokens (reasoning part apart), and the cost
  in USD where the provider reports it (OpenRouter does), summed over every run.

Runs go several at once, each in a fresh process with its own twin, DDS domain and log
(`suites/<name>/runs/<ts>/run_NN_<task>.log`): as many as free memory (`bench_run_mb` each) and the
performance cores (one kept free) allow, at most `bench_parallel_max`. The first line says how many and why.

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
* The tools the model gets are the same in every task (`g1/tools/`), plus `play_routine` for the
  upper-body routines. If the task needs a motion the menu lacks, that is a tool or a routine to
  write, not a note to add.
