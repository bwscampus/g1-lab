# Writing a task

A task is a folder with an instruction and its context. No code: the model reads the
instruction, the tools do the moving, you give the verdict.

```
g1 new task find_the_marker                                  # copies tasks/_template/
$EDITOR tasks/find_the_marker/task.json
g1 task show tasks/find_the_marker                           # the g1 run it stands for, and the score so far
g1 task run  tasks/find_the_marker --env sim --headless      # one run
g1 task eval tasks/find_the_marker -n 5 --env sim --headless # N runs, then the two numbers
```

## task.json

```json
{
  "instruction": "find the marker and stop in front of it",
  "scene": "room",
  "objects": ["marker@1.2,-0.8"],
  "safety_notes": ["a low shelf stands 1 m behind the start; do not walk backward"],
  "demo": "runs/20260930-101500_sim_find-the-marker_success",
  "demo_mode": null,
  "refs": ["marker.jpg"],
  "success": "human",
  "max_decisions": 30,
  "max_time_s": 600,
  "notes": "for people; the model never sees this"
}
```

| key | meaning |
|---|---|
| `instruction` | what the model is asked to do, one sentence. The only required key. |
| `scene`, `objects` | the sim room (`g1 scene fetch` once) and what to put in it: `name@x,y[,z]`, names from `g1 scene status` (`mug`, `marker`, `cracker_box`, `mustard`, `pencil`). The robot starts at the origin facing +x; the room spans x −2..4, y −3..3. Ignored on the robot. |
| `safety_notes` | persistent physical facts the camera cannot see, added to the system prompt. `--scene room` adds the room's own (walls, table, doorway). |
| `demo` | a demonstration shown on turn 0, relative to the task folder: a recorded run (its frames and, in `video+action` mode, the tools it picked), a phone video (keyframes picked by the model, or evenly with `--demo-select uniform`), or a `demo.json` bundle from `g1 demo prepare`. |
| `refs` | photos of the goal, shown on turn 0. |
| `success` | who decides: `human` — the terminal asks `Task result [s success / f failed]` after the run. |
| `max_decisions`, `max_time_s` | the budget; a rejected reply costs a decision too. |
| `notes` | for people. |

Flags after the folder override the file for one run: `g1 task run tasks/x --env sim
--headless --max-decisions 10 --no-demo --echo`.

## What a run produces

`tasks/<name>/runs/<ts>_<env>_<instruction>_<outcome>/` — every frame the model decided on
(`step_NNNN.png`, lossless), every decision with the raw reply, the execution results, the
conversation, the system prompt and schemas, the joint state at 20 Hz, token usage — and one
line in `tasks/<name>/results.jsonl`. `g1 episode DIR` prints the step table and the
`--tools` chain that replays it; `g1 run --tools replay --episode DIR` replays it with no
camera and no model. A good run is the next run's demonstration.

`g1 task eval` prints GPT-Policy's two numbers: the success rate and the decisions each
successful run took. Both are only as good as the verdicts you give.

## Writing a good task

* Say what done looks like ("stop in front of it", "with the mug in the lower third of the
  image"). `done` is the model's conclusion; it will call it when its image matches your
  words.
* Put in `safety_notes` what the head camera cannot see: it looks 47° down, so a table at
  chest height behind the robot is invisible until it is too late.
* Start in sim with `--headless`, then `mjpython -m g1 task run ...` to watch, then the
  robot with `--walk` only after reading `docs/running-on-the-robot.md`.
* The tools the model gets are the same in every task: `g1 tools`. If the task needs a
  motion the menu lacks, that is a tool to write, not a note to add.
