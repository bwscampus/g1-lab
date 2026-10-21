# Tasks

A task is a folder with what to do and its context, no code: `task.json` holds either an
`instruction` (the vision model decides, needs `HF_TOKEN`) or a fixed `tools` chain (no model).

```
g1 run tpose --sim                    # on the twin
g1 run tpose --mode gantry            # on the robot, hanging; --mode standing on its own feet (may walk)
g1 run find_the_mug --sim             # the model searches the room, in the MuJoCo window
g1 run find_the_mug --mode standing --view   # on the robot, its head camera in a browser
g1 run point_at_the_laptop --sim      # demo: start beside the table, turn the waist to the laptop, point at it (gantry too)
g1 run find_and_point_at_the_laptop --sim   # demo: explore the room, walk to the table, point at the laptop (standing only)
g1 run sit_in_the_chair --sim         # demo: find the chair, line up, turn around, back up and sit (standing mode only on the robot)
```

Runs land under `tasks/NAME/runs/` (git-ignored), each with its `trace.npz`, and one line per run
in `tasks/NAME/results.jsonl`. See `docs/writing-a-task.md`.
