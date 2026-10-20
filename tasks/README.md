# Tasks

A task is a folder with what to do and its context, no code: `task.json` holds either an
`instruction` (the vision model decides, needs `HF_TOKEN`) or a fixed `tools` chain (no model).

```
g1 run tpose --sim                    # on the twin
g1 run tpose --mode gantry            # on the robot, hanging; --mode standing on its own feet (may walk)
g1 run find_the_mug --sim             # the model searches the room, in the MuJoCo window
g1 run find_the_mug --mode standing --view   # on the robot, its head camera in a browser
```

Runs land under `tasks/NAME/runs/` (git-ignored), each with its `trace.npz`, and one line per run
in `tasks/NAME/results.jsonl`. See `docs/writing-a-task.md`.
