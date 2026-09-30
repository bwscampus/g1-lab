# Tasks

A task is a folder with an instruction and its context — no code. `g1 new task NAME` copies
`_template/`; edit `task.json`; then:

```
g1 task show tasks/NAME                       # what the model will be given, and the score so far
g1 task run  tasks/NAME --env sim --headless  # one run; flags after the folder override the file
g1 task eval tasks/NAME -n 5 --env sim --headless   # N runs: success rate, decisions to success
```

Runs land under `tasks/NAME/runs/` (git-ignored) and one line per run in `tasks/NAME/results.jsonl`.
See `docs/writing-a-task.md`.
