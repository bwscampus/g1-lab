# Tasks

A task is a folder with what to do and its context, no code: `task.json` holds an `instruction`
(the vision model decides, one tool at a time). A fixed motion with no model is a routine
(`routines/`, `docs/writing-a-routine.md`); `g1 run` plays either by name.

```
g1 run point_at --sim                 # point at {object} (default: the laptop); g1 start fills it: "point at the water bottle"
g1 run find_the_mug --sim             # the model searches the room, in the MuJoCo window
g1 run find_the_mug --mode standing --view   # on the robot, its head camera in a browser
g1 run point_at_the_laptop --sim      # demo: start beside the table, turn the waist to the laptop, point at it (gantry too)
g1 run find_and_point_at_the_laptop --sim   # demo: explore the room, walk to the table, point at the laptop (standing only)
g1 run point_at_what_you_hear --sim   # spoken requests: "point at the mug", then the next, until "stop" (hears; gantry too)
g1 run sit_in_the_chair --sim         # demo: find the chair, line up, turn around, back up and sit (standing mode only on the robot)
```

Runs land under `tasks/NAME/runs/` (git-ignored), each with its `trace.npz`, and one line per run
in `tasks/NAME/results.jsonl`. A successful run a person promotes becomes a demo in
`tasks/NAME/demos/` (committed), shown to the model on the task's next runs. See `docs/writing-a-task.md`.
Each `task.json` lists the `modes` it runs in: the ones that walk or sit are standing only.
