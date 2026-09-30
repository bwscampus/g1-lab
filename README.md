# g1-lab

A Unitree G1 (29-DoF) that a vision-language model drives one **tool** at a time, in
GPT-Policy's shape ([arXiv:2609.19138](https://arxiv.org/abs/2609.19138), see
[Acknowledgements](#acknowledgements)): the model looks at the head camera and the joint
state, picks a tool from a menu and fills in its arguments; the host plans the motion,
checks it against the joint and speed limits, runs it, waits for the joints to settle and
reports what happened; the model looks again. The same tools chain from the command line
without a model, in the MuJoCo sim and on the robot.

```
g1 run  --env sim --tools tpose --headless                        # one tool, fully checked, ~1 s
g1 run  --env sim --tools walk_forward:0.5,hold:1,turn:45 --headless   # a chain: each starts where the last ended
mjpython -m g1 run --env sim --tools tpose                         # the same in the viewer (macOS needs mjpython)
g1 task run tasks/find_the_mug --env sim --headless               # the model decides (needs $HF_TOKEN)
g1 run  --env robot --tools tpose --iface <iface> --mode standing  # live
g1 run  ... --view            # the head camera live in a browser (http://127.0.0.1:8765), sim or robot
g1 run  ... --record          # every camera frame to camera.mp4 in the run directory (PyAV)
```

Two words to know, both GPT-Policy's:

* a **tool** is a bounded program with parameters — a name, a prompt, a JSON-schema
  `params` and a `segments()` method in one Python class. It ends by construction (a motion
  has a duration) and never watches the camera while it runs: the model is the closed loop.
  `g1 tools` lists them; **students write tools** (`g1 new tool NAME`).
* a **task** is a folder with an instruction and its context — the sim scene, safety notes,
  a demonstration, the budget — and no code. `g1 task run` runs it, `g1 task eval` scores
  it; **students write tasks** (`g1 new task NAME`).

## Setup

Python 3.10+ on macOS or Linux.

```
conda create -n g1 python=3.10 -y && conda activate g1     # or a venv
git clone <this repo> g1-lab && cd g1-lab
pip install -e ".[sim,dev]"                                # MuJoCo (with mjpython on macOS), pytest
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git ~/Robotics/mujoco_menagerie
```

The sim looks for `~/Robotics/mujoco_menagerie/unitree_g1/scene.xml`; set `G1_MJCF` to keep
the clone elsewhere. Then:

```
g1 tools                                   # the menu the model sees, and the CLI-only presets
g1 run --env sim --tools tpose --headless  # a checked run: the joint table and PASS
pytest                                     # ~100 tests, all through headless sim (~3 min)
```

For the model: `cp .env.example .env` and put a Hugging Face token in it (`HF_TOKEN`; the
model is `VLM_MODEL`, see `docs/vision-model.md`). Then `g1 scene fetch` once (~35 MB of
room textures and object meshes, git-ignored) and `g1 task run tasks/find_the_mug --env sim
--headless`, or `g1 decide runs/head.png --instruction "find the mug"` for one decision on a
saved frame. The robot stage needs `unitree_sdk2py` and, for the camera,
`unitree_webrtc_connect` plus `pip install -e ".[camera]"` (neither is on PyPI); see
`docs/running-on-the-robot.md`.

## Write a tool

`g1 new tool bow` writes `g1/tools/bow.py` from the template — a parametric bow — and it is
in the menu on the next run. The whole tool:

```python
class Bow(Tool):
    name = "bow"
    prompt = ("Bow from the waist by angle_deg degrees over down_s seconds, hold the bow for hold_s, "
              "then straighten up over up_s. A greeting toward a person in front of the robot.")
    params = {"angle_deg": num(5.0, 30.0, 15.0, "how far to bend forward, degrees"),
              "down_s": num(0.5, 5.0, 1.5, "seconds to bend down"),
              "hold_s": num(0.0, limit("hold_seconds_max"), 1.0, "seconds to stay bowed"),
              "up_s": num(0.5, 5.0, 1.5, "seconds to straighten up")}

    def segments(self):
        bowed = {**STAND, WAIST_PITCH: math.radians(self.angle_deg)}
        return (Segment(bowed, self.down_s, label="bowing"),
                Segment(bowed, self.hold_s, label="holding the bow"),
                Segment(STAND, self.up_s, label="straightening up"))
```

Parameters are numbers with ranges, in seconds, metres and degrees — never a `slow|fast`
switch — so one tool serves every situation. `{limit:name}` in the prompt and
`limit("name")` in a range come from `configs/limits.json`, so what the model is told never
differs from what the host enforces. Try it, chain it, offer it:

```
g1 run --env sim --tools bow --headless                       # the defaults
g1 run --env sim --tools bow:20:1:0.5:1,hold:1,bow:angle_deg=10 --headless
g1 tools                                                      # it is in the menu
```

`docs/writing-a-tool.md` has the rest: the joints and their conventions, base motion, the
onboard gestures, what `check` and the monitor do with it, and why no tool watches the
camera.

## Write a task

`g1 new task find_the_marker` copies `tasks/_template/`; `tasks/find_the_marker/task.json`:

```json
{"instruction": "find the marker and stop in front of it",
 "scene": "room", "objects": ["marker@1.2,-0.8"],
 "safety_notes": ["a low shelf stands 1 m behind the start; do not walk backward"],
 "demo": null, "max_decisions": 30, "max_time_s": 600,
 "notes": "for people; the model never sees this"}
```

```
g1 task show tasks/find_the_marker                          # what the model will be given
g1 task run  tasks/find_the_marker --env sim --headless     # one run; you give the verdict at the end
g1 task eval tasks/find_the_marker -n 5 --env sim --headless   # success rate, decisions to success
```

Runs land under the task's `runs/` with every frame, decision and result; a good run is the
next run's demonstration (`"demo": "runs/<dir>"`). See `docs/writing-a-task.md`.

## Layout

```
g1/
  cli.py          g1 run | task | tools | limits | camera | decide | episode | demo | scene | new
  core/           config (joints, limits table, stand pose), limits (every tunable number), action
                  (Action, Obs, Segment, Runnable), poses, images (RGB, Pillow), worker
  tools/          STUDENTS WRITE: one file per tool, found by import; base.py is the contract,
                  move / arms / gestures / control are the built-ins, _template.py is `g1 new tool`
  envs/           base (Env), monitor (the per-tick safety check), sim (MuJoCo), scene (the room), robot
  camera.py       the robot's head camera over WebRTC; DirCamera replays frames; the --view and --record taps
  vlm.py          the Hugging Face client
  agent/          agent (settle -> snapshot -> think -> act -> feedback), decider (the model I/O
                  contract), demo (turn-0 demonstrations), episode (the run record), task
tasks/            STUDENTS WRITE: one folder per task (task.json, runs/, results.jsonl)
configs/limits.json   every limit and default, with its unit and where the number came from
docs/             architecture, writing-a-tool, writing-a-task, running-on-the-robot, vision-model
tests/            pytest, mirroring g1/; everything runs through headless sim
```

**Every number is in one file.** Speeds, the ranges the model may ask for, timeouts, settle
tolerances, the token budget, the safe-return timing: `configs/limits.json`, each entry with
a unit, a note and a `source` (`guess` = never verified, review these first). `g1 limits`
prints the table; `G1_LIMITS=other.json g1 ...` swaps it for a run.

**Ctrl-C is safe.** An interrupted run (Ctrl-C, `--max-time`, an error) never drops the arms
where they are: the runner brings them from the last commanded pose to the stand pose over
3 s, then fades the arm_sdk weight to 0 over 2 s so the onboard controller takes over, base
stopped from the first tick. Further Ctrl-Cs during that return, and during the robot's
hand-over, are ignored — there is no forced release.

## Acknowledgements

The decision loop in this repo is inspired by **GPT-Policy**, *In-Context Robot
Learning with VLM Agents* ([arXiv:2609.19138](https://arxiv.org/abs/2609.19138),
[code](https://github.com/cheng-haha/GPT-Policy)). These ideas are theirs:

* a frozen vision-language model choosing one tool per turn from a catalog, with a `note`
  on every action; every tool a fixed motion planned by the host from the arguments, so the
  model is the only closed loop
* one structured observation per turn, carrying the measured state and the result of the
  previous action as feedback
* errors returned to the model as feedback instead of retried
* demonstrations compiled into the first turn as context
* a human assigning the success label after the run, and the run trace (events,
  transcript, protocol, states, usage)

This repo reimplements that design for a Unitree G1. It contains none of their code, which
had no licence granted when this was written. The executor, the tools, the joint monitor,
the safe return and the simulator are this repo's own, and any fault in how their design
was adapted is ours.

```bibtex
@article{cheng2026incontext,
  title   = {In-Context Robot Learning with VLM Agents},
  author  = {Cheng, Dongzhou and Yi, Taoran and Fang, Ye and Zhang, Xingwu and
             Feng, Fan and Li, Yixuan and Zhuang, Gengxiong and Wang, Rongze and
             Yang, Shuai and Song, Wei and Xue, Weizhi and Wu, Minyan and
             Gui, Jie and Wang, Jiaqi and Wu, Tong},
  journal = {arXiv preprint arXiv:2609.19138},
  year    = {2026}
}
```
