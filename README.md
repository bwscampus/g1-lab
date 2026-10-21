# g1-lab

A Unitree G1 (29-DoF) that a vision-language model drives one **tool** at a time, in
GPT-Policy's shape ([arXiv:2609.19138](https://arxiv.org/abs/2609.19138), see
[Acknowledgements](#acknowledgements)): the model looks at the head camera and the joint
state, picks a tool from a menu and fills in its arguments; the host plans the motion,
checks it against the joint and speed limits, runs it, waits for the joints to settle and
reports what happened; the model looks again. The same tools chain without a model, on a
MuJoCo twin of the robot and on the robot.

```
g1 run tpose --sim                   # a task on the twin, in the MuJoCo window: the robot code, over DDS
g1 run find_the_mug --sim            # the model searches a room (needs HF_TOKEN)
g1 run point_at_the_mug --sim        # demo: the waist finds the mug, the right arm points at it
g1 run sit_in_the_chair --sim        # demo: find a chair, back up to it and sit (the twin does not stand up again)
g1 run tpose --mode gantry           # the robot, hanging in a gantry (never walks)
g1 run find_the_mug --mode standing --view   # the robot on its own feet (may walk); its camera in a browser
g1 status                            # read-only: is it the link, or is the robot refusing?
g1 twin fetch                        # once: the twin's model, walker and room
g1 trace compare RUN_A RUN_B         # how far the twin is from the robot on the same task
g1 limits [guess]                    # every number and default as one table (or only the guesses)
```

That is the whole command line. Everything else lives in a file: what to do in the task's
`task.json`, every number and default (the model included) in `configs/limits.json`, and in
`.env` only what is private to one machine: the model's key and the robot's address and key. `G1_VERBOSE=1` prints every SDK call and a health line a second.

Two words to know, both GPT-Policy's:

* a **tool** is a bounded program with parameters — a name, a prompt, a JSON-schema
  `params` and a `segments()` method in one Python class. It ends by construction (a motion
  has a duration) and never watches the camera while it runs: the model is the closed loop.
  **Students write tools**: a file in `g1/tools/`.
* a **task** is a folder under `tasks/` with a `task.json` and no code: an `instruction` the
  model works on (with the scene, safety notes and the budget), or a fixed `tools` chain that
  runs without a model. **Students write tasks.**

The menu the model gets: `move`, `arm_path`, `hold`, `check`, `say` (the onboard
text-to-speech), `wave_hand`/`shake_hand` (onboard gestures), `done`, `give_up`.

## Install

Python 3.10+ on macOS or Linux (developed on macOS, Python 3.10 in conda).

### 1. The twin — everyone

```
conda create -n g1 python=3.10 -y && conda activate g1     # or: python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/bwscampus/g1-lab.git && cd g1-lab
pip install -e ".[sim,dev]"                                # numpy, jsonschema, Pillow, MuJoCo (+ mjpython on macOS), onnxruntime, pytest
g1 twin fetch                                              # the twin's G1 model, walker and room, into assets/
```

`--sim` needs `unitree_sdk2py` too (step 3, the control part): the twin is a separate process the
robot code talks to over DDS on the loopback interface, exactly as it talks to the G1.

The twin is unitree_mujoco's 29-DoF G1 (free base, torque motors at the real limits) behind a
stand-in for the onboard controller, which answers LocoClient and AudioClient, blends arm_sdk,
stands, and walks with NVIDIA's GR00T whole-body controller (two small networks, Balance and
Walk, on the CPU; they see the arms, so a moving arm doesn't knock it over). Nothing is pinned or
slid: a base command makes it step, the arms sag at arm_sdk's kp 60 as the robot's do, and a bad
motion can knock it over, which fails the run. `g1 twin fetch` copies the model from a
`unitree_mujoco` clone next to this repo if there is one, else downloads it and the walker from
pinned commits (sha256-checked; the walker's weights are under the NVIDIA Open Model License). Check:

```
g1 run tpose --sim                   # the MuJoCo window opens, the arms go out to a T and back, PASS, it closes
pytest                               # ~120 tests on the twin in-process, headless (~2 min); pytest -m dds: over DDS
```

### 2. The model — to run a task with an instruction

```
cp .env.example .env            # git-ignored; exported variables win over it
```

Put a Hugging Face token in it (`HF_TOKEN=hf_...`; a fine-grained token with "Make calls to
Inference Providers", from https://huggingface.co/settings/tokens). The model and every other
setting are in `configs/limits.json` (`vlm_model`; `g1 limits` prints them all; see
`docs/vision-model.md`). Then:

```
g1 run find_the_mug --sim            # a whole task on the twin; recorded, no questions asked
```

### 3. The robot — control, then the camera

**Network.** Plug into the G1 (or join its network) so the laptop has an address on
`192.168.123.x`; `ifconfig`/`ip a` shows the interface (`en7`, `eth0`, ...). Put it in `.env`
as `UNITREE_IFACE=en7`, and the robot's address as `UNITREE_ROBOT_IP=192.168.123.161` for the
camera.

**Control: `unitree_sdk2py`.** Not on PyPI. It needs CycloneDDS 0.10.x, which on macOS (and
on Linux without a system package) is built from source first:

```
cd ~/Robotics
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install && cmake --build . --target install
export CYCLONEDDS_HOME=~/Robotics/cyclonedds/install            # needed for the next step

cd ~/Robotics
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .                      # installs cyclonedds==0.10.2 against CYCLONEDDS_HOME
```

Check without moving anything: `g1 status` (read-only; prints the LowState rate, the FSM and
a one-line verdict — `docs/running-on-the-robot.md` says what each means). Then the first
motion, on a robot already standing under its own controller:

```
g1 run tpose --mode standing         # takes the arms where they are, raises and lowers them, hands back
```

**Camera: `unitree_webrtc_connect`.** Also not on PyPI; it brings `aiortc` and PyAV, which
decode the robot's H.264 stream. (It also pulls in `opencv-python` as a dependency of its own;
this repo never imports it — see `docs/architecture.md` — and the two coexist as long as
nothing loads `cv2` into a running process.)

```
cd ~/Robotics
git clone https://github.com/legion1581/go2_webrtc_connect.git
cd go2_webrtc_connect && pip install -e .
cd ~/Robotics/g1-lab && pip install -e ".[camera]"
```

Firmware 1.5.1 and later needs the robot's AES-128 key; it is per device and does not change.
Fetch it once with the account the robot is bound to in the Unitree Explorer app and put it in
`.env`:

```
unitree-fetch-aes-key --email you@example.com --sn <serial> --quiet      # --region cn for the Chinese cloud
# .env:  UNITREE_AES_128_KEY=<32 hex characters>
```

Close the Unitree app (the robot takes one WebRTC client), then the real thing, spotter on L2+B.
`--view` puts the head camera in a browser at `http://127.0.0.1:8765` while it runs (the app
cannot watch at the same time, so this is the only live picture):

```
g1 run find_the_mug --mode standing --view
```

`docs/running-on-the-robot.md` has the pre-flight for walking, what the two modes do, and
what to do when the robot seems stuck.

## Write a tool

A file in `g1/tools/` with a `Tool` subclass is in the menu on the next run. A whole tool:

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
        bowed = {WAIST_PITCH: math.radians(self.angle_deg)}
        return (Segment(bowed, self.down_s, label="bowing"),
                Segment(bowed, self.hold_s, label="holding the bow"),
                Segment({WAIST_PITCH: 0.0}, self.up_s, label="straightening up"))
```

Parameters are numbers with ranges, in seconds, metres and degrees — never a `slow|fast`
switch — so one tool serves every situation. A tool touches only the joints it is about and
leaves them where it ends: nothing moves the arms to a neutral pose at the start or end of a
run, or while walking. `{limit:name}` in the prompt and `limit("name")` in a range come from
`configs/limits.json`, so what the model is told never differs from what the host enforces.
Try it with a task: `tasks/bow/task.json` = `{"tools": "bow:20:1:0.5:1,hold:1,bow:angle_deg=10"}`,
then `g1 run bow --sim`.

`docs/writing-a-tool.md` has the rest: the joints and their conventions, base motion, the
onboard gestures, what `check` and the monitor do with it, and why no tool watches the
camera.

## Write a task

`tasks/find_the_marker/task.json`:

```json
{"instruction": "find the marker and stop in front of it",
 "scene": "room", "objects": ["marker@1.2,-0.8"],
 "safety_notes": ["a low shelf stands 1 m behind the start; do not walk backward"],
 "max_decisions": 30, "max_time_s": 600,
 "notes": "for people; the model never sees this"}
```

```
g1 run find_the_marker --sim         # one run on the twin; on the robot it asks to save it, then the verdict
```

Every run lands under the task's `runs/` with every frame, decision and result and its
`trace.npz`, and adds a line to `results.jsonl`. See `docs/writing-a-task.md`.

## Layout

```
g1/
  cli.py          g1 run | status | twin | trace; run_settings() (what an env reads), run(), safe_return
  core/           config (joints, limits table, stand pose), limits (every tunable number), action
                  (Action, Obs, Segment, Runnable), poses, images (RGB, Pillow), worker
  tools/          STUDENTS WRITE: one file per tool, found by import; base.py is the contract,
                  move / arms / gestures / speech / control are the built-ins
  envs/           base (Env), monitor (the per-tick safety check), robot (RobotEnv: the G1 or the twin),
                  link (the SDK seam), sim (the in-process twin the tests use), scene (the room), trace
  twin/           the twin: model, walker, onboard controller, services, loopback, DDS bridge, TwinProcess
  camera.py       the robot's head camera over WebRTC; the twin's over shared memory; DirCamera replays frames
  vlm.py          the Hugging Face client
  agent/          agent (settle -> snapshot -> think -> act -> feedback), decider (the model I/O
                  contract), episode (the run record), task (task.json)
tasks/            STUDENTS WRITE: one folder per task (task.json, runs/, results.jsonl)
configs/limits.json   every limit and default, with its unit and where the number came from
docs/             architecture, writing-a-tool, writing-a-task, running-on-the-robot, vision-model
tests/            pytest, mirroring g1/; everything runs on the twin in-process
```

**Every number is in one file.** Speeds, the ranges the model may ask for, timeouts, settle
tolerances, the model and its token budget, the robot's bring-up and teardown waits, the twin's
gains and stand-ins: `configs/limits.json`, each entry with a unit, a note and a `source`
(`guess` = never verified, review these first). `g1 limits` prints it as one table.
`G1_LIMITS=other.json g1 ...` swaps it for a run.

**Ctrl-C is safe.** An interrupted run (Ctrl-C, the task's `max_time_s`, an error) never drops
the arms where they are: the runner brings them from the last commanded pose to the stand pose
over 3 s, then fades the arm_sdk weight to 0 over 2 s so the onboard controller takes over,
base stopped from the first tick. Further Ctrl-Cs during that return, and during the robot's
hand-over, are ignored — there is no forced release. With `--sim` the twin keeps running until
that is done.

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
