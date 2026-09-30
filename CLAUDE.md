# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Unitree G1 (29-DoF) driven one **tool** at a time by a vision-language model, in
GPT-Policy's shape (arXiv:2609.19138): the model picks a tool from a menu and fills in its
arguments; the host plans a fixed motion from them, checks it, runs it, waits for the joints
to settle and reports. The same tools chain from the CLI without a model. Two envs, chosen
with `--env`: `sim` (MuJoCo, and the checks that gate a run; `--headless` is the fast
pre-flight) and `robot` (live via `unitree_sdk2py`). Run sim before the robot for anything new.

## Vocabulary (GPT-Policy's; use these words in code, CLI, docs)

- **tool** — one class in `g1/tools/`: `name`, `prompt`, `params` (JSON-schema fragments), `segments()`.
  A fixed motion (or a query) computed from the arguments; ends by construction; **never watches
  the camera while it runs** — the model is the closed loop. Visible to the model by default.
- **instruction** — the goal text. **task** — a folder (`tasks/<name>/task.json`) with an instruction and
  its context, no code. **observation / decision / execution result / verdict** — as in the paper.
- Retired, do not reintroduce: *skill*, *policy*, *primitive*, *routine*, *goal*, a catalog file, a
  tool that servos on vision, a per-tool timeout.

## Commands

```
pip install -e ".[sim,dev]"          # unitree_sdk2py / unitree_webrtc_connect come from their repos for --env robot

g1 run --env sim --tools tpose --headless                        # one tool, fully checked
g1 run --env sim --tools walk_forward:0.5,hold:1,turn:45 --headless   # a chain (--pause between)
g1 run --env sim --tools 'move:1:0.3:-45,arm_path:waypoints=[{"joints":{"left_elbow":-0.4}}]' --headless
mjpython -m g1 run --env sim --tools tpose                        # macOS: the viewer only works under mjpython
g1 run --env robot --tools tpose --iface <iface_or_ip> --mode gantry|standing   # --mode is required
g1 run --env robot --tools search -i "find the mug" --iface <iface> --mode standing --camera-ip <ip> --walk

HF_TOKEN=hf_... g1 run --env sim --scene room --tools search --instruction "find the mug" \
        --sim-objects mug@1.5,1.2 --camera-size 720x1280 --headless --max-time 600
g1 task run tasks/find_the_mug --env sim --headless             # the same from a task folder; records under tasks/<name>/runs/
g1 task eval tasks/find_the_mug -n 5 --env sim --headless       # success rate, decisions to success
g1 task show tasks/find_the_mug
g1 run --env sim --tools replay --episode runs/<dir> --headless  # a saved run, no camera or model
g1 run ... --tools search --demo runs/<earlier run> | walk.mp4 | out/demo.json   # demonstration on turn 0
g1 run ... --view [PORT]     # the head camera live at http://127.0.0.1:8765 (MJPEG, browser); works in sim
g1 run ... --record [PATH]   # every frame to camera.mp4 in the run dir (PyAV, H.264, stamps on the env clock)

g1 tools [--json | --joints | --prompts]   # the menu; the catalog the model reads; the joint table
g1 limits [--source guess]                 # every tunable number (configs/limits.json)
g1 new tool NAME | g1 new task NAME        # scaffold from g1/tools/_template.py | tasks/_template/
g1 decide runs/head.png --instruction "find the mug"   # one real model decision from a frame
g1 camera [--save PNG]                     # head camera smoke test, no robot control
g1 episode runs/<dir>                      # step table + the --tools chain that replays it
g1 demo prepare|show ; g1 scene fetch|status
python -m g1 ...  ==  g1 ...   ; --env/--tools fall back to $G1_ENV/$G1_TOOLS

pytest                               # all tests; everything through headless sim (~3 min)
pytest tests/tools/test_tools.py::test_every_tool_pair_is_continuous
```

## Layout

```
g1/cli.py          one command; run_parser() + run() + safe_return; subcommands dispatch to each module's main()
g1/core/           config (joint table, STAND_Q, UPPER_BODY, LOCO_METHODS, BASE_VEL_MAX), limits (configs/limits.json),
                   action (Action, Obs, Segment, Runnable, ease, EPS), poses, images (Pillow, RGB), worker
g1/tools/          __init__: discovery (TOOLS), menu(), the three renderings, parse_tool/parse_chain/chain_spec, Chain
                   base: the Tool contract + the segment player + validate_args + num/integer/text/flag/limit
                   move (move, walk_forward, turn) · arms (arm_path, look, tpose, sixseven) · gestures (wave_hand,
                   shake_hand) · control (hold, check + dry_run, done, give_up, takeover, handback) · _template (bow)
g1/envs/           base (Env, EnvAbort, shield_sigint), monitor (JointMonitor), sim, scene (the room, assets/), robot
g1/camera.py       Frame, Camera (+subscribe), DirCamera, WebRTCCamera, Viewer (--view), Recorder (--record)
g1/vlm.py          VLMClient (Hugging Face), load_dotenv
g1/agent/          agent (Agent, build_search, build_replay, AGENTS), decider (AgentContext/AgentTurn/Decision,
                   observation(), instructions(), VLMDecider), demo, episode (EpisodeWriter, StepRecord), task
tasks/             _template/, find_the_mug/; runs and results.jsonl inside are git-ignored
docs/              architecture, writing-a-tool, writing-a-task, running-on-the-robot, vision-model
tests/             mirrors g1/ (core, tools, envs, agent) + test_camera, test_cli, test_vlm; doubles.py (RedBallDecider,
                   red_blob, sim_env, scripted_sim); conftest keeps tests away from the real .env
```

Absolute imports (`from g1.core.config import ...`); tests are a package (`tests/__init__.py`), pytest gets the
root via `pythonpath`. `limits.ROOT` is the repo root (configs/, tasks/, assets/, .env all resolve from it).

## Rules that are easy to break

- **Tool metadata lives in the class, nowhere else.** Prompt + `params` + code in one file; the menu is
  generated by import (`g1/tools/__init__.py:discover`). Limits are referenced (`limit("name")`,
  `{limit:name}`) and resolved when the schema is read, never copied. Every limit is in
  `configs/limits.json` with a unit, a note and a source; a name the file lacks fails at import.
- **Parameters are numbers with ranges** (`num(lo, hi, default)`), unit-suffixed (`_s`, `_m`, `_deg`,
  `_rad`, or `seconds`), never enums for a continuous quantity; a property without a default is
  required; `note` is added automatically to motion and query tools (required of the model, "" for code).
  `tests/tools/test_tools.py::test_parameters_are_numbers_with_ranges_not_switches` enforces this.
- **A tool never overrides `step`** and never reads the camera. The player in `Tool` interpolates
  `segments()`. Query tools implement `query(cmd, joints)`; terminal tools end the run.
- **Seed from the last commanded q, never the measured one** when composing (Chain composes segments;
  the Agent seeds each tool from its own `_cmd`). On the robot the measured pose lags by gravity sag.
- **Every movement is dry-run before it executes** (`g1/tools/control.py:dry_run` through a
  `JointMonitor` from the commanded pose); a violation is `motion_not_executed`, spends a decision,
  moves nothing. **Errors are feedback**: no re-ask loop.
- **Ctrl-C runs the safe return to completion** (`cli.safe_return`: last commanded pose → STAND 3 s,
  weight → 0 over 2 s, base stopped, SIGINT shielded). Never add a second-Ctrl-C release or a keypress
  escape. Never call the SDK from `step`. Walking is `LocoClient.Move` behind `--walk` only; no
  low-level leg control; only `WaveHand`/`ShakeHand` may be called (`config.LOCO_METHODS`).
- **Frames are RGB everywhere**; image I/O only through `g1/core/images.py` (Pillow). OpenCV is not
  used and must not be imported (its bundled ffmpeg collides with PyAV's, which decodes the robot's video).
- **Taps never touch the tick.** `--view`/`--record` subscribe to the env's camera slot (`Env.source()`,
  `Camera.subscribe`: called on the producer's thread, must only enqueue), run on their own threads and
  drop frames when behind; `cli.run` starts them after `env.reset()` and stops them inside `with env`
  and before `program.close()` (the Agent's close renames the run directory the mp4 sits in). The robot
  takes one WebRTC client, so the view must come out of our process.
- **Hugging Face only** for the model (`HF_TOKEN`, `VLM_MODEL`, `VLM_BASE_URL`); no other providers,
  no Anthropic SDK. Fakes are test doubles only (`tests/doubles.py`); `search` always uses the real model.
- Sim blend semantics are emulated (`cmd = (1-w)*hold + w*target`), `hold` = the Menagerie stand
  keyframe; the sim pins the pelvis and slides it on a base command (`--free-base` aborts on one).
- Joint indexing is DDS order (= Menagerie actuator order); index 29 is the arm_sdk weight slot.

## The decision step (`g1/agent/agent.py`)

takeover → settle (0.03 rad, 0.05 rad/s, 10 ticks, ≥ 0.5 s, 3 s timeout) → snapshot (a fresh frame) →
think (background `Decider.request`; the loop holds) → act (the whole tool; a record every `STEP_MAX`
= 3 s) → settle → feedback (`previous_result`: residuals, base error, settle report | the error) → …
`done`/`give_up` end it; `check` answers via `dry_run`; overload retried with backoff re-observing;
quota or `step_timeout` fails the run; `max_decisions` ends as `budget_exhausted`; `close()` asks the
human verdict and renames the run directory (`_success`/`_failed`/`_unreviewed`/`_interrupted`).
Records: `runs/<ts>_<env>_<instruction>_<outcome>/` with `episode.json`, `step_NNNN.json/.png`,
`events.jsonl`, `transcript.json`, `protocol.json`, `states.jsonl`, `usage.json*`, `status.json`.
The observation JSON and the system prompt are in `g1/agent/decider.py`; `Decision.parse` validates a
whole JSON object against the tool's schema and clamps numbers with a note.
