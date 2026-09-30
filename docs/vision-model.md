# The vision-language model

The model runs only when the agent asks: one request per decision, on a background thread,
with the robot standing still. Backend: **Hugging Face Inference Providers** (`g1/vlm.py`),
any model on the router that takes images and returns JSON.

```
HF_TOKEN        a fine-grained token with "Make calls to Inference Providers"
VLM_MODEL       optional: the model id (default Qwen/Qwen3-VL-30B-A3B-Instruct, verified live);
                a suffix such as ":deepinfra" pins one of the router's providers
VLM_BASE_URL    optional: another Hugging Face endpoint, e.g. a dedicated Inference Endpoint
VLM_MAX_TOKENS  optional: the reply budget, reasoning included (default: limit max_tokens)
```

Put them in `.env` in the repo root (`cp .env.example .env`; git-ignored; exported
variables win) or export them. `--model` overrides the model for one run. Test it on a saved
frame before spending a run:

```
g1 decide runs/head.png --instruction "find the mug"      # streams the reply, prints the parsed decision
g1 decide runs/head.png --instruction "find the mug" --no-base   # the menu without the walking tools
```

## What the model is sent

The system prompt, once per run (`g1/agent/decider.py:instructions`): the robot and camera
conventions, the joint table and sign conventions, the scene's hidden obstacles
(`safety_notes`), the rules, and the tool catalog as bullets and as JSON. Then per
decision one user message: the observation JSON —

```
instruction, images [{name, width, height, captured_age_s}],
state {joint_pos, joint_vel, joint_torque, waist_yaw_deg, base_pose_cmd, base_pose_env},
extra {env_step, decisions_left, can_walk, attempt}, previous_result
```

— and the head image (JPEG at `image_width_px`). Structured output is requested as
`json_schema` (the output schema of the menu), stepped down to `json_object`, then none,
when a provider rejects it. The conversation is the model's memory: the system prompt once,
then every observation and the model's own reply verbatim; only the last
`live_image_window` (8) observations keep their image, text is never pruned, a turn-0
demonstration is never pruned. `--fresh-turns` makes every decision a fresh chat.

The reply must be one whole JSON object, `{"name": ..., "arguments": {...}}`, naming a tool
from the menu and satisfying its parameter schema (jsonschema Draft 2020-12; numbers out of
range are clamped and the model is told). Anything else is `invalid_selection` in the next
`previous_result`, and costs a decision.

## Cost and pacing

Roughly 0.5–1.5k tokens per image plus the text; a 30-decision run is on the order of
30–60k tokens, more with a reasoning model (its thinking counts against `max_tokens`; the
budget in `configs/limits.json` is a ceiling, not a cost). A reasoning model also takes
longer per decision (10–30 s measured) — raise `--step-timeout` if it times out. `g1 task
show` sums tokens per run from `results.jsonl`.

The sim outruns a real model when headless; it does not matter for the agent (the robot
stands still while the model thinks, on either clock), but `--realtime 1` paces a headless
run to wall-clock if you want the timings to mean what they would on the robot.

## Demonstrations on turn 0

Before the first observation the model can be shown a previous episode, prefixed
`HISTORICAL DEMONSTRATION` so it is read as reference and never as pending commands:

```
--demo runs/<earlier run>    its step PNGs are the keyframes; video+action by default:
                             the tool picked on each frame, joint angles at 1 Hz, the base pose
--demo walk_to_mug.mp4       a phone video: ffmpeg samples 2 fps, the model picks <= 8 keyframes per
                             30 s window with a stage and a reason (--demo-select uniform: evenly spaced)
--demo out/demo.json         a bundle compiled once with `g1 demo prepare`
--ref mug.png                a goal photo, labelled; repeatable
--input-json request.json    {"instruction", "content": ["text", {"image": p, "label": l}, {"video": p, "mode": m}]}
```

`--demo-frames` caps the keyframes (12, max 24); video keyframes are cached by content hash
under `runs/.cache/video`; the exact request is archived in the run as `input/input.json`.
A run made with `--record` leaves `camera.mp4` in its directory — the full 15 fps stream,
not just the decision frames — and that file is a demonstration too (`--demo
runs/<dir>/camera.mp4`); in sim it plays at sim speed, so a headless run is watchable.

```
g1 demo prepare --instruction "find the mug" --demo runs/<good run> out/   # compile once
g1 demo show out/demo.json                                                 # what the model gets
```
