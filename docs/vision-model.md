# The vision-language model

The model runs only when the agent asks: one request per decision, on a background thread,
with the robot standing still. Backend: **Hugging Face Inference Providers** (`g1/vlm.py`),
any model on the router that takes images and returns JSON.

```
HF_TOKEN        a fine-grained token with "Make calls to Inference Providers"
VLM_BASE_URL    optional: another Hugging Face endpoint, e.g. a dedicated Inference Endpoint
```

The rest is in `configs/limits.json` (`g1 limits`), not `.env`:

```
vlm_model         the model id; a suffix such as ":deepinfra" pins one of the router's providers
                  (Qwen/Qwen3-VL-30B-A3B-Instruct is the verified fast fallback)
reasoning_effort  sent as reasoning_effort (none|minimal|low|medium|high, or default to leave it
                  out); dropped if the server rejects it
max_tokens        the reply budget, reasoning included
```

Setting `VLM_MODEL`, `VLM_MAX_TOKENS` or `VLM_REASONING_EFFORT` in `.env` is refused with a pointer to
the file, so an old `.env` never silently overrides it.

Thinking time is most of a decision's wall time. Left to the provider, DeepSeek-V4.1-Flash
spent 3 to 52 s per decision and once ran into the 60 s `step_timeout_s`; the agent prints the
median and longest think at the end of every run, and `usage.jsonl` records the effort sent.

Put the token in `.env` in the repo root (`cp .env.example .env`; git-ignored; exported
variables win) or export it. A task with an `instruction` uses them; try one on the twin
before spending a robot run: `g1 run find_the_mug --sim`.

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
`live_image_window` (8, in `configs/limits.json`) observations keep their image; text is never
pruned.

The reply must be one whole JSON object, `{"name": ..., "arguments": {...}}`, naming a tool
from the menu and satisfying its parameter schema (jsonschema Draft 2020-12; numbers out of
range are clamped and the model is told). Anything else is `invalid_selection` in the next
`previous_result`, and costs a decision.

## Cost and pacing

Roughly 0.5–1.5k tokens per image plus the text; a 30-decision run is on the order of
30–60k tokens, more with a reasoning model (its thinking counts against `max_tokens`; the
budget in `configs/limits.json` is a ceiling, not a cost). A reasoning model also takes
longer per decision (10–30 s measured) — raise `step_timeout_s` in `configs/limits.json` if
it times out. Each run's tokens are in its line of the task's `results.jsonl`. `--sim` runs in
real time, so the timings mean what they would on the robot.
