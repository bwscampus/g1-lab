# Running on the robot

```
g1 run tpose --mode standing           # a robot standing under its own controller; may walk
g1 run tpose --mode gantry             # a robot hanging in a gantry; never walks
g1 run find_the_mug --mode standing --view   # the model decides; its camera in a browser
```

Run the task with `--sim` first; the stages are not chained automatically. `--sim` *is* the
robot path: the same `RobotEnv` code, the same DDS, the same mode, against the twin in its own
process on the loopback interface:

```
g1 run find_the_mug --sim --mode standing
```

The robot's address comes from `.env`, never a flag: `UNITREE_IFACE` (the laptop's
interface on the robot's network, or the robot's IP), `UNITREE_ROBOT_IP` (the head camera),
`UNITREE_AES_128_KEY` (the camera, firmware ≥ 1.5.1), and `UNITREE_DOMAIN` if your robot is
not on DDS domain 0.

To see how far the twin is from your robot, run the same task on both (in the gantry, arms
only) and compare the two run directories:

```
g1 run tpose --mode gantry
g1 run tpose --sim --mode gantry
g1 trace compare tasks/tpose/runs/<robot run> tasks/tpose/runs/<twin run>
```

## Install

`unitree_sdk2py` (from [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python))
for control (and for `--sim`); for the camera, `unitree_webrtc_connect` (from
[go2_webrtc_connect](https://github.com/legion1581/go2_webrtc_connect)) plus `pip install -e
".[camera]"`. Neither is on PyPI.

## Pre-flight

* the robot is NOT in debug mode; clear space around the arms; someone on the remote with
  L2+B ready
* `--mode standing` lets the tools and the model walk: on the floor, ~2 m clear in every
  direction, no tether to snag, the spotter's thumb on L2+B. In `--mode gantry` the walking
  tools are not offered to the model and a task that drives the base is refused up front.
* the camera: close the Unitree app (the robot takes one WebRTC client). Expect
  `(720, 1280, 3)` at ~15 fps.
* the spotter's view: add `--view` and open `http://127.0.0.1:8765` (a free port if that one is
  taken; the URL is printed). Because the robot takes one WebRTC client, this is the only live
  picture there is. It reads the same latest-only frame slot the agent does, on its own thread,
  and adds no latency to control; the stream itself is ~100–200 ms behind reality.

## `--mode` (required, no default)

* `gantry`: full bring-up and shutdown for a robot hanging in a gantry. Damp → FSM 4
  (locked stand) → FSM 200 (main operation) → run → release the arms → Damp. Never walks.
* `standing`: for a robot already standing under its own controller. Records the current
  FSM, goes to FSM 200, runs, releases the arms, returns to the recorded FSM. Never damps.
  May walk.

Entering FSM 200 is what gives this process the arms and the base, so the bring-up prints
`entering control mode (FSM 200, main operation)` — or `already in control mode` when the
robot is there, which skips the transition and its wait. There is no countdown. The arms are
taken where they are (a 2 s weight ramp, nothing moves) and released where the last tool
left them (a 2 s fade); the onboard controller takes them from there. The camera, when used,
connects **before** any FSM change so a bad link fails before the robot is touched, and is
closed last.

## The speaker

`say(text)` speaks through the robot's onboard text-to-speech (`AudioClient.TtsMaker`) at the
volume set on the robot. The first time, hear it and correct the guesses in
`configs/limits.json`: `speech_chars_per_s` (how long `say` waits per character) and
`tts_speaker_id` (which voice). A robot without the voice service, or a failed call, prints a
warning and the run goes on.

## A task that hears

A task with `"hears": true` (`point_at_what_you_hear`) listens through the robot's own speech
recognition: switch voice interaction on in the Unitree app first (then close the app, so the
camera is free). At bring-up the run prints `microphone: listening on rt/audio_msg`, and at the end
how many utterances it heard and dropped. Speak clearly, near the head, one request at a time:
"point at the mug", then "now the laptop", then "stop". What you said is in each step's
`query` (`heard`). The robot's own assistant may also answer you; if that gets in the way, see
the note below.

## The robot's audio, as measured (2026-10-30)

Everything goes through the robot's `voice` service and its microphone, on the robot's
network (`UNITREE_IFACE`), read with `unitree_sdk2py` (`g1.audio.AudioClient`):

- **Speaker:** `PlayStream(app_name, stream_id, pcm)` plays raw **16 kHz mono 16-bit
  little-endian PCM**: no header and no rate field, so any other rate plays at the wrong
  speed. Files are converted on the host. The robot buffers and plays on its own: `TtsMaker`
  returned in ~2 ms, `PlayStream` with 3-4 s of audio in ~0.1 s, `PlayStop` in ~2 ms, all with
  code 0. Nothing waits for the sound to end.
- **Speech recognition:** the robot transcribes what it hears and publishes it on DDS
  `rt/audio_msg` (`std_msgs String`, JSON). An utterance arrives as
  `{"text": "Hello, robot.", "language": "<|en|>", "index": 28, "timestamp": ..., "is_final": false, "confidence": 0.5, ...}`;
  `is_final` and `confidence` did not vary. The same topic sends `{"play_state": 1}` and
  `{"play_state": 0}` when the robot's speaker starts and stops. Short or noisy speech comes
  back mislabelled as `<|zh|>` or `<|ja|>`.
- **Raw microphone:** UDP multicast `239.168.123.161:5555`, 16 kHz mono 16-bit, 5120-byte
  packets. The speaker's own output is mostly cancelled from it.
- **Both need voice interaction switched on in the Unitree app.** Without it, `rt/audio_msg`
  and the multicast stay silent, with no error. With it on, they keep working in dev mode
  (motion service released). The robot's own assistant then also answers what it hears.
- Leaving dev mode (`MotionSwitcherClient.SelectMode("ai")`) brings the robot back in FSM 0,
  zero torque; `LocoClient.Damp()` puts it in damp.

If the built-in assistant gets in the way (it answers, or plays over us), take the raw
multicast stream and do our own speech-to-text instead of `rt/audio_msg`, as
[unitree_converse](https://github.com/SaxionMechatronics/unitree_converse) does.

## What can and cannot move

Only the waist and arms are commanded (arm_sdk, 50 Hz, the blend weight in
`motor_cmd[29]`); the legs balance under the onboard controller. Base motion is
`LocoClient.Move` — a 1 s dead-man command re-sent from a 10 Hz thread, clamped to
`BASE_VEL_MAX`, `StopMove` when a tool's base command clears and again before the arms are
released. Move works in FSM 200, so walking adds no FSM transition. The onboard gestures a
tool may call are `WaveHand` and `ShakeHand`, and one FSM: `Sit` (the `sit` tool, `--mode
standing` only), sent as `SetFsmId(3)`; after it the agent refuses every walking and onboard tool,
and the teardown sends `SetFsmId(801)` (sit to stand), waits up to `sit_stand_timeout_s` for a
standing FSM and returns to 200, or warns that the robot is still seated. Try the sit by hand
first (a `{"tools": "sit"}` task, a chair already behind the heels, a spotter, the remote ready)
and time it into `sit_seconds`. Any other FSM, damp, torque and squat are unreachable from a
tool or a model reply; so are raw audio playback, the volume and the
LED strip (`AUDIO_METHODS` allows only `TtsMaker`). The joint monitor runs report-only over
the measured angles (teardown already releases the arms; stopping mid-run is its own risk) and
`report()` counts tick overruns. With `--sim` the same monitor is strict and a fall fails the run.

## Unitree's stored moves (`g1 moves`)

The robot carries Unitree's own moves; `g1 moves` lists, checks and plays them, by hand (the
model never reaches them). What is installed depends on the robot and its firmware, so the robot
is asked every time (`GetMimicMotion`, `GetActionList`); the twin has none.

```
g1 moves                       # read-only: the FSM and everything stored, with durations
g1 moves check bow arm:hug     # read-only: installed or not (exit 1 when one is missing)
g1 moves fsm walk              # FSM switches along their order, shown and confirmed first
g1 moves play bow 504 arm:20   # from standing (FSM 200/500/501/801/802), in order, confirmed first
```

- **Mimic motions** (whole body): `SetFsmId(550<id>)`, e.g. 516 bow = `SetFsmId(550516)`. Played on our G1: 570, 516, 531-533, 502. The list
  comes from the FSM API (7100) with `{"fsm_id": 550, "api_id": 1}`, which unitree_sdk2 (C++)
  wraps as `LocoClient::GetMimicMotion` and the Python SDK does not (`link.mimic`). Ids found on our
  G1 (2026-11-02): 4xx get up / lie down, 5xx gestures (507-573) and dances (502-506, 574), 6xx dances (601-605).
- **Arm actions** (upper body, the `arm` service): `ExecuteAction(id)`; 99 releases one that holds
  its last keyframe (they let go after at most 20 s on their own). Some run only in FSM 500/501 or
  only on some `mode_machine` (the LowState field; ours is 5): the list says so.
- **Taught actions** (recorded in the app): `ExecuteCustomAction(name)`, `StopCustomAction`.
- **FSM order**: 0 -> 1 (damp) -> 4 (locked stand, hanging) -> 501 (walk) or 200; 200 <-> 501;
  200/501 -> 3 (sit) -> 801 (stand up) -> 200/501. Walking is **501** on our 29-DoF G1 (3-DoF waist):
  it takes `SetFsmId(500)` with code 0 and stays in 4 (500 is the 1-DoF-waist G1's). Damp drops a
  standing robot, so it is only ever the end of a switch, never a way through (200 -> 4 has no path:
  use the remote). `g1 moves fsm walk` is 501.
- **A move, as measured (2026-11-03)**: 501 -> 550 while it plays -> 802 when it ends; the next move
  plays from 802, so a list chains with no FSM step between. The play waits for the robot to leave
  550, not for the listed duration: the boxing moves (531-533, listed 2.6-3.9 s) hold their last
  pose and stayed in 550 for about 23 s. 802 -> 501 (`g1 moves fsm walk`) is allowed but not yet tried.
- Ctrl-C during a move stops it (a mimic motion: back to the FSM it started in; an arm action:
  release) and that stop finishes. Times: `moves_settle_s`, `moves_overrun_s`, `arm_action_s` (guesses: time them).

Unitree's reference: unitree_sdk2 `include/unitree/robot/g1/loco/g1_loco_client.hpp`
(GetMimicMotion) and `include/unitree/robot/g1/arm/g1_arm_action_client.hpp`; the G1 sport
services page, support.unitree.com/home/en/G1_developer/sport_services_interface.

## Conversation demo (`g1 talk`)

`g1 talk` stands the robot ready (the `g1 moves fsm walk` order, if it is not standing), asks once
whether moves may play, then listens. Each time someone speaks and pauses (`talk_quiet_s`), what the
robot's speech recognition heard goes to the model (`configs/model.json`), which picks one of three
tools: `say` (spoken with `TtsMaker`), `move` (a stored gesture or dance, ids 5xx/6xx only: never a
get-up or lie-down) or `ignore` (not meant for it). It never runs a task, takes the arms, walks or
switches the FSM after the start. Needs voice interaction on in the app (or `rt/audio_msg` stays
silent) and the model's key in `.env` on the machine that runs it. Ctrl-C ends it (a move playing is
stopped first). What it heard, its choice, the model's time and what it said are printed.

## Ctrl-C

An interrupted run (Ctrl-C, the task's `max_time_s`, any exception) never drops the arms where
they are: the runner brings them from the last commanded pose to the stand pose over 3 s, then
fades the arm_sdk weight to 0 over 2 s, base stopped from the first tick, and only then
leaves the env. Ctrl-C is ignored during that return and during the hand-over; there is
deliberately no second-Ctrl-C escape, so an accidental key press can never cause an unsafe
manoeuvre. Your emergency stop is the remote (L2+B), not the keyboard.

## When the robot seems stuck

"It rejects every state change and stops streaming" is two different failures, and they
need different fixes. Everything — the LowState stream and the *replies* to `SetFsmId` —
travels over the same DDS transport, so:

| what you see | meaning | fix |
|---|---|---|
| requests fail fast with a service code, LowState keeps streaming | the sport service **refuses**: a protection state (after a fall, a joint fault) or another controller owns the FSM (the app, the remote's mode) | close the app, check the remote, clear the fault (often a power cycle) |
| requests time out (3104) or cannot be sent (3102), and LowState stops | the **link** is gone: wrong or flapping interface, WiFi vs Ethernet, the robot's services restarted, a leftover DDS participant from a process that was killed hard | fix the interface/subnet (`192.168.123.x`), restart the run, restart the robot's services |

`g1 status` tells them apart without touching the robot:

```
g1 status                          # read-only: no arm_sdk, no FSM change; UNITREE_IFACE from .env
lowstate: 1491 messages in 3s on en7 = 497/s, newest 2 ms old, joint0 q -0.012, battery 81%
rpc:      GetFsmId -> 0 (ok) in 0.004s, FSM 200 (main operation)
verdict:  link ok, FSM 200 (main operation)
```

The other verdicts are `no link: …`, `link ok … but requests fail with <code>: the service
refuses`, and `link ok; FSM 1 (damp) — not a standing state`.

A run checks every SDK call it makes: a failed transition during bring-up stops the run
**before** arm_sdk is touched, with the code named and the same hint; a robot reporting
damp/zero-torque is refused by `--mode standing`; at the end the recorded FSM is restored
only when it was a standing state. Every call is recorded (client, name, args, code, elapsed):
in the step file it happened in, or, at bring-up, in `summary.json`, so a stuck run shows the last thing
the robot answered. `G1_VERBOSE=1` prints all of it live: each call with its code as it
happens, the bring-up step by step, and once a second

```
t=  12.0  lowstate 498/s (age 2 ms)  ticks 600 overruns 0  base Move ok 38 ms  camera 180 frames, age 60 ms
```

## The AES key (camera, firmware ≥ 1.5.1)

The head camera streams over WebRTC and newer firmware requires the robot's AES-128 key. It
is per device and does not change; fetch it once with the account the robot is bound to in
the Unitree Explorer app (`unitree-fetch-aes-key` comes with `unitree_webrtc_connect`):

```
unitree-fetch-aes-key --email you@example.com                      # lists every bound robot
unitree-fetch-aes-key --email you@example.com --sn <serial> --quiet  # only that robot's key
```

`--region cn` for the Chinese cloud. Put the key and the address in `.env`:

```
UNITREE_ROBOT_IP=192.168.123.161
UNITREE_AES_128_KEY=<32 hex characters>
```

If the camera does not connect: the key belongs to another robot, the app is still connected,
or the robot is not reachable at that address (`ping`).
