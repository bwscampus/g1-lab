# Running on the robot

```
g1 run --env robot --tools tpose --iface <iface_or_ip> --mode standing
g1 run --env robot --tools walk_forward:0.5 --iface <iface> --mode standing --walk
g1 task run tasks/find_the_mug --env robot --iface <iface> --mode standing --camera-ip <ip> --walk
```

Run it in sim first (`--env sim --headless`); the stages are not chained automatically.
`--iface` (the laptop's interface on the robot's network, or the robot's IP) and
`--camera-ip` default to `UNITREE_IFACE` and `UNITREE_ROBOT_IP` from `.env`, so with those
set the flags can be left out.

## Install

`unitree_sdk2py` (from [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python))
for control; for the camera, `unitree_webrtc_connect` (from
[go2_webrtc_connect](https://github.com/legion1581/go2_webrtc_connect)) plus `pip install -e
".[camera]"`. Neither is on PyPI; neither is needed for sim.

## Pre-flight

* the robot is NOT in debug mode; clear space around the arms; someone on the remote with
  L2+B ready
* walking (`--walk`): on the floor or hoisted with the feet touching, ~2 m clear in every
  direction, no tether to snag, the spotter's thumb on L2+B. Without `--walk` the walking
  tools are simply not offered to the model and a chain that drives the base is refused up
  front.
* the camera: close the Unitree app (the robot takes one WebRTC client), then `g1 camera`
  — frame size, fps and gaps for 5 s, no robot control. Expect `(720, 1280, 3)` at ~15 fps.
* the spotter's view: add `--view` to any run and open `http://127.0.0.1:8765` on the laptop
  (`--view 0` picks a free port; the URL is printed). Because the robot takes one WebRTC
  client, this is the only live picture there is — the app cannot watch at the same time.
  It reads the same latest-only frame slot the agent does, on its own thread, and adds no
  latency to control; the stream itself is ~100–200 ms behind reality. `--record` writes
  every frame to `camera.mp4` in the run directory (PyAV, H.264, real-time stamps).

## `--mode` (required, no default)

* `gantry`: full bring-up and shutdown for a robot hanging in a gantry. Damp → FSM 4
  (locked stand) → FSM 200 (main operation) → run → release the arms → Damp.
* `standing`: for a robot already standing under its own controller. Records the current
  FSM, goes to FSM 200, runs, releases the arms, returns to the recorded FSM. Never damps.

Entering FSM 200 is what gives this process the arms and the base, so the bring-up prints
`entering control mode (FSM 200, main operation)` — or `already in control mode` when the
robot is there, which skips the transition and its wait. There is no countdown. The arms are
taken where they are (a 2 s weight ramp, nothing moves) and released where the last tool
left them (a 2 s fade); the onboard controller takes them from there. The camera, when used,
connects **before** any FSM change so a bad link fails before the robot is touched, and is
closed last.

## The speaker

`say(text)` speaks through the robot's onboard text-to-speech (`AudioClient.TtsMaker`);
`--volume 0-100` sets the volume before the run. The first time, hear it and correct the
guesses in `configs/limits.json`: `speech_chars_per_s` (how long `say` waits per character)
and `tts_speaker_id` (which voice):

```
g1 run --env robot --tools 'say:text=hello, I am looking for the mug' --iface <iface> --mode standing --volume 60
```

A robot without the voice service, or a failed call, prints a warning and the run goes on.

## What can and cannot move

Only the waist and arms are commanded (arm_sdk, 50 Hz, the blend weight in
`motor_cmd[29]`); the legs balance under the onboard controller. Base motion is
`LocoClient.Move` — a 1 s dead-man command re-sent from a 10 Hz thread, clamped to
`BASE_VEL_MAX`, `StopMove` when a tool's base command clears and again before the arms are
released. Move works in FSM 200, so walking adds no FSM transition. The onboard gestures a
tool may call are `WaveHand` and `ShakeHand`, nothing else; FSM, damp, torque, sit and squat
are unreachable from a tool or a model reply; so are raw audio playback, the volume and the
LED strip (`AUDIO_METHODS` allows only `TtsMaker`). The joint monitor runs report-only over
the measured angles (teardown already releases the arms; stopping mid-run is its own risk) and
`report()` counts tick overruns.

## Ctrl-C

An interrupted run (Ctrl-C, `--max-time`, any exception) never drops the arms where they
are: the runner brings them from the last commanded pose to the stand pose over 3 s, then
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
g1 status --iface <iface>          # read-only: no arm_sdk, no FSM change
lowstate: 2487 messages in 5s = 497/s, newest 2 ms old, joint0 q -0.012, battery 81%
rpc:      GetFsmId -> 0 (ok) in 0.004s, FSM 200 (main operation)
verdict:  link ok, FSM 200 (main operation)
```

The other verdicts are `no link: …`, `link ok … but requests fail with <code>: the service
refuses`, and `link ok; FSM 1 (damp) — not a standing state`. `--json` for scripts.

A run checks every SDK call it makes: a failed transition during bring-up stops the run
**before** arm_sdk is touched, with the code named and the same hint; a robot reporting
damp/zero-torque is refused by `--mode standing`; at the end the recorded FSM is restored
only when it was a standing state. Every call is written to the run's `events.jsonl` as an
`sdk_call` line (client, name, args, code, elapsed), so a stuck run shows the last thing
the robot answered. `--verbose` / `-v` prints all of it live: each call with its code as
it happens, the bring-up step by step, and once a second

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

If `g1 camera` fails: the key belongs to another robot, the app is still connected, or the
robot is not reachable at that address (`ping`).
