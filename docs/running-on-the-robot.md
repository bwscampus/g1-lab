# Running on the robot

```
g1 run --env robot --tools tpose --iface <iface_or_ip> --mode standing
g1 run --env robot --tools walk_forward:0.5 --iface <iface> --mode standing --walk
g1 task run tasks/find_the_mug --env robot --iface <iface> --mode standing --camera-ip <ip> --walk
```

Run it in sim first (`--env sim --headless`); the stages are not chained automatically.

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

## `--mode` (required, no default)

* `gantry`: full bring-up and shutdown for a robot hanging in a gantry. Damp → FSM 4
  (locked stand) → FSM 200 (main operation) → run → release the arms → Damp.
* `standing`: for a robot already standing under its own controller. Records the current
  FSM, goes to FSM 200, runs, releases the arms, returns to the recorded FSM. Never damps.

Both hand the arms back on exit. The camera, when used, connects **before** any FSM change
so a bad link fails before the robot is touched, and is closed last.

## What can and cannot move

Only the waist and arms are commanded (arm_sdk, 50 Hz, the blend weight in
`motor_cmd[29]`); the legs balance under the onboard controller. Base motion is
`LocoClient.Move` — a 1 s dead-man command re-sent from a 10 Hz thread, clamped to
`BASE_VEL_MAX`, `StopMove` when a tool's base command clears and again before the arms are
released. Move works in FSM 200, so walking adds no FSM transition. The onboard gestures a
tool may call are `WaveHand` and `ShakeHand`, nothing else; FSM, damp, torque, sit and squat
are unreachable from a tool or a model reply. The joint monitor runs report-only over the
measured angles (teardown already releases the arms; stopping mid-run is its own risk) and
`report()` counts tick overruns.

## Ctrl-C

An interrupted run (Ctrl-C, `--max-time`, any exception) never drops the arms where they
are: the runner brings them from the last commanded pose to the stand pose over 3 s, then
fades the arm_sdk weight to 0 over 2 s, base stopped from the first tick, and only then
leaves the env. Ctrl-C is ignored during that return and during the hand-over; there is
deliberately no second-Ctrl-C escape, so an accidental key press can never cause an unsafe
manoeuvre. Your emergency stop is the remote (L2+B), not the keyboard.

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
