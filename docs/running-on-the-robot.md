# Running on the robot

```
g1 run tpose --mode standing           # a routine, on a robot standing under its own controller
g1 run tpose --mode gantry             # a robot hanging in a gantry; never walks
g1 start --mode standing               # the session: it listens, you type, it chooses (below)
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

To see how far the twin is from your robot, run the same routine on both (in the gantry, arms
only) and compare the two run directories:

```
g1 run tpose --mode gantry
g1 run tpose --sim --mode gantry
g1 trace compare routines/tpose/runs/<robot run> routines/tpose/runs/<twin run>
```

## Install

`unitree_sdk2py` (from [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python))
for control (and for `--sim`); for the camera from a laptop, `unitree_webrtc_connect` (from
[go2_webrtc_connect](https://github.com/legion1581/go2_webrtc_connect)) plus `pip install -e
".[camera]"`. Neither is on PyPI.

## Pre-flight

* the robot is NOT in debug mode; clear space around the arms; someone on the remote with
  L2+B ready
* `--mode standing` lets the tools and the model walk: on the floor, ~2 m clear in every
  direction, no tether to snag, the spotter's thumb on L2+B. In `--mode gantry` the walking
  tools are not offered to the model and a task that drives the base is refused up front.
* the camera from a laptop: close the Unitree app (the robot takes one WebRTC client). Expect
  `(720, 1280, 3)` at ~15 fps. On the robot's Jetson the app can stay open (see below).
* the spotter's view: add `--view` and open `http://127.0.0.1:8765` (a free port if that one is
  taken; the URL is printed). From a laptop, because the robot takes one WebRTC client, this is
  the only live picture there is. It reads the same latest-only frame slot the agent does, on its own thread,
  and adds no latency to control; the stream itself is ~100–200 ms behind reality.

## Running on the robot's Jetson

The head camera (a RealSense D435i) is plugged into the robot's Jetson (`192.168.123.164` on
the robot's network; it is usually also on Wi-Fi). Unitree's video hub there
(`/unitree/module/video_hub_pc4`) encodes it in hardware, 1280x720 H.264 at 15 fps, and
multicasts it as RTP to `230.1.1.1:1720` on `eth0`; the main board (`.161`) relays the same
stream over WebRTC to the app and to a laptop. On the Jetson, `g1 run` takes the multicast
directly (`DirectCamera`: no relay hop, no jitter buffer, no one-client limit) and decodes it with
the Jetson's hardware decoder in a `gst-launch-1.0` child process. It chooses this by itself when
the video hub is installed and GStreamer is on the PATH, and prints
`camera: direct on the Jetson (...)`; anywhere else it uses WebRTC as before.

The WebRTC camera does not work in a run on the Jetson: its Python decoder shares the process
with `rt/lowstate` (~900 messages/s), and on the Jetson's cores that starves it for the GIL
(15 fps alone, ~0 fps in a run, `failed (no_frame)`).

Set up the Jetson once: the repo and `pip install -e .` in a Python 3.10 env with
`unitree_sdk2py`, and `.env` with `UNITREE_IFACE=eth0` (the interface name: GStreamer joins the
multicast on it) and the model's key. `UNITREE_ROBOT_IP` and `UNITREE_AES_128_KEY` are not
needed there. Run from a laptop over Wi-Fi and watch through a tunnel:

```
ssh -L 8765:127.0.0.1:8765 unitree@<the Jetson's Wi-Fi address>
g1 run point_at_the_laptop --mode gantry --view      # then http://127.0.0.1:8765 on the laptop
```

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
first (`g1 moves fsm sit`, a chair already behind the heels, a spotter, the remote ready)
and time it into `sit_seconds`. Any other FSM, damp, torque and squat are unreachable from a
tool or a model reply; so are raw audio playback, the volume and the
LED strip (`AUDIO_METHODS` allows only `TtsMaker`). The joint monitor runs report-only over
the measured angles (teardown already releases the arms; stopping mid-run is its own risk) and
`report()` counts tick overruns. With `--sim` the same monitor is strict and a fall fails the run.

## Built-ins: Unitree's stored moves (`g1 moves`)

The robot carries Unitree's own moves; `g1 moves` lists, checks and plays them, by hand (the
model never reaches them). What is installed depends on the robot and its firmware, so the robot
is asked every time (`GetMimicMotion`, `GetActionList`); the twin has none.

```
g1 moves                       # read-only: the FSM and everything stored, with durations
g1 moves check bow arm:hug     # read-only: installed or not (exit 1 when one is missing)
g1 moves fsm walk              # FSM switches along their order, shown and confirmed first
g1 moves play bow 504 arm:20   # from standing (FSM 200/500/501/801/802), in order, confirmed first
g1 moves walk 0.5              # also: side 0.3 (left), turn -90 (right); walk mode (501) first, confirmed first
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
  200/501 -> 3 (sit) -> 801 -> 200/501. **801 is run mode** (walk-run locomotion in Unitree's FSM table; the
  sit teardown stands up into it). `g1 moves walk` and `g1 start` walk in limit `walk_fsm`, 501: in 801 and
  802 SetVelocity returned 0 and the robot did not move (2026-11-03).
  `g1 moves fsm run` switches to it. Walk mode is **501** on our 29-DoF G1 (3-DoF waist):
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

## The lidar (`g1 lidar`)

The head lidar is a Livox MID-360 at `192.168.123.120`. It sends its points and IMU only to the main
board (`.161:56301`/`:56401`), never to the Jetson or a laptop, so the feed comes from Unitree's onboard
services: `lidar_driver` and `unitree_slam` (off by default on 2026-11-05). `g1 lidar` and `/lidar on`
switch on whichever is off (`RobotStateClient.ServiceSwitch`, only these two), wait until the service
list (`ServiceList`, status 0 = running) shows both running (at most `lidar_service_start_s`), and at the
end switch off the ones they switched on, so the robot is left as it was found. A service already on
stays on.

`g1 lidar` then starts a mapping session on Unitree's SLAM (`slam_operate` 1801,
`slam_type` indoor) and reads `rt/unitree/slam_mapping/points` (PointCloud2, frame `map`, ~1,000 points
a scan at 10 Hz) with the pose from `rt/unitree/slam_mapping/odom`. It maps until Ctrl-C (at most
`lidar_max_s`), printing a line a second (frame, points, rate, bounds, pose, yaw, map size; the first
scan also prints the cloud's fields), then always ends the session (Ctrl-C, a `kill` and a dropped
ssh all end it the same way; the teardown can't be cut short) (1802; Unitree's own map lands on
the main board, which has no ssh, so it is out of reach), closes SLAM (1901) and saves **our map as
`runs/lidar/<time>/map.pcd`** (5 cm voxels, map frame). SLAM's origin is the MID-360's IMU at the start,
X forward, Z up: on the gantry the floor is at about z -1.3 m.

**The live view.** While it maps, `g1 lidar` serves a 3D view at `http://127.0.0.1:8766`
(`lidar_view_port`; the URL is printed), like the camera's `--view`: the map dimmed and the latest scan
bright, both coloured by height above the floor, a 1 m floor grid, and the robot as a red dot with a
heading line. Drag to orbit, shift- or right-drag to pan, the wheel to zoom; Top, 3D, Follow robot,
Scan and Map buttons. It is plain canvas (nothing fetched from the internet). Run on the Jetson and
watch from a laptop over Wi-Fi:

```
ssh -L 8766:127.0.0.1:8766 unitree@<the Jetson's Wi-Fi address>
cd ~/g1-lab && g1 lidar                  # then open http://127.0.0.1:8766 on the laptop; Ctrl-C ends it
scp unitree@<the Jetson>:g1-lab/runs/lidar/<time>/map.pcd .    # open in CloudCompare or Open3D
```

The raw driver cloud (`rt/utlidar/cloud_livox_mid360`, `livox_frame`, ~20,000 points at 10 Hz) is not
levelled (the lidar is mounted tilted), which is why the map is built from SLAM's points.

For a 3D map, `g1/lidar.py:Scan` holds `points` (N x 4 float32: x, y, z in m, intensity) and `pose`
(4 x 4, map <- body); `Scan.map_points()` gives the points in the map frame whichever frame the cloud
came in; `MapAccumulator` voxel-downsamples scans (`lidar_voxel_m`) and `write_pcd` saves the result for
Open3D, PCL or CloudCompare.

Unitree's warnings: don't use the app's mapping or navigation at the same time; keep maps under 45 m in
X and Y (beyond it SLAM can load the onboard computer enough to affect locomotion); avoid violent
motion (it loses localisation). Sources: Unitree's SLAM and navigation services interface, and the
service list and lidar parameters read from this robot.

## The session (`g1 start`)

```
g1 start --mode gantry           # hanging: routines and tasks only (each damps at the end)
g1 start --mode standing         # on its own feet: built-ins, walking, everything
```

A Claude-Code-style terminal: an input box at the bottom, output scrolling above it, a status line
under it (mode, FSM, mic, camera and lidar with their URLs, what is running). The robot listens (its
speech recognition) and you type: **plain text is chat**, exactly as if you had said it. For each
thing said or typed the model (`configs/model.json`) makes one choice:

* `say`: an answer (spoken with `TtsMaker`, English voice: `tts_speaker_id` 1), and nothing else;
* `move` / `walk`: a **built-in**, standing mode only: a stored gesture or dance (ids 5xx/6xx, never
  a get-up or lie-down), or a turn, walk and sidestep capped by `talk_turn_max_deg`, `talk_walk_max_m`,
  `talk_side_max_m` in walk mode (501, `walk_fsm`). It cannot see while it talks: it walks only when
  asked, and the cap is the only thing between a request and a wall;
* `play_routine`: a **routine** from `routines/`; `run_task`: a **task** from `tasks/`, with the camera;
* `ignore`: not meant for it.

It only chooses: it never writes a routine or a task. A routine or task **the model** chooses is
confirmed at the keyboard (`y` then Enter; anything else or Esc is no). **You** can call any skill
directly, without the model and without the question:

| command | does |
|---|---|
| `do a T-pose`, `/chat ...` | chat: as if said |
| `/skills` | everything, marked available or not in this mode, and why |
| `/routine NAME` · `/task NAME` | run one now |
| `/move ID\|NAME` · `/walk FWD [SIDE [TURN]]` · `/say TEXT` | a built-in now; the robot says it |
| `/mode gantry\|standing` | switch mode: it **never moves the robot**; it asks you to confirm it is hanging / standing (standing also needs a standing FSM) |
| `/mic on\|off` | handle what the robot hears like chat, or ignore it (a task that hears needs it on) |
| `/camera on\|off` | the head camera, live at `http://127.0.0.1:8765` (`ssh -L 8765:...` from a laptop); tasks reuse it |
| `/lidar on\|off` | switches the lidar services on if needed, SLAM mapping and its live 3D view at `:8766`; off (and exit) saves `runs/lidar/<time>/map.pcd` and switches off the services it switched on |
| `/status` · `/stop` · `/help` · `/quit` | the `g1 status` probe; Esc; the list; exit |

A routine or task runs through `g1 run`'s own path in the session's mode (standing: back to walk
mode first if a built-in left it in 802; gantry: the gantry bring-up and damp), sharing the session's
microphone and camera, kept as `_unreviewed` (no questions mid-session; label it later). A task's
`done` summary is spoken; a refusal (the wrong mode, the robot not standing, a routine's dry run) is
said, printed, and the session goes on. Settings and skills typed while one runs wait for it to end;
`/mode` asks you to stop it first.

**Stopping and exiting:**

| | nothing running | a skill running |
|---|---|---|
| **Esc** | nothing | stop it to its neutral: a built-in move back to the FSM it started from, a walk to zero velocity, a routine or task through the safe return (below); the session goes on |
| **Ctrl-C** | clears the line; on an empty line "press Ctrl-C again to exit", a second within 2 s (`console_ctrl_c_window_s`) exits | stops it like Esc (a reflex Ctrl-C never does nothing while the robot moves) |
| `/quit`, Ctrl-D, `kill` (SIGTERM), a dropped ssh (SIGHUP) | exit | stop it first, then exit |
| the remote's **L2+B** | damp: the emergency stop, whatever the code is doing | |

Every stop is SIGINT to the process (the terminal runs in its own thread; skills run on the main
thread), the path the safe return is shielded on: a stop or exit pressed during the return waits until
the robot is back to neutral. On exit the lidar's mapping ends and its map is saved and the camera
closes. Needs voice interaction on in the app for the mic, the model's key in `.env` for chat (without
it, only direct commands work), and `prompt_toolkit` (in `pip install -e .`).

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
