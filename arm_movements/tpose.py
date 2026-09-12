"""
Arm control on the G1 via the arm_sdk topic, with ai_sport still balancing.

    python g1_arm_sdk_wave.py <network_interface_or_ip>

Pre-flight:
  - Robot standing on the floor in FSM 200 (bring up from the app or your
    working LocoClient sequence). Do NOT put it in debug mode.
  - Clear space around the arms; they will rise to shoulder height.
  - Someone on the remote, L2+B ready.

How it works:
  - We publish LowCmd_ on "rt/arm_sdk" at 50 Hz.
  - motor_cmd[29].q is the "arm sdk weight": 1.0 = our targets take over
    the arms, 0.0 = controller has them. We ramp it up at the start and
    back down at the end so the handover is smooth.
  - Arm joints are indices 15-28 (identity mapping to Menagerie, as you
    verified). We hold waist 12-14 at their current reading.
"""
import sys
import time
import math

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from arms import ArmSdk

# Joint indices (29-DoF G1)
WAIST = [12, 13, 14]
L_ARM = [15, 16, 17, 18, 19, 20, 21]   # shoulder P/R/Y, elbow, wrist R/P/Y
R_ARM = [22, 23, 24, 25, 26, 27, 28]
ARM_JOINTS = L_ARM + R_ARM
CTRL_JOINTS = WAIST + ARM_JOINTS
WEIGHT_IDX = 29                        # motor_cmd[29].q carries the arm-sdk weight

KP, KD = 60.0, 1.5
DT = 0.02                              # 50 Hz

# Neutral / default pose: arms hanging at the sides, elbows slightly bent, waist zero.
# Every run starts by moving to this from the current pose, and ends by returning to it.
NEUTRAL = {
    12: 0.0, 13: 0.0, 14: 0.0,                                              # waist
    15: 0.0, 16: 0.0, 17: 0.0, 18: 1.5, 19: 0.0, 20: 0.0, 21: 0.0,         # left
    22: 0.0, 23: 0.0, 24: 0.0, 25: 1.5, 26: 0.0, 27: 0.0, 28: 0.0,        # right
}

# Target pose: T-pose, arms straight out to the sides at shoulder height, elbows straight
ARMS_UP = {
    15: 0.0, 16: 1.57, 17: 0.0, 18: 1.47, 19: 0.0, 20: 0.0, 21: 0.0,   # left
    22: 0.0, 23: -1.57, 24: 0.0, 25: 1.47, 26: 0.0, 27: 0.0, 28: 0.0,  # right
}

def run_segment(arm, start, goal, duration, weight_fn):
    """Interpolate joint dict start->goal over duration, publishing at 50 Hz."""
    steps = int(duration / DT)
    for i in range(steps):
        a = (i + 1) / steps
        a = 0.5 - 0.5 * math.cos(math.pi * a)          # smooth ease
        q = {j: start[j] + a * (goal[j] - start[j]) for j in CTRL_JOINTS}
        arm.send(q, weight_fn(a))
        time.sleep(DT)


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <network_interface_or_ip>")
        sys.exit(1)
    ChannelFactoryInitialize(0, sys.argv[1])

    # --- Bring-up to FSM 200 (main operation control). Feet must be on the floor. ---
    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print("Damp")
    loco.Damp()
    time.sleep(1.0)
    print("FSM 4 (locked stand)")
    loco.SetFsmId(4)
    time.sleep(7.0)
    print("FSM 200 (main operation)")
    loco.SetFsmId(200)
    time.sleep(3.0)
    print("fsm:", loco.GetFsmId(), " (robot should be balancing on its own now)")
    for s in (3, 2, 1):
        print(f"Taking over arms in {s}...")
        time.sleep(1.0)

    arm = ArmSdk()

    try:
        run_arm_sequence(arm)
    finally:
        # Whatever happened (normal end, Ctrl-C, exception): release the arms,
        # then damp, so no stale takeover is left on the robot.
        print("Releasing arms")
        arm.release(1.0)
        print("Damp")
        loco.Damp()
        time.sleep(1.0)
        print("fsm:", loco.GetFsmId(), " Done.")


def run_arm_sequence(arm):
    # 0. Clear any stale arm_sdk takeover from a previous run, then read a fresh pose.
    print("Releasing any stale arm_sdk state")
    arm.release(1.0)
    q0 = arm.fresh_state()
    hold = {j: float(q0[j]) for j in CTRL_JOINTS}    # current pose incl. waist
    print("Got fresh LowState. Taking over arms.")

    # 1. Ramp weight 0->1 while holding current pose (2 s). Nothing should move.
    run_segment(arm, hold, hold, 2.0, weight_fn=lambda a: a)

    # 2. Move from wherever the arms are to the defined NEUTRAL pose (3 s).
    print("Moving to neutral")
    run_segment(arm, hold, NEUTRAL, 3.0, weight_fn=lambda a: 1.0)

    # 3. Raise arms to the side (3 s)
    up = dict(NEUTRAL); up.update(ARMS_UP)
    run_segment(arm, NEUTRAL, up, 3.0, weight_fn=lambda a: 1.0)

    # # 4. Wave both hands: oscillate both elbows +/- 0.4 rad for 4 s
    # print("Waving")
    # t0 = time.time()
    # while time.time() - t0 < 4.0:
    #     q = dict(up)
    #     phase = 0.4 * math.sin(2 * math.pi * 1.0 * (time.time() - t0))
    #     q[18] = ARMS_UP[18] + phase    # left elbow
    #     q[25] = ARMS_UP[25] + phase    # right elbow
    #     arm.send(q, 1.0)
    #     time.sleep(DT)

    # 5. Lower arms back to NEUTRAL (3 s)
    print("Returning to neutral")
    run_segment(arm, up, NEUTRAL, 3.0, weight_fn=lambda a: 1.0)

    # 6. Ramp weight 1->0 (2 s) at NEUTRAL so ai_sport takes the arms back smoothly,
    #    then publish an explicit release so nothing stale is left behind.
    run_segment(arm, NEUTRAL, NEUTRAL, 2.0, weight_fn=lambda a: 1.0 - a)
    print("Released arms to controller.")


if __name__ == "__main__":
    main()