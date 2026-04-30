#!/usr/bin/env python3
"""Test YAM arm movements and gripper.

Usage:
    uv run python test_arm.py --channel can0
    uv run python test_arm.py --channel can0 --read-only
"""

import argparse
import signal
import sys
import time

import numpy as np

from i2rt.robots.get_robot import get_yam_robot


def smooth_move(robot, target, duration_s=2.0):
    """Move to target using the built-in interpolation."""
    print(f"  Moving to target over {duration_s:.1f}s...")
    robot.move_joints(np.array(target, dtype=np.float64), time_interval_s=duration_s)


def print_pos(robot, label="Current"):
    pos = robot.get_joint_pos()
    names = ["Shoulder Pan", "Shoulder Pitch", "Elbow",
             "Wrist 1", "Wrist 2", "Wrist 3", "Gripper"]
    print(f"\n  {label} joint positions:")
    for i, (name, val) in enumerate(zip(names, pos)):
        print(f"    [{i}] {name:15s}: {val:+.4f} rad")
    return pos


def wait(prompt):
    input(f"\n>>> {prompt} Press Enter to proceed (Ctrl+C to abort)... ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--read-only", action="store_true", help="Just read positions, no movement")
    args = parser.parse_args()

    print(f"Connecting to YAM arm on {args.channel}...")
    robot = get_yam_robot(channel=args.channel)

    def cleanup(*_):
        print("\nStopping...")
        robot.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup)

    if args.read_only:
        print("Read-only mode (Ctrl+C to exit)")
        while True:
            print_pos(robot)
            time.sleep(0.5)
        return

    # --- Test sequence ---
    home = robot.get_joint_pos().copy()
    print_pos(robot, "Starting")

    # 1. Shoulder pan test
    wait("Test shoulder pan (joint 0, ±0.3 rad).")
    target = home.copy()
    target[0] += 0.3
    smooth_move(robot, target, 2.0)
    time.sleep(0.5)
    target[0] -= 0.6
    smooth_move(robot, target, 2.0)
    time.sleep(0.5)
    smooth_move(robot, home, 2.0)

    # 2. Shoulder pitch test
    wait("Test shoulder pitch (joint 1, +0.3 rad).")
    target = home.copy()
    target[1] += 0.3
    smooth_move(robot, target, 2.0)
    time.sleep(0.5)
    smooth_move(robot, home, 2.0)

    # 3. Elbow test
    wait("Test elbow (joint 2, +0.3 rad).")
    target = home.copy()
    target[2] += 0.3
    smooth_move(robot, target, 2.0)
    time.sleep(0.5)
    smooth_move(robot, home, 2.0)

    # 4. Wrist test
    wait("Test wrist joints (joints 3-5, ±0.3 rad each).")
    for j in [3, 4, 5]:
        target = home.copy()
        target[j] += 0.3
        print(f"  Joint {j} +0.3")
        smooth_move(robot, target, 1.5)
        time.sleep(0.3)
        target[j] -= 0.6
        print(f"  Joint {j} -0.3")
        smooth_move(robot, target, 1.5)
        time.sleep(0.3)
        smooth_move(robot, home, 1.5)

    # 5. Gripper test
    wait("Test gripper (close then open).")
    target = home.copy()
    target[6] = 0.0
    print("  Closing gripper...")
    smooth_move(robot, target, 1.5)
    time.sleep(1.0)
    target[6] = 0.8
    print("  Opening gripper...")
    smooth_move(robot, target, 1.5)
    time.sleep(0.5)

    # Return home
    print("\nReturning to start position...")
    smooth_move(robot, home, 2.0)

    print_pos(robot, "Final")
    print("\nDone! Closing robot...")
    robot.close()


if __name__ == "__main__":
    main()
