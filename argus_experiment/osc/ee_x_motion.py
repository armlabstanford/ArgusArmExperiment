"""
Move the YAM end effector +0.1 m in base-frame X then back to start.

Usage:
    # Simulation
    python argus_experiment/osc/ee_x_motion.py --sim

    # Real hardware
    python argus_experiment/osc/ee_x_motion.py --channel can0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "robots_realtime" / "dependencies" / "i2rt"))

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType


#TODO
START_POSE = np.array([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]])


def run(
    robot,
    xml_path: str,
    ee_site: str,
    distance: float,
    n_steps: int,
    dt: float,
) -> None:
    kin = Kinematics(xml_path, ee_site)

    # Warm up: let the robot settle and read a stable joint state
    time.sleep(0.5)
    q0 = robot.get_joint_pos()
    n_arm = 6  # YAM always has 6 arm joints

    # Current EE pose as the trajectory origin
    start_pose = kin.fk(q0[:n_arm])  # (4, 4)
    print(f"Start EE position (x, y, z): {start_pose[:3, 3].round(4)}")

    # Build waypoints: go forward n_steps, come back n_steps
    alphas = np.concatenate([np.linspace(0.0, 1.0, n_steps), np.linspace(1.0, 0.0, n_steps)])
    waypoints = []
    for alpha in alphas:
        wp = start_pose.copy()
        wp[0, 3] += alpha * distance
        waypoints.append(wp)

    print(f"Executing {len(waypoints)} waypoints over {len(waypoints) * dt:.2f} s ...")

    init_q = q0[:n_arm].copy()
    for i, target_pose in enumerate(waypoints):
        ok, ik_q = kin.ik(target_pose, ee_site, init_q=init_q)
        if not ok:
            print(f"  IK failed at waypoint {i}, holding last solution")

        cmd = robot.get_joint_pos().copy()
        cmd[:n_arm] = ik_q[:n_arm]
        robot.command_joint_pos(cmd)

        init_q = ik_q[:n_arm]  # warm-start next IK from this solution
        time.sleep(dt)

    print("Motion complete.")


def main() -> None:
    arm_choices = [a.value for a in ArmType]
    gripper_choices = [g.value for g in GripperType]

    parser = argparse.ArgumentParser(description="Move YAM EE ±0.1 m in base-frame X")
    parser.add_argument("--arm", default="yam", choices=arm_choices)
    parser.add_argument("--gripper", default="no_gripper", choices=gripper_choices)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--dt", type=float, default=0.02, help="Timestep between waypoints (s)")
    parser.add_argument("--distance", type=float, default=0.1, help="X displacement in metres")
    parser.add_argument("--steps", type=int, default=50, help="Waypoints per leg of the motion")
    parser.add_argument("--site", type=str, default=None, help="EE site name (auto-detected if omitted)")
    args = parser.parse_args()

    arm = ArmType.from_string_name(args.arm)
    gripper = GripperType.from_string_name(args.gripper)

    if args.site is not None:
        site = args.site
    elif gripper == GripperType.YAM_TEACHING_HANDLE:
        site = "tcp_site"
    else:
        site = "grasp_site"

    robot = get_yam_robot(
        channel=args.channel,
        arm_type=arm,
        gripper_type=gripper,
        sim=args.sim,
    )

    run(robot, robot.xml_path, site, args.distance, args.steps, args.dt)


if __name__ == "__main__":
    main()
