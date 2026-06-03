"""
Execute a straight-line end-effector trajectory in base frame.

Edit the TRAJECTORY CONFIG section below to set direction, distance, velocity, etc.
Then run:
    python argus_experiment/osc/ee_line_traj.py --sim
    python argus_experiment/osc/ee_line_traj.py --channel can0
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

# ---------------------------------------------------------------------------
# TRAJECTORY CONFIG — edit these
# ---------------------------------------------------------------------------

# Direction vector in base frame (will be normalised). Examples:
#   +X forward : [1, 0, 0]
#   -Z down    : [0, 0, -1]
#   diagonal   : [1, 0, -1]
DIRECTION = np.array([1.0, 0.0, 0.0])

DISTANCE = 0.1          # total displacement along DIRECTION (metres)
VELOCITY = 0.05         # cartesian speed (m/s)
DT = 0.02               # control timestep (s)

RETURN_TO_START = True  # move back to start pose after reaching target
HOLD_TIME = 0.0         # seconds to hold at target before returning

# Set to a float (kg) to override link_6 inertial for gravity comp accuracy.
# Leave as None if no payload.
EE_MASS = None

# ---------------------------------------------------------------------------
# Robot config — override via CLI flags if needed
# ---------------------------------------------------------------------------
DEFAULT_ARM = "yam"
DEFAULT_GRIPPER = "no_gripper"

N_ARM_JOINTS = 6


def build_waypoints(
    start_pose: np.ndarray,
) -> list[np.ndarray]:
    unit = DIRECTION / np.linalg.norm(DIRECTION)
    n_steps = max(2, int(round(DISTANCE / (VELOCITY * DT))))

    forward = []
    for i in range(n_steps):
        wp = start_pose.copy()
        wp[:3, 3] += unit * DISTANCE * (i / (n_steps - 1))
        forward.append(wp)

    n_hold = max(0, int(round(HOLD_TIME / DT)))
    hold = [forward[-1].copy()] * n_hold

    reverse = [wp.copy() for wp in reversed(forward)] if RETURN_TO_START else []

    return forward + hold + reverse


def execute_waypoints(robot, kin: Kinematics, ee_site: str, waypoints: list[np.ndarray]) -> None:
    init_q = robot.get_joint_pos()[:N_ARM_JOINTS].copy()

    for i, target_pose in enumerate(waypoints):
        ok, ik_q = kin.ik(target_pose, ee_site, init_q=init_q)
        if not ok:
            print(f"  Warning: IK did not converge at waypoint {i}/{len(waypoints)}, holding last solution")

        cmd = robot.get_joint_pos().copy()
        cmd[:N_ARM_JOINTS] = ik_q[:N_ARM_JOINTS]
        robot.command_joint_pos(cmd)

        init_q = ik_q[:N_ARM_JOINTS]
        time.sleep(DT)


def run(robot, xml_path: str, ee_site: str) -> None:
    kin = Kinematics(xml_path, ee_site)
    time.sleep(0.5)

    start_pose = kin.fk(robot.get_joint_pos()[:N_ARM_JOINTS])
    unit = DIRECTION / np.linalg.norm(DIRECTION)
    end_pos = start_pose[:3, 3] + unit * DISTANCE

    print(f"EE start : {start_pose[:3, 3].round(4)}")
    print(f"EE target: {end_pos.round(4)}")
    print(f"Direction: {unit.round(4)}, distance={DISTANCE:.3f} m, velocity={VELOCITY:.3f} m/s")

    waypoints = build_waypoints(start_pose)
    print(f"Executing {len(waypoints)} waypoints over {len(waypoints) * DT:.2f} s ...")

    execute_waypoints(robot, kin, ee_site, waypoints)
    print("Done.")


def main() -> None:
    arm_choices = [a.value for a in ArmType]
    gripper_choices = [g.value for g in GripperType]

    parser = argparse.ArgumentParser(description="Straight-line EE trajectory in base frame")
    parser.add_argument("--arm", default=DEFAULT_ARM, choices=arm_choices)
    parser.add_argument("--gripper", default=DEFAULT_GRIPPER, choices=gripper_choices)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--site", default=None, help="EE site name (auto-detected if omitted)")
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
        ee_mass=EE_MASS,
    )

    run(robot, robot.xml_path, site)


if __name__ == "__main__":
    main()
