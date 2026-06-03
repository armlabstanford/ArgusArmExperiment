"""
Rotate the end effector around a base-frame axis while holding position fixed.

Edit the TRAJECTORY CONFIG section below, then run:
    python argus_experiment/osc/ee_rot_traj.py --sim
    python argus_experiment/osc/ee_rot_traj.py --channel can0
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

# Rotation axis in base frame.
# Use a string shorthand or a custom 3-vector (will be normalised).
#   "roll"  → rotate around base X
#   "pitch" → rotate around base Y
#   "yaw"   → rotate around base Z
AXIS = "yaw"

ANGLE = 45.0              # total rotation angle (degrees, can be negative)
ANGULAR_VELOCITY = 20.0   # rotation speed (degrees/s)
DT = 0.02                 # control timestep (s)

RETURN_TO_START = True    # rotate back to start orientation after reaching target
HOLD_TIME = 0.0           # seconds to hold at target before returning

# Set to a float (kg) to override link_6 inertial for gravity comp accuracy.
EE_MASS = None

# ---------------------------------------------------------------------------
# Robot config — override via CLI flags if needed
# ---------------------------------------------------------------------------
DEFAULT_ARM = "yam"
DEFAULT_GRIPPER = "no_gripper"

N_ARM_JOINTS = 6

_AXIS_MAP = {
    "roll":  np.array([1.0, 0.0, 0.0]),
    "pitch": np.array([0.0, 1.0, 0.0]),
    "yaw":   np.array([0.0, 0.0, 1.0]),
}


def _rot_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix for angle theta (rad) around unit axis."""
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c),      x*y*(1-c) - z*s,   x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,    c + y*y*(1-c),      y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,    z*y*(1-c) + x*s,    c + z*z*(1-c)  ],
    ])


def _resolve_axis() -> np.ndarray:
    if isinstance(AXIS, str):
        key = AXIS.lower()
        if key not in _AXIS_MAP:
            raise ValueError(f"AXIS string must be one of {list(_AXIS_MAP)}, got {AXIS!r}")
        return _AXIS_MAP[key]
    vec = np.asarray(AXIS, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        raise ValueError("AXIS vector must be non-zero")
    return vec / norm


def build_waypoints(start_pose: np.ndarray) -> list[np.ndarray]:
    axis_vec = _resolve_axis()
    n_steps = max(2, int(round(abs(ANGLE) / (ANGULAR_VELOCITY * DT))))
    thetas = np.linspace(0.0, np.radians(ANGLE), n_steps)

    R_start = start_pose[:3, :3]
    pos = start_pose[:3, 3]

    forward = []
    for theta in thetas:
        wp = np.eye(4)
        wp[:3, 3] = pos
        wp[:3, :3] = _rot_matrix(axis_vec, theta) @ R_start
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
    axis_vec = _resolve_axis()

    print(f"EE position (held fixed): {start_pose[:3, 3].round(4)}")
    print(f"Rotation axis: {axis_vec.round(4)} ({AXIS if isinstance(AXIS, str) else 'custom'})")
    print(f"Angle={ANGLE:.1f} deg, velocity={ANGULAR_VELOCITY:.1f} deg/s")

    waypoints = build_waypoints(start_pose)
    print(f"Executing {len(waypoints)} waypoints over {len(waypoints) * DT:.2f} s ...")

    execute_waypoints(robot, kin, ee_site, waypoints)
    print("Done.")


def main() -> None:
    arm_choices = [a.value for a in ArmType]
    gripper_choices = [g.value for g in GripperType]

    parser = argparse.ArgumentParser(description="Rotational EE trajectory in base frame")
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
