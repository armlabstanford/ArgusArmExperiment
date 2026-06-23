"""
Execute a straight-line end-effector trajectory in base frame, with optional
move-to-start, MuJoCo sim viewer, return-to-start, and return-to-home.

Generalizes ee_x_motion.py: arbitrary direction, configurable velocity and
distance. Direction is given in base frame and normalized.

Lifecycle: record the robot's actual initial pose ("home") -> move to the
canonical TRAJ_START_QPOS -> run the line trajectory -> return to the
recorded home pose, so the arm finishes exactly where it physically began.

Usage:
    # Simulation (opens a MuJoCo viewer window)
    python argus_experiment/osc/ee_line_traj.py --sim

    # +X, 15 cm, slow
    python argus_experiment/osc/ee_line_traj.py --sim --direction 1 0 0 --distance 0.15 --velocity 0.03

    # straight down 10 cm on hardware
    python argus_experiment/osc/ee_line_traj.py --channel can0 --direction 0 0 -1

    # diagonal, no trajectory return, no final home return
    python argus_experiment/osc/ee_line_traj.py --sim --direction 1 0 -1 --no-return --no-home
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


N_ARM = 6  # YAM always has 6 arm joints

# Canonical trajectory-start configuration (6D arm joints), in radians.
# This is the pose the line trajectory is launched from (a well-conditioned,
# non-singular config), distinct from "home" which is the robot's actual
# pose at script start.
TRAJ_START_QPOS = np.array([
    +0.0883,  # [0] Shoulder Pan
    +0.5694,  # [1] Shoulder Pitch
    +0.5758,  # [2] Elbow
    +0.0090,  # [3] Wrist 1
    +0.1844,  # [4] Wrist 2
    -0.1036,  # [5] Wrist 3
])

START_TOL = 0.05        # rad; per-joint tolerance for "already at target pose"
START_MOVE_TIME = 3.0   # s; duration of a smooth joint-space move


def _sync(viewer) -> None:
    """Refresh the passive viewer if one is attached and still open."""
    if viewer is not None and viewer.is_running():
        viewer.sync()


def move_to_qpos(
    robot,
    target_q: np.ndarray,
    dt: float,
    viewer=None,
    move_time: float = START_MOVE_TIME,
    tol: float = START_TOL,
    label: str = "target",
) -> np.ndarray:
    """Smoothly interpolate the arm joints from the current pose to target_q.

    No-op (beyond a state re-read) if already within `tol`. Returns the
    achieved joint state. Used for both the trajectory-start move and the
    final return-to-home, so both share one tested interpolation.
    """
    q0 = robot.get_joint_pos()
    arm_err = np.max(np.abs(q0[:N_ARM] - target_q[:N_ARM]))
    if arm_err <= tol:
        print(f"Arm already within {tol} rad of {label} (err {arm_err:.4f}).")
        return q0

    print(f"Arm is {arm_err:.4f} rad from {label} (tol {tol}); moving ...")
    steps = max(2, int(round(move_time / dt)))
    for i in range(steps + 1):
        alpha = i / steps
        cmd = q0.copy()
        cmd[:N_ARM] = (1 - alpha) * q0[:N_ARM] + alpha * target_q[:N_ARM]
        robot.command_joint_pos(cmd)
        _sync(viewer)
        time.sleep(move_time / steps)
    time.sleep(0.5)                 # settle
    return robot.get_joint_pos()    # re-read so callers see the true achieved pose


def build_waypoints(
    start_pose: np.ndarray,
    direction: np.ndarray,
    distance: float,
    velocity: float,
    dt: float,
    hold_time: float,
    return_to_start: bool,
) -> list[np.ndarray]:
    """Straight-line Cartesian waypoints along `direction` for `distance`,
    at `velocity`. Orientation is held fixed at start_pose's rotation."""
    unit = direction / np.linalg.norm(direction)
    n_steps = max(2, int(round(distance / (velocity * dt))))

    forward = []
    for i in range(n_steps):
        wp = start_pose.copy()
        wp[:3, 3] += unit * distance * (i / (n_steps - 1))
        forward.append(wp)

    n_hold = max(0, int(round(hold_time / dt)))
    hold = [forward[-1].copy() for _ in range(n_hold)]

    reverse = [wp.copy() for wp in reversed(forward)] if return_to_start else []

    return forward + hold + reverse


def execute_waypoints(robot, kin, ee_site, waypoints, init_q, dt, viewer=None) -> None:
    """Warm-started IK tracking. On IK failure, hold the last good solution
    (command nothing new) rather than pushing a bad solution."""
    for i, target_pose in enumerate(waypoints):
        ok, ik_q = kin.ik(target_pose, ee_site, init_q=init_q)
        if not ok:
            print(f"  IK failed at waypoint {i} (target x,y,z={target_pose[:3, 3].round(4)}); holding last good solution")
            continue
        robot.command_joint_pos(ik_q[:N_ARM])
        _sync(viewer)
        init_q = ik_q[:N_ARM]
        time.sleep(dt)


def run(
    robot,
    xml_path: str,
    ee_site: str,
    direction: np.ndarray,
    distance: float,
    velocity: float,
    dt: float,
    hold_time: float,
    return_to_start: bool,
    viewer=None,
) -> None:
    kin = Kinematics(xml_path, ee_site)
    time.sleep(0.5)

    # Record the robot's ACTUAL initial pose. This is "home" — we return here
    # at the end so the arm finishes exactly where it physically began.
    home_q = robot.get_joint_pos()[:N_ARM].copy()
    print(f"Recorded home pose: {home_q.round(4)}")

    # Move to the canonical trajectory-start config; use the achieved pose as origin.
    q0 = move_to_qpos(robot, TRAJ_START_QPOS, dt, viewer=viewer, label="trajectory start")
    start_pose = kin.fk(q0[:N_ARM])

    unit = direction / np.linalg.norm(direction)
    end_pos = start_pose[:3, 3] + unit * distance
    print(f"EE start : {start_pose[:3, 3].round(4)}")
    print(f"EE target: {end_pos.round(4)}")
    print(f"Direction: {unit.round(4)}, distance={distance:.3f} m, velocity={velocity:.3f} m/s")

    waypoints = build_waypoints(
        start_pose, direction, distance, velocity, dt, hold_time, return_to_start
    )
    print(f"Executing {len(waypoints)} waypoints over {len(waypoints) * dt:.2f} s ...")

    execute_waypoints(robot, kin, ee_site, waypoints, q0[:N_ARM].copy(), dt, viewer=viewer)
    print("Motion complete.")

    # Final step: return to the recorded home pose (joint-space interpolation).
    print(f"Returning to recorded home pose: {home_q.round(4)} ...")
    move_to_qpos(robot, home_q, dt, viewer=viewer, label="home")
    print("At home.")


def make_sim_viewer(robot):
    """Attach a passive MuJoCo viewer over SimRobot's internal model/data.
    Returns None on any failure so the run continues headless."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as e:
        print(f"mujoco not importable ({e}); running headless.")
        return None

    model = getattr(robot, "_model", None)
    data = getattr(robot, "_data", None)
    if model is None or data is None:
        print("SimRobot exposes no _model/_data; running headless.")
        return None

    try:
        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        return viewer
    except Exception as e:
        print(f"Could not launch viewer ({e}); running headless.")
        return None


def main() -> None:
    arm_choices = [a.value for a in ArmType]

    parser = argparse.ArgumentParser(description="Straight-line EE trajectory in base frame")
    parser.add_argument("--arm", default="yam", choices=arm_choices)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--no-view", action="store_true", help="Run sim headless (no viewer window)")
    parser.add_argument("--site", default="grasp_site", help="EE site name")

    parser.add_argument("--direction", type=float, nargs=3, default=[0.0, 1.0, 0.0],
                        metavar=("X", "Y", "Z"), help="Base-frame direction (normalized)")
    parser.add_argument("--distance", type=float, default=0.1, help="Displacement along direction (m)")
    parser.add_argument("--velocity", type=float, default=0.05, help="Cartesian speed (m/s)")
    parser.add_argument("--dt", type=float, default=0.02, help="Control timestep (s)")
    parser.add_argument("--hold", type=float, default=0.0, help="Hold time at target (s)")
    parser.add_argument("--no-return", action="store_true", help="Do not reverse the line trajectory")
    args = parser.parse_args()

    direction = np.array(args.direction, dtype=float)
    if np.linalg.norm(direction) < 1e-9:
        parser.error("--direction must be a nonzero vector")

    arm = ArmType.from_string_name(args.arm)

    robot = get_yam_robot(
        channel=args.channel,
        arm_type=arm,
        gripper_type=GripperType.from_string_name("no_gripper"),
        sim=args.sim,
    )

    viewer = make_sim_viewer(robot) if (args.sim and not args.no_view) else None

    try:
        run(
            robot, robot.xml_path, args.site,
            direction=direction,
            distance=args.distance,
            velocity=args.velocity,
            dt=args.dt,
            hold_time=args.hold,
            return_to_start=not args.no_return,
            viewer=viewer,
        )

        if viewer is not None:
            print("Sim done — close the viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    finally:
        if viewer is not None:
            viewer.close()
        robot.close()


if __name__ == "__main__":
    main()