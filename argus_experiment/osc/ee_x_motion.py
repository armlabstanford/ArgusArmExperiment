"""
Move the YAM end effector +0.1 m in base-frame X then back to start.

Usage:
    # Simulation (opens a MuJoCo viewer window)
    python argus_experiment/osc/ee_x_motion.py --sim

    # Simulation, headless (no window)
    python argus_experiment/osc/ee_x_motion.py --sim --no-view

    # Real hardware
    python argus_experiment/osc/ee_x_motion.py --channel can0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "robots_realtime" / "dependencies" / "i2rt"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "argus_experiment" / "calibration"))

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType
from camera_frame import add_camera_cli_args, resolve_motion_frame


N_ARM = 6  # YAM always has 6 arm joints

# Desired start configuration (6D arm joints), in radians.
# START_QPOS = np.array([
#     -1.5471,  # [0] Shoulder Pan
#     +0.7246,  # [1] Shoulder Pitch
#     +0.6334,  # [2] Elbow
#     +0.2104,  # [3] Wrist 1
#     +0.0162,  # [4] Wrist 2
#     +0.0235,  # [5] Wrist 3
# ])

START_QPOS = np.array([
    +0.0883,  # [0] Shoulder Pan
    +0.5694,  # [1] Shoulder Pitch
    +0.5758,  # [2] Elbow
    +0.0090,  # [3] Wrist 1
    +0.1844,  # [4] Wrist 2
    -0.1036,  # [5] Wrist 3
])

START_TOL = 0.05       # rad; per-joint tolerance for "already at start"
START_MOVE_TIME = 3.0  # s; duration of the smooth move to the start pose


def _sync(viewer) -> None:
    """Refresh the passive viewer if one is attached and still open."""
    if viewer is not None and viewer.is_running():
        viewer.sync()


def run(
    robot,
    xml_path: str,
    ee_site: str,
    distance: float,
    n_steps: int,
    dt: float,
    viewer=None,
) -> None:
    kin = Kinematics(xml_path, ee_site)

    # Warm up: let the robot settle and read a stable joint state
    time.sleep(0.5)
    q0 = robot.get_joint_pos()

    # --- Move to the desired start pose if we're not already there ---
    arm_err = np.max(np.abs(q0[:N_ARM] - START_QPOS))
    if arm_err > START_TOL:
        print(f"Arm is {arm_err:.4f} rad from start (tol {START_TOL}); moving to start pose ...")
        steps = 50
        for i in range(steps + 1):
            alpha = i / steps
            cmd = q0.copy()
            cmd[:N_ARM] = (1 - alpha) * q0[:N_ARM] + alpha * START_QPOS
            robot.command_joint_pos(cmd)
            _sync(viewer)
            time.sleep(START_MOVE_TIME / steps)
        time.sleep(0.5)               # settle
        q0 = robot.get_joint_pos()    # re-read so the FK origin is the true start
    else:
        print(f"Arm already within {START_TOL} rad of start pose (err {arm_err:.4f}).")

    # Current EE pose as the trajectory origin
    start_pose = kin.fk(q0[:N_ARM])  # (4, 4)
    print(f"Start EE position (x, y, z): {start_pose[:3, 3].round(4)}")
    print(f"Target EE position (x, y, z): {(start_pose[:3, 3] + np.array([distance, 0, 0])).round(4)}")

    # Build waypoints: go forward n_steps, come back n_steps
    alphas = np.concatenate([np.linspace(0.0, 1.0, n_steps), np.linspace(1.0, 0.0, n_steps)])
    waypoints = []
    for alpha in alphas:
        wp = start_pose.copy()
        wp[0, 3] += alpha * distance
        waypoints.append(wp)

    print(f"Executing {len(waypoints)} waypoints over {len(waypoints) * dt:.2f} s ...")

    init_q = q0[:N_ARM].copy()
    for i, target_pose in enumerate(waypoints):
        ok, ik_q = kin.ik(target_pose, ee_site, init_q=init_q)
        if not ok:
            print(f"  IK failed at waypoint {i} (target x,y,z={target_pose[:3, 3].round(4)}); holding last good solution")
            continue  # init_q still holds the last good solution

        robot.command_joint_pos(ik_q[:N_ARM])
        _sync(viewer)
        init_q = ik_q[:N_ARM]  # warm-start next IK from this solution
        time.sleep(dt)

    print("Motion complete.")


def make_sim_viewer(robot):
    """Attach a passive MuJoCo viewer over SimRobot's internal model/data.

    SimRobot is headless: it holds an MjModel/MjData (as `_model`/`_data`) and
    runs mj_forward on command_joint_pos, but never opens a window. We attach
    our own passive viewer over those same objects and sync each step.
    Returns None on any failure so the run continues headless.
    """
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

    parser = argparse.ArgumentParser(description="Move YAM EE ±0.1 m in base-frame X")
    parser.add_argument("--arm", default="yam", choices=arm_choices)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--dt", type=float, default=0.02, help="Timestep between waypoints (s)")
    parser.add_argument("--distance", type=float, default=0.1, help="X displacement in metres")
    parser.add_argument("--steps", type=int, default=50, help="Waypoints per leg of the motion")
    parser.add_argument("--site", type=str, default="grasp_site", help="EE site name")
    parser.add_argument("--no-view", action="store_true", help="Run sim headless (no viewer window)")
    add_camera_cli_args(parser)
    args = parser.parse_args()

    arm = ArmType.from_string_name(args.arm)

    robot = get_yam_robot(
        channel=args.channel,
        arm_type=arm,
        gripper_type=GripperType.from_string_name("no_gripper"),
        sim=args.sim,
    )

    xml_path, site = resolve_motion_frame(robot, args)

    viewer = make_sim_viewer(robot) if (args.sim and not args.no_view) else None

    try:
        run(robot, xml_path, site, args.distance, args.steps, args.dt, viewer=viewer)

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