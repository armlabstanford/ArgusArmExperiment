"""
Rotate the YAM end effector in place about a body-frame axis (roll/pitch/yaw),
with move-to-start, MuJoCo sim viewer, return-to-start, and return-to-home.

Orientation analog of ee_line_traj.py: instead of translating the EE along a
base-frame direction, this holds EE position fixed and sweeps its orientation
about one of its own axes. "Body frame" means the tool spins in place about
its own X (roll), Y (pitch), or Z (yaw).

Lifecycle: record the robot's actual initial pose ("home") -> move to the
canonical TRAJ_START_QPOS -> run the rotation trajectory -> return to the
recorded home pose.

Usage:
    # Simulation (opens a MuJoCo viewer window)
    python argus_experiment/osc/ee_rot_traj.py --sim

    # roll +30 deg, slow
    python argus_experiment/osc/ee_rot_traj.py --sim --axis roll --angle 0.5236 --ang-velocity 0.2

    # yaw -45 deg on hardware
    python argus_experiment/osc/ee_rot_traj.py --channel can0 --axis yaw --angle -0.7854

    # pitch sweep, no reverse, no final home return
    python argus_experiment/osc/ee_rot  _traj.py --sim --axis pitch --angle 0.5 --no-return --no-home
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
ARGUS_MASS = 0.178

# Canonical trajectory-start configuration (6D arm joints), in radians.
# Well-conditioned, non-singular launch pose; distinct from "home" (the
# robot's actual pose at script start).
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

# Body-frame rotation axes: roll about EE X, pitch about EE Y, yaw about EE Z.
_AXES = {
    "roll":  np.array([1.0, 0.0, 0.0]),
    "pitch": np.array([0.0, 1.0, 0.0]),
    "yaw":   np.array([0.0, 0.0, 1.0]),
}


def _sync(viewer) -> None:
    """Refresh the passive viewer if one is attached and still open."""
    if viewer is not None and viewer.is_running():
        viewer.sync()


def _axis_angle_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation: 3x3 rotation matrix for `angle` (rad) about unit `axis`."""
    a = axis / np.linalg.norm(axis)
    x, y, z = a
    c, s, C = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])


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
    final return-to-home.
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


def build_orientation_waypoints(
    start_pose: np.ndarray,
    axis: np.ndarray,
    total_angle: float,
    ang_velocity: float,
    dt: float,
    hold_time: float,
    return_to_start: bool,
) -> list[np.ndarray]:
    """Rotate the EE in place about a BODY-frame axis, holding position fixed.

    Orientation at waypoint i is R_start @ R(axis, alpha * total_angle), i.e.
    the incremental rotation is applied in the EE's own frame (post-multiply).
    Position is held at the start pose's translation.
    """
    n_steps = max(2, int(round(abs(total_angle) / (ang_velocity * dt))))
    R_start = start_pose[:3, :3]
    p_start = start_pose[:3, 3]

    forward = []
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        wp = np.eye(4)
        wp[:3, :3] = R_start @ _axis_angle_to_R(axis, alpha * total_angle)  # body-frame
        wp[:3, 3] = p_start                                                  # position fixed
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
            print(f"  IK failed at waypoint {i}; holding last good solution")
            continue
        robot.command_joint_pos(ik_q[:N_ARM])
        _sync(viewer)
        init_q = ik_q[:N_ARM]
        time.sleep(dt)


def _R_to_rpy(R: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (XYZ) from a rotation matrix, for logging only."""
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def run(
    robot,
    xml_path: str,
    ee_site: str,
    axis_name: str,
    total_angle: float,
    ang_velocity: float,
    dt: float,
    hold_time: float,
    return_to_start: bool,
    viewer=None,
) -> None:
    kin = Kinematics(xml_path, ee_site)
    time.sleep(0.5)

    # Record the robot's ACTUAL initial pose ("home") to return to at the end.
    home_q = robot.get_joint_pos()[:N_ARM].copy()
    print(f"Recorded home pose: {home_q.round(4)}")

    # Move to the canonical trajectory-start config; use achieved pose as origin.
    q0 = move_to_qpos(robot, TRAJ_START_QPOS, dt, viewer=viewer, label="trajectory start")
    start_pose = kin.fk(q0[:N_ARM])

    axis = _AXES[axis_name]
    print(f"EE position (held fixed): {start_pose[:3, 3].round(4)}")
    print(f"Start RPY: {_R_to_rpy(start_pose[:3, :3]).round(4)}")
    print(f"Axis: {axis_name} (body frame), sweep {total_angle:+.4f} rad "
          f"at {ang_velocity:.3f} rad/s")

    waypoints = build_orientation_waypoints(
        start_pose, axis, total_angle, ang_velocity, dt, hold_time, return_to_start
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

    parser = argparse.ArgumentParser(description="Rotate YAM EE in place about a body-frame axis")
    parser.add_argument("--arm", default="yam", choices=arm_choices)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--no-view", action="store_true", help="Run sim headless (no viewer window)")
    parser.add_argument("--site", default="grasp_site", help="EE site name")

    parser.add_argument("--axis", choices=list(_AXES.keys()), default="roll",
                        help="Body-frame rotation axis (roll=X, pitch=Y, yaw=Z)")
    parser.add_argument("--angle", type=float, default=0.5236,
                        help="Total angle to sweep (radians; default ~30 deg)")
    parser.add_argument("--ang-velocity", type=float, default=0.2,
                        help="Angular speed (rad/s)")
    parser.add_argument("--dt", type=float, default=0.02, help="Control timestep (s)")
    parser.add_argument("--hold", type=float, default=0.0, help="Hold time at target (s)")
    parser.add_argument("--no-return", action="store_true", help="Do not reverse the rotation")
    args = parser.parse_args()

    if args.ang_velocity <= 0:
        parser.error("--ang-velocity must be positive")

    arm = ArmType.from_string_name(args.arm)

    robot = get_yam_robot(
        channel=args.channel,
        arm_type=arm,
        gripper_type=GripperType.from_string_name("no_gripper"),
        sim=args.sim,
        ee_mass=ARGUS_MASS,
    )

    viewer = make_sim_viewer(robot) if (args.sim and not args.no_view) else None

    try:
        run(
            robot, robot.xml_path, args.site,
            axis_name=args.axis,
            total_angle=args.angle,
            ang_velocity=args.ang_velocity,
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