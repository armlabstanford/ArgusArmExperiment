"""
Sinusoidal rotational end-effector motion about a body-frame axis.

Lifecycle:
  record home -> move to TRAJ_START_QPOS -> run N sinusoidal periods
  centered on that EE orientation -> return to recorded home.

The sinusoid is: angle(t) = AMPLITUDE * sin(2π * t / T)
  where T = 2π * AMPLITUDE / ang_velocity  (so peak angular speed = --ang-velocity).

Usage:
    # Simulation — roll axis (default)
    python argus_experiment/osc/ee_rot_sinusoidal.py --sim

    # Pitch axis, faster
    python argus_experiment/osc/ee_rot_sinusoidal.py --sim --axis pitch --ang-velocity 0.5

    # Yaw on hardware, 3 periods
    python argus_experiment/osc/ee_rot_sinusoidal.py --channel can0 --axis yaw --periods 3

    # Headless sim
    python argus_experiment/osc/ee_rot_sinusoidal.py --sim --no-view
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "robots_realtime" / "dependencies" / "i2rt"))

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType


N_ARM = 6
ARGUS_MASS = 0.178

TRAJ_START_QPOS = np.array([
    +0.0883,  # [0] Shoulder Pan
    +0.5694,  # [1] Shoulder Pitch
    +0.5758,  # [2] Elbow
    +0.0090,  # [3] Wrist 1
    +0.1844,  # [4] Wrist 2
    -0.1036,  # [5] Wrist 3
])

AMPLITUDE   = 0.7854    # rad — ±π/4 (45 deg) peak excursion
N_PERIODS   = 2
START_TOL   = 0.05      # rad
START_MOVE_TIME = 3.0   # s

_AXES = {
    "roll":  np.array([1.0, 0.0, 0.0]),
    "pitch": np.array([0.0, 1.0, 0.0]),
    "yaw":   np.array([0.0, 0.0, 1.0]),
}


def _sync(viewer) -> None:
    if viewer is not None and viewer.is_running():
        viewer.sync()


def _axis_angle_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation matrix for `angle` (rad) about unit `axis`."""
    a = axis / np.linalg.norm(axis)
    x, y, z = a
    c, s, C = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C  ],
    ])


def move_to_qpos(robot, target_q, dt, viewer=None, move_time=START_MOVE_TIME,
                 tol=START_TOL, label="target") -> np.ndarray:
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
    time.sleep(0.5)
    return robot.get_joint_pos()


def build_sinusoidal_waypoints(
    center_pose: np.ndarray,
    axis: np.ndarray,
    amplitude: float,
    ang_velocity: float,
    n_periods: int,
    dt: float,
) -> list[np.ndarray]:
    """
    Sinusoidal rotation: angle(t) = amplitude * sin(ωt)  applied in body frame.
    ω = ang_velocity / amplitude  →  peak angular speed = ang_velocity.
    Position is held fixed at center_pose[:3, 3].
    """
    omega = ang_velocity / amplitude      # rad/s — sets peak angular speed
    period = 2 * np.pi / omega            # s per cycle
    total_time = n_periods * period
    n_steps = max(2, int(round(total_time / dt)))
    t = np.linspace(0.0, total_time, n_steps, endpoint=False)

    R_center = center_pose[:3, :3]
    p_center = center_pose[:3, 3]

    waypoints = []
    for ti in t:
        theta = amplitude * np.cos(omega * ti)  # cos: starts at +amplitude, zero velocity
        wp = np.eye(4)
        wp[:3, :3] = R_center @ _axis_angle_to_R(axis, theta)  # body-frame rotation
        wp[:3, 3]  = p_center                                   # position fixed
        waypoints.append(wp)
    return waypoints


def execute_waypoints(robot, kin, ee_site, waypoints, init_q, dt, viewer=None,
                      records: list | None = None) -> np.ndarray:
    """Returns the last successful joint config (warm-start for next call)."""
    t = 0.0
    for i, target_pose in enumerate(waypoints):
        ok, ik_q = kin.ik(target_pose, ee_site, init_q=init_q)
        if not ok:
            print(f"  IK failed at waypoint {i}; holding last good solution")
            t += dt
            continue
        robot.command_joint_pos(ik_q[:N_ARM])
        _sync(viewer)
        if records is not None:
            pose = kin.fk(ik_q[:N_ARM])
            records.append((t, pose[:3, 3].copy(), _R_to_rpy(pose[:3, :3]).copy()))
        init_q = ik_q[:N_ARM]
        time.sleep(dt)
        t += dt
    return init_q


def _R_to_rpy(R: np.ndarray) -> np.ndarray:
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def plot_ee_recording(records: list, axis_name: str) -> None:
    if not records:
        return
    times         = np.array([r[0] for r in records])
    positions     = np.array([r[1] for r in records])  # (N, 3)
    rpys          = np.array([r[2] for r in records])  # (N, 3)
    ang_velocities = np.gradient(rpys, times, axis=0)  # (N, 3)  rad/s

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"Rotational sinusoidal EE 6-DOF  [axis: {axis_name}]", fontsize=12)

    ax = axes[0]
    for j, label in enumerate(["x", "y", "z"]):
        ax.plot(times, positions[:, j], label=label)
    ax.set_ylabel("Position (m)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("EE position (should be constant)")

    ax = axes[1]
    for j, label in enumerate(["roll", "pitch", "yaw"]):
        ax.plot(times, rpys[:, j], label=label)
    ax.set_ylabel("Angle (rad)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("EE orientation (RPY)")

    ax = axes[2]
    for j, label in enumerate(["ω_roll", "ω_pitch", "ω_yaw"]):
        ax.plot(times, ang_velocities[:, j], label=label)
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("EE angular velocity")

    plt.tight_layout()
    plt.show()


def run(robot, xml_path, ee_site, axis_name, ang_velocity, n_periods, dt, viewer=None) -> None:
    kin = Kinematics(xml_path, ee_site)
    time.sleep(0.5)

    home_q = robot.get_joint_pos()[:N_ARM].copy()
    print(f"Recorded home pose: {home_q.round(4)}")

    q0 = move_to_qpos(robot, TRAJ_START_QPOS, dt, viewer=viewer, label="trajectory start")
    center_pose = kin.fk(q0[:N_ARM])

    axis = _AXES[axis_name]
    omega = ang_velocity / AMPLITUDE
    period = 2 * np.pi / omega
    print(f"EE position (fixed): {center_pose[:3, 3].round(4)}")
    print(f"Start RPY          : {_R_to_rpy(center_pose[:3, :3]).round(4)} rad")
    print(f"Axis               : {axis_name} (body frame)")
    print(f"Amplitude          : ±{AMPLITUDE:.4f} rad  (±{np.degrees(AMPLITUDE):.1f} deg)")
    print(f"Peak angular speed : {ang_velocity:.3f} rad/s  |  period: {period:.2f} s  |  periods: {n_periods}")

    # Ramp from center orientation to +AMPLITUDE edge before starting cosine sinusoid.
    n_ramp = max(2, int(round(AMPLITUDE / (ang_velocity * dt))))
    R_center, p_center = center_pose[:3, :3], center_pose[:3, 3]
    edge_waypoints = []
    for i in range(n_ramp + 1):
        wp = np.eye(4)
        wp[:3, :3] = R_center @ _axis_angle_to_R(axis, AMPLITUDE * (i / n_ramp))
        wp[:3, 3]  = p_center
        edge_waypoints.append(wp)
    print(f"Ramping to sinusoid edge ({n_ramp} steps) ...")
    init_q = execute_waypoints(robot, kin, ee_site, edge_waypoints, q0[:N_ARM].copy(), dt, viewer=viewer)

    waypoints = build_sinusoidal_waypoints(center_pose, axis, AMPLITUDE, ang_velocity, n_periods, dt)
    print(f"Executing {len(waypoints)} waypoints over {len(waypoints) * dt:.2f} s ...")

    records: list = []
    execute_waypoints(robot, kin, ee_site, waypoints, init_q, dt,
                      viewer=viewer, records=records)
    print("Motion complete.")
    plot_ee_recording(records, axis_name)

    print(f"Returning to home: {home_q.round(4)} ...")
    move_to_qpos(robot, home_q, dt, viewer=viewer, label="home")
    print("At home.")


def make_sim_viewer(robot):
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as e:
        print(f"mujoco not importable ({e}); running headless.")
        return None
    model = getattr(robot, "_model", None)
    data  = getattr(robot, "_data",  None)
    if model is None or data is None:
        print("SimRobot exposes no _model/_data; running headless.")
        return None
    try:
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        return viewer
    except Exception as e:
        print(f"Could not launch viewer ({e}); running headless.")
        return None


def main() -> None:
    arm_choices = [a.value for a in ArmType]

    parser = argparse.ArgumentParser(
        description="Sinusoidal rotational EE motion about a body-frame axis"
    )
    parser.add_argument("--arm", default="yam", choices=arm_choices)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--no-view", action="store_true", help="Run sim headless (no viewer)")
    parser.add_argument("--site", default="grasp_site")

    parser.add_argument("--axis", choices=list(_AXES.keys()), default="roll",
                        help="Body-frame rotation axis (roll=X, pitch=Y, yaw=Z); default roll")
    parser.add_argument("--ang-velocity", type=float, default=0.2,
                        help="Peak angular speed (rad/s); default 0.2")
    parser.add_argument("--periods", type=int, default=N_PERIODS,
                        help=f"Number of sinusoidal periods; default {N_PERIODS}")
    parser.add_argument("--dt", type=float, default=0.02, help="Control timestep (s)")
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
        run(robot, robot.xml_path, args.site,
            axis_name=args.axis,
            ang_velocity=args.ang_velocity,
            n_periods=args.periods,
            dt=args.dt,
            viewer=viewer)

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
