"""
Sawtooth (triangle-wave) linear end-effector motion along a base-frame axis.

Unlike the sinusoidal script, the EE moves at *constant* velocity between the
two endpoints, producing a triangle wave in position and a square wave in velocity.

Lifecycle:
  record home -> move to TRAJ_START_QPOS -> ramp to negative edge at velocity
  -> N periods of constant-velocity sweeps between ±AMPLITUDE -> return home.

One period = one full round trip (−A → +A → −A).

Usage:
    # Simulation — X axis (default)
    python argus_experiment/osc/ee_linear_sawtooth.py --sim

    # Y axis, slower
    python argus_experiment/osc/ee_linear_sawtooth.py --sim --direction 0 1 0 --velocity 0.05

    # Z axis on hardware
    python argus_experiment/osc/ee_linear_sawtooth.py --channel can0 --direction 0 0 1

    # Headless sim
    python argus_experiment/osc/ee_linear_sawtooth.py --sim --no-view
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

AMPLITUDE       = 0.2   # m — half peak-to-peak (0.4 m total sweep)
N_PERIODS       = 2
START_TOL       = 0.05  # rad
START_MOVE_TIME = 3.0   # s


def _R_to_rpy(R: np.ndarray) -> np.ndarray:
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def _sync(viewer) -> None:
    if viewer is not None and viewer.is_running():
        viewer.sync()


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


def build_sawtooth_waypoints(
    center_pose: np.ndarray,
    direction: np.ndarray,
    amplitude: float,
    velocity: float,
    n_periods: int,
    dt: float,
) -> list[np.ndarray]:
    """
    Triangle-wave waypoints at constant velocity.
    Starts at −amplitude (after ramp), sweeps to +amplitude then back, N times.
    Each leg has exactly n_leg steps so every inter-waypoint step = velocity * dt.
    """
    unit = direction / np.linalg.norm(direction)
    n_leg = max(2, int(round(2 * amplitude / (velocity * dt))))  # steps per half-period

    neg_edge = center_pose.copy()
    neg_edge[:3, 3] -= unit * amplitude
    pos_edge = center_pose.copy()
    pos_edge[:3, 3] += unit * amplitude

    waypoints = []
    for _ in range(n_periods):
        # Leg 1: −A → +A
        for i in range(n_leg):
            alpha = i / n_leg
            wp = center_pose.copy()
            wp[:3, 3] = (1 - alpha) * neg_edge[:3, 3] + alpha * pos_edge[:3, 3]
            waypoints.append(wp)
        # Leg 2: +A → −A
        for i in range(n_leg):
            alpha = i / n_leg
            wp = center_pose.copy()
            wp[:3, 3] = (1 - alpha) * pos_edge[:3, 3] + alpha * neg_edge[:3, 3]
            waypoints.append(wp)

    # Final point: land exactly on −A
    waypoints.append(neg_edge)
    return waypoints


def execute_waypoints(robot, kin, ee_site, waypoints, init_q, dt, viewer=None,
                      records: list | None = None) -> np.ndarray:
    """Returns the last successful joint config (warm-start for next call)."""
    t = 0.0
    for i, target_pose in enumerate(waypoints):
        ok, ik_q = kin.ik(target_pose, ee_site, init_q=init_q)
        if not ok:
            print(f"  IK failed at waypoint {i} "
                  f"(target x,y,z={target_pose[:3, 3].round(4)}); holding last good solution")
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


def plot_recording(records: list, direction: np.ndarray) -> None:
    if not records:
        return
    unit       = direction / np.linalg.norm(direction)
    times      = np.array([r[0] for r in records])
    positions  = np.array([r[1] for r in records])  # (N, 3)
    rpys       = np.array([r[2] for r in records])  # (N, 3)
    velocities = np.gradient(positions, times, axis=0)  # (N, 3)  m/s

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"Sawtooth EE 6-DOF  [direction {unit.round(2)}]", fontsize=12)

    ax = axes[0]
    for j, label in enumerate(["x", "y", "z"]):
        ax.plot(times, positions[:, j], label=label)
    ax.set_ylabel("Position (m)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("EE position")

    ax = axes[1]
    for j, label in enumerate(["roll", "pitch", "yaw"]):
        ax.plot(times, rpys[:, j], label=label)
    ax.set_ylabel("Angle (rad)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("EE orientation (RPY)")

    ax = axes[2]
    for j, label in enumerate(["vx", "vy", "vz"]):
        ax.plot(times, velocities[:, j], label=label)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("EE linear velocity  (should be ±constant on motion axis)")

    plt.tight_layout()
    plt.show()


def run(robot, xml_path, ee_site, direction, velocity, n_periods, dt, viewer=None) -> None:
    kin = Kinematics(xml_path, ee_site)
    time.sleep(0.5)

    home_q = robot.get_joint_pos()[:N_ARM].copy()
    print(f"Recorded home pose: {home_q.round(4)}")

    q0 = move_to_qpos(robot, TRAJ_START_QPOS, dt, viewer=viewer, label="trajectory start")
    center_pose = kin.fk(q0[:N_ARM])

    unit = direction / np.linalg.norm(direction)
    n_leg = max(2, int(round(2 * AMPLITUDE / (velocity * dt))))
    period = 2 * n_leg * dt
    print(f"EE center    : {center_pose[:3, 3].round(4)}")
    print(f"Direction    : {unit.round(4)}")
    print(f"Amplitude    : ±{AMPLITUDE:.3f} m  (peak-to-peak {2*AMPLITUDE:.3f} m)")
    print(f"Velocity     : {velocity:.3f} m/s (constant)  |  period: {period:.2f} s  |  periods: {n_periods}")

    # Ramp from center to negative edge (−A) at the same velocity before the sweep.
    n_ramp = max(2, int(round(AMPLITUDE / (velocity * dt))))
    ramp_waypoints = []
    for i in range(n_ramp + 1):
        wp = center_pose.copy()
        wp[:3, 3] -= unit * AMPLITUDE * (i / n_ramp)
        ramp_waypoints.append(wp)
    print(f"Ramping to negative edge ({n_ramp} steps) ...")
    init_q = execute_waypoints(robot, kin, ee_site, ramp_waypoints, q0[:N_ARM].copy(), dt, viewer=viewer)

    waypoints = build_sawtooth_waypoints(center_pose, direction, AMPLITUDE, velocity, n_periods, dt)
    print(f"Executing {len(waypoints)} waypoints over {len(waypoints) * dt:.2f} s ...")

    records: list = []
    execute_waypoints(robot, kin, ee_site, waypoints, init_q, dt, viewer=viewer, records=records)
    print("Motion complete.")
    # plot_recording(records, direction)

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
        description="Sawtooth (constant-velocity) linear EE motion along a base-frame axis"
    )
    parser.add_argument("--arm", default="yam", choices=arm_choices)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--no-view", action="store_true", help="Run sim headless (no viewer)")
    parser.add_argument("--site", default="grasp_site")

    parser.add_argument("--direction", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"), help="Base-frame direction (normalised)")
    parser.add_argument("--velocity", type=float, default=0.05,
                        help="Constant Cartesian speed (m/s); default 0.05")
    parser.add_argument("--periods", type=int, default=N_PERIODS,
                        help=f"Number of round-trip periods; default {N_PERIODS}")
    parser.add_argument("--dt", type=float, default=0.02, help="Control timestep (s)")
    args = parser.parse_args()

    direction = np.array(args.direction, dtype=float)
    if np.linalg.norm(direction) < 1e-9:
        parser.error("--direction must be a nonzero vector")
    if args.velocity <= 0:
        parser.error("--velocity must be positive")

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
            direction=direction,
            velocity=args.velocity,
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
