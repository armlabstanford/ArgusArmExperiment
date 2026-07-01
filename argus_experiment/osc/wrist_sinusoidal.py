"""
Sinusoidal direct joint-space motion on the YAM wrist joints (3, 4, 5).

Unlike ee_rot_sinusoidal (which uses IK to rotate about the EE position),
this commands the wrist motors directly — no IK involved.

Joint indices:
    3 — Wrist 1  (forearm rotation)
    4 — Wrist 2  (wrist pitch)
    5 — Wrist 3  (wrist roll)

Sinusoid: q_i(t) = q_start_i + AMPLITUDE * sin(ωt)
  where ω = ang_velocity / AMPLITUDE  (peak joint speed = --ang-velocity).

Lifecycle:
  record home -> move to TRAJ_START_QPOS -> run N sinusoidal periods
  on selected wrist joint(s) -> return to recorded home.

Usage:
    # Simulation — wrist 3 (default)
    python argus_experiment/osc/wrist_sinusoidal.py --sim

    # Wrist 1
    python argus_experiment/osc/wrist_sinusoidal.py --sim --wrist wrist1

    # All three wrists together
    python argus_experiment/osc/wrist_sinusoidal.py --sim --wrist all

    # Hardware, wrist 2, faster
    python argus_experiment/osc/wrist_sinusoidal.py --channel can0 --wrist wrist2 --ang-velocity 0.5
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

AMPLITUDE   = 0.7854    # rad — ±π/4 (±45 deg) peak excursion
N_PERIODS   = 2
START_TOL   = 0.05      # rad
START_MOVE_TIME = 3.0   # s

_WRIST_JOINTS = {
    "wrist1": [3],
    "wrist2": [4],
    "wrist3": [5],
    "all":    [3, 4, 5],
}


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


def run_wrist_sinusoidal(
    robot,
    joint_indices: list[int],
    amplitude: float,
    ang_velocity: float,
    n_periods: int,
    dt: float,
    viewer=None,
    kin=None,
    records: list | None = None,
) -> None:
    """Sinusoidally oscillate the selected wrist joints about their TRAJ_START_QPOS values."""
    omega = ang_velocity / amplitude
    period = 2 * np.pi / omega
    total_time = n_periods * period
    n_steps = max(2, int(round(total_time / dt)))
    t_vals = np.linspace(0.0, total_time, n_steps, endpoint=False)

    q_start = robot.get_joint_pos().copy()

    # Ramp joints from center to +amplitude edge before starting cosine sinusoid.
    n_ramp = max(2, int(round(amplitude / (ang_velocity * dt))))
    for i in range(n_ramp + 1):
        cmd = q_start.copy()
        for idx in joint_indices:
            cmd[idx] = q_start[idx] + amplitude * (i / n_ramp)
        robot.command_joint_pos(cmd)
        _sync(viewer)
        time.sleep(dt)

    for ti in t_vals:
        cmd = q_start.copy()
        delta = amplitude * np.cos(omega * ti)  # cos: starts at +amplitude, zero velocity
        for idx in joint_indices:
            cmd[idx] = q_start[idx] + delta
        robot.command_joint_pos(cmd)
        _sync(viewer)
        if records is not None and kin is not None:
            pose = kin.fk(cmd[:N_ARM])
            records.append((ti, cmd[joint_indices].copy(),
                            pose[:3, 3].copy(), _R_to_rpy(pose[:3, :3]).copy()))
        time.sleep(dt)


def plot_wrist_recording(records: list, joint_indices: list[int]) -> None:
    if not records:
        return
    joint_names = {3: "Wrist 1", 4: "Wrist 2", 5: "Wrist 3"}
    names = [joint_names[i] for i in joint_indices]

    times      = np.array([r[0] for r in records])
    angles     = np.array([r[1] for r in records])  # (N, len(joint_indices))
    positions  = np.array([r[2] for r in records])  # (N, 3)
    rpys       = np.array([r[3] for r in records])  # (N, 3)
    velocities = np.gradient(angles, times, axis=0)  # (N, len(joint_indices))  rad/s

    # Figure 1 — commanded joint angles + velocity
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig1.suptitle(f"Wrist sinusoidal — commanded joints {names}", fontsize=12)

    ax = axes1[0]
    for j, name in enumerate(names):
        ax.plot(times, angles[:, j], label=name)
    ax.set_ylabel("Joint angle (rad)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("Commanded wrist joint angles")

    ax = axes1[1]
    for j, name in enumerate(names):
        ax.plot(times, velocities[:, j], label=f"ω {name}")
    ax.set_ylabel("Joint velocity (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("Wrist joint velocity")

    fig1.tight_layout()

    # Figure 2 — EE 6-DOF pose (FK)
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig2.suptitle(f"Wrist sinusoidal — EE 6-DOF (FK)  [joints {names}]", fontsize=12)

    ax = axes2[0]
    for j, label in enumerate(["x", "y", "z"]):
        ax.plot(times, positions[:, j], label=label)
    ax.set_ylabel("Position (m)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("EE position")

    ax = axes2[1]
    for j, label in enumerate(["roll", "pitch", "yaw"]):
        ax.plot(times, rpys[:, j], label=label)
    ax.set_ylabel("Angle (rad)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("EE orientation (RPY)")

    fig2.tight_layout()
    plt.show()


def run(robot, xml_path, ee_site, joint_indices, ang_velocity, n_periods, dt, viewer=None) -> None:
    kin = Kinematics(xml_path, ee_site)
    time.sleep(0.5)

    home_q = robot.get_joint_pos()[:N_ARM].copy()
    print(f"Recorded home pose: {home_q.round(4)}")

    move_to_qpos(robot, TRAJ_START_QPOS, dt, viewer=viewer, label="trajectory start")

    omega = ang_velocity / AMPLITUDE
    period = 2 * np.pi / omega
    joint_names = {3: "Wrist 1", 4: "Wrist 2", 5: "Wrist 3"}
    names_str = ", ".join(joint_names[i] for i in joint_indices)
    print(f"Joint(s)          : {names_str}  (indices {joint_indices})")
    print(f"Amplitude         : ±{AMPLITUDE:.4f} rad  (±{np.degrees(AMPLITUDE):.1f} deg)")
    print(f"Peak joint speed  : {ang_velocity:.3f} rad/s  |  period: {period:.2f} s  |  periods: {n_periods}")
    total_steps = max(2, int(round(n_periods * period / dt)))
    print(f"Executing {total_steps} steps over {total_steps * dt:.2f} s ...")

    records: list = []
    run_wrist_sinusoidal(robot, joint_indices, AMPLITUDE, ang_velocity, n_periods, dt,
                         viewer=viewer, kin=kin, records=records)
    print("Motion complete.")
    plot_wrist_recording(records, joint_indices)

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
        description="Sinusoidal direct wrist-joint motion (no IK)"
    )
    parser.add_argument("--arm", default="yam", choices=arm_choices)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--no-view", action="store_true", help="Run sim headless (no viewer)")
    parser.add_argument("--site", default="grasp_site", help="EE site name for FK")

    parser.add_argument("--wrist", choices=list(_WRIST_JOINTS.keys()), default="wrist3",
                        help="Which wrist joint(s) to oscillate (default: wrist3)")
    parser.add_argument("--ang-velocity", type=float, default=3,
                        help="Peak joint speed (rad/s); default 0.2")
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
        run(
            robot,
            xml_path=robot.xml_path,
            ee_site=args.site,
            joint_indices=_WRIST_JOINTS[args.wrist],
            ang_velocity=args.ang_velocity,
            n_periods=args.periods,
            dt=args.dt,
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
