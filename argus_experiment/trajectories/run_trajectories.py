"""
Generalized trajectory runner: executes a queue of trajectory commands back to back.

Each command in TRAJECTORY_QUEUE specifies a wave type (sinusoidal/sawtooth),
a motion type (linear/angular), an axis, and a speed. For every command the
arm runs the full cycle:

    record home -> move to TRAJ_START_QPOS -> run trajectory -> return home

then holds briefly at home before starting the next command's cycle.

All motion is executed in the link_6 CAMERA frame: camera_site is injected
from the hand-eye result (--handeye) and used for both FK (trajectory center)
and IK (targeting) — angular axes (roll/pitch/yaw) rotate about the camera's
own axes, not the gripper's.

Generalizes ee_linear_sinusoidal / ee_linear_sawtooth / ee_rot_sinusoidal /
ee_rot_sawtooth. Arm is fixed to yam on channel can0. The queue is passed on
the command line via repeated --traj WAVE:MOTION:AXIS:SPEED tokens. A trial
name (positional arg) titles the output data folder.

Usage:
    # via the shell wrapper (edit the --traj lines there)
    bash argus_experiment/trajectories/run_trajectories.sh TRIAL_NAME --sim

    # or directly
    python argus_experiment/trajectories/run_trajectories.py TRIAL_NAME --sim \
        --traj sinusoidal:linear:x:0.5 \
        --traj sawtooth:angular:roll:0.2
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "robots_realtime" / "dependencies" / "i2rt"))
sys.path.insert(0, str(_REPO / "argus_experiment" / "calibration"))

from camera_frame import add_camera_site, load_handeye
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType

DEFAULT_HANDEYE = _REPO / "argus_experiment" / "calibration" / "hand_eye_result.yaml"
DEFAULT_OUT_DIR = _REPO / "argus_experiment" / "trajectories" / "recordings"


# ---------------------------------------------------------------------------
# Trajectory command, parsed from a --traj WAVE:MOTION:AXIS:SPEED CLI token.
#
#   wave   : "sinusoidal" | "sawtooth"
#   motion : "linear" | "angular"
#   axis   : linear  -> "x", "y", "z"           (base-frame direction)
#            angular -> "roll", "pitch", "yaw"   (body-frame rotation axis)
#   speed  : linear  -> m/s   (peak for sinusoidal, constant for sawtooth)
#            angular -> rad/s  (peak for sinusoidal, constant for sawtooth)
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryCommand:
    wave: Literal["sinusoidal", "sawtooth"]
    motion: Literal["linear", "angular"]
    axis: str
    speed: float


# ---------------------------------------------------------------------------
# Constants (shared with the ee_* reference scripts)
# ---------------------------------------------------------------------------
N_ARM = 6
ARGUS_MASS = 0.731219  # 0.178 camera+mount + 0.553219 linear_4310 body (fingers removed)

# TRAJ_START_QPOS = np.array([
#     +0.0883,  # [0] Shoulder Pan
#     +0.5694,  # [1] Shoulder Pitch
#     +0.5758,  # [2] Elbow
#     +0.0090,  # [3] Wrist 1
#     +0.1844,  # [4] Wrist 2
#     -0.1036,  # [5] Wrist 3
# ])

# Previous start pose: [1.6268, 0.5804, 1.1751, -0.5827, 0.0463, 0.0757].
# Its y stroke ran through a wrist singularity -- sigma_min(J) collapsed from
# 0.026 to 5e-5 around y = -0.025 m and IK drove joint 4 past its yaml limit,
# so ~8% of y waypoints failed to converge. This pose keeps sigma_min >= 0.039
# across all six strokes with >= 0.65 rad of joint-limit margin; the camera sits
# 9.4 cm away from the old start and rotated 25.6 deg.
TRAJ_START_QPOS = np.array([
    +1.8188,  # [0] Shoulder Pan
    +0.9233,  # [1] Shoulder Pitch
    +1.2121,  # [2] Elbow
    -0.3515,  # [3] Wrist 1
    -0.0100,  # [4] Wrist 2
    +0.4339,  # [5] Wrist 3
])

LINEAR_AMPLITUDE  = 0.1       # m   — half peak-to-peak (matches ee_linear_*)
ANGULAR_AMPLITUDE = 0.5236/3 #TODO    # rad — peak excursion ~30 deg (matches ee_rot_*)

N_PERIODS       = 2
START_TOL       = 0.05        # rad
START_MOVE_TIME = 3.0         # s
HOME_HOLD_TIME  = 1.0         # s — pause at home between queued trajectories

AXES = {
    "x":     np.array([1.0, 0.0, 0.0]),
    "y":     np.array([0.0, 1.0, 0.0]),
    "z":     np.array([0.0, 0.0, 1.0]),
    "roll":  np.array([1.0, 0.0, 0.0]),
    "pitch": np.array([0.0, 1.0, 0.0]),
    "yaw":   np.array([0.0, 0.0, 1.0]),
}
LINEAR_AXES = ("x", "y", "z")
ANGULAR_AXES = ("roll", "pitch", "yaw")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
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


def _R_to_wxyz(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    w = max(w, 1e-8)
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class Recorder:
    """Samples measured joint q/qdot and camera pose continuously over the run.

    Camera pose is T_base_cam = FK(grasp_site) @ X, where X = T_ee_cam from the
    hand-eye result. joint_vel is the robot's measured velocity (authoritative on
    hardware; the sim reports zero, so use t + q for finite differences there).
    """

    def __init__(self, robot, kin, X: np.ndarray):
        self._robot = robot
        self._kin = kin
        self._X = X
        self._t0 = time.time()
        self.t, self.q, self.qdot = [], [], []
        self.cam_pos, self.cam_quat = [], []
        self.traj_idx, self.phase = [], []

    def sample(self, traj_idx: int, phase: str) -> None:
        obs = self._robot.get_observations()
        q = np.asarray(obs["joint_pos"], dtype=float)[:N_ARM]
        qd = np.asarray(obs["joint_vel"], dtype=float)[:N_ARM]
        # X = T_ee_cam is defined relative to grasp_site (what hand-eye calibration used
        # for FK), so always resolve grasp_site here regardless of kin's default site
        # (which may be camera_site when --camera drives the actual trajectory).
        T_cam = self._kin.fk(q, "grasp_site") @ self._X
        self.t.append(time.time() - self._t0)
        self.q.append(q)
        self.qdot.append(qd)
        self.cam_pos.append(T_cam[:3, 3].copy())
        self.cam_quat.append(_R_to_wxyz(T_cam[:3, :3]))
        self.traj_idx.append(traj_idx)
        self.phase.append(phase)

    def save(self, npz_path, queue, periods, dt, end_dwell=0.0, home_hold=HOME_HOLD_TIME) -> None:
        np.savez(
            npz_path,
            t=np.array(self.t),                                   # (N,)   seconds from run start
            q=np.array(self.q),                                   # (N, 6) measured joint positions
            qdot=np.array(self.qdot),                             # (N, 6) measured joint velocities
            cam_pos=np.array(self.cam_pos),                       # (N, 3) camera position (base frame)
            cam_quat=np.array(self.cam_quat),                     # (N, 4) camera orientation wxyz
            traj_idx=np.array(self.traj_idx, dtype=int),          # (N,)   index into the queue
            phase=np.array(self.phase),                          # (N,)   to_start|ramp|trajectory|to_home|hold
            traj_wave=np.array([c.wave for c in queue]),          # (K,)   per-command info; index via traj_idx
            traj_motion=np.array([c.motion for c in queue]),      # (K,)
            traj_axis=np.array([c.axis for c in queue]),          # (K,)
            traj_speed=np.array([c.speed for c in queue], float),  # (K,)
            periods=periods, dt=dt, argus_mass=ARGUS_MASS,
            end_dwell=end_dwell,                                  # s, hold at each end of travel
            home_hold=home_hold,                                  # s, hold at home between commands
        )


def sanitize_trial_name(name: str) -> str:
    """Make a trial name filesystem-safe: spaces/slashes/etc -> underscore."""
    safe = re.sub(r"[^\w\-]+", "_", name.strip()).strip("_")
    return safe or "trial"


def write_description(txt_path, queue, args, npz_name: str) -> None:
    """Human-readable summary of the queue that produced the accompanying npz."""
    lines = [
        "Trajectory Queue Recording",
        "=" * 27,
        f"Trial name  : {args.trial_name}",
        f"Timestamp   : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Mode        : {'sim' if args.sim else 'hardware'} (arm=yam, channel=can0)",
        "Frame       : camera (camera_site, injected from hand-eye result)",
        f"Periods     : {args.periods}",
        f"Control dt  : {args.dt} s",
        f"End dwell   : {args.end_dwell} s (hold at each end of travel)",
        f"Home hold   : {args.home_hold} s (between queued trajectories)",
        f"Hand-eye    : {args.handeye}",
        f"ARGUS_MASS  : {ARGUS_MASS} kg",
        f"Data file   : {npz_name}",
        "",
        f"Queue ({len(queue)} commands, run in order):",
    ]
    for i, cmd in enumerate(queue):
        unit = "m/s" if cmd.motion == "linear" else "rad/s"
        lines.append(f"  [{i}] {cmd.wave:10s} {cmd.motion:8s} axis={cmd.axis:6s} speed={cmd.speed} {unit}")
    lines += [
        "",
        "npz keys: t, q, qdot, cam_pos, cam_quat, traj_idx, phase,",
        "          traj_wave, traj_motion, traj_axis, traj_speed, periods, dt, argus_mass,",
        "          end_dwell, home_hold",
        "  traj_idx indexes into the queue above (0-based).",
        "  phase is one of: to_start | ramp | trajectory | to_home | hold",
    ]
    Path(txt_path).write_text("\n".join(lines) + "\n")


def move_to_qpos(robot, target_q, dt, viewer=None, move_time=START_MOVE_TIME,
                 tol=START_TOL, label="target",
                 recorder: Optional[Recorder] = None, traj_idx: int = -1, phase: str = "") -> np.ndarray:
    q0 = robot.get_joint_pos()
    arm_err = np.max(np.abs(q0[:N_ARM] - target_q[:N_ARM]))
    if arm_err <= tol:
        print(f"  Arm already within {tol} rad of {label} (err {arm_err:.4f}).")
        return q0
    print(f"  Arm is {arm_err:.4f} rad from {label} (tol {tol}); moving ...")
    steps = max(2, int(round(move_time / dt)))
    for i in range(steps + 1):
        alpha = i / steps
        cmd = q0.copy()
        cmd[:N_ARM] = (1 - alpha) * q0[:N_ARM] + alpha * target_q[:N_ARM]
        robot.command_joint_pos(cmd)
        _sync(viewer)
        if recorder is not None:
            recorder.sample(traj_idx, phase)
        time.sleep(move_time / steps)
    time.sleep(0.5)
    return robot.get_joint_pos()


def execute_waypoints(robot, kin, ee_site, waypoints, init_q, dt, viewer=None,
                      recorder: Optional[Recorder] = None, traj_idx: int = -1, phase: str = "") -> np.ndarray:
    """Warm-started IK tracking. Returns the last successful joint config.

    Paces off an absolute deadline instead of sleeping dt per step. IK + command +
    recorder cost ~0.5-1 ms, and time.sleep(dt) would add that on top of every
    step, so the arm would track slower than the commanded speed -- negligible at
    dt=0.02 but ~30% low by dt=0.002. A step that overruns dt does not push the
    following ones late; overruns are counted and reported instead, since they
    mean the commanded speed was not actually achieved.
    """
    n_overrun = n_fail = 0
    next_t = time.perf_counter() + dt
    for i, target_pose in enumerate(waypoints):
        ok, ik_q = kin.ik(target_pose, ee_site, init_q=init_q)
        if not ok:
            n_fail += 1
            # Hold the last good command, but still consume this step's time slot so
            # an IK failure does not silently compress the trajectory timeline.
            print(f"    IK failed at waypoint {i}; holding last good solution")
        else:
            robot.command_joint_pos(ik_q[:N_ARM])
            init_q = ik_q[:N_ARM]
        _sync(viewer)
        if recorder is not None:
            recorder.sample(traj_idx, phase)
        slack = next_t - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            n_overrun += 1
        next_t += dt
    if n_overrun:
        pct = 100.0 * n_overrun / max(1, len(waypoints))
        # A failed IK solve runs to its iteration cap (~35 ms vs ~1 ms warm-started),
        # so it blows the deadline on its own; that is a reachability problem, not a
        # dt that is too small. Only point at dt when the loop is otherwise clean.
        cause = (f"{n_fail} IK failure(s) stalled the loop -- the target is near a "
                 f"kinematic limit" if n_fail else f"the loop cannot sustain dt={dt}s; increase --dt")
        print(f"    WARNING: {n_overrun}/{len(waypoints)} steps ({pct:.0f}%) overran "
              f"dt={dt}s; {cause}")
    return init_q


# ---------------------------------------------------------------------------
# waypoint builders — linear (from ee_linear_sinusoidal / ee_linear_sawtooth)
# ---------------------------------------------------------------------------
def build_linear_sinusoidal(center_pose, unit, amplitude, velocity, n_periods, dt) -> list[np.ndarray]:
    """pos(t) = center + amplitude * cos(ωt) * unit;  ω = velocity / amplitude."""
    omega = velocity / amplitude
    total_time = n_periods * (2 * np.pi / omega)
    n_steps = max(2, int(round(total_time / dt)))
    t = np.linspace(0.0, total_time, n_steps, endpoint=False)
    waypoints = []
    for ti in t:
        wp = center_pose.copy()
        wp[:3, 3] += amplitude * np.cos(omega * ti) * unit  # cos: starts at +A, zero velocity
        waypoints.append(wp)
    return waypoints


def build_linear_sawtooth(center_pose, unit, amplitude, velocity, n_periods, dt,
                          end_dwell: float = 0.0) -> list[np.ndarray]:
    """Triangle wave at constant velocity between −amplitude and +amplitude.

    end_dwell (s) holds the arm at each edge before it reverses. The hold is emitted
    as repeated edge waypoints so execute_waypoints keeps ticking at dt and the
    recorder keeps sampling through the pause.
    """
    n_leg = max(2, int(round(2 * amplitude / (velocity * dt))))
    n_dwell = max(0, int(round(end_dwell / dt)))
    neg_edge = center_pose.copy(); neg_edge[:3, 3] -= unit * amplitude
    pos_edge = center_pose.copy(); pos_edge[:3, 3] += unit * amplitude
    waypoints = []
    for _ in range(n_periods):
        waypoints += [neg_edge.copy() for _ in range(n_dwell)]
        for i in range(n_leg):
            alpha = i / n_leg
            wp = center_pose.copy()
            wp[:3, 3] = (1 - alpha) * neg_edge[:3, 3] + alpha * pos_edge[:3, 3]
            waypoints.append(wp)
        waypoints += [pos_edge.copy() for _ in range(n_dwell)]
        for i in range(n_leg):
            alpha = i / n_leg
            wp = center_pose.copy()
            wp[:3, 3] = (1 - alpha) * pos_edge[:3, 3] + alpha * neg_edge[:3, 3]
            waypoints.append(wp)
    waypoints.append(neg_edge)
    waypoints += [neg_edge.copy() for _ in range(n_dwell)]
    return waypoints


def build_linear_ramp(center_pose, unit, amplitude, velocity, dt, to: str) -> list[np.ndarray]:
    """Ramp from center to the +amplitude ('pos') or −amplitude ('neg') edge."""
    sign = 1.0 if to == "pos" else -1.0
    n_ramp = max(2, int(round(amplitude / (velocity * dt))))
    waypoints = []
    for i in range(n_ramp + 1):
        wp = center_pose.copy()
        wp[:3, 3] += sign * unit * amplitude * (i / n_ramp)
        waypoints.append(wp)
    return waypoints


# ---------------------------------------------------------------------------
# waypoint builders — angular (from ee_rot_sinusoidal / ee_rot_sawtooth)
# ---------------------------------------------------------------------------
def build_angular_sinusoidal(center_pose, axis, amplitude, ang_velocity, n_periods, dt) -> list[np.ndarray]:
    """angle(t) = amplitude * cos(ωt) applied in body frame; position held fixed."""
    omega = ang_velocity / amplitude
    total_time = n_periods * (2 * np.pi / omega)
    n_steps = max(2, int(round(total_time / dt)))
    t = np.linspace(0.0, total_time, n_steps, endpoint=False)
    R_center, p_center = center_pose[:3, :3], center_pose[:3, 3]
    waypoints = []
    for ti in t:
        theta = amplitude * np.cos(omega * ti)
        wp = np.eye(4)
        wp[:3, :3] = R_center @ _axis_angle_to_R(axis, theta)
        wp[:3, 3] = p_center
        waypoints.append(wp)
    return waypoints


def build_angular_sawtooth(center_pose, axis, amplitude, ang_velocity, n_periods, dt,
                           end_dwell: float = 0.0) -> list[np.ndarray]:
    """Triangle wave rotation at constant angular velocity; position held fixed.

    end_dwell (s) holds at each angular edge, same scheme as build_linear_sawtooth.
    """
    n_leg = max(2, int(round(2 * amplitude / (ang_velocity * dt))))
    n_dwell = max(0, int(round(end_dwell / dt)))
    R_center, p_center = center_pose[:3, :3], center_pose[:3, 3]

    def wp_at(theta):
        wp = np.eye(4)
        wp[:3, :3] = R_center @ _axis_angle_to_R(axis, theta)
        wp[:3, 3] = p_center
        return wp

    waypoints = []
    for _ in range(n_periods):
        waypoints += [wp_at(-amplitude) for _ in range(n_dwell)]
        for i in range(n_leg):
            a = i / n_leg
            waypoints.append(wp_at((1 - a) * (-amplitude) + a * amplitude))
        waypoints += [wp_at(amplitude) for _ in range(n_dwell)]
        for i in range(n_leg):
            a = i / n_leg
            waypoints.append(wp_at((1 - a) * amplitude + a * (-amplitude)))
    waypoints.append(wp_at(-amplitude))
    waypoints += [wp_at(-amplitude) for _ in range(n_dwell)]
    return waypoints


def build_angular_ramp(center_pose, axis, amplitude, ang_velocity, dt, to: str) -> list[np.ndarray]:
    """Ramp orientation from center to the +amplitude ('pos') or −amplitude ('neg') edge."""
    sign = 1.0 if to == "pos" else -1.0
    n_ramp = max(2, int(round(amplitude / (ang_velocity * dt))))
    R_center, p_center = center_pose[:3, :3], center_pose[:3, 3]
    waypoints = []
    for i in range(n_ramp + 1):
        wp = np.eye(4)
        wp[:3, :3] = R_center @ _axis_angle_to_R(axis, sign * amplitude * (i / n_ramp))
        wp[:3, 3] = p_center
        waypoints.append(wp)
    return waypoints


# ---------------------------------------------------------------------------
# per-command cycle: home -> TRAJ_START_QPOS -> trajectory -> home (hold)
# ---------------------------------------------------------------------------
def run_command(robot, kin, ee_site, cmd: TrajectoryCommand, n_periods: int, dt: float,
                viewer=None, recorder: Optional[Recorder] = None, traj_idx: int = -1,
                end_dwell: float = 0.0, home_hold: float = HOME_HOLD_TIME) -> None:
    print(f"\n=== {cmd.wave} {cmd.motion}  axis={cmd.axis}  speed={cmd.speed} ===")

    home_q = robot.get_joint_pos()[:N_ARM].copy()
    print(f"  Recorded home pose: {home_q.round(4)}")

    q0 = move_to_qpos(robot, TRAJ_START_QPOS, dt, viewer=viewer, label="trajectory start",
                      recorder=recorder, traj_idx=traj_idx, phase="to_start")
    center_pose = kin.fk(q0[:N_ARM], ee_site)
    init_q = q0[:N_ARM].copy()

    if cmd.motion == "linear":
        if cmd.axis not in LINEAR_AXES:
            raise ValueError(f"linear axis must be one of {LINEAR_AXES}, got {cmd.axis!r}")
        unit = AXES[cmd.axis]
        if cmd.wave == "sinusoidal":
            ramp = build_linear_ramp(center_pose, unit, LINEAR_AMPLITUDE, cmd.speed, dt, to="pos")
            main_wps = build_linear_sinusoidal(center_pose, unit, LINEAR_AMPLITUDE, cmd.speed, n_periods, dt)
        else:
            ramp = build_linear_ramp(center_pose, unit, LINEAR_AMPLITUDE, cmd.speed, dt, to="neg")
            main_wps = build_linear_sawtooth(center_pose, unit, LINEAR_AMPLITUDE, cmd.speed, n_periods, dt,
                                             end_dwell=end_dwell)
    elif cmd.motion == "angular":
        if cmd.axis not in ANGULAR_AXES:
            raise ValueError(f"angular axis must be one of {ANGULAR_AXES}, got {cmd.axis!r}")
        axis = AXES[cmd.axis]
        if cmd.wave == "sinusoidal":
            ramp = build_angular_ramp(center_pose, axis, ANGULAR_AMPLITUDE, cmd.speed, dt, to="pos")
            main_wps = build_angular_sinusoidal(center_pose, axis, ANGULAR_AMPLITUDE, cmd.speed, n_periods, dt)
        else:
            ramp = build_angular_ramp(center_pose, axis, ANGULAR_AMPLITUDE, cmd.speed, dt, to="neg")
            main_wps = build_angular_sawtooth(center_pose, axis, ANGULAR_AMPLITUDE, cmd.speed, n_periods, dt,
                                              end_dwell=end_dwell)
    else:
        raise ValueError(f"motion must be 'linear' or 'angular', got {cmd.motion!r}")

    print(f"  Ramping to trajectory edge ({len(ramp)} steps) ...")
    init_q = execute_waypoints(robot, kin, ee_site, ramp, init_q, dt, viewer=viewer,
                               recorder=recorder, traj_idx=traj_idx, phase="ramp")

    print(f"  Executing {len(main_wps)} waypoints over {len(main_wps) * dt:.2f} s ...")
    execute_waypoints(robot, kin, ee_site, main_wps, init_q, dt, viewer=viewer,
                      recorder=recorder, traj_idx=traj_idx, phase="trajectory")
    print("  Motion complete.")

    print(f"  Returning to home: {home_q.round(4)} ...")
    move_to_qpos(robot, home_q, dt, viewer=viewer, label="home",
                 recorder=recorder, traj_idx=traj_idx, phase="to_home")
    print(f"  At home. Holding {home_hold:.1f}s ...")
    for _ in range(max(1, int(round(home_hold / dt)))):
        _sync(viewer)
        if recorder is not None:
            recorder.sample(traj_idx, "hold")
        time.sleep(dt)


# ---------------------------------------------------------------------------
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


def parse_traj_command(spec: str) -> TrajectoryCommand:
    """Parse a "wave:motion:axis:speed" CLI token into a TrajectoryCommand."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected WAVE:MOTION:AXIS:SPEED, got {spec!r}")
    wave, motion, axis, speed_s = parts
    if wave not in ("sinusoidal", "sawtooth"):
        raise argparse.ArgumentTypeError(f"wave must be sinusoidal|sawtooth, got {wave!r}")
    if motion not in ("linear", "angular"):
        raise argparse.ArgumentTypeError(f"motion must be linear|angular, got {motion!r}")
    valid_axes = LINEAR_AXES if motion == "linear" else ANGULAR_AXES
    if axis not in valid_axes:
        raise argparse.ArgumentTypeError(f"{motion} axis must be one of {valid_axes}, got {axis!r}")
    try:
        speed = float(speed_s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"speed must be a number, got {speed_s!r}")
    if speed <= 0:
        raise argparse.ArgumentTypeError(f"speed must be positive, got {speed}")
    return TrajectoryCommand(wave, motion, axis, speed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a queue of EE trajectory commands")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--periods", type=int, default=N_PERIODS,
                        help=f"Number of sinusoidal periods; default {N_PERIODS}")
    parser.add_argument("--dt", type=float, default=0.02, help="Control timestep (s)")
    parser.add_argument("--end-dwell", type=float, default=0.0, dest="end_dwell",
                        help="Seconds to hold at each end of travel before reversing "
                             "(sawtooth only); default 0 = no pause")
    parser.add_argument("--home-hold", type=float, default=HOME_HOLD_TIME, dest="home_hold",
                        help=f"Seconds to hold at home between queued trajectories; "
                             f"default {HOME_HOLD_TIME}")
    parser.add_argument("--traj", action="append", dest="queue", type=parse_traj_command,
                        required=True, metavar="WAVE:MOTION:AXIS:SPEED",
                        help="Trajectory command, e.g. sinusoidal:linear:x:0.5 . Repeatable; "
                             "runs in the order given.")
    parser.add_argument("--handeye", default=str(DEFAULT_HANDEYE),
                        help="hand_eye_result.yaml for the camera transform")
    parser.add_argument("trial_name", help="Trial name; titles the output data folder")
    parser.add_argument("--out", default=None,
                        help="Output directory for this run's data + description.txt "
                             "(overrides the trial-name/timestamp default)")
    args = parser.parse_args()

    # X = T_ee_cam drives BOTH the executed trajectory frame (camera_site, injected
    # below) and the Recorder's logged camera pose. If unavailable, fall back to
    # identity (camera_site == grasp_site) rather than crashing, but warn loudly
    # since this silently changes what frame the arm actually moves in.
    try:
        X = load_handeye(args.handeye)
        print(f"Running trajectories in CAMERA frame (hand-eye: {args.handeye})")
    except (FileNotFoundError, KeyError) as e:
        X = np.eye(4)
        print(f"WARNING: could not load hand-eye ({e}); camera_site will equal grasp_site.")

    robot = get_yam_robot(
        channel="can0",
        arm_type=ArmType.YAM,
        gripper_type=GripperType.from_string_name("no_gripper"),
        sim=args.sim,
        ee_mass=ARGUS_MASS,
    )
    xml_path = add_camera_site(robot.xml_path, X)
    site = "camera_site"
    kin = Kinematics(xml_path, site)
    time.sleep(0.5)

    recorder = Recorder(robot, kin, X)
    viewer = make_sim_viewer(robot) if args.sim else None

    try:
        print(f"Running {len(args.queue)} queued trajectories ...")
        for i, cmd in enumerate(args.queue):
            print(f"\n--- Queue item {i + 1}/{len(args.queue)} ---")
            run_command(robot, kin, site, cmd, args.periods, args.dt,
                        viewer=viewer, recorder=recorder, traj_idx=i,
                        end_dwell=args.end_dwell, home_hold=args.home_hold)
        print("\nQueue complete.")

        if viewer is not None:
            print("Sim done — close the viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    finally:
        if args.out is not None:
            out_dir = Path(args.out)
        else:
            out_dir = DEFAULT_OUT_DIR / f"{sanitize_trial_name(args.trial_name)}_cam"
        out_dir.mkdir(parents=True, exist_ok=True)

        npz_path = out_dir / "trajectory_data.npz"
        txt_path = out_dir / "trajectories.txt"
        recorder.save(npz_path, args.queue, args.periods, args.dt,
                      end_dwell=args.end_dwell, home_hold=args.home_hold)
        write_description(txt_path, args.queue, args, npz_path.name)
        print(f"Recorded {len(recorder.t)} samples -> {out_dir}/")
        print(f"  {npz_path.name}, {txt_path.name}")
        if viewer is not None:
            viewer.close()
        robot.close()


if __name__ == "__main__":
    main()
