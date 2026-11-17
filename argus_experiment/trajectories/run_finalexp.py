"""
Final experiment: one recording of one (axis, waveform, speed) condition.

A round is 3 axes x 4 speeds x 2 waveforms = 24 recordings, each with its own
ARGUS bag and mocap bag. run_finalexp.sh (repo root) runs one round per
invocation, back to back in the order `schedule --round N` prints, launching
this script once per recording:

    robot up -> TRAJ_START_QPOS
      -> --on-ready hook: start ARGUS + mocap; returns once both are recording
      -> hold still STILL_TIME s at the start pose (OpenVINS initialization)
      -> ramp start pose -> -A in RAMP_TIME s (half-cosine, same for every condition)
      -> sweep: DWELL at -A, then cycles of [leg to +A, DWELL, leg to -A, DWELL]
      -> --on-done hook: stop mocap + ARGUS; returns once both have closed
      -> back to the pose the arm started in, robot closed

The speed v (deg/s) is the sawtooth's constant sweep speed. Both waveforms
share amplitude A and leg time 2A/v, so they have the same period
T = 4A/v + 2*DWELL, the same cycle count SWEEP_TIME / T and the same mean leg
speed v. Only the velocity profile differs: constant with sharp reversals, or
a half-cosine whose peak is pi/2 = 1.57x v. The sweep time is fixed rather
than the cycle count, so every condition drifts for the same time, and every
condition has the same rest schedule. With A = 30 deg and DWELL = 0.5 s:

    v (deg/s)      30    60    120   240
    leg (s)        2     1     0.5   0.25
    T (s)          5     3     2     1.5
    cycles         6     10    15    20
    sine peak      47    94    188   377   (deg/s)

Each recording: STILL_TIME + RAMP_TIME + DWELL + SWEEP_TIME = 42.5 s of arm
time after both recordings are up. All rotation is about the camera's own axes
(camera_site from the hand-eye result), as in run_trajectories.py. No robot
data is saved.

Usage:
    python run_finalexp.py schedule --round 2   # round 2's randomized run order
    python run_finalexp.py check         # offline IK + joint-speed check of all 24 conditions
    python run_finalexp.py run NAME --axis pitch --wave sinusoidal --speed 120 --sim
"""

import argparse
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------
AXES_ORDER    = ("roll", "pitch", "yaw")    # camera-frame rotation axes
WAVES         = ("sawtooth", "sinusoidal")
WAVE_TAG      = {"sawtooth": "saw", "sinusoidal": "sin"}
SPEEDS_DPS    = (30, 60, 120, 240)          # deg/s: sawtooth speed = mean leg speed of both waves
SEED          = 1117                        # fixed, so each round's order is reproducible and resumable

AMPLITUDE_DEG = 30.0      # deg - the sweep runs -A..+A about the start pose
DWELL         = 0.5       # s   - rest at every edge, same for every condition
SWEEP_TIME    = 30.0      # s   - sweep cycles per recording, same for every condition
STILL_TIME    = 10.0      # s   - still at the start pose once both recordings are up
RAMP_TIME     = 2.0       # s   - start pose -> -A, same for every condition
DT            = 0.0025    # s   - control timestep, same for every condition


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def make_schedule(round_no: int, seed: int = SEED) -> list[tuple]:
    """(name, axis, wave, speed_dps, round) for the 24 recordings of one round, in run order.

    Every round is its own random permutation of all 24 conditions (seeded by
    the round number), so each condition runs once per round and slow drifts
    over a session (temperature, battery, Motive calibration) spread evenly
    across repeats instead of piling onto one.
    """
    if round_no < 1:
        raise ValueError(f"round must be >= 1, got {round_no}")
    rng = np.random.default_rng([seed, round_no])
    conditions = [(a, w, v) for a in AXES_ORDER for w in WAVES for v in SPEEDS_DPS]
    return [(f"fe_r{round_no}_{i + 1:02d}_{a}_{WAVE_TAG[w]}_{v}dps", a, w, v, round_no)
            for i, (a, w, v) in enumerate(conditions[k] for k in rng.permutation(len(conditions)))]


# ---------------------------------------------------------------------------
# Angle profile
# ---------------------------------------------------------------------------
@dataclass
class Profile:
    theta: np.ndarray   # rad, one per dt step: ramp, first dwell, then the cycles
    n_head: int         # steps of ramp + first dwell (theta[:n_head])
    n_cycle: int        # steps per cycle
    n_cycles: int
    leg_time: float     # s, one -A <-> +A leg
    period: float       # s, two legs + two dwells
    peak_dps: float     # deg/s, peak commanded angular speed


def _whole(x: float, what: str) -> int:
    n = int(round(x))
    if n < 1 or abs(x - n) > 1e-6:
        raise ValueError(f"{what} = {x:.6g} is not a whole number; adjust the design constants")
    return n


def build_profile(wave: str, speed_dps: float, amplitude_deg: float = AMPLITUDE_DEG,
                  dwell: float = DWELL, sweep_time: float = SWEEP_TIME,
                  ramp_time: float = RAMP_TIME, dt: float = DT) -> Profile:
    """Rotation angle about the sweep axis, one sample per dt.

    Each leg's samples end on the edge, so a dwell of n steps holds the edge for
    exactly n*dt and a leg of n steps takes exactly n*dt. The cycle count and the
    leg length must both come out whole, so speed and timing are exact rather
    than rounded.
    """
    A = np.deg2rad(amplitude_deg)
    leg_time = 2 * amplitude_deg / speed_dps
    period = 2 * (leg_time + dwell)
    n_cycles = _whole(sweep_time / period, f"sweep time / period ({sweep_time} s / {period} s)")
    n_leg = _whole(leg_time / dt, f"leg time / dt ({leg_time} s / {dt} s)")
    n_dwell = int(round(dwell / dt))
    n_ramp = max(1, int(round(ramp_time / dt)))

    u = np.arange(1, n_leg + 1) / n_leg
    if wave == "sawtooth":
        s = u                                     # constant speed v
    elif wave == "sinusoidal":
        s = 0.5 * (1 - np.cos(np.pi * u))         # half-cosine: mean v, peak (pi/2) v
    else:
        raise ValueError(f"wave must be one of {WAVES}, got {wave!r}")
    up = -A + 2 * A * s                           # -A -> +A; the return leg is -up
    lo, hi = np.full(n_dwell, -A), np.full(n_dwell, A)
    cycle = np.concatenate([up, hi, -up, lo])

    r = np.arange(1, n_ramp + 1) / n_ramp
    ramp = -A * 0.5 * (1 - np.cos(np.pi * r))     # 0 -> -A, zero speed at both ends
    head = np.concatenate([ramp, lo])

    peak = speed_dps * (np.pi / 2 if wave == "sinusoidal" else 1.0)
    return Profile(np.concatenate([head, np.tile(cycle, n_cycles)]), len(head), len(cycle),
                   n_cycles, leg_time, period, peak)


def profile_poses(center_pose: np.ndarray, axis: np.ndarray, theta: np.ndarray, rt) -> list[np.ndarray]:
    """Camera poses rotated by theta about the camera-frame `axis`; position fixed."""
    R_c, p_c = center_pose[:3, :3], center_pose[:3, 3]
    poses = []
    for th in theta:
        wp = np.eye(4)
        wp[:3, :3] = R_c @ rt._axis_angle_to_R(axis, th)
        wp[:3, 3] = p_c
        poses.append(wp)
    return poses


# ---------------------------------------------------------------------------
# Robot setup (imports deferred so `schedule` runs with numpy alone)
# ---------------------------------------------------------------------------
def _import_rt():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_trajectories as rt  # also puts i2rt and calibration/ on sys.path
    return rt


def _make_robot(rt, sim: bool, handeye: str):
    from camera_frame import add_camera_site, load_handeye
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import ArmType, GripperType

    try:
        X = load_handeye(handeye)
        print(f"Running in CAMERA frame (hand-eye: {handeye})")
    except (FileNotFoundError, KeyError) as e:
        X = np.eye(4)
        print(f"WARNING: could not load hand-eye ({e}); camera_site will equal grasp_site.")

    robot = get_yam_robot(
        channel="can0",
        arm_type=ArmType.YAM,
        gripper_type=GripperType.from_string_name("no_gripper"),
        sim=sim,
        ee_mass=rt.ARGUS_MASS,
    )
    site = "camera_site"
    kin = Kinematics(add_camera_site(robot.xml_path, X), site)
    return robot, kin, site


def _event(name: str, label: str) -> None:
    # Wall-clock times on this machine, the same clock as the mocap header.stamp,
    # so the run log can cut each mocap bag into still / ramp / sweep.
    print(f"EVENT {name} {label} {time.time():.3f}", flush=True)


def _hook(cmd: str, label: str) -> bool:
    if not cmd:
        return True
    print(f"  [{label}] {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    if rc != 0:
        print(f"  [{label}] FAILED (exit {rc})", flush=True)
    return rc == 0


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_schedule(args) -> None:
    for row in make_schedule(args.round, args.seed):
        print("\t".join(str(x) for x in row))


def cmd_run(args) -> int:
    prof = build_profile(args.wave, args.speed, sweep_time=args.sweep, dt=args.dt)
    print(f"\n=== {args.name}: {args.wave} {args.axis}, {args.speed:g} deg/s "
          f"(peak {prof.peak_dps:.0f}), A = {AMPLITUDE_DEG:g} deg ===")
    print(f"  leg {prof.leg_time:g} s, dwell {DWELL:g} s, period {prof.period:g} s, "
          f"{prof.n_cycles} cycles in {args.sweep:g} s")
    print(f"  after the recordings start: still {args.still:g} s, ramp {RAMP_TIME:g} s, "
          f"sweep {DWELL + args.sweep:g} s (first dwell + cycles)")

    rt = _import_rt()
    robot, kin, site = _make_robot(rt, args.sim, args.handeye)
    time.sleep(0.5)
    viewer = rt.make_sim_viewer(robot) if args.sim and not args.headless else None
    home_q = robot.get_joint_pos()[:rt.N_ARM].copy()
    print(f"  Recorded home pose: {home_q.round(4)}")

    status, recording = 0, False
    try:
        q0 = rt.move_to_qpos(robot, rt.TRAJ_START_QPOS, args.dt, viewer=viewer, label="trajectory start")
        center = kin.fk(q0[:rt.N_ARM], site)
        poses = profile_poses(center, rt.AXES[args.axis], prof.theta, rt)

        recording = True  # from here on, always run --on-done, even if --on-ready failed half way
        if not _hook(args.on_ready, "start recordings"):
            status = 3
            return status
        _event(args.name, "recording")

        print(f"  Holding still {args.still:g} s at the start pose ...", flush=True)
        rt.hold_still(args.still, args.dt, viewer=viewer)

        _event(args.name, "motion_start")
        print(f"  Ramp + sweep: {len(poses)} waypoints over {len(poses) * args.dt:.2f} s ...", flush=True)
        rt.execute_waypoints(robot, kin, site, poses, q0, args.dt, viewer=viewer)
        _event(args.name, "motion_end")
        print("  Sweep complete.")
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        status = 130
    finally:
        # Finish the cleanup even on a second Ctrl-C: stopping half way would leave
        # the recordings running or the arm out at the start pose when the robot
        # closes. The hooks inherit this, so they finish too. Use the e-stop to halt.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if recording:
            if not _hook(args.on_done, "stop recordings"):
                status = status or 4
            _event(args.name, "stopped")
        try:
            print(f"  Returning to home: {home_q.round(4)} ...")
            rt.move_to_qpos(robot, home_q, args.dt, viewer=viewer, label="home")
        finally:
            if viewer is not None:
                viewer.close()
            robot.close()
    return status


def cmd_check(args) -> int:
    """IK the ramp, first dwell and two cycles of every condition from TRAJ_START_QPOS.

    Later cycles repeat these, so this covers every pose. Peak joint speeds are
    finite differences of the warm-started IK, i.e. the velocity feedforward
    execute_waypoints would send. Above MAX_FF_VEL it clips.
    """
    rt = _import_rt()
    robot, kin, site = _make_robot(rt, sim=True, handeye=args.handeye)
    robot.close()
    q_start = rt.TRAJ_START_QPOS[:rt.N_ARM].copy()
    center = kin.fk(q_start, site)

    print(f"\nTRAJ_START_QPOS = {q_start.round(4)}, A = {AMPLITUDE_DEG:g} deg, dt = {args.dt} s, "
          f"MAX_FF_VEL = {rt.MAX_FF_VEL} rad/s")
    print(f"{'axis':6s} {'wave':10s} {'v':>4s} {'peak':>5s} {'T':>5s} {'cyc':>4s} "
          f"{'IK fail':>7s} {'max|qd|':>8s} {'joint':>5s} {'clipped':>7s}")
    bad = 0
    for axis in AXES_ORDER:
        for wave in WAVES:
            for v in SPEEDS_DPS:
                prof = build_profile(wave, v, dt=args.dt)
                theta = prof.theta[:prof.n_head + 2 * prof.n_cycle]
                q, qs, fails = q_start.copy(), [q_start.copy()], 0
                for wp in profile_poses(center, rt.AXES[axis], theta, rt):
                    ok, sol = kin.ik(wp, site, init_q=q)
                    if ok:
                        q = np.asarray(sol, dtype=float)[:rt.N_ARM]
                    else:
                        fails += 1
                    qs.append(q.copy())
                qd = np.abs(np.diff(np.array(qs), axis=0)) / args.dt
                j = int(np.argmax(qd.max(axis=0)))
                clipped = int(np.sum(np.any(qd > rt.MAX_FF_VEL, axis=1)))
                bad += fails > 0 or clipped > 0
                flag = "  <--" if fails or clipped else ""
                print(f"{axis:6s} {wave:10s} {v:4d} {prof.peak_dps:5.0f} {prof.period:5.2f} "
                      f"{prof.n_cycles:4d} {fails:7d} {qd.max():8.2f} {j + 1:5d} {clipped:7d}{flag}",
                      flush=True)
    print(f"\n{bad} of {len(AXES_ORDER) * len(WAVES) * len(SPEEDS_DPS)} conditions have IK "
          f"failures or clipped feedforward." if bad else "\nAll conditions OK.")
    return 1 if bad else 0


def main() -> None:
    default_handeye = str(Path(__file__).resolve().parents[1] / "calibration" / "hand_eye_result.yaml")
    p = argparse.ArgumentParser(description="Final experiment: one recording per condition")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("schedule", help="Print one round's run order: name, axis, wave, speed_dps, round (TSV)")
    s.add_argument("--round", type=int, default=1)
    s.add_argument("--seed", type=int, default=SEED)

    c = sub.add_parser("check", help="Offline IK and joint-speed check of every condition")
    c.add_argument("--dt", type=float, default=DT)
    c.add_argument("--handeye", default=default_handeye)

    r = sub.add_parser("run", help="Run one recording")
    r.add_argument("name", help="Recording name (for the log)")
    r.add_argument("--axis", required=True, choices=AXES_ORDER)
    r.add_argument("--wave", required=True, choices=WAVES)
    r.add_argument("--speed", required=True, type=float, help="deg/s (sawtooth speed, mean for sinusoidal)")
    r.add_argument("--sim", action="store_true")
    r.add_argument("--headless", action="store_true", help="With --sim, no MuJoCo viewer")
    r.add_argument("--on-ready", default="", dest="on_ready",
                   help="Shell command run at the start pose; must exit 0 once recordings are up")
    r.add_argument("--on-done", default="", dest="on_done",
                   help="Shell command run after the sweep; should exit once recordings are closed")
    r.add_argument("--still", type=float, default=STILL_TIME, help=argparse.SUPPRESS)  # test overrides
    r.add_argument("--sweep", type=float, default=SWEEP_TIME, help=argparse.SUPPRESS)
    r.add_argument("--dt", type=float, default=DT, help=argparse.SUPPRESS)
    r.add_argument("--handeye", default=default_handeye)

    args = p.parse_args()
    if args.cmd == "schedule":
        cmd_schedule(args)
    elif args.cmd == "check":
        sys.exit(cmd_check(args))
    else:
        sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
