"""
Still experiment: one recording holding one random pose for HOLD_TIME s.

A round is the 10 poses of run_still_random.sh, one recording each, in pose
order. run_stillexp.sh (repo root) runs one round per invocation, launching this
script once per recording with the same protocol as the final experiment:

    robot up -> check the home -> pose path -> move to pose K
      -> --on-ready hook: start ARGUS + mocap; returns once both are recording
      -> hold still HOLD_TIME s
      -> --on-done hook: stop mocap + ARGUS; returns once both have closed
      -> back to the pose the arm started in, robot closed

The poses come from run_still.sample_random_poses with run_still_random.sh's
settings (seed 0; +-1.0 rad shoulder pan and +-0.3 rad on joints 2-6 around
TRAJ_START_QPOS; link_6 and the camera at least 0.10 m up), so they are the
still_random poses as long as TRAJ_START_QPOS is unchanged. Each recording
checks the joint-space path from the arm's actual home before it moves. No robot
data is saved.

Usage:
    python run_stillexp.py poses                  # print the 10 poses
    python run_stillexp.py schedule --round 1     # one round's run order
    python run_stillexp.py run NAME --pose 3 --sim
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from run_finalexp import _event, _hook, _import_rt

# run_still_random.sh settings: these reproduce its poses.
N_POSES    = 5
SEED       = 0       #       RNG seed for the pose set
PAN_RANGE  = 1.0     # rad - max joint-1 (shoulder pan) offset from TRAJ_START_QPOS
RANGE      = 0.3     # rad - max offset from TRAJ_START_QPOS for joints 2-6
MIN_HEIGHT = 0.10    # m   - lowest allowed link_6 / camera z (base frame)

HOLD_TIME  = 60.0    # s   - still at the pose once both recordings are up
DT         = 1/300    # s   - control timestep for the moves (as run_still.sh)


def make_schedule(round_no: int) -> list[tuple]:
    """(name, pose, round) for the N_POSES recordings of one round, in run order."""
    if round_no < 1:
        raise ValueError(f"round must be >= 1, got {round_no}")
    return [(f"still_r{round_no}_pose{k:02d}", k, round_no) for k in range(1, N_POSES + 1)]


def _make_robot(rt, sim: bool, handeye: str):
    """Robot, camera-site kinematics and the camera-site XML (the pose checks use it)."""
    from camera_frame import add_camera_site, load_handeye
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import ArmType, GripperType

    try:
        X = load_handeye(handeye)
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
    xml_path = add_camera_site(robot.xml_path, X)
    return robot, Kinematics(xml_path, "camera_site"), xml_path


def _poses(xml_path):
    import run_still as rs
    checker = rs.PoseChecker(xml_path)
    return rs.sample_random_poses(checker, N_POSES, RANGE, PAN_RANGE, MIN_HEIGHT, SEED), checker


def cmd_schedule(args) -> None:
    for row in make_schedule(args.round):
        print("\t".join(str(x) for x in row))


def cmd_poses(args) -> int:
    rt = _import_rt()
    robot, _, xml_path = _make_robot(rt, sim=True, handeye=args.handeye)
    robot.close()
    poses, _ = _poses(xml_path)
    print(f"\nstill_random poses (seed {SEED}) around TRAJ_START_QPOS = {rt.TRAJ_START_QPOS.round(4)}:")
    for k, q in enumerate(poses, 1):
        print(f"  pose {k:2d}: {q.round(4)}")
    return 0


def cmd_run(args) -> int:
    if not 1 <= args.pose <= N_POSES:
        raise SystemExit(f"--pose must be 1..{N_POSES}, got {args.pose}")
    print(f"\n=== {args.name}: still at random pose {args.pose} for {args.hold:g} s ===")

    rt = _import_rt()
    import run_still as rs
    robot, _, xml_path = _make_robot(rt, args.sim, args.handeye)
    time.sleep(0.5)
    viewer = rt.make_sim_viewer(robot) if args.sim and not args.headless else None
    home_q = robot.get_joint_pos()[:rt.N_ARM].copy()
    print(f"  Recorded home pose: {home_q.round(4)}")

    status, recording = 0, False
    try:
        poses, checker = _poses(xml_path)
        target = poses[args.pose - 1]
        print(f"  Pose {args.pose}: {target.round(4)}")
        why = rs.check_path(checker, home_q, target, MIN_HEIGHT)
        if why is not None:
            print(f"  ERROR: path home -> pose {args.pose} fails: {why}. Nothing was moved.")
            return 2

        rt.move_to_qpos(robot, target, args.dt, viewer=viewer, label=f"pose {args.pose}")

        recording = True  # from here on, always run --on-done, even if --on-ready failed half way
        if not _hook(args.on_ready, "start recordings"):
            status = 3
            return status
        _event(args.name, "recording")

        print(f"  Holding still {args.hold:g} s ...", flush=True)
        rt.hold_still(args.hold, args.dt, viewer=viewer)
        _event(args.name, "hold_end")
        print("  Hold complete.")
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        status = 130
    finally:
        # As in run_finalexp.py: finish stopping the recordings and homing even on a
        # second Ctrl-C. Use the e-stop to halt.
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


def main() -> None:
    default_handeye = str(Path(__file__).resolve().parents[1] / "calibration" / "hand_eye_result.yaml")
    p = argparse.ArgumentParser(description="Still experiment: one recording per random pose")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("schedule", help="Print one round's run order: name, pose, round (TSV)")
    s.add_argument("--round", type=int, default=1)

    c = sub.add_parser("poses", help="Print the 10 random poses")
    c.add_argument("--handeye", default=default_handeye)

    r = sub.add_parser("run", help="Run one recording")
    r.add_argument("name", help="Recording name (for the log)")
    r.add_argument("--pose", required=True, type=int, help=f"Pose number, 1..{N_POSES}")
    r.add_argument("--sim", action="store_true")
    r.add_argument("--headless", action="store_true", help="With --sim, no MuJoCo viewer")
    r.add_argument("--on-ready", default="", dest="on_ready",
                   help="Shell command run at the pose; must exit 0 once recordings are up")
    r.add_argument("--on-done", default="", dest="on_done",
                   help="Shell command run after the hold; should exit once recordings are closed")
    r.add_argument("--hold", type=float, default=HOLD_TIME, help=argparse.SUPPRESS)  # test override
    r.add_argument("--dt", type=float, default=DT, help=argparse.SUPPRESS)
    r.add_argument("--handeye", default=default_handeye)

    args = p.parse_args()
    if args.cmd == "schedule":
        cmd_schedule(args)
    elif args.cmd == "poses":
        sys.exit(cmd_poses(args))
    else:
        sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
