"""
Static-hold runner: moves between home and a target pose with no trajectory
motion, holding still at each end.

Records the arm's actual initial pose as home, then repeats N times:

    home -> target pose -> hold POSE_HOLD s -> return home -> hold HOME_HOLD s

The target is TRAJ_START_QPOS every repeat, or with --random a different random
pose each repeat. Random poses are drawn up front, uniformly within ±--range rad
of TRAJ_START_QPOS per joint, from a fixed --seed so the sim test and the
hardware run visit the same poses. A pose is kept only if it is
  - at least JOINT_MARGIN inside the XML joint limits,
  - free of self-collision (the base/link1 meshes always overlap, so that pair
    is ignored), and
  - has link_6 and the camera at least --min-height m above the base plane.
Before any motion, the joint-space path from the actual home to each pose is
checked for self-collision and for dipping below min(--min-height, its height
at home); a failure aborts the run rather than silently changing the poses.

Shares TRAJ_START_QPOS, the joint-space moves, the hold loop and the Recorder
with run_trajectories.py, so the npz has the same t/q/qdot/cam_pos/cam_quat
layout. traj_idx is the repeat index (0-based).

Usage:
    # via the shell wrappers
    bash argus_experiment/trajectories/run_still.sh TRIAL_NAME --sim
    bash argus_experiment/trajectories/run_still_random.sh TRIAL_NAME --sim

    # or directly
    python argus_experiment/trajectories/run_still.py TRIAL_NAME --sim --repeats 2 --random --seed 3
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

# run_trajectories also puts i2rt and calibration/ on sys.path for the imports below.
from run_trajectories import (
    ARGUS_MASS, DEFAULT_HANDEYE, DEFAULT_OUT_DIR, N_ARM, TRAJ_START_QPOS,
    Recorder, hold_still, make_sim_viewer, move_to_qpos, sanitize_trial_name,
)
from camera_frame import add_camera_site, load_handeye
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType

N_REPEATS      = 10
POSE_HOLD_TIME = 10.0  # s — hold at the target pose
HOME_HOLD_TIME = 5.0   # s — hold at home after each return

# --random sampling
RANDOM_RANGE  = 0.3    # rad — max per-joint offset from TRAJ_START_QPOS
MIN_HEIGHT    = 0.10   # m   — lowest allowed link_6 / camera z (base frame)
JOINT_MARGIN  = 0.1    # rad — keep this far inside the XML joint range
MAX_TRIES     = 10000  # candidate draws before giving up
N_PATH_CHECKS = 50     # samples along each home <-> pose joint-space path
IGNORED_CONTACTS = {frozenset(("world", "link1"))}  # base/link1 meshes always overlap


class PoseChecker:
    """Self-collision and height checks on the (camera-site) MuJoCo model."""

    def __init__(self, xml_path: str):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.lo = self.m.jnt_range[:N_ARM, 0] + JOINT_MARGIN
        self.hi = self.m.jnt_range[:N_ARM, 1] - JOINT_MARGIN

    def _forward(self, q) -> None:
        self.d.qpos[:N_ARM] = q
        mujoco.mj_forward(self.m, self.d)

    def collisions(self, q) -> set[frozenset]:
        """Body pairs in contact at q, minus IGNORED_CONTACTS."""
        self._forward(q)
        body = lambda g: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g])
        pairs = {frozenset((body(c.geom1), body(c.geom2))) for c in self.d.contact[:self.d.ncon]}
        return pairs - IGNORED_CONTACTS

    def min_height(self, q) -> float:
        """Lower of the link_6 origin and camera_site z (base frame)."""
        self._forward(q)
        return min(self.d.body("link_6").xpos[2], self.d.site("camera_site").xpos[2])


def sample_random_poses(checker: PoseChecker, n: int, rng_range: float, min_height: float,
                        seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(MAX_TRIES):
        q = TRAJ_START_QPOS + rng.uniform(-rng_range, rng_range, N_ARM)
        if (np.all(q >= checker.lo) and np.all(q <= checker.hi)
                and not checker.collisions(q) and checker.min_height(q) >= min_height):
            poses.append(q)
            if len(poses) == n:
                return np.array(poses)
    raise RuntimeError(f"only {len(poses)}/{n} valid poses in {MAX_TRIES} draws; "
                       f"try a smaller --range or --min-height")


def check_path(checker: PoseChecker, home_q, target_q, min_height: float) -> str | None:
    """None if the home -> target joint-space path is clear, else why it is not.

    Contacts already present at home are ignored (the arm is physically there, so
    they are mesh-approximation artifacts), as is dipping no lower than home.
    """
    home_pairs = checker.collisions(home_q)
    floor = min(min_height, checker.min_height(home_q))
    for a in np.linspace(0.0, 1.0, N_PATH_CHECKS + 1):
        q = (1 - a) * home_q + a * target_q
        if hits := checker.collisions(q) - home_pairs:
            return f"self-collision {sorted(map(sorted, hits))} at {a:.0%} of the way"
        if (h := checker.min_height(q)) < floor:
            return f"link_6/camera dips to z={h:.3f} m at {a:.0%} of the way"
    return None


def run_cycle(robot, home_q, target_q, dt, pose_hold, home_hold, viewer=None,
              recorder: Recorder | None = None, idx: int = -1, label: str = "target pose") -> None:
    move_to_qpos(robot, target_q, dt, viewer=viewer, label=label,
                 recorder=recorder, traj_idx=idx, phase="to_pose")
    print(f"  At {label}. Holding {pose_hold:.1f}s ...")
    hold_still(pose_hold, dt, viewer=viewer, recorder=recorder, traj_idx=idx, phase="pose_hold")

    move_to_qpos(robot, home_q, dt, viewer=viewer, label="home",
                 recorder=recorder, traj_idx=idx, phase="to_home")
    print(f"  At home. Holding {home_hold:.1f}s ...")
    hold_still(home_hold, dt, viewer=viewer, recorder=recorder, traj_idx=idx, phase="hold")


def save(recorder: Recorder, npz_path, args, home_q: np.ndarray, targets: np.ndarray) -> None:
    np.savez(
        npz_path,
        t=np.array(recorder.t),                          # (N,)   seconds from run start
        q=np.array(recorder.q),                          # (N, 6) measured joint positions
        qdot=np.array(recorder.qdot),                    # (N, 6) measured joint velocities
        cam_pos=np.array(recorder.cam_pos),              # (N, 3) camera position (base frame)
        cam_quat=np.array(recorder.cam_quat),            # (N, 4) camera orientation wxyz
        traj_idx=np.array(recorder.traj_idx, dtype=int), # (N,)   repeat index
        phase=np.array(recorder.phase),                  # (N,)   to_pose|pose_hold|to_home|hold
        target_qpos=targets,                             # (R, 6) pose held during pose_hold, per repeat
        home_qpos=home_q,                                # (6,)   recorded initial pose
        repeats=args.repeats, dt=args.dt, argus_mass=ARGUS_MASS,
        pose_hold=args.pose_hold,                        # s, hold at the target pose
        home_hold=args.home_hold,                        # s, hold at home after each return
        random=args.random, seed=args.seed,              # --random sampling settings
        range=args.range, min_height=args.min_height,
    )


def write_description(txt_path, args, npz_name: str, home_q: np.ndarray, targets: np.ndarray) -> None:
    if args.random:
        target_line = (f"random, seed={args.seed}, ±{args.range} rad/joint around TRAJ_START_QPOS, "
                       f"min height {args.min_height} m")
    else:
        target_line = "TRAJ_START_QPOS every repeat"
    lines = [
        "Static Hold Recording",
        "=" * 21,
        f"Trial name  : {args.trial_name}",
        f"Timestamp   : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Mode        : {'sim' if args.sim else 'hardware'} (arm=yam, channel=can0)",
        f"Targets     : {target_line}",
        f"Repeats     : {args.repeats}",
        f"Pose hold   : {args.pose_hold} s (at the target pose)",
        f"Home hold   : {args.home_hold} s (after each return home)",
        f"Control dt  : {args.dt} s",
        f"Home qpos   : {home_q.round(4).tolist()}",
        f"Hand-eye    : {args.handeye}",
        f"ARGUS_MASS  : {ARGUS_MASS} kg",
        f"Data file   : {npz_name}",
        "",
        "Each repeat: home -> target pose -> pose_hold -> home -> hold",
        "Target qpos per repeat:",
        *(f"  [{i}] {q.round(4).tolist()}" for i, q in enumerate(targets)),
        "",
        "npz keys: t, q, qdot, cam_pos, cam_quat, traj_idx, phase, target_qpos, home_qpos,",
        "          repeats, dt, argus_mass, pose_hold, home_hold, random, seed, range, min_height",
        "  traj_idx is the repeat index (0-based); target_qpos[traj_idx] is that repeat's pose.",
        "  phase is one of: to_pose | pose_hold | to_home | hold",
    ]
    Path(txt_path).write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated static holds at a target pose and home")
    parser.add_argument("trial_name", help="Trial name; titles the output data folder")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help=f"Number of pose/home cycles; default {N_REPEATS}")
    parser.add_argument("--pose-hold", type=float, default=POSE_HOLD_TIME, dest="pose_hold",
                        help=f"Seconds to hold at the target pose; default {POSE_HOLD_TIME}")
    parser.add_argument("--home-hold", type=float, default=HOME_HOLD_TIME, dest="home_hold",
                        help=f"Seconds to hold at home after each return; default {HOME_HOLD_TIME}")
    parser.add_argument("--dt", type=float, default=0.02, help="Control / recording timestep (s)")
    parser.add_argument("--random", action="store_true",
                        help="Hold a different random pose each repeat instead of TRAJ_START_QPOS")
    parser.add_argument("--seed", type=int, default=0, help="--random: RNG seed; default 0")
    parser.add_argument("--range", type=float, default=RANDOM_RANGE,
                        help=f"--random: max per-joint offset from TRAJ_START_QPOS (rad); "
                             f"default {RANDOM_RANGE}")
    parser.add_argument("--min-height", type=float, default=MIN_HEIGHT, dest="min_height",
                        help=f"--random: lowest allowed link_6 / camera z (m, base frame); "
                             f"default {MIN_HEIGHT}")
    parser.add_argument("--handeye", default=str(DEFAULT_HANDEYE),
                        help="hand_eye_result.yaml for the logged camera pose")
    parser.add_argument("--out", default=None,
                        help="Output directory for this run's data + description "
                             "(overrides the trial-name default)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.range <= 0:
        parser.error("--range must be positive")

    try:
        X = load_handeye(args.handeye)
    except (FileNotFoundError, KeyError) as e:
        X = np.eye(4)
        print(f"WARNING: could not load hand-eye ({e}); logged camera pose will equal grasp_site.")

    robot = get_yam_robot(
        channel="can0",
        arm_type=ArmType.YAM,
        gripper_type=GripperType.from_string_name("no_gripper"),
        sim=args.sim,
        ee_mass=ARGUS_MASS,
    )
    xml_path = add_camera_site(robot.xml_path, X)
    kin = Kinematics(xml_path, "camera_site")
    time.sleep(0.5)

    # Record the robot's ACTUAL initial pose as home; every cycle returns here.
    home_q = robot.get_joint_pos()[:N_ARM].copy()
    print(f"Recorded home pose: {home_q.round(4)}")

    if args.random:
        checker = PoseChecker(xml_path)
        try:
            targets = sample_random_poses(checker, args.repeats, args.range, args.min_height, args.seed)
            for i, q in enumerate(targets):
                if (why := check_path(checker, home_q, q, args.min_height)) is not None:
                    raise RuntimeError(f"path home -> random pose {i + 1} fails: {why}; "
                                       f"try another --seed")
        except RuntimeError as e:
            robot.close()
            raise SystemExit(f"ERROR: {e}. Nothing was moved.")
        print(f"Random poses (seed={args.seed}, ±{args.range} rad around TRAJ_START_QPOS):")
        for i, q in enumerate(targets):
            print(f"  [{i + 1}] {q.round(4)}")
    else:
        targets = np.tile(TRAJ_START_QPOS, (args.repeats, 1))

    recorder = Recorder(robot, kin, X)
    viewer = make_sim_viewer(robot) if args.sim else None

    try:
        for i, target_q in enumerate(targets):
            print(f"\n--- Repeat {i + 1}/{args.repeats} ---")
            label = f"random pose {i + 1}" if args.random else "trajectory start"
            run_cycle(robot, home_q, target_q, args.dt, args.pose_hold, args.home_hold,
                      viewer=viewer, recorder=recorder, idx=i, label=label)
        print("\nAll repeats complete.")

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
        save(recorder, npz_path, args, home_q, targets)
        write_description(txt_path, args, npz_path.name, home_q, targets)
        print(f"Recorded {len(recorder.t)} samples -> {out_dir}/")
        print(f"  {npz_path.name}, {txt_path.name}")
        if viewer is not None:
            viewer.close()
        robot.close()


if __name__ == "__main__":
    main()
