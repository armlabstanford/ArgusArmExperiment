#!/usr/bin/env python3
"""Plot each Vicon sub-trial in the Vicon world frame.

Usage:
    python3 plot_vicon_subtrials.py TRIAL_NAME [--show] [--out fig.png]

Reads the sub-trials written by align_data.py
(data/TRIAL/TRIAL_subtrials/subtrial_*.npz); if that folder is absent it
segments TRIAL_pose.npz directly with the same home-dwell logic.

One row of panels per sub-trial, all in the Vicon frame:

    [ 3D path (mm) ]   [ position x/y/z vs time ]   [ orientation r/p/y vs time ]

Position is the marker-cluster centroid (mm); orientation is the rigid-body
rotation R relative to frame 0, shown as intrinsic XYZ Euler angles (deg).
"""

import sys
import argparse
from pathlib import Path

import numpy as np


def resolve(arg: str):
    here = Path(__file__).resolve().parent
    base = Path(arg)
    if base.suffix == ".npz":
        d = base.parent
        trial = base.stem.replace("_pose", "")
    else:
        d = here / "data" / arg
        trial = arg
    return d, trial


def rotmat_to_euler_xyz(R):
    """(N,3,3) -> (N,3) intrinsic XYZ Euler angles (radians)."""
    sy = np.clip(-R[:, 2, 0], -1.0, 1.0)
    ry = np.arcsin(sy)
    rx = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    rz = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    lock = np.abs(sy) > 1.0 - 1e-9
    if lock.any():
        rx[lock] = np.arctan2(-R[lock, 1, 2], R[lock, 1, 1])
        rz[lock] = 0.0
    return np.column_stack([rx, ry, rz])


def load_subtrials(d, trial):
    """Return list of dicts {t, pos_mm, R} from saved sub-trials or by segmenting."""
    sub_dir = d / f"{trial}_subtrials"
    files = sorted(sub_dir.glob("subtrial_*.npz")) if sub_dir.is_dir() else []
    if files:
        out = []
        for f in files:
            z = np.load(f)
            out.append({"t": z["vicon_t"], "pos_mm": z["vicon_pos_mm"],
                        "R": z["vicon_R"] if z["vicon_R"].ndim == 3 else None})
        print(f"Loaded {len(out)} sub-trials from {sub_dir}/")
        return out

    # Fallback: segment the pose npz directly.
    from align_data import load_vicon, segment
    pose_npz = d / f"{trial}_pose.npz"
    if not pose_npz.is_file():
        sys.exit(f"No sub-trials folder and no pose npz found under {d}")
    vic = load_vicon(pose_npz)
    segs = segment(vic["t"], vic["pos_m"], use_home=True)
    print(f"Segmented {len(segs)} sub-trials from {pose_npz.name}")
    return [{"t": vic["t"][a:b], "pos_mm": vic["pos_mm"][a:b],
             "R": vic["R"][a:b] if vic["R"] is not None else None} for a, b in segs]


def set_equal_aspect(ax, p):
    mins, maxs = p.min(0), p.max(0)
    c = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial", help="TRIAL_NAME or path to *_pose.npz")
    ap.add_argument("--out", type=Path, help="figure path (default in trial dir)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    d, trial = resolve(args.trial)
    subs = load_subtrials(d, trial)
    if not subs:
        sys.exit("No sub-trials to plot.")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis_names = ("x", "y", "z")
    ang_names = ("roll", "pitch", "yaw")
    colors = ("tab:red", "tab:green", "tab:blue")
    rows = len(subs)
    fig = plt.figure(figsize=(15, 3.3 * rows))

    for r, s in enumerate(subs):
        t = s["t"] - s["t"][0]
        p = s["pos_mm"]

        # 3D path in Vicon frame
        ax = fig.add_subplot(rows, 3, r * 3 + 1, projection="3d")
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="tab:blue", lw=1.2)
        ax.scatter(*p[0], c="k", s=25, marker="o")
        ax.scatter(*p[-1], c="k", s=45, marker="*")
        set_equal_aspect(ax, p)
        ax.set_title(f"sub {r}: 3D path  ({s['t'][0]:.1f}-{s['t'][-1]:.1f}s)", fontsize=9)
        ax.set_xlabel("x[mm]", fontsize=7); ax.set_ylabel("y[mm]", fontsize=7)
        ax.set_zlabel("z[mm]", fontsize=7)

        # position vs time
        ax = fig.add_subplot(rows, 3, r * 3 + 2)
        for k in range(3):
            ax.plot(t, p[:, k], color=colors[k], lw=1, label=axis_names[k])
        ax.set_title(f"sub {r}: position (Vicon frame)", fontsize=9)
        ax.set_xlabel("t [s]", fontsize=8); ax.set_ylabel("pos [mm]", fontsize=8)
        ax.legend(fontsize=7, ncol=3)

        # orientation vs time
        ax = fig.add_subplot(rows, 3, r * 3 + 3)
        if s["R"] is not None and s["R"].ndim == 3:
            eul = np.degrees(rotmat_to_euler_xyz(s["R"]))
            for k in range(3):
                ax.plot(t, eul[:, k], color=colors[k], lw=1, label=ang_names[k])
            ax.legend(fontsize=7, ncol=3)
        else:
            ax.text(0.5, 0.5, "no orientation", ha="center", transform=ax.transAxes)
        ax.set_title(f"sub {r}: orientation", fontsize=9)
        ax.set_xlabel("t [s]", fontsize=8); ax.set_ylabel("angle [deg]", fontsize=8)

    fig.suptitle(f"{trial}: Vicon sub-trials (Vicon frame)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    out = args.out or d / f"{trial}_vicon_subtrials.png"
    fig.savefig(out, dpi=130)
    print(f"Saved: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
