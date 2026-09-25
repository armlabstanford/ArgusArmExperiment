#!/usr/bin/env python3
"""Discretize the VIO odometry into sub-trials and view per-sub-trial behaviour.

Usage:
    python3 discretize_vio.py TRIAL_NAME
    python3 discretize_vio.py path/to/TRIAL_odometry.npz [--out fig.png] [--show]

Sub-trials are found from the VIO's *own* rest-dwells: the object sits at a home
/ rest pose for ~1 s between motions, so a stationary gap splits one sub-trial
from the next.  Each sub-trial gets its own row of panels:

    [ 3D path ]   [ x/y/z vs time ]   [ speed vs time ]

Panels auto-scale independently so each sub-trial is legible even when the VIO
scale differs wildly between them.  Diverged samples (non-finite, |pos| huge, or
negative pose covariance) are drawn in red so filter blow-ups are obvious.
"""

import sys
import argparse
from pathlib import Path

import numpy as np

from align_data import load_vio   # divergence-aware VIO loader

# ---- rest/dwell tunables --------------------------------------------------
SPEED_THRESH_MS = 0.03   # below this speed => stationary (at rest/home)
REST_S = 1.0             # nominal rest duration between motions
MIN_DWELL_S = 0.5        # a stationary gap this long splits two sub-trials
MIN_MOVE_S = 2.0         # ignore motion bursts shorter than this
SMOOTH_S = 0.10          # speed smoothing window


def resolve_odom(arg: str) -> tuple[Path, str]:
    p = Path(arg)
    if p.suffix == ".npz":
        return p, p.stem.replace("_odometry", "")
    here = Path(__file__).resolve().parent
    return here / "data" / arg / f"{arg}_odometry.npz", arg


def speed_of(t, pos):
    dt = float(np.median(np.diff(t)))
    v = np.linalg.norm(np.gradient(pos, axis=0) / dt, axis=1)
    w = max(1, int(round(SMOOTH_S / dt)))
    return np.convolve(v, np.ones(w) / w, mode="same"), dt


def fill_short_gaps(mask, max_gap):
    """Set interior False runs shorter than max_gap samples to True."""
    m = mask.astype(bool).copy()
    n = len(m)
    i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and j < n and (j - i) < max_gap:   # interior short gap
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def segment_by_dwell(t, pos, valid, dt, speed):
    """Return list of (start, end) sub-trials split by >=MIN_DWELL_S rests."""
    moving = (speed > SPEED_THRESH_MS) & valid
    moving = fill_short_gaps(moving, int(round(MIN_DWELL_S / dt)))
    e = np.diff(moving.astype(int))
    s = np.where(e == 1)[0] + 1
    en = np.where(e == -1)[0] + 1
    if moving[0]:
        s = np.r_[0, s]
    if moving[-1]:
        en = np.r_[en, len(moving)]
    return [(a, b) for a, b in zip(s, en) if (b - a) * dt >= MIN_MOVE_S]


def main():
    global SPEED_THRESH_MS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("odom", help="TRIAL_NAME or path to *_odometry.npz")
    ap.add_argument("--out", type=Path, help="figure path (default next to npz)")
    ap.add_argument("--show", action="store_true", help="show interactively")
    ap.add_argument("--speed-thresh", type=float, default=SPEED_THRESH_MS)
    args = ap.parse_args()

    SPEED_THRESH_MS = args.speed_thresh

    odom_npz, trial = resolve_odom(args.odom)
    if not odom_npz.is_file():
        sys.exit(f"odometry npz not found: {odom_npz}")

    vio = load_vio(odom_npz)
    t, pos, ok = vio["t"], vio["pos_m"], vio["ok"]
    vend = vio["valid_end"]
    valid = np.zeros(len(t), bool)
    valid[:vend] = True

    sp, dt = speed_of(t, pos)
    subs = segment_by_dwell(t, pos, valid, dt, sp)

    print(f"Trial: {trial}")
    print(f"VIO: {len(t)} msgs, {t[-1]:.1f}s @ ~{1/dt:.0f}Hz")
    print(f"Valid (pre-divergence) window: 0..{t[vend-1] if vend>0 else 0:.2f}s "
          f"({int(ok.sum())}/{len(ok)} samples ok)")
    print(f"Sub-trials found (VIO rest-dwells, rest~{REST_S}s): {len(subs)}")
    for i, (a, b) in enumerate(subs):
        rng = np.linalg.norm(pos[a:b] - pos[a], axis=1).max()
        print(f"  [{i}] {t[a]:6.2f} - {t[b-1]:6.2f}s  ({(b-a)*dt:4.1f}s)  "
              f"max disp {rng:.3f} m")
    if not subs:
        print("  (none — VIO likely diverged before any clean rest-bounded motion)")

    make_plot(t, pos, sp, ok, subs, trial, args, odom_npz)


def make_plot(t, pos, sp, ok, subs, trial, args, odom_npz):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = max(1, len(subs))
    fig = plt.figure(figsize=(15, 3.2 * rows))
    axis_names = ("x", "y", "z")
    colors = ("tab:red", "tab:green", "tab:blue")

    if not subs:                      # fallback: whole trace
        subs = [(0, len(t))]

    for r, (a, b) in enumerate(subs):
        tt = t[a:b] - t[a]
        p = pos[a:b]
        bad = ~ok[a:b]

        # --- 3D path ---
        ax = fig.add_subplot(rows, 3, r * 3 + 1, projection="3d")
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="0.5", lw=1)
        if bad.any():
            ax.scatter(p[bad, 0], p[bad, 1], p[bad, 2], c="r", s=8, label="diverged")
        ax.scatter(*p[0], c="k", s=25, marker="o")
        ax.set_title(f"sub {r}: 3D path  ({t[a]:.1f}-{t[b-1]:.1f}s)", fontsize=9)
        ax.set_xlabel("x[m]", fontsize=7); ax.set_ylabel("y[m]", fontsize=7)
        ax.set_zlabel("z[m]", fontsize=7)

        # --- position vs time ---
        ax = fig.add_subplot(rows, 3, r * 3 + 2)
        for k in range(3):
            ax.plot(tt, p[:, k], color=colors[k], lw=1, label=axis_names[k])
        _shade_bad(ax, tt, bad)
        ax.set_title(f"sub {r}: position", fontsize=9)
        ax.set_xlabel("t [s]", fontsize=8); ax.set_ylabel("pos [m]", fontsize=8)
        ax.legend(fontsize=7, ncol=3)

        # --- speed vs time ---
        ax = fig.add_subplot(rows, 3, r * 3 + 3)
        ax.plot(tt, sp[a:b], color="k", lw=1)
        ax.axhline(SPEED_THRESH_MS, color="tab:orange", ls="--", lw=0.8,
                   label=f"rest thr {SPEED_THRESH_MS}")
        _shade_bad(ax, tt, bad)
        ax.set_title(f"sub {r}: speed", fontsize=9)
        ax.set_xlabel("t [s]", fontsize=8); ax.set_ylabel("speed [m/s]", fontsize=8)
        ax.legend(fontsize=7)
        # ax.set_yscale("log")
        ax.set_ylim(bottom=1e-4, top=5)

    fig.suptitle(f"{trial}: VIO discretized into sub-trials", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = args.out or odom_npz.with_name(f"{trial}_vio_discretized.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved: {out}")
    if args.show:
        plt.show()


def _shade_bad(ax, tt, bad):
    if bad.any():
        ax.axvspan(tt[bad][0], tt[-1], color="r", alpha=0.08, label="diverged")


if __name__ == "__main__":
    main()
