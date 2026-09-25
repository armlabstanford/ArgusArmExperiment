#!/usr/bin/env python3
"""Align Vicon (c3d) and VIO (argus) trajectories and split into sub-trials.

Usage:
    python3 align_data.py TRIAL_NAME
    python3 align_data.py path/to/TRIAL_pose.npz path/to/TRIAL_odometry.npz

Method
------
The two streams share no clock (Vicon c3d stores only frame numbers @ 100 Hz;
VIO odometry is ROS/unix time @ ~20 Hz) and have different durations, so they
cannot be aligned by timestamp.  Instead:

  1. Segment each stream into sub-trials by detecting *home dwells* — the object
     starts at a rest "home" pose and returns to it between motions.  Each
     excursion away from home is one sub-trial.  Vicon uses distance-to-home +
     stationarity; VIO (which drifts, so absolute position is unreliable) uses
     stationarity only.
  2. Pair sub-trials by order (Vicon #i <-> VIO #i).
  3. Within each pair, resample both onto a common normalised-time grid and solve
     a similarity transform (Umeyama sim3: scale + R + t) that maps VIO into the
     Vicon mm world frame.

The VIO EKF can diverge (position -> huge, covariance diagonals negative).  Such
samples are detected and excluded; sub-trials whose VIO window is invalid are
still saved (Vicon-only) so the segmentation is always available.

Output
------
    data/TRIAL/TRIAL_subtrials/subtrial_00.npz ...   (per sub-trial)
    data/TRIAL/TRIAL_subtrials/summary.txt
    data/TRIAL/TRIAL_subtrials/overview.png
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# ---- tunables (overridable via CLI) ---------------------------------------
HOME_WINDOW_S = 1.0      # initial window used to estimate the home pose
HOME_RADIUS_M = 0.05     # within this of home counts as "at home" (Vicon)
SPEED_THRESH_MS = 0.03   # below this speed counts as stationary
MIN_MOVE_S = 2.0         # ignore excursions shorter than this
MIN_PEAK_DISP_M = 0.05   # excursion must reach at least this far from home (Vicon)
SMOOTH_S = 0.07          # speed smoothing window
RESAMPLE_N = 200         # samples per sub-trial on the normalised-time grid
VIO_POS_LIMIT_M = 1e3    # |position| above this => diverged
# A sim3 fit is trusted only if VIO(m)->Vicon(mm) scale and residual are sane.
TRUST_SCALE_RANGE = (300.0, 3000.0)   # ~1000 expected (metric VIO)
TRUST_RMSE_MM = 50.0


# ---- io -------------------------------------------------------------------
def resolve(args):
    """Return (pose_npz, odom_npz, out_dir, trial_name)."""
    here = Path(__file__).resolve().parent
    if len(args) == 2:
        pose, odom = Path(args[0]), Path(args[1])
        trial = pose.stem.replace("_pose", "")
        return pose, odom, pose.parent, trial
    trial = args[0]
    base = Path(trial)
    if base.suffix == ".npz":                     # a pose npz was passed
        pose = base
        odom = base.with_name(base.name.replace("_pose", "_odometry"))
        return pose, odom, base.parent, base.stem.replace("_pose", "")
    d = here / "data" / trial
    return (d / f"{trial}_pose.npz", d / f"{trial}_odometry.npz", d, trial)


def fill_nan(p):
    """Linearly interpolate NaN gaps, column-wise."""
    p = np.asarray(p, float).copy()
    for k in range(p.shape[1]):
        c = p[:, k]
        m = np.isnan(c)
        if m.any() and (~m).any():
            c[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), c[~m])
    return p


def load_vicon(pose_npz):
    d = np.load(pose_npz)
    fr = float(d["frame_rate"]) if "frame_rate" in d.files else 100.0
    pos_mm = fill_nan(d["centroid"])
    t = np.arange(len(pos_mm)) / fr
    R = np.asarray(d["R"], float) if "R" in d.files else None
    return {"t": t, "pos_mm": pos_mm, "pos_m": pos_mm / 1000.0, "R": R, "rate": fr}


def load_vio(odom_npz):
    d = np.load(odom_npz)
    t = np.asarray(d["t_rel"], float)
    pos = np.asarray(d["position"], float)
    R = None
    if "orientation_wxyz" in d.files:
        R = quat_wxyz_to_R(np.asarray(d["orientation_wxyz"], float))
    # Divergence: non-finite, huge position, or negative covariance diagonal.
    ok = np.isfinite(pos).all(axis=1) & (np.linalg.norm(pos, axis=1) < VIO_POS_LIMIT_M)
    if "pose_covariance" in d.files:
        diag = np.diagonal(np.asarray(d["pose_covariance"], float), axis1=1, axis2=2)
        ok &= (diag >= 0).all(axis=1)
    # Trust only the valid prefix (VIO does not recover after divergence).
    valid_end = int(np.argmin(ok)) if not ok.all() else len(ok)
    if ok.all():
        valid_end = len(ok)
    elif not ok[0]:
        valid_end = 0
    return {"t": t, "pos_m": pos, "R": R, "ok": ok, "valid_end": valid_end}


def quat_wxyz_to_R(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ---- segmentation ---------------------------------------------------------
def speed(t, pos):
    dt = np.median(np.diff(t))
    v = np.linalg.norm(np.gradient(pos, axis=0) / dt, axis=1)
    w = max(1, int(round(SMOOTH_S / dt)))
    return np.convolve(v, np.ones(w) / w, mode="same"), dt


def _runs(moving, dt):
    e = np.diff(moving.astype(int))
    s = np.where(e == 1)[0] + 1
    en = np.where(e == -1)[0] + 1
    if moving[0]:
        s = np.r_[0, s]
    if moving[-1]:
        en = np.r_[en, len(moving)]
    return [(a, b) for a, b in zip(s, en) if (b - a) * dt >= MIN_MOVE_S]


def segment(t, pos, use_home, valid=None):
    """Return list of (start_idx, end_idx) excursions away from home."""
    v, dt = speed(t, pos)
    moving = v > SPEED_THRESH_MS
    home = None
    if use_home:
        n0 = max(1, int(HOME_WINDOW_S / dt))
        home = np.median(pos[:n0], axis=0)
        moving |= np.linalg.norm(pos - home, axis=1) > HOME_RADIUS_M
    if valid is not None:
        moving &= valid
    runs = _runs(moving, dt)
    if home is not None:   # drop jitter that never actually leaves home
        runs = [(a, b) for a, b in runs
                if np.linalg.norm(pos[a:b] - home, axis=1).max() >= MIN_PEAK_DISP_M]
    return runs


# ---- alignment ------------------------------------------------------------
def umeyama_sim3(X, Y):
    """Find s, R, t with Y ~= s*R@X + t (row-vector point sets). Returns rmse too."""
    n = len(X)
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    Sigma = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    varX = (Xc ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / varX)
    t = muY - s * R @ muX
    resid = Y - (s * (X @ R.T) + t)
    rmse = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return s, R, t, rmse


def resample(t, arr, n):
    tt = np.linspace(t[0], t[-1], n)
    out = np.column_stack([np.interp(tt, t, arr[:, k]) for k in range(arr.shape[1])])
    return tt, out


def sample_R(t, R, tt):
    """Nearest-index rotation sampling onto times tt."""
    if R is None:
        return None
    idx = np.clip(np.searchsorted(t, tt), 0, len(t) - 1)
    return R[idx]


# ---- main -----------------------------------------------------------------
def main():
    global HOME_RADIUS_M, SPEED_THRESH_MS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="TRIAL_NAME, or POSE.npz [ODOM.npz]")
    ap.add_argument("--home-radius", type=float, default=HOME_RADIUS_M)
    ap.add_argument("--speed-thresh", type=float, default=SPEED_THRESH_MS)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    HOME_RADIUS_M = args.home_radius
    SPEED_THRESH_MS = args.speed_thresh

    pose_npz, odom_npz, out_base, trial = resolve(args.paths)
    if not pose_npz.is_file():
        sys.exit(f"pose npz not found: {pose_npz}")
    have_vio = odom_npz.is_file()

    vic = load_vicon(pose_npz)
    vio = load_vio(odom_npz) if have_vio else None

    out_dir = out_base / f"{trial}_subtrials"
    out_dir.mkdir(exist_ok=True)

    lines = [f"Trial: {trial}", f"Vicon: {pose_npz.name}  ({len(vic['t'])} frames @ {vic['rate']}Hz, {vic['t'][-1]:.1f}s)"]

    # --- VIO health report ---
    vio_seg = []
    if vio is not None:
        n_ok = int(vio["ok"].sum())
        vend = vio["valid_end"]
        t_valid = vio["t"][vend - 1] if vend > 0 else 0.0
        lines.append(f"VIO:   {odom_npz.name}  ({len(vio['t'])} msgs, {vio['t'][-1]:.1f}s)")
        lines.append(f"VIO valid samples: {n_ok}/{len(vio['ok'])}; trusted window: "
                     f"0..{t_valid:.2f}s (first divergence at index {vend})")
        if vend > 5:
            valid_mask = np.zeros(len(vio["t"]), bool)
            valid_mask[:vend] = True
            vio_seg = segment(vio["t"], vio["pos_m"], use_home=False, valid=valid_mask)
    else:
        lines.append("VIO:   (odometry npz not found — Vicon-only segmentation)")

    # --- segment Vicon (master) ---
    vic_seg = segment(vic["t"], vic["pos_m"], use_home=True)
    lines.append("")
    lines.append(f"Vicon sub-trials: {len(vic_seg)}")
    for i, (a, b) in enumerate(vic_seg):
        lines.append(f"  [{i}] {vic['t'][a]:6.2f} - {vic['t'][b-1]:6.2f}s  ({(b-a)/vic['rate']:.1f}s)")
    lines.append(f"VIO excursions in valid window: {len(vio_seg)}")

    n_pairs = min(len(vic_seg), len(vio_seg))
    if have_vio and n_pairs < len(vic_seg):
        lines.append(f"NOTE: only {n_pairs} sub-trial(s) have usable VIO; the rest are "
                     f"saved Vicon-only (VIO diverged / no matching excursion).")

    # --- per sub-trial ---
    summaries = []
    for i, (a, b) in enumerate(vic_seg):
        rec = dict(
            index=i,
            vicon_t=vic["t"][a:b],
            vicon_pos_mm=vic["pos_mm"][a:b],
            vicon_R=(vic["R"][a:b] if vic["R"] is not None else np.empty((0,))),
            vicon_interval_s=np.array([vic["t"][a], vic["t"][b - 1]]),
        )
        aligned = False
        if i < n_pairs:
            va, vb = vio_seg[i]
            vt, vp = vio["t"][va:vb], vio["pos_m"][va:vb]
            # common normalised-time grid -> correspondences
            _, vic_rs = resample(vic["t"][a:b], vic["pos_mm"][a:b], RESAMPLE_N)
            tt, vio_rs = resample(vt, vp, RESAMPLE_N)
            s, R, t, rmse = umeyama_sim3(vio_rs, vic_rs)      # VIO(m) -> Vicon(mm)
            vio_aligned = (s * (vp @ R.T) + t)
            trusted = (TRUST_SCALE_RANGE[0] <= s <= TRUST_SCALE_RANGE[1]
                       and rmse <= TRUST_RMSE_MM)
            rec.update(
                vio_t=vt, vio_pos_m=vp,
                vio_R=(sample_R(vio["t"], vio["R"], vt) if vio["R"] is not None else np.empty((0,))),
                vio_pos_aligned_mm=vio_aligned,
                vio_interval_s=np.array([vt[0], vt[-1]]),
                align_scale=s, align_R=R, align_t=t, align_rmse_mm=rmse,
                align_trusted=trusted,
                grid_norm=np.linspace(0, 1, RESAMPLE_N),
                vicon_pos_rs_mm=vic_rs, vio_pos_rs_aligned_mm=(s * (vio_rs @ R.T) + t),
            )
            aligned = True
            flag = "OK" if trusted else "UNTRUSTED (VIO unreliable)"
            lines.append(f"  aligned [{i}]: scale={s:.1f}  rmse={rmse:.1f}mm  -> {flag}")
        np.savez_compressed(out_dir / f"subtrial_{i:02d}.npz", vio_aligned=aligned, **rec)
        summaries.append((i, aligned))

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {len(vic_seg)} sub-trials to {out_dir}/")

    if not args.no_plot:
        make_plot(vic, vio, vic_seg, vio_seg, n_pairs, out_dir, trial)


def make_plot(vic, vio, vic_seg, vio_seg, n_pairs, out_dir, trial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 6))

    # Left: Vicon trajectory coloured by sub-trial.
    ax = fig.add_subplot(121, projection="3d")
    p = vic["pos_mm"]
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color="0.8", lw=0.5)
    cmap = plt.get_cmap("tab10")
    for i, (a, b) in enumerate(vic_seg):
        seg = p[a:b]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=cmap(i % 10), lw=2, label=f"sub {i}")
    ax.set_title(f"{trial}: Vicon sub-trials")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    ax.legend(loc="upper left", fontsize=8)

    # Right: aligned overlay for the first aligned sub-trial (if any).
    ax2 = fig.add_subplot(122, projection="3d")
    if n_pairs > 0 and vio is not None:
        a, b = vic_seg[0]
        _, vic_rs = resample(vic["t"][a:b], vic["pos_mm"][a:b], RESAMPLE_N)
        va, vb = vio_seg[0]
        _, vio_rs = resample(vio["t"][va:vb], vio["pos_m"][va:vb], RESAMPLE_N)
        s, R, t, rmse = umeyama_sim3(vio_rs, vic_rs)
        va_al = s * (vio_rs @ R.T) + t
        ax2.plot(vic_rs[:, 0], vic_rs[:, 1], vic_rs[:, 2], "b", lw=2, label="Vicon")
        ax2.plot(va_al[:, 0], va_al[:, 1], va_al[:, 2], "r--", lw=2, label="VIO->Vicon")
        ax2.set_title(f"sub 0 aligned (rmse {rmse:.1f}mm)")
        ax2.legend()
    else:
        ax2.text2D(0.5, 0.5, "no VIO-alignable sub-trial\n(VIO diverged)",
                   ha="center", transform=ax2.transAxes)
        ax2.set_title("aligned overlay")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]"); ax2.set_zlabel("z [mm]")

    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=140)
    print(f"Saved plot: {out_dir/'overview.png'}")


if __name__ == "__main__":
    main()
