"""
Evaluate and robustly re-solve a hand-eye calibration from hand_eye_result.raw.npz.

Checks
  1. Board consistency: the fixed board's pose in the base frame,
     T_base_board_i = T_base_ee_i @ X @ T_cam_board_i, should be identical for
     every capture. Its spread (mm / deg) is the real accuracy metric.
  2. Method agreement: TSAI / PARK / HORAUD / DANIILIDIS should agree to ~mm/deg.
  3. Convention check: also tries inverted EE poses (in case FK returned T_ee_base).
  4. Rotation diversity of the arm motions (ill-conditioning check).
  5. Robust re-solve: drops the worst capture until all residuals are within
     tolerance, then tests each dropped capture for a 180-deg checkerboard
     corner-order flip and re-admits it if the flip explains it.

Usage:
    python argus_experiment/calibration/evaluate_handeye.py \
        argus_experiment/calibration/hand_eye_result.raw.npz \
        --cols 10 --rows 7 --square 0.036
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def se3(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).ravel(); return T


def inv(T):
    Ti = np.eye(4); Ti[:3, :3] = T[:3, :3].T; Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]; return Ti


def ang(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def solve(Tbe, Tct, method="PARK"):
    R, t = cv2.calibrateHandEye([T[:3, :3] for T in Tbe], [T[:3, 3] for T in Tbe],
                                [T[:3, :3] for T in Tct], [T[:3, 3] for T in Tct],
                                method=METHODS[method])
    return se3(R, t)


def consistency(Tbe, Tct, X):
    """Per-capture deviation of T_base_board from its robust mean: (mm, deg)."""
    Tbt = np.array([a @ X @ b for a, b in zip(Tbe, Tct)])
    p_ref = np.median(Tbt[:, :3, 3], axis=0)
    U, _, Vt = np.linalg.svd(Tbt[:, :3, :3].sum(0))
    R_ref = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    mm = 1e3 * np.linalg.norm(Tbt[:, :3, 3] - p_ref, axis=1)
    deg = np.array([ang(R_ref.T @ T[:3, :3]) for T in Tbt])
    return mm, deg


def summarize(tag, Tbe, Tct):
    print(f"\n--- {tag} ({len(Tbe)} poses) ---")
    Xs = {}
    for name in METHODS:
        X = solve(Tbe, Tct, name)
        mm, deg = consistency(Tbe, Tct, X)
        Xs[name] = X
        print(f"  {name:11s} t={X[:3, 3].round(4)}  board spread: "
              f"median {np.median(mm):6.1f} mm / {np.median(deg):5.2f} deg, "
              f"max {mm.max():7.1f} mm / {deg.max():6.2f} deg")
    names = list(Xs)
    dt = max(1e3 * np.linalg.norm(Xs[a][:3, 3] - Xs[b][:3, 3]) for a in names for b in names)
    dr = max(ang(Xs[a][:3, :3].T @ Xs[b][:3, :3]) for a in names for b in names)
    print(f"  method disagreement: {dt:.1f} mm / {dr:.2f} deg  (good: < ~3 mm / < ~0.5 deg)")
    return Xs


def rotation_diversity(Tbe):
    R0 = Tbe[0][:3, :3]
    rel = [ang(R0.T @ T[:3, :3]) for T in Tbe]
    axes = []
    for i in range(len(Tbe)):
        for j in range(i + 1, len(Tbe)):
            Rij = Tbe[i][:3, :3].T @ Tbe[j][:3, :3]
            if ang(Rij) > 5:
                rv, _ = cv2.Rodrigues(Rij)
                axes.append(rv.ravel() / np.linalg.norm(rv))
    sv = np.linalg.svd(np.array(axes), compute_uv=False) if axes else np.zeros(3)
    sv = sv / (sv[0] + 1e-12)
    print(f"\nEE rotation diversity: max angle from first pose {max(rel):.1f} deg; "
          f"rotation-axis spread (normalized singular values) {sv.round(2)}")
    print("  (want max angle >~ 30-45 deg and all three singular values well above ~0.2)")


def _screw(T):
    """(angle_deg, pitch_m, axis) of a rigid motion."""
    rv, _ = cv2.Rodrigues(T[:3, :3])
    th = np.linalg.norm(rv)
    axis = rv.ravel() / th if th > 1e-9 else np.zeros(3)
    return np.degrees(th), float(axis @ T[:3, 3]), axis


def screw_invariants(Tbe, Tct, min_deg=10.0, min_pitch=0.005):
    """
    X-independent check. For captures i, j:
        A = inv(Tbe_i) @ Tbe_j      (EE motion)
        B = Tct_i @ inv(Tct_j)      (camera motion)
    satisfy A X = X B, so A and B MUST share rotation angle and screw pitch.
    """
    n = len(Tbe)

    def angle_err(lag):
        errs = []
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = i + lag, j + lag
                if not (0 <= ci < n and 0 <= cj < n):
                    continue
                a, _, _ = _screw(inv(Tbe[i]) @ Tbe[j])
                b, _, _ = _screw(Tct[ci] @ inv(Tct[cj]))
                if a > min_deg:
                    errs.append(abs(a - b))
        return np.array(errs)

    print("\nX-independent check (arm motion vs camera motion between capture pairs):")
    e0 = angle_err(0)
    print(f"  rotation-angle mismatch: median {np.median(e0):.2f} deg, "
          f"90th pct {np.percentile(e0, 90):.2f} deg, {100 * np.mean(e0 > 2):.0f}% of pairs > 2 deg")
    print("    (good data: median < ~0.3 deg)")
    lags = {k: np.median(angle_err(k)) for k in range(-3, 4)}
    print("  pairing-lag test (median mismatch if camera capture is shifted by k):  "
          + "  ".join(f"k={k:+d}: {v:.2f}" for k, v in lags.items()))

    # per-capture badness and translation-scale (pitch) ratio
    bad = np.zeros(n)
    cnt = np.zeros(n)
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            a, pa, _ = _screw(inv(Tbe[i]) @ Tbe[j])
            b, pb, _ = _screw(Tct[i] @ inv(Tct[j]))
            if a <= min_deg:
                continue
            e = abs(a - b) > 2
            bad[[i, j]] += e; cnt[[i, j]] += 1
            if not e and abs(pa) > min_pitch:
                ratios.append(pb / pa)
    if ratios:
        r = np.array(ratios)
        print(f"  screw-pitch ratio camera/arm over angle-consistent pairs: median {np.median(r):.3f} "
              f"(IQR {np.percentile(r, 25):.2f}-{np.percentile(r, 75):.2f}, n={len(r)})")
        print("    (must be ~1.0; ~2.0 or ~0.5 => camera translations scaled: intrinsics or --square)")
    frac = np.divide(bad, cnt, out=np.zeros(n), where=cnt > 0)
    worst = np.argsort(-frac)[:10]
    print("  captures most often inconsistent (capture #: % of its pairs bad): "
          + ", ".join(f"#{i + 1}:{100 * frac[i]:.0f}%" for i in worst))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--cols", type=int, default=10)
    p.add_argument("--rows", type=int, default=7)
    p.add_argument("--square", type=float, default=0.036)
    p.add_argument("--tol-mm", type=float, default=5.0)
    p.add_argument("--tol-deg", type=float, default=1.0)
    p.add_argument("--min-poses", type=int, default=10)
    p.add_argument("--method", default="PARK", choices=list(METHODS))
    p.add_argument("--out", default=None,
                   help="Write cleaned result YAML (default: <npz dir>/hand_eye_result_clean.yaml)")
    a = p.parse_args()

    d = np.load(a.npz)
    Tbe = np.array([se3(R, t) for R, t in zip(d["R_g2b"], d["t_g2b"])])
    Tct = np.array([se3(R, t) for R, t in zip(d["R_t2c"], d["t_t2c"])])
    n = len(Tbe)
    print(f"Loaded {n} captures from {a.npz}")
    dist = np.linalg.norm(Tct[:, :3, 3], axis=1)
    print(f"Board distance from camera: {dist.min():.3f}-{dist.max():.3f} m")

    summarize("raw data, FK as T_base_ee", Tbe, Tct)
    summarize("convention check: FK inverted (T_ee_base)", np.array([inv(T) for T in Tbe]), Tct)
    rotation_diversity(Tbe)
    screw_invariants(Tbe, Tct)

    # --- robust re-solve: drop the worst capture until all are within tolerance
    keep = list(range(n))
    while len(keep) > a.min_poses:
        X = solve(Tbe[keep], Tct[keep], a.method)
        mm, deg = consistency(Tbe[keep], Tct[keep], X)
        score = mm / a.tol_mm + deg / a.tol_deg
        w = int(np.argmax(score))
        if mm[w] <= a.tol_mm and deg[w] <= a.tol_deg:
            break
        keep.pop(w)
    dropped = sorted(set(range(n)) - set(keep))

    # --- test dropped captures for a 180-deg corner-order flip about the board normal
    W, H = (a.cols - 1) * a.square, (a.rows - 1) * a.square
    T_flip = se3(np.diag([-1.0, -1.0, 1.0]), [W, H, 0.0])  # board frame at opposite corner
    X = solve(Tbe[keep], Tct[keep], a.method)
    flipped = []
    for i in dropped:
        mm, deg = consistency(Tbe[keep + [i]], np.concatenate([Tct[keep], [Tct[i] @ T_flip]]), X)
        if mm[-1] <= a.tol_mm and deg[-1] <= a.tol_deg:
            flipped.append(i)
    for i in flipped:
        Tct[i] = Tct[i] @ T_flip
    keep = sorted(keep + flipped)
    still_bad = sorted(set(dropped) - set(flipped))

    print(f"\nRobust re-solve ({a.method}, tol {a.tol_mm} mm / {a.tol_deg} deg):")
    print(f"  kept {len(keep)}/{n}; repaired {len(flipped)} flipped detections "
          f"(capture #s {[i + 1 for i in flipped]})")
    print(f"  rejected {len(still_bad)} (capture #s {[i + 1 for i in still_bad]}) "
          "— check these in handeye_captures/")
    Xs = summarize("cleaned data", Tbe[keep], Tct[keep])

    X = Xs[a.method]
    out = Path(a.out) if a.out else Path(a.npz).parent / "hand_eye_result_clean.yaml"
    mm, deg = consistency(Tbe[keep], Tct[keep], X)
    doc = {
        "method": a.method, "num_poses": len(keep), "source": str(Path(a.npz).resolve()),
        "kept_capture_numbers": [i + 1 for i in keep],
        "flipped_capture_numbers": [i + 1 for i in flipped],
        "rejected_capture_numbers": [i + 1 for i in still_bad],
        "board_spread": {"median_mm": float(np.median(mm)), "max_mm": float(mm.max()),
                         "median_deg": float(np.median(deg)), "max_deg": float(deg.max())},
        "T_ee_cam": {"matrix": X.tolist()},
        "all_methods": {k: v.tolist() for k, v in Xs.items()},
    }
    with open(out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"\nSaved cleaned result -> {out}  (use with --handeye {out})")


if __name__ == "__main__":
    main()