"""
Eye-in-hand hand-eye calibration for the YAM arm with a link_6-mounted camera.

Solves X = T_ee_cam (camera pose in the end-effector / grasp_site frame) using
cv2.calibrateHandEye. The checkerboard stays FIXED; you jog the compliant arm by
hand to ~15-20 varied poses and capture at each. Rotation variety matters most —
tilt/roll the wrist a lot, don't just translate.

Intrinsics are read from a Kalibr camchain YAML (default:
arm_calib_dynamic-camchain-imucam.yaml, camera 'cam0'). That calibration is
camera_model=pinhole + distortion_model=equidistant, i.e. the Kannala-Brandt
KB4 model, which is exactly OpenCV's cv2.fisheye model (k1..k4). Detected
corners are undistorted with cv2.fisheye before solvePnP. The legacy
{camera: {fx, fy, cx, cy, distortion, ...}} format is still accepted.

The intrinsics are only valid at the resolution they were calibrated at
(640x400 for cam0). The Arducam OV9281 has no 640x400 mode, so capture at
--capture-res 1280x800 and either K is rescaled 2x (default; distortion is
scale-invariant), or add --resize-to-calib to downscale frames to 640x400 and
use K as calibrated.
The script aborts if the camera doesn't deliver the requested resolution.

Workflow:
  1. Tape the checkerboard down where the camera can see it.
  2. Run this script (arm becomes compliant — gravity-comp, zero stiffness).
  3. At each pose: move the arm by hand, hold steady, press ENTER to capture.
     'u' undo last, 'q' finish & solve.
  4. Result saved to --out (X = T_ee_cam) plus a ready-to-paste MuJoCo camera_site.

Board convention: --cols/--rows are INNER corners = (squares per side - 1).
This board is 8x11 squares @ 36 mm -> 10x7 inner corners (matches the "10x7"
printed on its side).

Usage:
    python argus_experiment/calibration/hand_eye_calibration.py \
        --channel can0 --device /dev/video3 --capture-res 1280x800 \
        --intrinsics argus_experiment/calibration/arm_calib_dynamic-camchain-imucam.yaml \
        --cam-key cam0 --cols 10 --rows 7 --square 0.036
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "robots_realtime" / "dependencies" / "i2rt"))
sys.path.insert(0, str(_REPO / "robots_realtime"))  # for robots_realtime.sensors

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType

N_ARM = 6

# The camera module is always used with no_gripper, whose grasp_site sits at
# pos "0 0 0" quat "0 1 0 0" relative to link_6. Rather than hardcode it, the
# link_6-referenced camera_site is composed from the live model at solve time.


# ---------------------------------------------------------------------------
# small SO3 / SE3 helpers
# ---------------------------------------------------------------------------
def wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def R_to_wxyz(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-8:  # fallback for 180-deg rotations
        w = 1e-8
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def make_SE3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


# ---------------------------------------------------------------------------
# intrinsics
# ---------------------------------------------------------------------------
_FISHEYE_MODELS = ("equidistant", "fisheye", "kb4")
_RADTAN_MODELS = ("radtan", "plumb_bob", "radial-tangential")


def load_intrinsics(path: Path, cam_key: str = "cam0"):
    """
    Load K and distortion from a Kalibr camchain YAML (or the legacy format).

    Kalibr camchain layout (per camera, e.g. cam0):
        camera_model:      pinhole
        intrinsics:        [fu, fv, pu, pv]
        distortion_model:  equidistant | radtan | none
        distortion_coeffs: [k1, k2, k3, k4]   (equidistant)
                           [k1, k2, r1, r2]   (radtan == OpenCV k1,k2,p1,p2)
        resolution:        [width, height]

    pinhole + equidistant is Kannala-Brandt KB4 == cv2.fisheye.

    Returns (K, D, is_fisheye, (width, height), info_dict).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "camera" in cfg:  # legacy single-camera format
        cam = cfg["camera"]
        fx, fy, cx, cy = (float(cam[k]) for k in ("fx", "fy", "cx", "cy"))
        dist = np.asarray(cam["distortion"], dtype=float)
        dist_model = cam.get("distortion_model", "equidistant").lower()
        cam_model = "pinhole"
        w, h = int(cam["width"]), int(cam["height"])
        if dist_model in _FISHEYE_MODELS:
            dist = dist[:4]
        src_key = "camera"
    else:  # Kalibr camchain
        if cam_key not in cfg:
            raise KeyError(f"{cam_key!r} not found in {path} (available: {list(cfg)})")
        cam = cfg[cam_key]
        cam_model = cam.get("camera_model", "pinhole").lower()
        if cam_model != "pinhole":
            raise ValueError(
                f"Kalibr camera_model {cam_model!r} is not supported "
                "(only 'pinhole' with equidistant/radtan/none distortion).")
        fx, fy, cx, cy = (float(v) for v in cam["intrinsics"])
        dist = np.asarray(cam.get("distortion_coeffs", []), dtype=float)
        dist_model = cam.get("distortion_model", "radtan").lower()
        w, h = (int(v) for v in cam["resolution"])
        src_key = cam_key

    if dist_model in _FISHEYE_MODELS:
        if dist.size != 4:
            raise ValueError(f"equidistant model needs 4 coeffs, got {dist.size}")
        is_fisheye = True
        D = dist.reshape(4, 1)
    elif dist_model in _RADTAN_MODELS:
        is_fisheye = False
        D = dist.reshape(-1, 1)
    elif dist_model == "none":
        is_fisheye = False
        D = np.zeros((4, 1))
    else:
        raise ValueError(f"Unsupported distortion_model {dist_model!r}")

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])
    info = {
        "source": str(Path(path).resolve()),
        "cam_key": src_key,
        "camera_model": cam_model,
        "distortion_model": dist_model,
        "intrinsics_fx_fy_cx_cy": [fx, fy, cx, cy],
        "distortion_coeffs": D.ravel().tolist(),
        "resolution_wh": [w, h],
    }
    return K, D, is_fisheye, (w, h), info


def scale_intrinsics(K: np.ndarray, src_wh, dst_wh) -> np.ndarray:
    """
    Rescale K for a uniformly resized image (e.g. calibrated at 640x400 on 2x
    downscaled frames, used at native 1280x800). Distortion (KB4 or radtan)
    acts on normalized coordinates, so D is unchanged.

    Uses the pixel-centre convention (integer coords = pixel centres), so for a
    scale s:  c_dst = s * (c_src + 0.5) - 0.5.
    """
    (sw, sh), (dw, dh) = src_wh, dst_wh
    sx, sy = dw / sw, dh / sh
    if abs(sx - sy) > 1e-9:
        raise ValueError(
            f"Capture {dw}x{dh} vs calibrated {sw}x{sh} changes aspect ratio "
            f"(sx={sx:.4f}, sy={sy:.4f}) — that's a crop, not a resize; "
            "intrinsics can't be rescaled. Recalibrate at this resolution.")
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
    K2[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
    return K2


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------
class V4L2Camera:
    """
    Minimal direct V4L2 capture. Opens exactly the given device (no fallback
    scanning), sets FOURCC *before* width/height (required by many UVC cams,
    incl. the Arducam OV9281), and returns RGB frames.
    """

    def __init__(self, device, resolution, fourcc="YUYV", fps=None):
        self.device = device
        dev = int(device) if str(device).isdigit() else str(device)
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open {device}. Is another process using it? "
                             f"Check:  fuser -v {device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def mode(self) -> str:
        cc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        cc = "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4))
        return (f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} {cc} "
                f"@ {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps")

    def read_rgb(self, timeout: float = 3.0) -> np.ndarray:
        t0 = time.time()
        while time.time() - t0 < timeout:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                if frame.ndim == 2:
                    return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raise RuntimeError(f"No frame from {self.device} within {timeout:.0f} s")

    def stop(self) -> None:
        self.cap.release()


def solve_pnp_target(corners, objp, K, D, is_fisheye):
    """Return (R_target2cam, t_target2cam) via PnP. Fisheye-aware."""
    if is_fisheye:
        # Undistort detected corners into ideal-pinhole pixels (P=K), then PnP
        # with zero distortion. corners: (N,1,2).
        undist = cv2.fisheye.undistortPoints(corners.astype(np.float64), K, D, P=K)
        ok, rvec, tvec = cv2.solvePnP(objp, undist, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


# ---------------------------------------------------------------------------
# capture + solve
# ---------------------------------------------------------------------------
def run(args) -> None:
    K, D, is_fisheye, (w, h), intr_info = load_intrinsics(Path(args.intrinsics), args.cam_key)
    print(f"Intrinsics [{intr_info['cam_key']}]: fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f} @ {w}x{h} "
          f"| {intr_info['camera_model']}/{intr_info['distortion_model']} "
          f"({'cv2.fisheye KB4' if is_fisheye else 'OpenCV pinhole'})")
    print(f"  D = {D.ravel().round(6)}")

    # Capture resolution: default = calibrated resolution. If the camera can't
    # run that mode, capture at a larger same-aspect mode and either
    #   (a) --resize-to-calib: downscale each frame to the calibrated size and
    #       use K unchanged (matches the images Kalibr saw), or
    #   (b) default: keep full-res frames and rescale K (better corner precision).
    if args.capture_res:
        cw, ch = (int(v) for v in args.capture_res.lower().split("x"))
    else:
        cw, ch = w, h
    cam_wh = (cw, ch)          # what the camera is asked to deliver
    resize_wh = None           # if set, frames are resized to this before detection
    if cam_wh != (w, h):
        K_scaled = scale_intrinsics(K, (w, h), cam_wh)  # also validates aspect ratio
        intr_info["capture_resolution_wh"] = list(cam_wh)
        if args.resize_to_calib:
            resize_wh = (w, h)
            print(f"  Capturing at {cw}x{ch}, resizing frames to {w}x{h} (INTER_AREA); "
                  "K used as calibrated.")
            intr_info["frames_resized_to_wh"] = [w, h]
        else:
            K = K_scaled
            print(f"  Capturing at {cw}x{ch}: K rescaled x{cw / w:g} -> "
                  f"fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        intr_info["K_used"] = K.tolist()

    # Checkerboard object points (inner-corner grid), Z=0 plane.
    cols, rows = args.cols, args.rows
    objp = np.zeros((cols * rows, 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= args.square  # metres
    pattern = (cols, rows)

    out_dir = Path(args.out).resolve().parent
    dbg_dir = out_dir / "handeye_captures"
    dbg_dir.mkdir(parents=True, exist_ok=True)

    # Open the camera and verify resolution BEFORE making the arm compliant:
    # intrinsics are only valid at the calibrated resolution.
    camera = V4L2Camera(args.device, cam_wh, fourcc=args.fourcc)
    probe = camera.read_rgb()
    ph, pw = probe.shape[:2]
    print(f"  Camera {args.device}: {camera.mode()}")
    if (pw, ph) != cam_wh:
        camera.stop()
        raise SystemExit(
            f"Camera delivers {pw}x{ph} but {cam_wh[0]}x{cam_wh[1]} was requested. Check the device "
            "and its supported modes (v4l2-ctl -d <dev> --list-formats-ext); pass "
            "--capture-res to match a supported mode with the calibrated aspect ratio.")

    robot = get_yam_robot(
        channel=args.channel,
        arm_type=ArmType.from_string_name(args.arm),
        gripper_type=GripperType.from_string_name("no_gripper"),
        sim=args.sim,
        zero_gravity_mode=True,  # compliant: gravity-comp with zero stiffness
        ee_mass=0.178,
    )
    kin = Kinematics(robot.xml_path, args.site)
    time.sleep(0.5)

    print("\n" + "=" * 68)
    print("Arm is COMPLIANT (gravity-comp). Jog it by hand to each pose.")
    print("Keep the checkerboard FIXED for the entire session.")
    print("  [ENTER] capture   'u' undo last   'q' finish & solve")
    print("Aim for 15-20 poses with LOTS of orientation variety.")
    print("=" * 68 + "\n")

    R_g2b, t_g2b, R_t2c, t_t2c, poses_q = [], [], [], [], []

    def flush_and_grab():
        for _ in range(5):  # drop buffered frames
            frame = camera.read_rgb()
        if resize_wh is not None:
            frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_AREA)
        return frame

    while True:
        cmd = input(f"[{len(R_g2b)} captured] ENTER=capture, u=undo, q=finish: ").strip().lower()
        if cmd == "q":
            break
        if cmd == "u":
            if R_g2b:
                for lst in (R_g2b, t_g2b, R_t2c, t_t2c, poses_q):
                    lst.pop()
                print("  undid last capture.")
            continue
        rgb = flush_and_grab()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print("  checkerboard NOT found — adjust view/lighting and retry.")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))

        pnp = solve_pnp_target(corners, objp, K, D, is_fisheye)
        if pnp is None:
            print("  solvePnP failed — retry.")
            continue
        R_tc, t_tc = pnp

        q = robot.get_joint_pos()[:N_ARM].copy()
        T_be = kin.fk(q)  # base <- grasp_site
        R_g2b.append(T_be[:3, :3]); t_g2b.append(T_be[:3, 3])
        R_t2c.append(R_tc);         t_t2c.append(t_tc.ravel())
        poses_q.append(q)

        # Save an annotated capture for later inspection.
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.drawChessboardCorners(vis, pattern, corners, found)
        cv2.imwrite(str(dbg_dir / f"capture_{len(R_g2b):02d}.png"), vis)
        print(f"  captured #{len(R_g2b)}  target dist={np.linalg.norm(t_tc):.3f} m")

    n = len(R_g2b)
    if n < 3:
        print(f"\nOnly {n} captures — need >=3 (>=10 recommended). Aborting.")
        robot.close(); camera.stop()
        return

    print(f"\nSolving hand-eye from {n} poses ...")
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    results = {}
    for name, m in methods.items():
        R_x, t_x = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=m)
        results[name] = make_SE3(R_x, t_x)
        rpy = np.array(cv2.RQDecomp3x3(R_x)[0])
        print(f"  {name:11s} t={t_x.ravel().round(4)}  rpy_deg={rpy.round(2)}")

    # Primary = PARK by default (jointly solves R,t; robust general choice).
    X = results[args.method]  # T_ee_cam in grasp_site frame
    print(f"\nUsing method '{args.method}' as primary result.")

    # Compose to a link_6-referenced transform for a MuJoCo camera_site.
    # Read the reference site's actual pose from the live (combined) model so
    # this is correct for whatever build was used (no_gripper for the camera).
    import xml.etree.ElementTree as ET
    _link6 = ET.parse(robot.xml_path).getroot().find(".//body[@name='link_6']")
    _s = next(s for s in _link6.findall("site") if s.get("name") == args.site)
    T_l6_grasp = make_SE3(
        wxyz_to_R(np.array([float(v) for v in _s.get("quat", "1 0 0 0").split()])),
        np.array([float(v) for v in _s.get("pos", "0 0 0").split()]),
    )
    T_l6_cam = T_l6_grasp @ X
    site_pos = T_l6_cam[:3, 3]
    site_quat = R_to_wxyz(T_l6_cam[:3, :3])

    _save_result(Path(args.out), X, T_l6_cam, results, args, n, intr_info)
    np.savez(Path(args.out).with_suffix(".raw.npz"),
             R_g2b=np.array(R_g2b), t_g2b=np.array(t_g2b),
             R_t2c=np.array(R_t2c), t_t2c=np.array(t_t2c),
             poses_q=np.array(poses_q), K=K, D=D)

    print("\n" + "=" * 68)
    print(f"X = T_ee_cam (grasp_site frame):\n{np.array2string(X, precision=5)}")
    print("\nPaste this camera_site under <body name=\"link_6\"> in the YAM XML:")
    print(f'  <site name="camera_site" '
          f'pos="{site_pos[0]:.6f} {site_pos[1]:.6f} {site_pos[2]:.6f}" '
          f'quat="{site_quat[0]:.6f} {site_quat[1]:.6f} {site_quat[2]:.6f} {site_quat[3]:.6f}" '
          f'size="0.005" rgba="0 0 1 1"/>')
    print("\nThen rotate about the camera with:  --site camera_site")
    print(f"Saved: {args.out}  and  {Path(args.out).with_suffix('.raw.npz')}")
    print(f"Capture previews: {dbg_dir}")
    print("=" * 68)
    print("\nSupport the arm before it powers down.")
    robot.close()
    camera.stop()


def _save_result(path: Path, X, T_l6_cam, results, args, n, intr_info) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "method": args.method,
        "num_poses": n,
        "board": {"cols": args.cols, "rows": args.rows, "square_m": args.square},
        "intrinsics": intr_info,  # provenance: which calibration produced X
        "T_ee_cam": {  # camera in grasp_site (EE) frame — this is X
            "matrix": X.tolist(),
            "translation": X[:3, 3].tolist(),
            "quaternion_wxyz": R_to_wxyz(X[:3, :3]).tolist(),
        },
        "T_link6_cam": {  # camera in link_6 frame — for the MuJoCo site
            "translation": T_l6_cam[:3, 3].tolist(),
            "quaternion_wxyz": R_to_wxyz(T_l6_cam[:3, :3]).tolist(),
        },
        "all_methods": {k: v.tolist() for k, v in results.items()},
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def main() -> None:
    arm_choices = [a.value for a in ArmType]
    calib_dir = _REPO / "argus_experiment" / "calibration"
    default_intr = calib_dir / "arm_calib_dynamic-camchain-imucam.yaml"
    default_out = calib_dir / "hand_eye_result.yaml"

    p = argparse.ArgumentParser(description="Eye-in-hand hand-eye calibration for YAM + link_6 camera")
    p.add_argument("--arm", default="yam", choices=arm_choices)
    p.add_argument("--channel", default="can0")
    p.add_argument("--sim", action="store_true", help="Compliant sim robot (no real camera stream)")
    p.add_argument("--device", default="/dev/video0", help="OpenCV camera device path/index")
    p.add_argument("--fourcc", default="YUYV", choices=["YUYV", "MJPG"],
                   help="Pixel format. YUYV = uncompressed (best corners); MJPG if YUYV "
                        "can't reach the resolution over USB")
    p.add_argument("--intrinsics", default=str(default_intr),
                   help="Kalibr camchain YAML (or legacy {camera: ...} YAML)")
    p.add_argument("--cam-key", default="cam0",
                   help="Camera entry in the Kalibr camchain (cam0, cam1, ...)")
    p.add_argument("--capture-res", default=None,
                   help="Capture WxH if the camera can't run the calibrated resolution, "
                        "e.g. 1280x800 for intrinsics calibrated at 640x400 (K is rescaled)")
    p.add_argument("--resize-to-calib", action="store_true",
                   help="With --capture-res: downscale frames to the calibrated resolution "
                        "and use K unchanged, instead of rescaling K")
    p.add_argument("--site", default="grasp_site", help="EE site used for FK (calibration frame)")
    # Board: 8x11 squares @ 36 mm  ->  10x7 INNER corners (squares - 1 each side).
    p.add_argument("--cols", type=int, default=10, help="Inner corners along width (= squares_w - 1)")
    p.add_argument("--rows", type=int, default=7, help="Inner corners along height (= squares_h - 1)")
    p.add_argument("--square", type=float, default=0.036, help="Checkerboard square size (m)")
    p.add_argument("--method", default="PARK",
                   choices=["TSAI", "PARK", "HORAUD", "DANIILIDIS"],
                   help="Primary calibrateHandEye method")
    p.add_argument("--out", default=str(default_out))
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()