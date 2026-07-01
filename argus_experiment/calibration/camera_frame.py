"""
Integrate a hand-eye result (X = T_ee_cam) into the YAM MuJoCo model as a
`camera_site` under link_6, so IK/FK can target the camera frame directly.

Model-agnostic: reads the reference site's actual pose from the (already
combined) robot.xml_path, so it stays correct whether the arm was built with
no_gripper, a real gripper, etc. — avoiding the hardcoded-offset pitfall.

Typical use in a motion script:

    from argus_experiment.calibration.camera_frame import load_handeye, add_camera_site
    X = load_handeye("argus_experiment/calibration/hand_eye_result.yaml")
    xml_path = add_camera_site(robot.xml_path, X)          # inject camera_site
    kin = Kinematics(xml_path, "camera_site")              # rotate about the camera
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


def _wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _R_to_wxyz(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    w = max(w, 1e-8)
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _se3(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).ravel(); return T


def load_handeye(path) -> np.ndarray:
    """Load X = T_ee_cam (4x4) from a hand_eye_result.yaml."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    return np.array(doc["T_ee_cam"]["matrix"], dtype=float)


def _read_site_pose(link6: ET.Element, site_name: str) -> np.ndarray:
    """T_link6_site from the site element's pos/quat (MuJoCo defaults if absent)."""
    for site in link6.findall("site"):
        if site.get("name") == site_name:
            pos = np.array([float(v) for v in site.get("pos", "0 0 0").split()])
            quat = np.array([float(v) for v in site.get("quat", "1 0 0 0").split()])
            return _se3(_wxyz_to_R(quat), pos)
    raise ValueError(f"site {site_name!r} not found under link_6")


def add_camera_site(xml_path, X, ref_site="grasp_site", site_name="camera_site"):
    """
    Inject `camera_site` under link_6 using hand-eye X expressed in `ref_site`.
    Returns the path to a new temp XML (original untouched).

    X = T_ref_cam (camera in the ref_site frame; ref_site is what FK used
    during calibration, i.e. grasp_site by default).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    link6 = root.find(".//body[@name='link_6']") or root.find(".//body[@name='link6']")
    if link6 is None:
        raise ValueError("link_6 body not found in model")

    T_l6_ref = _read_site_pose(link6, ref_site)
    T_l6_cam = T_l6_ref @ X

    site = ET.SubElement(link6, "site")
    site.set("name", site_name)
    site.set("pos", " ".join(f"{v:.6f}" for v in T_l6_cam[:3, 3]))
    site.set("quat", " ".join(f"{v:.6f}" for v in _R_to_wxyz(T_l6_cam[:3, :3])))
    site.set("size", "0.005")
    site.set("rgba", "0 0 1 1")

    out = tempfile.NamedTemporaryFile(suffix=".xml", prefix="yam_cam_", delete=False).name
    tree.write(out)
    return out


if __name__ == "__main__":
    # Quick check against the current YAM no_gripper model + saved result.
    import sys
    _REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPO / "robots_realtime" / "dependencies" / "i2rt"))
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    robot = get_yam_robot(channel="can0", arm_type=ArmType.YAM,
                          gripper_type=GripperType.NO_GRIPPER, sim=True, ee_mass=0.178)
    X = load_handeye(_REPO / "argus_experiment" / "calibration" / "hand_eye_result.yaml")
    new_xml = add_camera_site(robot.xml_path, X)
    tree = ET.parse(new_xml)
    cam = tree.getroot().find(".//site[@name='camera_site']")
    print("camera_site pos :", cam.get("pos"))
    print("camera_site quat:", cam.get("quat"))
