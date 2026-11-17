#!/usr/bin/env python3
"""
bag2c3d.py

Export the labeled markers of MOCAP4ROS2 (OptiTrack) rigid bodies from a
recorded ROS 2 bag to a C3D file that OpenSim can load.

Each rigid body's markers become C3D points named <segment>_<marker_index>,
e.g. torso_1 ... torso_7, r_humerus_1 ... r_humerus_6. The segment name comes
from the body's Motive Streaming ID (SEGMENTS below; override with --name).
Bodies with no name are kept as rb<ID>_<marker_index>.

The bag is read with the pure-Python `rosbags` library, so ROS and the mocap
msgs workspace do not need to be sourced (sqlite3 and mcap bags both work):
  pip install --user rosbags ezc3d
  python3 bag2c3d.py ~/ArgusArmExperiment/mocap_data/dynamic1      # -> mocap_data/dynamic1.c3d
  python3 bag2c3d.py ~/ArgusArmExperiment/mocap_data/dynamic1 -o d1.c3d --axes=-x,z,y

Timing
  Frames go on a uniform --rate grid using header.stamp: each message lands
  round(dt * rate) frames after the previous one. Frames the driver never
  received become gaps rather than silently shortening the trial.
  The ROS stamp of frame 0 is saved as the string parameter MOCAP:START_STAMP.

Long trials
  ezc3d, and so OpenSim, reads at most 65535 frames from a C3D (9.1 min at
  120 Hz) and ignores TRIAL:ACTUAL_END_FIELD and POINT:LONG_FRAMES. A longer
  trial is written as <out>_part1.c3d, <out>_part2.c3d, ...: the fewest
  equal-length parts of at most --max-frames each, all with the same markers.
  Each part's MOCAP:START_STAMP is the stamp of its own frame 0 (trial start
  + FRAME_OFFSET / rate); MOCAP:PART ("2/3") and MOCAP:FRAME_OFFSET (the
  part's first frame in the whole trial) say where it sits.

Missing data
  Bodies Motive lost and markers absent from a frame (or streamed as exactly
  0,0,0) are written as invalid points (residual -1), which OpenSim reads as NaN.

Axes
  Motive streams Z-up. OpenSim is Y-up with X forward and Z to the subject's
  right. --axes gives each output axis as a signed input axis: --axes=-x,z,y
  means X_out = -x, Y_out = z, Z_out = y. Only proper rotations are accepted.
  POINT:X_SCREEN / Y_SCREEN record the resulting up axis for viewers.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from bag_to_csv import id_sort_key, parse_rigidbodies

# Motive Streaming ID -> OpenSim segment name. Order here = marker order in the C3D.
SEGMENTS = {
    "47": "torso",
    "48": "r_humerus",
    "45": "r_forearm",
    "42": "r_acromion",
    "49": "l_humerus",
    "46": "l_forearm",
    "41": "l_acromion",
}

# Field layout of mocap4r2_msgs (constants omitted; they don't affect decoding).
MSG_DEFS = {
    "mocap4r2_msgs/msg/Marker": """
int8 id_type
int32 marker_index
string marker_name
geometry_msgs/Point translation
""",
    "mocap4r2_msgs/msg/Markers": """
std_msgs/Header header
uint32 frame_number
mocap4r2_msgs/Marker[] markers
""",
    "mocap4r2_msgs/msg/RigidBody": """
string rigid_body_name
mocap4r2_msgs/Marker[] markers
geometry_msgs/Pose pose
""",
    "mocap4r2_msgs/msg/RigidBodies": """
std_msgs/Header header
uint32 frame_number
mocap4r2_msgs/RigidBody[] rigidbodies
""",
}

UNIT_SCALE = {"mm": 1000.0, "m": 1.0}
MAX_FRAMES = 65535  # most frames ezc3d (OpenSim) reads back from one C3D


# --------------------------------------------------------------------------- #
# Bag reading (rosbags, no ROS needed)
# --------------------------------------------------------------------------- #
def iter_bag(uri, topic):
    """Yield deserialized RigidBodies messages from the bag."""
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    except ImportError:
        sys.exit("Needs the rosbags library: pip install --user rosbags")

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for name, text in MSG_DEFS.items():
        types.update(get_types_from_msg(text, name))
    typestore.register(types)

    path = Path(uri)
    if path.is_file() and (path.parent / "metadata.yaml").is_file():
        path = path.parent  # rosbags opens the bag directory

    with AnyReader([path], default_typestore=typestore) as reader:
        conns = {c.topic: c for c in reader.connections}
        listing = "\n  ".join(f"{c.topic} [{c.msgtype}]" for c in conns.values())
        if topic:
            if topic not in conns:
                sys.exit(f"Topic '{topic}' not in bag. Topics:\n  {listing}")
        else:
            cands = [t for t, c in conns.items() if c.msgtype.endswith("/RigidBodies")]
            if not cands:
                sys.exit(f"No */RigidBodies topic found. Topics:\n  {listing}")
            if len(cands) > 1:
                print(f"[warn] several RigidBodies topics {cands}, using {cands[0]} "
                      f"(pick with --topic)", file=sys.stderr)
            topic = cands[0]

        conn = conns[topic]
        if conn.msgtype not in MSG_DEFS:
            sys.exit(f"{topic} is {conn.msgtype}; only mocap4r2_msgs is supported.")
        print(f"[info] reading {topic} [{conn.msgtype}] from {path}", file=sys.stderr)
        for c, _, raw in reader.messages(connections=[conn]):
            yield reader.deserialize(raw, c.msgtype)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def parse_axes(spec):
    """'-x,z,y' -> 3x3 rotation R with p_out = R @ p_in."""
    parts = [p.strip().lower() for p in spec.split(",")]
    R = np.zeros((3, 3))
    try:
        assert len(parts) == 3
        for row, p in enumerate(parts):
            sign = -1.0 if p.startswith("-") else 1.0
            R[row, "xyz".index(p.lstrip("+-"))] = sign
    except (AssertionError, ValueError):
        sys.exit(f"--axes '{spec}': expected three signed axes like -x,z,y")
    if not np.allclose(np.abs(R).sum(axis=0), 1):
        sys.exit(f"--axes '{spec}': each of x, y, z must appear exactly once")
    if np.linalg.det(R) < 0:
        sys.exit(f"--axes '{spec}' is a mirror image, not a rotation; "
                 f"flip the sign of one axis")
    return R


def axis_name(v):
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + "XYZ"[i]


def frame_indices(stamps, rate):
    """Gap-aware frame index for each message: dropped frames leave holes."""
    steps = np.maximum(1, np.rint(np.diff(stamps) * rate)).astype(int)
    return np.concatenate([[0], np.cumsum(steps)])


def build_points(frames, names, rate):
    """frames: list of (stamp, bodies).

    Returns labels, descriptions, xyz (n_frames, n_markers, 3) in m (NaN = missing),
    the stamp of frame 0, and the number of messages.
    """
    layout = {}  # rb_id -> set(marker_index)
    for _, bodies in frames:
        for rb_id, _, markers in bodies:
            layout.setdefault(rb_id, set()).update(m[0] for m in markers)

    order = [rb for rb in names if rb in layout]
    order += sorted((rb for rb in layout if rb not in names), key=id_sort_key)
    unnamed = [rb for rb in order if rb not in names]
    if unnamed:
        print(f"[warn] no segment name for rigid bodies {unnamed}; "
              f"exporting as rb<ID> (add --name ID=segment)", file=sys.stderr)
    missing = [rb for rb in names if rb not in layout]
    if missing:
        print("[warn] named bodies never seen in bag: "
              + ", ".join(f"{rb} ({names[rb]})" for rb in missing), file=sys.stderr)

    col, labels, descs = {}, [], []
    for rb in order:
        seg = names.get(rb, f"rb{rb}")
        for idx in sorted(layout[rb]):
            col[(rb, idx)] = len(labels)
            labels.append(f"{seg}_{idx}")
            descs.append(f"rigid body {rb} marker {idx}")

    stamps = np.array([s for s, _ in frames])
    fidx = frame_indices(stamps, rate)
    xyz = np.full((fidx[-1] + 1, len(labels), 3), np.nan)
    for f, (_, bodies) in zip(fidx, frames):
        for rb_id, _, markers in bodies:
            for idx, x, y, z in markers:
                xyz[f, col[(rb_id, idx)]] = (x, y, z)

    for rb in order:
        print(f"[info] {names.get(rb, f'rb{rb}'):>12} (id {rb}): "
              f"{len(layout[rb])} markers", file=sys.stderr)
    return labels, descs, xyz, stamps[0], len(fidx)


def write_c3d(out, labels, descs, xyz, rate, units, R, meta):
    import ezc3d

    pts = xyz @ R.T * UNIT_SCALE[units]           # (F, M, 3)
    valid = ~np.isnan(pts).any(axis=2)            # (F, M)
    data = np.ones((4, pts.shape[1], pts.shape[0]))
    data[:3] = np.nan_to_num(pts).transpose(2, 1, 0)

    c = ezc3d.c3d()
    point = c["parameters"]["POINT"]
    point["RATE"]["value"] = np.array([float(rate)])
    point["LABELS"]["value"] = labels
    point["DESCRIPTIONS"]["value"] = descs
    point["UNITS"]["value"] = [units]
    c.add_parameter("POINT", "X_SCREEN", [axis_name(R[:, 0])])  # Motive +x after --axes
    c.add_parameter("POINT", "Y_SCREEN", [axis_name(R[:, 2])])  # Motive up (+z) after --axes
    for key, value in meta.items():
        c.add_parameter("MOCAP", key, value)
    c["data"]["points"] = data
    c["data"]["meta_points"] = {"residuals": np.where(valid.T, 0.0, -1.0)[None]}
    c.write(str(out))
    return valid


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="bag directory (or .db3/.mcap file)")
    ap.add_argument("-o", "--out", help="output C3D (default: <bag>.c3d)")
    ap.add_argument("-t", "--topic", help="RigidBodies topic (default: auto-detect)")
    ap.add_argument("--rate", type=float, default=120.0,
                    help="Motive frame rate in Hz (default: 120)")
    ap.add_argument("--units", choices=sorted(UNIT_SCALE), default="mm",
                    help="C3D point units (default: mm; OpenSim converts either)")
    ap.add_argument("--axes", default="x,y,z",
                    help="output axes as signed input axes, e.g. --axes=-x,z,y "
                         "(default: x,y,z = Motive frame, Z up)")
    ap.add_argument("--name", nargs="+", default=[], metavar="ID=SEGMENT",
                    help="add or override segment names, e.g. --name 50=pelvis")
    ap.add_argument("--rb", nargs="+", metavar="ID",
                    help="only export these streaming IDs, e.g. --rb 47 48")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                    help=f"split longer trials into parts (default and max: {MAX_FRAMES})")
    args = ap.parse_args()
    if not 1 <= args.max_frames <= MAX_FRAMES:
        sys.exit(f"--max-frames must be between 1 and {MAX_FRAMES}")

    uri = args.bag.rstrip("/")
    out = args.out or (os.path.splitext(uri)[0] if os.path.isfile(uri) else uri) + ".c3d"
    R = parse_axes(args.axes)
    names = dict(SEGMENTS)
    for pair in args.name:
        rb, sep, seg = pair.partition("=")
        if not sep or not rb or not seg:
            sys.exit(f"--name '{pair}': expected ID=SEGMENT")
        names[rb.strip()] = seg.strip()
    keep = set(args.rb) if args.rb else None
    if keep is not None:
        names = {rb: seg for rb, seg in names.items() if rb in keep}

    frames = []
    for msg in iter_bag(uri, args.topic):
        stamp, _, bodies = parse_rigidbodies(msg, drop_zeros=True)
        if keep is not None:
            bodies = [b for b in bodies if b[0] in keep]
        frames.append((stamp, bodies))
    if not frames:
        sys.exit("[error] topic had no messages")

    labels, descs, xyz, t0, n_msgs = build_points(frames, names, args.rate)
    if not labels:
        sys.exit("[error] no rigid body markers in bag")
    n_frames = xyz.shape[0]
    if n_frames > n_msgs:
        print(f"[info] {n_frames - n_msgs} of {n_frames} frames were never received "
              f"(left as gaps)", file=sys.stderr)

    meta = {
        "SOURCE_BAG": [os.path.basename(uri)],
        "START_STAMP": [f"{t0:.9f}"],
        "SEGMENTS": [f"{rb}={seg}" for rb, seg in names.items()],
        "AXES": [args.axes],
    }
    n_parts = -(-n_frames // args.max_frames)
    bounds = np.linspace(0, n_frames, n_parts + 1).round().astype(int)
    stem = out[:-4] if out.lower().endswith(".c3d") else out
    if n_parts > 1:
        print(f"[info] {n_frames} frames is over the {args.max_frames}-frame limit; "
              f"writing {n_parts} parts", file=sys.stderr)
    valid = []
    for p, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        path, part_meta = out, meta
        if n_parts > 1:
            path = f"{stem}_part{p}.c3d"
            part_meta = dict(meta, START_STAMP=[f"{t0 + a / args.rate:.9f}"],
                             PART=[f"{p}/{n_parts}"], FRAME_OFFSET=[str(a)])
        valid.append(write_c3d(path, labels, descs, xyz[a:b], args.rate, args.units, R,
                               part_meta))
        print(f"[info] wrote {len(labels)} markers x {b - a} frames "
              f"({(b - a) / args.rate:.2f} s @ {args.rate:g} Hz, {args.units}) -> {path}",
              file=sys.stderr)
    valid = np.concatenate(valid)

    worst = sorted(zip(valid.mean(axis=0), labels))[:3]
    print("[info] least tracked: " + ", ".join(f"{l} {100 * v:.1f}%" for v, l in worst),
          file=sys.stderr)


if __name__ == "__main__":
    main()
