#!/usr/bin/env python3
"""
bag_to_csv.py

Export MOCAP4ROS2 (OptiTrack) rigid body poses and their markers' 3D positions
from a recorded ROS 2 bag to CSV.

Expects a mocap4r2_msgs/msg/RigidBodies topic (older mocap_msgs also works), where
  rigid_body_name = Motive Streaming ID (e.g. "1")
  pose            = position (m) + orientation quaternion
  markers[]       = that body's labeled markers, keyed by marker_index

Run in a terminal where ROS 2 and the mocap msgs package are sourced:
  source /opt/ros/<distro>/setup.bash
  source ~/<ws>/install/setup.bash
  python3 bag_to_csv.py /home/argus/ArgusArmExperiment/mocap_bags/dynamic1 -o dynamic1.csv

Formats
  --format wide (default): one row per message (mocap frame)
      t_s, stamp_s, frame_number,
      rb<ID>_px, rb<ID>_py, rb<ID>_pz, rb<ID>_qx, rb<ID>_qy, rb<ID>_qz, rb<ID>_qw,
      rb<ID>_m<idx>_x, rb<ID>_m<idx>_y, rb<ID>_m<idx>_z, ...
    Columns cover every rigid body / marker seen anywhere in the bag; blank = missing.
  --format long: one row per (frame, rigid body, item), item = pose or a marker
      t_s, stamp_s, frame_number, rb_id, item, marker_index, x, y, z, qx, qy, qz, qw
"""

import argparse
import csv
import glob
import os
import sys

POSE_FIELDS = ["px", "py", "pz", "qx", "qy", "qz", "qw"]


# --------------------------------------------------------------------------- #
# Message parsing (pure python, no ROS imports, so it can be unit-tested)
# --------------------------------------------------------------------------- #
def id_sort_key(s):
    """Sort streaming IDs numerically when possible ("2" before "10")."""
    try:
        return (0, int(s), "")
    except (TypeError, ValueError):
        return (1, 0, str(s))


def is_zero(*vals):
    return all(v == 0.0 for v in vals)


def parse_rigidbodies(msg, drop_zeros=False):
    """Return (stamp_s, frame_number, bodies) for one RigidBodies message.

    bodies: list of (rb_id, pose_tuple_or_None, [(marker_index, x, y, z), ...])
    """
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    frame = getattr(msg, "frame_number", "")

    rb_list = getattr(msg, "rigidbodies", None)
    if rb_list is None:
        rb_list = getattr(msg, "rigid_bodies", [])

    bodies = []
    for rb in rb_list:
        rb_id = str(rb.rigid_body_name).strip()
        p, q = rb.pose.position, rb.pose.orientation
        pose = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if drop_zeros and is_zero(p.x, p.y, p.z):
            pose = None  # untracked body streamed as zeros

        markers = []
        for i, m in enumerate(getattr(rb, "markers", [])):
            idx = getattr(m, "marker_index", i)
            pt = getattr(m, "translation", None) or getattr(m, "position")
            if drop_zeros and is_zero(pt.x, pt.y, pt.z):
                continue  # occluded marker streamed as zeros
            markers.append((int(idx), pt.x, pt.y, pt.z))

        bodies.append((rb_id, pose, markers))
    return stamp, frame, bodies


# --------------------------------------------------------------------------- #
# Bag reading (ROS 2)
# --------------------------------------------------------------------------- #
def detect_storage(uri):
    if os.path.isfile(uri):
        return "mcap" if uri.endswith(".mcap") else "sqlite3"
    if glob.glob(os.path.join(uri, "*.mcap")):
        return "mcap"
    return "sqlite3"


def iter_bag(uri, topic, storage):
    """Yield (bag_time_ns, deserialized_msg) for the rigid body topic."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic:
        if topic not in types:
            sys.exit(f"Topic '{topic}' not in bag. Topics:\n  " +
                     "\n  ".join(f"{n} [{t}]" for n, t in types.items()))
    else:
        cands = [n for n, t in types.items() if t.endswith("/RigidBodies")]
        if not cands:
            sys.exit("No */RigidBodies topic found. Topics:\n  " +
                     "\n  ".join(f"{n} [{t}]" for n, t in types.items()))
        if len(cands) > 1:
            print(f"[warn] several RigidBodies topics {cands}, using {cands[0]} "
                  f"(pick with --topic)", file=sys.stderr)
        topic = cands[0]

    print(f"[info] reading {topic} [{types[topic]}] from {uri} ({storage})",
          file=sys.stderr)
    try:
        msg_cls = get_message(types[topic])
    except (AttributeError, ModuleNotFoundError, ValueError) as e:
        sys.exit(f"Can't import {types[topic]} ({e}). Source the workspace that "
                 f"builds the mocap msgs package before running.")

    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    while reader.has_next():
        _, data, t_ns = reader.read_next()
        yield t_ns, deserialize_message(data, msg_cls)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def fmt(v):
    return "" if v is None else repr(float(v))


def write_long(frames, out_path):
    n = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "stamp_s", "frame_number", "rb_id", "item",
                    "marker_index", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for t_s, stamp, frame, bodies in frames:
            for rb_id, pose, markers in bodies:
                if pose is not None:
                    w.writerow([fmt(t_s), f"{stamp:.9f}", frame, rb_id, "pose", ""]
                               + [fmt(v) for v in pose])
                for idx, x, y, z in markers:
                    w.writerow([fmt(t_s), f"{stamp:.9f}", frame, rb_id, "marker",
                                idx, fmt(x), fmt(y), fmt(z), "", "", "", ""])
            n += 1
    return n


def write_wide(frames, out_path):
    # frames is a list here: pass 1 finds every body/marker, pass 2 writes.
    layout = {}  # rb_id -> set(marker_index)
    for _, _, _, bodies in frames:
        for rb_id, _, markers in bodies:
            layout.setdefault(rb_id, set()).update(m[0] for m in markers)

    rb_ids = sorted(layout, key=id_sort_key)
    header = ["t_s", "stamp_s", "frame_number"]
    for rb in rb_ids:
        header += [f"rb{rb}_{c}" for c in POSE_FIELDS]
        for idx in sorted(layout[rb]):
            header += [f"rb{rb}_m{idx}_{a}" for a in "xyz"]
    col = {name: i for i, name in enumerate(header)}

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t_s, stamp, frame, bodies in frames:
            row = [""] * len(header)
            row[0], row[1], row[2] = fmt(t_s), f"{stamp:.9f}", frame
            for rb_id, pose, markers in bodies:
                if pose is not None:
                    for c, v in zip(POSE_FIELDS, pose):
                        row[col[f"rb{rb_id}_{c}"]] = fmt(v)
                for idx, x, y, z in markers:
                    for a, v in zip("xyz", (x, y, z)):
                        row[col[f"rb{rb_id}_m{idx}_{a}"]] = fmt(v)
            w.writerow(row)

    summary = ", ".join(f"rb{rb}: {len(layout[rb])} markers" for rb in rb_ids)
    print(f"[info] rigid bodies -> {summary}", file=sys.stderr)
    return len(frames)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="bag directory (or .db3/.mcap file)")
    ap.add_argument("-o", "--out", help="output CSV (default: <bag>_rigidbodies.csv)")
    ap.add_argument("-t", "--topic", help="RigidBodies topic (default: auto-detect)")
    ap.add_argument("--format", choices=["wide", "long"], default="wide")
    ap.add_argument("--storage", choices=["sqlite3", "mcap"],
                    help="bag storage plugin (default: auto-detect)")
    ap.add_argument("--rb", nargs="+", metavar="ID",
                    help="only export these streaming IDs, e.g. --rb 1 2 3")
    ap.add_argument("--drop-zeros", action="store_true",
                    help="treat poses/markers at exactly (0,0,0) as missing "
                         "(how untracked items often get streamed)")
    args = ap.parse_args()

    uri = args.bag.rstrip("/")
    out = args.out or f"{uri}_rigidbodies.csv"
    storage = args.storage or detect_storage(uri)
    keep = set(args.rb) if args.rb else None

    def frames_gen():
        t0 = None
        for t_ns, msg in iter_bag(uri, args.topic, storage):
            t0 = t_ns if t0 is None else t0
            stamp, frame, bodies = parse_rigidbodies(msg, args.drop_zeros)
            if keep is not None:
                bodies = [b for b in bodies if b[0] in keep]
            yield (t_ns - t0) * 1e-9, stamp, frame, bodies

    if args.format == "long":
        n = write_long(frames_gen(), out)
    else:
        n = write_wide(list(frames_gen()), out)

    if n == 0:
        print("[warn] topic had no messages", file=sys.stderr)
    print(f"[info] wrote {n} frames -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()