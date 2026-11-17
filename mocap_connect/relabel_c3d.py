#!/usr/bin/env python3
"""
relabel_c3d.py

Rename the markers bag2c3d.py writes to the segment names of a per-session
marker set, splitting a rigid body where some of its markers sit on another
segment: the knee markers carried by the thigh Argus modules (drift), or the
anatomical markers carried by the arm Argus modules (upperarm).

  python3 relabel_c3d.py ~/ArgusArmExperiment/mocap_data/drift?_part*.c3d
  python3 relabel_c3d.py ~/ArgusArmExperiment/mocap_data/upperarm?.c3d
  # -> mocap_data/labeled/<same name>

Each file uses the set in MARKER_SETS that shares the most streaming IDs with
it; --set picks one by name. Each marker is matched by the rigid body and
marker index it came from, read from POINT:DESCRIPTIONS ("rigid body 54
marker 5", written by bag2c3d.py) or, failing that, from an rbNN_k label.
DESCRIPTIONS are copied unchanged, so a labeled file still records each
marker's source and can be relabeled again after MARKER_SETS changes. Markers
of each output segment are numbered 1..n in file order, e.g. rb55_1 rb55_2
rb55_5 rb55_6 -> argus3_1 .. argus3_4.

Only the labels (and MOCAP:SEGMENTS) change; point data, residuals and frame
count are copied as they are.
"""

import argparse
import re
import sys
from pathlib import Path

import ezc3d

# Per session: "bodies" maps Motive Streaming ID -> segment name; "split" maps
# (Streaming ID, marker index) -> segment for markers that sit on another
# segment than the rest of their rigid body. Argus modules go by unit number.
MARKER_SETS = {
    "drift": {
        "bodies": {
            "51": "back",
            "60": "pelvis",
            "57": "l_foot",
            "59": "r_foot",
            "53": "argus0",  # pelvis
            "56": "argus1",  # left shank
            "58": "argus2",  # right shank
            "55": "argus3",  # right thigh; markers 3, 4 are on the knee
            "54": "argus4",  # left thigh; markers 5, 6 are on the knee
        },
        "split": {
            ("54", 5): "l_knee", ("54", 6): "l_knee",
            ("55", 3): "r_knee", ("55", 4): "r_knee",
        },
    },
    "upperarm": {
        "bodies": {
            "63": "l_acromion",
            "62": "r_acromion",
            "70": "thorax",
            "69": "argus0",  # sternum
            "67": "argus1",  # left radius; marker 5 is on the radius
            "65": "argus2",  # right radius; marker 5 is on the radius
            "66": "argus3",  # left humerus; markers 4, 6 are on the humerus
            "64": "argus4",  # right humerus; markers 3, 4 are on the humerus
        },
        "split": {
            ("67", 5): "l_radius",
            ("65", 5): "r_radius",
            ("66", 4): "l_humerus", ("66", 6): "l_humerus",
            ("64", 3): "r_humerus", ("64", 4): "r_humerus",
        },
    },
}

DESC_RE = re.compile(r"rigid body (\S+) marker (\d+)")
LABEL_RE = re.compile(r"rb(\S+)_(\d+)")


def source_of(label, desc):
    """(streaming ID, marker index) a column came from, or None."""
    m = DESC_RE.fullmatch(desc) or LABEL_RE.fullmatch(label)
    return (m[1], int(m[2])) if m else None


def pick_set(ids, name):
    """Marker set name: the one given, else the one sharing the most streaming IDs."""
    if name:
        return name
    overlap = {n: len(ids & s["bodies"].keys()) for n, s in MARKER_SETS.items()}
    best = max(overlap.values())
    winners = [n for n, k in overlap.items() if k == best]
    if best == 0 or len(winners) > 1:
        sys.exit(f"[error] can't tell the marker set from streaming IDs {sorted(ids)} "
                 f"(overlap {overlap}); pick one with --set")
    return winners[0]


def relabel(labels, sources, mset):
    """Return new labels, {segment: [source labels]}, and the unmatched labels."""
    new, groups, unmatched = [], {}, []
    for lab, src in zip(labels, sources):
        seg = None
        if src:
            seg = mset["split"].get(src) or mset["bodies"].get(src[0])
        if seg is None:
            new.append(lab)
            unmatched.append(lab)
            continue
        members = groups.setdefault(seg, [])
        members.append(f"rb{src[0]}_{src[1]}")
        new.append(f"{seg}_{len(members)}")
    return new, groups, unmatched


def process(src, dst, set_name):
    c = ezc3d.c3d(str(src))
    params = c["parameters"]
    point = params["POINT"]
    labels = [s.strip() for s in point["LABELS"]["value"]]
    descs = [s.strip() for s in point["DESCRIPTIONS"]["value"]] if "DESCRIPTIONS" in point else []
    descs += [""] * (len(labels) - len(descs))
    sources = [source_of(l, d) for l, d in zip(labels, descs)]

    ids = {s[0] for s in sources if s}
    set_name = pick_set(ids, set_name)
    mset = MARKER_SETS[set_name]
    new, groups, unmatched = relabel(labels, sources, mset)
    dupes = sorted({l for l in new if new.count(l) > 1})
    if dupes:
        sys.exit(f"[error] {src.name}: duplicate labels after relabeling: {dupes}")

    point["LABELS"]["value"] = new
    segments = ([f"{rb}={seg}" for rb, seg in mset["bodies"].items()]
                + [f"{rb}:{idx}={seg}" for (rb, idx), seg in mset["split"].items()])
    if "MOCAP" in params and "SEGMENTS" in params["MOCAP"]:
        params["MOCAP"]["SEGMENTS"]["value"] = segments
    else:
        c.add_parameter("MOCAP", "SEGMENTS", segments)

    dst.parent.mkdir(parents=True, exist_ok=True)
    c.write(str(dst))

    n_frames = c["data"]["points"].shape[2]
    print(f"[info] {src.name} (set {set_name}): {len(labels)} markers x {n_frames} frames -> {dst}")
    for seg, members in groups.items():
        print(f"         {seg:>10}: {' '.join(members)}")
    if unmatched:
        print(f"[warn] {src.name}: left unchanged (not in set {set_name}): {' '.join(unmatched)}")
    absent = [f"{rb} ({seg})" for rb, seg in mset["bodies"].items() if rb not in ids]
    if absent:
        print(f"[warn] {src.name}: bodies not in file: {', '.join(absent)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("c3d", nargs="+", type=Path, help="C3D file(s) from bag2c3d.py")
    ap.add_argument("-o", "--out", type=Path, help="output file (single input only)")
    ap.add_argument("-d", "--out-dir", type=Path,
                    help="output folder (default: labeled/ next to each input)")
    ap.add_argument("--set", choices=sorted(MARKER_SETS),
                    help="marker set (default: the one sharing the most streaming IDs)")
    args = ap.parse_args()
    if args.out and len(args.c3d) > 1:
        sys.exit("-o takes one input; use --out-dir for several")

    for src in args.c3d:
        dst = args.out or (args.out_dir or src.parent / "labeled") / src.name
        if dst.resolve() == src.resolve():
            sys.exit(f"[error] output would overwrite the input {src}")
        process(src, dst, args.set)


if __name__ == "__main__":
    main()
