#!/usr/bin/env python3
"""
vicon_frame_time_logger.py

Connects to the live Vicon DataStream and logs ONLY frame_number + Unix
timestamps for each frame -- no marker/segment data. Intended to be paired
afterward with an offline-processed C3D (Reconstruct -> Label -> Gap Fill)
by matching on frame_number, since live and offline frame numbers for the
same trial correspond 1:1.

REQUIRES:
    - pyvicon-datastream (`pip install pyvicon-datastream`), a third-party
      Python wrapper over the Vicon DataStream SDK.
    - This machine's clock synced via NTP/chrony to the same reference used
      by your other data-collection machines.
    - Nexus running in Live mode on the target host, DataStream Server
      enabled (default port 801).

USAGE:
    python vicon_frame_time_logger.py --host 171.164.119.64:801 --out session1_times.csv --fps 100

OUTPUT COLUMNS:
    frame_number       - Vicon frame index (matches the offline C3D's frame numbering)
    recv_unix_time      - host wall-clock time when this frame was received (must be NTP-synced)
    latency_total_s      - DataStream's reported end-to-end pipeline latency for this frame
    corrected_unix_time   - recv_unix_time minus latency_total_s (estimated true event time)
    smoothed_unix_time    - least-squares fit of corrected_unix_time vs frame_number;
                            use THIS column when joining to the offline C3D -- it removes
                            host receive-loop jitter while preserving true frame cadence.
"""

import argparse
import csv
import sys
import time

try:
    import pyvicon_datastream as pv
except ImportError:
    sys.exit(
        "Could not import pyvicon_datastream. Install it with "
        "`pip install pyvicon-datastream`."
    )


def connect(host):
    client = pv.PyViconDatastream()
    print(f"Connecting to {host} ...")
    ret = client.connect(host)
    if ret != pv.Result.Success or not client.is_connected():
        sys.exit(f"Failed to connect to Vicon DataStream Server ({ret}).")
    client.set_stream_mode(pv.StreamMode.ClientPull)
    print("Connected.")
    return client


def record(client, duration_s=None):
    rows = []
    start_wall = time.time()

    print("Recording frame times... press Ctrl+C to stop.")
    try:
        while True:
            if duration_s is not None and (time.time() - start_wall) > duration_s:
                break

            ret = client.get_frame()
            recv_time = time.time()  # host wall clock -- must be NTP-synced
            if ret != pv.Result.Success:
                continue  # e.g. Result.NoFrame -- server hasn't produced a new frame yet

            frame_number = client.get_frame_number()
            latency = client.get_latency_total()
            corrected_time = recv_time - latency

            rows.append({
                "frame_number": frame_number,
                "recv_unix_time": recv_time,
                "latency_total_s": latency,
                "corrected_unix_time": corrected_time,
            })
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"Captured {len(rows)} frames.")
    return rows


def smooth_timestamps(rows, fps_hint=None):
    """
    Fit corrected_unix_time = slope * frame_number + intercept via ordinary
    least squares. Removes host-loop jitter, preserves true frame cadence.
    """
    if not rows:
        return rows

    xs = [r["frame_number"] for r in rows]
    ys = [r["corrected_unix_time"] for r in rows]
    n = len(xs)

    if n < 2:
        for r in rows:
            r["smoothed_unix_time"] = r["corrected_unix_time"]
        return rows

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    slope = cov / var if var else 0.0
    intercept = mean_y - slope * mean_x

    if fps_hint:
        expected_slope = 1.0 / fps_hint
        if expected_slope and abs(slope - expected_slope) / expected_slope > 0.05:
            print(
                f"Warning: fitted frame interval ({slope * 1000:.3f} ms) differs "
                f"from expected ({expected_slope * 1000:.3f} ms) by >5%. "
                f"Check for dropped frames, a wrong --fps value, or clock issues."
            )

    for r in rows:
        r["smoothed_unix_time"] = slope * r["frame_number"] + intercept

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Log Vicon frame numbers + Unix timestamps only (no marker data)."
    )
    parser.add_argument(
        "--host", default="localhost:801",
        help="Vicon DataStream server address, e.g. 171.164.119.64:801",
    )
    parser.add_argument("--out", default="frame_times.csv", help="Output CSV path")
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Recording duration in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Known Vicon capture rate, used only to sanity-check the timestamp fit",
    )
    args = parser.parse_args()

    client = connect(args.host)
    rows = record(client, duration_s=args.duration)
    rows = smooth_timestamps(rows, fps_hint=args.fps)
    client.disconnect()

    if not rows:
        print("No frames captured, nothing to save.")
        return

    fieldnames = ["frame_number", "recv_unix_time", "latency_total_s",
                  "corrected_unix_time", "smoothed_unix_time"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved frame/time log to {args.out}")


if __name__ == "__main__":
    main()