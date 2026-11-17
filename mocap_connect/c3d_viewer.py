#!/usr/bin/env python3
"""
c3d_viewer.py

Play back the markers in a C3D file in the browser (viser), with a checkbox to
show/hide every marker and every segment.

  python3 c3d_viewer.py ~/ArgusArmExperiment/mocap_data/dynamic1.c3d
  # then open http://localhost:8080

Markers are grouped into segments by label prefix (torso_1, torso_2 -> torso),
which is how bag2c3d.py names them. A marker missing on a frame (NaN or
residual < 0) is hidden on that frame. The up axis is read from POINT:Y_SCREEN
when the file has it (bag2c3d.py writes it); override with --up.
"""

import argparse
import threading
import time
from pathlib import Path

import ezc3d
import numpy as np
import viser

UNIT_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
SPEEDS = {"0.1x": 0.1, "0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
           (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207)]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_c3d(path):
    """Return labels, rate (Hz), xyz (n_frames, n_markers, 3) in m with NaN = missing, up axis."""
    c = ezc3d.c3d(str(path))
    point = c["parameters"]["POINT"]
    labels = [s.strip() for s in point["LABELS"]["value"]]
    rate = float(c["header"]["points"]["frame_rate"])

    units = (point["UNITS"]["value"] or ["mm"])[0].strip().lower() if "UNITS" in point else "mm"
    if units not in UNIT_TO_M:
        print(f"[warn] unknown POINT:UNITS '{units}', assuming mm")
    xyz = c["data"]["points"][:3].transpose(2, 1, 0) * UNIT_TO_M.get(units, 1e-3)
    xyz[c["data"]["meta_points"]["residuals"][0].T < 0] = np.nan

    up = "+z"
    if "Y_SCREEN" in point and point["Y_SCREEN"]["value"]:
        up = point["Y_SCREEN"]["value"][0].strip().lower()
    return labels, rate, xyz, up


def segment_of(label):
    head, sep, tail = label.rpartition("_")
    return head if sep and tail.isdigit() else "other"


def uv_sphere(n_lat=8, n_lon=12):
    """Unit sphere mesh: (vertices (V, 3), faces (F, 3)), outward winding."""
    t, p = np.meshgrid(np.linspace(0, np.pi, n_lat + 1),
                       np.linspace(0, 2 * np.pi, n_lon, endpoint=False), indexing="ij")
    verts = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            faces += [(a, a + n_lon, b), (b, a + n_lon, b + n_lon)]
    return verts.reshape(-1, 3).astype(np.float32), np.array(faces, dtype=np.uint32)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("c3d", help="C3D file")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--up", choices=["+x", "+y", "+z", "-x", "-y", "-z"],
                    help="up axis (default: from POINT:Y_SCREEN, else +z)")
    args = ap.parse_args()

    labels, rate, xyz, up = load_c3d(args.c3d)
    up = args.up or up
    n_frames, n_markers, _ = xyz.shape
    valid = ~np.isnan(xyz).any(axis=2)                    # (F, M)
    print(f"[info] {Path(args.c3d).name}: {n_markers} markers x {n_frames} frames "
          f"@ {rate:g} Hz ({n_frames / rate:.2f} s), up {up}")

    segments = {}                                         # segment -> [marker columns]
    for j, lab in enumerate(labels):
        segments.setdefault(segment_of(lab), []).append(j)
    colors = np.zeros((n_markers, 3), dtype=np.uint8)
    for k, cols in enumerate(segments.values()):
        colors[cols] = PALETTE[k % len(PALETTE)]

    server = viser.ViserServer(host=args.host, port=args.port)

    # ---- Scene ----------------------------------------------------------- #
    up_axis = "xyz".index(up[1])
    up_vec = np.zeros(3)
    up_vec[up_axis] = 1.0 if up[0] == "+" else -1.0
    server.scene.set_up_direction(up)
    server.scene.add_grid("/grid", width=4.0, height=4.0, cell_size=0.1, section_size=0.5,
                          plane={0: "yz", 1: "xz", 2: "xy"}[up_axis])
    server.scene.add_frame("/origin", axes_length=0.15, axes_radius=0.004)

    center = np.nanmean(xyz.reshape(-1, 3), axis=0) if valid.any() else np.zeros(3)
    side = np.eye(3)[(up_axis + 1) % 3]
    server.initial_camera.look_at = tuple(center)
    server.initial_camera.position = tuple(center - 1.8 * side + np.cross(up_vec, side) * 0.8
                                           + 0.6 * up_vec)
    server.initial_camera.up = tuple(up_vec)

    verts, faces = uv_sphere()
    spheres = server.scene.add_batched_meshes_simple(
        "/markers", verts, faces,
        batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (n_markers, 1)),
        batched_positions=np.zeros((n_markers, 3)),
        batched_scales=np.zeros(n_markers),
        batched_colors=colors, lod="off", cast_shadow=False, receive_shadow=False)
    label_nodes = [server.scene.add_label(f"/labels/{lab}", lab, visible=False,
                                          font_screen_scale=0.7, anchor="bottom-left")
                   for lab in labels]

    # ---- GUI ------------------------------------------------------------- #
    with server.gui.add_folder("Playback"):
        play_btn = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(n_frames - 1, 1),
                                             step=1, initial_value=0)
        speed = server.gui.add_dropdown("Speed", list(SPEEDS), initial_value="1x")
        loop = server.gui.add_checkbox("Loop", True)
        status = server.gui.add_markdown("")
    with server.gui.add_folder("Display"):
        size = server.gui.add_slider("Marker size (mm)", min=2, max=40, step=1,
                                     initial_value=14)
        show_labels = server.gui.add_checkbox("Labels", False)
    marker_cbs = []
    with server.gui.add_folder("Markers"):
        all_btns = server.gui.add_button_group("All", ["Show all", "Hide all"])
        seg_cbs = []
        for seg, cols in segments.items():
            with server.gui.add_folder(seg, expand_by_default=False):
                seg_cb = server.gui.add_checkbox(f"{seg} (all)", True)
                cbs = [server.gui.add_checkbox(
                           labels[j], True,
                           hint=f"tracked in {100 * valid[:, j].mean():.1f}% of frames")
                       for j in cols]
            marker_cbs += cbs
            seg_cbs.append(seg_cb)

            def set_segment(_, seg_cb=seg_cb, cbs=cbs):
                for cb in cbs:
                    cb.value = seg_cb.value
            seg_cb.on_update(set_segment)

    lock = threading.Lock()

    def render(_=None):
        with lock:
            f = int(frame_slider.value)
            shown = valid[f] & np.array([cb.value for cb in marker_cbs])
            pos = np.nan_to_num(xyz[f]).astype(np.float32)
            with server.atomic():
                spheres.batched_positions = pos
                spheres.batched_scales = np.where(shown, size.value * 5e-4, 0.0).astype(np.float32)
                if show_labels.value:
                    for j, node in enumerate(label_nodes):
                        node.visible = bool(shown[j])
                        if shown[j]:
                            node.position = tuple(pos[j])
            status.content = (f"frame **{f}** / {n_frames - 1} &nbsp; t = {f / rate:.3f} s"
                              f" &nbsp; {int(shown.sum())}/{n_markers} markers")

    def on_labels(_):
        if not show_labels.value:
            with server.atomic():
                for node in label_nodes:
                    node.visible = False
        render()

    def on_all(_):
        for seg_cb in seg_cbs:
            seg_cb.value = all_btns.value == "Show all"

    playing = threading.Event()

    def on_play(_):
        if playing.is_set():
            playing.clear()
            play_btn.label, play_btn.icon = "Play", viser.Icon.PLAYER_PLAY
        else:
            if frame_slider.value >= n_frames - 1:
                frame_slider.value = 0
            playing.set()
            play_btn.label, play_btn.icon = "Pause", viser.Icon.PLAYER_PAUSE

    frame_slider.on_update(render)
    size.on_update(render)
    show_labels.on_update(on_labels)
    all_btns.on_click(on_all)
    play_btn.on_click(on_play)
    for cb in marker_cbs:
        cb.on_update(render)
    render()

    # ---- Playback loop --------------------------------------------------- #
    t_frame, last = 0.0, time.perf_counter()
    while True:
        time.sleep(1 / 60)
        now = time.perf_counter()
        dt, last = now - last, now
        if not playing.is_set():
            continue
        if int(t_frame) != frame_slider.value:            # user moved the slider
            t_frame = float(frame_slider.value)
        t_frame += dt * rate * SPEEDS[speed.value]
        if t_frame > n_frames - 1:
            if loop.value:
                t_frame %= n_frames
            else:
                t_frame = n_frames - 1
                on_play(None)
        if int(t_frame) != frame_slider.value:
            frame_slider.value = int(t_frame)


if __name__ == "__main__":
    main()
