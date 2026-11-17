"""Browser (viser) stand-in for the MuJoCo passive viewer, for --sim over ssh.

Duck-types the three calls run_trajectories.py makes on the viewer:
is_running(), sync(), close(). Draws the arm's mesh geoms plus an axes frame at
the camera (grasp_site @ X), refreshed from the SimRobot's MjData.
"""

import time

import mujoco
import numpy as np
import viser


class ViserSimViewer:
    def __init__(self, model, data, X=None, port=8080, max_hz=30.0):
        self._m, self._d, self._X = model, data, X
        self._min_period = 1.0 / max_hz
        self._last = 0.0
        self._server = viser.ViserServer(port=port)
        self._server.scene.add_grid("/grid", width=2.0, height=2.0)

        self._geoms = []
        for g in range(model.ngeom):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[g]
            v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            h = self._server.scene.add_mesh_simple(
                f"/robot/geom_{g}",
                model.mesh_vert[v0:v0 + nv].copy(),
                model.mesh_face[f0:f0 + nf].copy(),
                color=tuple(int(c * 255) for c in model.geom_rgba[g][:3]),
            )
            self._geoms.append((g, h))

        self._site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self._cam = None
        if X is not None and self._site >= 0:
            self._cam = self._server.scene.add_frame("/camera", axes_length=0.08, axes_radius=0.004)

        self._update()
        print(f"\nviser: open http://localhost:{port} in a browser "
              f"(forward the port if over ssh). Waiting for it to connect ...")
        while not self._server.get_clients():
            time.sleep(0.1)
        print("viser: browser connected, starting.\n")

    @staticmethod
    def _wxyz(mat9):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(mat9, dtype=float).ravel())
        return q

    def _update(self):
        with self._server.atomic():
            for g, h in self._geoms:
                h.position = self._d.geom_xpos[g].copy()
                h.wxyz = self._wxyz(self._d.geom_xmat[g])
            if self._cam is not None:
                T = np.eye(4)
                T[:3, :3] = self._d.site_xmat[self._site].reshape(3, 3)
                T[:3, 3] = self._d.site_xpos[self._site]
                T = T @ self._X
                self._cam.position = T[:3, 3]
                self._cam.wxyz = self._wxyz(T[:3, :3])

    def is_running(self):
        return True  # no window to close; Ctrl-C ends the run

    def sync(self):
        now = time.perf_counter()
        if now - self._last >= self._min_period:  # cap websocket traffic at max_hz
            self._last = now
            self._update()

    def close(self):
        self._server.stop()
