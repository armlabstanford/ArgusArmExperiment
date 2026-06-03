"""
Bimanual YAM arms trajectory optimization using PyRoki's solve_trajopt.
Plans smooth, collision-free joint-space trajectories to end-effector targets.
"""

import threading
from copy import deepcopy
from typing import Literal, Optional

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import viser
import viser.extras
import viser.transforms as vtf

from robots_realtime.robots.inverse_kinematics.pyroki_snippets._solve_ik import solve_ik
from robots_realtime.robots.inverse_kinematics.pyroki_snippets._trajopt import solve_trajopt
from robots_realtime.robots.viser.viser_base import TransformHandle, ViserAbstractBase


class YamTraj(ViserAbstractBase):
    """
    YAM robot visualization using trajectory optimization for motion planning.
    On each "Plan Trajectory" button press, solve_trajopt plans a smooth,
    collision-free path from the current joint config to the gizmo target.
    Between plans, falls back to per-frame IK so the gizmo stays live.
    """

    def __init__(
        self,
        rate: float = 100.0,
        viser_server: Optional[viser.ViserServer] = None,
        bimanual: bool = False,
        coordinate_frame: Literal["base", "world"] = "base",
    ):
        self.robot: Optional[pk.Robot] = None
        self.target_link_names = ["link_6"]
        self.joints = {"left": np.zeros(6)}
        self.coordinate_frame = coordinate_frame

        # Trajectory playback state — populated by _plan_trajectories()
        self.trajectory: dict[str, Optional[np.ndarray]] = {"left": None}
        self.traj_index: dict[str, int] = {"left": 0}
        self._planning = False

        if bimanual:
            self.target_link_names = self.target_link_names * 2
            self.joints["right"] = np.zeros(6)
            self.trajectory["right"] = None
            self.traj_index["right"] = 0

        super().__init__(rate, viser_server, bimanual=bimanual)

    def _setup_visualization(self):
        super()._setup_visualization()
        if self.bimanual:
            self.base_frame_right = self.viser_server.scene.add_frame("/base/base_right", show_axes=False)
            self.base_frame_right.position = (0.0, -0.61, 0.0)
            self.urdf_vis_right = viser.extras.ViserUrdf(
                self.viser_server, deepcopy(self.urdf), root_node_name="/base/base_right"
            )

    def _setup_solver_specific(self):
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.rest_pose = self.urdf.cfg
        self.robot_coll = pk.collision.RobotCollision.from_urdf(self.urdf)
        self.world_coll: list = []

    def _setup_gui(self):
        super()._setup_gui()

        self.timing_handle_left = self.viser_server.gui.add_number("Left Arm Time (ms)", 0.01, disabled=True)
        if self.bimanual:
            self.timing_handle_right = self.viser_server.gui.add_number("Right Arm Time (ms)", 0.01, disabled=True)

        self.timesteps_handle = self.viser_server.gui.add_slider(
            "Timesteps", min=10, max=100, step=5, initial_value=30
        )
        self.dt_handle = self.viser_server.gui.add_number(
            "dt (s)", initial_value=0.02, min=0.005, max=0.2, step=0.005
        )
        self.plan_button = self.viser_server.gui.add_button("Plan Trajectory")
        self.traj_status_handle = self.viser_server.gui.add_text(
            "Traj Status", initial_value="Idle", disabled=True
        )

        @self.plan_button.on_click
        def _plan(_):
            threading.Thread(target=self._plan_trajectories, daemon=True).start()

    def _initialize_transform_handles(self):
        if self.transform_handles["left"].control is not None:
            self.transform_handles["left"].control.position = (0.25, 0.0, 0.26)
            self.transform_handles["left"].control.wxyz = vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, np.pi / 2).wxyz
            self.transform_handles["left"].tcp_offset_frame.position = (0.0, 0.04, -0.13)

        if self.bimanual:
            if self.transform_handles["right"].control is not None:
                self.transform_handles["right"].control.remove()
                self.transform_handles["right"].tcp_offset_frame.remove()
            self.transform_handles["right"] = TransformHandle(
                tcp_offset_frame=self.viser_server.scene.add_frame(
                    "/base/base_righttarget_right/tcp_offset",
                    show_axes=False,
                    position=(0.0, 0.04, -0.13),
                    wxyz=vtf.SO3.from_rpy_radians(0.0, 0.0, 0.0).wxyz,
                ),
                control=self.viser_server.scene.add_transform_controls(
                    "/base/base_right/target_right",
                    scale=self.tf_size_handle.value,
                    position=(0.25, 0.0, 0.26),
                    wxyz=vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, np.pi / 2).wxyz,
                ),
            )

    def _update_optional_handle_sizes(self):
        pass

    def get_target_poses(self):
        target_poses = {}
        for side, handle in self.transform_handles.items():
            if handle.control is None:
                continue
            control_tf = vtf.SE3(np.array([*handle.control.wxyz, *handle.control.position]))
            tcp_offset_tf = vtf.SE3(np.array([*handle.tcp_offset_frame.wxyz, *handle.tcp_offset_frame.position]))
            target_poses[side] = control_tf @ tcp_offset_tf
        return target_poses

    def _get_current_ee_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, wxyz) of the target link via FK from current joints."""
        idx = list(self.joints.keys()).index(side)
        link_index = self.robot.links.names.index(self.target_link_names[idx])
        Ts = self.robot.forward_kinematics(jnp.array(self.joints[side]))
        se3 = jaxlie.SE3(Ts[link_index])
        return np.array(se3.translation()), np.array(se3.rotation().wxyz)

    def _plan_trajectories(self):
        """Plan trajopt trajectories from current configs to gizmo targets (background thread)."""
        if self._planning:
            return
        self._planning = True
        self.traj_status_handle.value = "Planning..."

        try:
            target_poses = self.get_target_poses()
            timesteps = int(self.timesteps_handle.value)
            dt = float(self.dt_handle.value)

            for idx, (side, target_tf) in enumerate(target_poses.items()):
                start_pos, start_wxyz = self._get_current_ee_pose(side)
                traj = solve_trajopt(
                    robot=self.robot,
                    robot_coll=self.robot_coll,
                    world_coll=self.world_coll,
                    target_link_name=self.target_link_names[idx],
                    start_position=start_pos,
                    start_wxyz=start_wxyz,
                    end_position=np.array(target_tf.translation()),
                    end_wxyz=np.array(target_tf.rotation().wxyz),
                    timesteps=timesteps,
                    dt=dt,
                )
                self.trajectory[side] = traj
                self.traj_index[side] = 0

            self.traj_status_handle.value = f"Done ({timesteps} steps)"
        except Exception as e:
            self.traj_status_handle.value = f"Error: {e}"
        finally:
            self._planning = False

    def solve_ik(self):
        """Step through a planned trajectory if available, otherwise fall back to per-frame IK."""
        if self.robot is None:
            return

        target_poses = self.get_target_poses()

        for side in list(self.joints.keys()):
            traj = self.trajectory.get(side)
            if traj is not None and self.traj_index[side] < len(traj):
                self.joints[side] = traj[self.traj_index[side]]
                self.traj_index[side] += 1
            elif side in target_poses:
                idx = list(self.joints.keys()).index(side)
                self.joints[side] = solve_ik(
                    robot=self.robot,
                    target_link_name=self.target_link_names[idx],
                    target_position=target_poses[side].translation(),
                    target_wxyz=target_poses[side].rotation().wxyz,
                )

    def update_visualization(self):
        if self.joints is not None:
            self.urdf_vis_left.update_cfg(self.joints["left"])
            if self.bimanual:
                self.urdf_vis_right.update_cfg(self.joints["right"])

    def home(self):
        self.joints["left"] = self.rest_pose.copy()
        self.trajectory["left"] = None
        if self.bimanual:
            self.joints["right"] = self.rest_pose.copy()
            self.trajectory["right"] = None

        self._initialize_transform_handles()

        self.urdf_vis_left.update_cfg(self.rest_pose)
        if self.bimanual:
            self.urdf_vis_right.update_cfg(self.rest_pose)

    def get_joint_positions(self) -> Optional[np.ndarray]:
        if self.bimanual:
            if self.joints["left"] is not None and self.joints["right"] is not None:
                return np.concatenate([self.joints["left"], self.joints["right"]])
            return None
        return self.joints["left"]

    def solve_ik_world(self, target_positions, target_wxyzs=None):
        return self.solve_ik_with_targets(target_positions, target_wxyzs, coordinate_frame="world")

    def solve_ik_base(self, target_positions, target_wxyzs=None):
        return self.solve_ik_with_targets(target_positions, target_wxyzs, coordinate_frame="base")


def main():
    viz = YamTraj(rate=100.0, bimanual=True)
    viz.run()


if __name__ == "__main__":
    main()
