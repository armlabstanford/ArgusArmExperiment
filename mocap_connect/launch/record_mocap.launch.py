"""Start the mocap4r2 OptiTrack driver and record its data to a bag.

Usage:
    ros2 launch ~/ArgusArmExperiment/mocap_connect/launch/record_mocap.launch.py bag_name:=trial_01

The bag is written to BAG_DIR/<bag_name>. Recording starts once the driver
reaches the active state. Ctrl-C stops the driver and closes the bag cleanly.
"""

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
import lifecycle_msgs.msg

# Where bags are saved. Kept outside the colcon workspace so colcon
# doesn't crawl through recordings when building.
BAG_DIR = os.path.expanduser('~/ArgusArmExperiment/mocap_data')

TOPICS = ['/markers', '/rigid_bodies']


def generate_launch_description():
    os.makedirs(BAG_DIR, exist_ok=True)

    params_file = os.path.join(
        get_package_share_directory('mocap4r2_optitrack_driver'),
        'config', 'mocap4r2_optitrack_driver_params.yaml')

    # No default value, so the launch fails immediately if bag_name is missing.
    bag_name_arg = DeclareLaunchArgument(
        'bag_name',
        description='Name of the bag to record (folder under ' + BAG_DIR + ')')
    bag_path = PathJoinSubstitution([BAG_DIR, LaunchConfiguration('bag_name')])

    driver_node = LifecycleNode(
        package='mocap4r2_optitrack_driver',
        executable='mocap4r2_optitrack_driver_main',
        name='mocap4r2_optitrack_driver_node',
        namespace='',
        output='screen',
        parameters=[params_file],
    )

    def change_state(transition_id):
        return EmitEvent(event=ChangeState(
            lifecycle_node_matcher=launch.events.matchers.matches_action(driver_node),
            transition_id=transition_id,
        ))

    # 1) Configure the driver as soon as it starts (connects to Motive).
    configure = change_state(lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE)

    # 2) Once configuring succeeds, activate it.
    #    start_state='configuring' makes this fire only after the initial
    #    configure, not after a manual deactivate.
    activate_after_configure = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=driver_node,
        start_state='configuring',
        goal_state='inactive',
        entities=[change_state(lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE)],
    ))

    # 3) Once the driver is active, start recording.
    record_after_activate = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=driver_node,
        start_state='activating',
        goal_state='active',
        entities=[
            LogInfo(msg=['Driver active, recording ', ' '.join(TOPICS), ' to ', bag_path]),
            ExecuteProcess(
                cmd=['ros2', 'bag', 'record', '-o', bag_path] + TOPICS,
                output='screen',
            ),
        ],
    ))

    return LaunchDescription([
        bag_name_arg,
        driver_node,
        activate_after_configure,
        record_after_activate,
        configure,
    ])