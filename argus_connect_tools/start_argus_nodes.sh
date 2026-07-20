#!/bin/bash

REMOTE_USER="argus"
REMOTE_HOST="192.168.50.10"

gnome-terminal --tab --title="usb_cam_node" -- bash -c "
ssh -t ${REMOTE_USER}@${REMOTE_HOST} './start_camera_node.sh' 
exec bash
"

gnome-terminal --tab --title="bmi270_fast_node" -- bash -c "
ssh -t ${REMOTE_USER}@${REMOTE_HOST} './start_imu_node.sh'
exec bash
"
