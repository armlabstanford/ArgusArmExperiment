#!/usr/bin/env bash
#
# Run a queue of EE trajectories on the YAM arm.
#
# All motion runs in the link_6 CAMERA frame (camera_site, injected from the
# hand-eye result) — roll/pitch/yaw rotate about the camera's own axes.
#
# First argument is the trial name, used to title the output data folder
# (recordings/<trial_name>_cam/).
#
# Edit the --traj lines below to set up the queue. Each is:
#     --traj WAVE:MOTION:AXIS:SPEED
#   WAVE   : sinusoidal | sawtooth
#   MOTION : linear  (AXIS x/y/z,        SPEED m/s)
#            angular (AXIS roll/pitch/yaw, SPEED rad/s)
#   SPEED  : peak for sinusoidal, constant for sawtooth
#
# Extra flags (--sim, --periods N, --dt S) are forwarded via "$@", e.g.:
#     bash argus_experiment/trajectories/run_trajectories.sh TRIAL_NAME --sim --periods 3 --dt 0.01
#
# Before EVERY new trajectory sequence, test in sim
#     bash argus_experiment/trajectories/run_trajectories.sh TRIAL_NAME --sim
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 TRIAL_NAME [--sim] [--periods N] [--dt S] ..." >&2
    exit 1
fi
TRIAL_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"



# VELOCITY=4.0
# DT=0.0025
# PERIODS=1
# END_DWELL=2
# HOME_HOLD=2

# python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
#     --traj "sawtooth:angular:roll:$VELOCITY" \
#     --traj "sawtooth:angular:pitch:$VELOCITY" \
#     --traj "sawtooth:angular:yaw:$VELOCITY" \
#     --periods "$PERIODS" \
#     --dt "$DT" \
#     --end-dwell "$END_DWELL" \
#     --home-hold "$HOME_HOLD" \
#     "$@"

VELOCITY=1.0
DT=0.005
PERIODS=1
END_DWELL=1
HOME_HOLD=1

python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
    --traj "sawtooth:linear:x:$VELOCITY" \
    --traj "sawtooth:linear:y:$VELOCITY" \
    --traj "sawtooth:linear:z:$VELOCITY" \
    --periods "$PERIODS" \
    --dt "$DT" \
    --end-dwell "$END_DWELL" \
    --home-hold "$HOME_HOLD" \
    "$@"

# # demo movements, will eventually load full sequence
# python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
#     --traj sawtooth:linear:y:0.8 \
#     --traj sawtooth:linear:z:0.8 \
#     --traj sawtooth:linear:x:0.8 \
#     "$@"


# python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
#     --traj sawtooth:angular:yaw:1.0 \
#     --traj sawtooth:angular:pitch:1.0 \
#     --traj sawtooth:angular:roll:1.0 \
#     "$@"

# python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
#     --traj sinusoidal:angular:roll:0.2 \
#     --traj sinusoidal:angular:pitch:0.2 \
#     --traj sinusoidal:angular:yaw:0.2 \
#     --traj sinusoidal:linear:x:0.1 \
#     --traj sinusoidal:linear:z:0.1 \
#     "$@"