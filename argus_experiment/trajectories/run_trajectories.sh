#!/usr/bin/env bash
#
# Run a queue of EE trajectories on the YAM arm.
#
# Edit the --traj lines below to set up the queue. Each is:
#     --traj WAVE:MOTION:AXIS:SPEED
#   WAVE   : sinusoidal | sawtooth
#   MOTION : linear  (AXIS x/y/z,        SPEED m/s)
#            angular (AXIS roll/pitch/yaw, SPEED rad/s)
#   SPEED  : peak for sinusoidal, constant for sawtooth
#
# Extra flags (--sim, --periods N, --dt S) are forwarded via "$@", e.g.:
#     bash argus_experiment/trajectories/run_trajectories.sh --sim --periods 3 --dt 0.01
# 
# Before EVERY new trajectory sequence, test in sim
#     bash argus_experiment/trajectories/run_trajectories.sh --sim
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# demo movements, will eventually load full sequence
# python "$SCRIPT_DIR/run_trajectories.py" \
#     --traj sinusoidal:linear:x:0.5 \
#     --traj sawtooth:linear:z:0.05 \
#     --traj sinusoidal:angular:yaw:0.1 \
#     --traj sawtooth:angular:roll:0.2 \
#     "$@"


python "$SCRIPT_DIR/run_trajectories.py" \
    --traj sawtooth:linear:z:0.05 \
    --traj sawtooth:angular:roll:0.2 \
    "$@"