#!/usr/bin/env bash
#
# Static holds at TRAJ_START_QPOS -- no trajectory motion.
#
# Records the arm's actual initial pose as home, then 10 times:
#   home -> TRAJ_START_QPOS -> hold 10 s -> return home -> hold 5 s
#
# Timing: ~3.5 s each way + 10 s + 5 s = ~22 s per repeat, ~3.7 min total.
#
# Usage:
#     bash argus_experiment/trajectories/run_still.sh TRIAL_NAME [--sim] [--dt S]
#
# Before EVERY new trajectory sequence, test in sim
#     bash argus_experiment/trajectories/run_still.sh TRIAL_NAME --sim
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 TRIAL_NAME [--sim] [--dt S] ..." >&2
    exit 1
fi
TRIAL_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPEATS=10      # number of start/home cycles
POSE_HOLD=1    # s    — hold at TRAJ_START_QPOS
HOME_HOLD=1     # s    — hold at home after each return
DT=0.02         # s    — control / recording timestep

python "$SCRIPT_DIR/run_still.py" "$TRIAL_NAME" \
    --repeats "$REPEATS" \
    --pose-hold "$POSE_HOLD" \
    --home-hold "$HOME_HOLD" \
    --dt "$DT" \
    "$@"
