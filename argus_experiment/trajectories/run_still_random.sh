#!/usr/bin/env bash
#
# Static holds at random poses -- no trajectory motion. Same cycle as run_still.sh,
# but each repeat holds a different random pose instead of TRAJ_START_QPOS.
#
# Records the arm's actual initial pose as home, then 10 times:
#   home -> random pose i -> hold 10 s -> return home -> hold 5 s
#
# Poses are uniform within ±RANGE rad of TRAJ_START_QPOS per joint, rejected if
# near a joint limit, self-colliding, or with link_6 / the camera below
# MIN_HEIGHT; paths from home are checked before anything moves. SEED is fixed so
# the sim test and the hardware run visit the same poses -- change it (and re-test
# in sim) for a new set.
#
# Timing: ~3.5 s each way + 10 s + 5 s = ~22 s per repeat, ~3.7 min total.
#
# Usage:
#     bash argus_experiment/trajectories/run_still_random.sh TRIAL_NAME [--sim] [--dt S] [--seed N]
#
# Before EVERY new trajectory sequence, test in sim
#     bash argus_experiment/trajectories/run_still_random.sh TRIAL_NAME --sim
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 TRIAL_NAME [--sim] [--dt S] [--seed N] ..." >&2
    exit 1
fi
TRIAL_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPEATS=10      # number of pose/home cycles, one random pose each
POSE_HOLD=1    # s    — hold at each random pose
HOME_HOLD=1     # s    — hold at home after each return
DT=0.02         # s    — control / recording timestep
SEED=0          #        RNG seed for the pose set
RANGE=0.3       # rad  — max per-joint offset from TRAJ_START_QPOS
MIN_HEIGHT=0.10 # m    — lowest allowed link_6 / camera z (base frame)

python "$SCRIPT_DIR/run_still.py" "$TRIAL_NAME" \
    --random \
    --repeats "$REPEATS" \
    --pose-hold "$POSE_HOLD" \
    --home-hold "$HOME_HOLD" \
    --dt "$DT" \
    --seed "$SEED" \
    --range "$RANGE" \
    --min-height "$MIN_HEIGHT" \
    "$@"
