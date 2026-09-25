#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 TRIAL_NAME [--sim] [--dt S] ..." >&2
    exit 1
fi
TRIAL_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VELOCITY=2.0
DT=0.0025
PERIODS=10
HOME_HOLD=20

python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
    --traj "sinusoidal:angular:roll:$VELOCITY" \
    --traj "sinusoidal:angular:pitch:$VELOCITY" \
    --traj "sinusoidal:angular:yaw:$VELOCITY" \
    --periods "$PERIODS" \
    --dt "$DT" \
    --home-hold "$HOME_HOLD" \
    "$@"
