#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 TRIAL_NAME [--sim] [--dt S] ..." >&2
    exit 1
fi
TRIAL_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VELOCITY=0.5
DT=0.005
PERIODS=10
END_DWELL=5
HOME_HOLD=20

python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
    --traj "sawtooth:linear:x:$VELOCITY" \
    --traj "sawtooth:linear:y:$VELOCITY" \
    --traj "sawtooth:linear:z:$VELOCITY" \
    --periods "$PERIODS" \
    --dt "$DT" \
    --end-dwell "$END_DWELL" \
    --home-hold "$HOME_HOLD" \
    "$@"
