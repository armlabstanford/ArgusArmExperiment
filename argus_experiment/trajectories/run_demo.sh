#!/usr/bin/env bash
#
# Demo: every trajectory type and axis, one after the other.
# Each one runs: home -> trajectory start -> 1 period -> home -> hold, then the next.
#
# Usage:
#     bash argus_experiment/trajectories/demo.sh TRIAL_NAME [--sim]
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 TRIAL_NAME [--sim] ..." >&2
    exit 1
fi
TRIAL_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIN_VEL=0.5     # m/s   — linear speed (constant for sawtooth, peak for sinusoid)
ANG_VEL=2.0     # rad/s — angular speed (constant for sawtooth, peak for sinusoid)
DT=0.005        # s     — control / recording timestep
PERIODS=1       #         periods per trajectory
END_DWELL=1     # s     — hold at each end of travel
HOME_HOLD=1     # s     — hold at home between trajectories

# Run order. Comment out a line with # to skip it.
TRAJS=(
    "sawtooth:linear:x:$LIN_VEL"
    "sawtooth:linear:y:$LIN_VEL"
    "sawtooth:linear:z:$LIN_VEL"
    "sinusoidal:linear:x:$LIN_VEL"
    "sinusoidal:linear:y:$LIN_VEL"
    "sinusoidal:linear:z:$LIN_VEL"
    "sawtooth:angular:roll:$ANG_VEL"
    "sawtooth:angular:pitch:$ANG_VEL"
    "sawtooth:angular:yaw:$ANG_VEL"
    "sinusoidal:angular:roll:$ANG_VEL"
    "sinusoidal:angular:pitch:$ANG_VEL"
    "sinusoidal:angular:yaw:$ANG_VEL"
)

TRAJ_ARGS=()
for t in "${TRAJS[@]}"; do
    TRAJ_ARGS+=(--traj "$t")
done

python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
    "${TRAJ_ARGS[@]}" \
    --periods "$PERIODS" \
    --dt "$DT" \
    --end-dwell "$END_DWELL" \
    --home-hold "$HOME_HOLD" \
    "$@"
