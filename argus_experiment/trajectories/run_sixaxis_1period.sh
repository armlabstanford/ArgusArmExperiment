#!/usr/bin/env bash
#
# One sinusoidal period on each of the six camera-frame axes, in order:
#     x -> y -> z -> roll -> pitch -> yaw
#
# Each axis runs the usual run_trajectories.py cycle:
#     home -> TRAJ_START_QPOS -> ramp to +A -> 1 period of A*cos(wt) -> home -> hold
#
# Amplitudes come from run_trajectories.py (LINEAR_AMPLITUDE = 0.1 m,
# ANGULAR_AMPLITUDE = 10 deg). Speeds match linsinusoid_v0.1 / angsinusoid_v0.2:
#   linear  0.1 m/s peak   -> period 2*pi*0.1/0.1      = 6.3 s
#   angular 0.2 rad/s peak -> period 2*pi*0.1745/0.2   = 5.5 s
# ~19 s per axis including the moves and holds, ~2 min total.
#
# Usage:
#     bash argus_experiment/trajectories/run_sixaxis_1period.sh TRIAL_NAME [--sim]
#
# Before EVERY new trajectory sequence, test in sim
#     bash argus_experiment/trajectories/run_sixaxis_1period.sh TRIAL_NAME --sim
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 TRIAL_NAME [--sim] [--dt S] ..." >&2
    exit 1
fi
TRIAL_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIN_VELOCITY=0.1    # m/s   — peak linear speed
ANG_VELOCITY=0.2    # rad/s — peak angular speed
DT=0.02             # s     — control timestep
PERIODS=1           # one full back-and-forth per axis
END_DWELL=0.5       # s     — hold at each end of travel
HOME_HOLD=5         # s     — hold at home between axes

python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
    --traj "sinusoidal:angular:roll:$ANG_VELOCITY" \
    --traj "sinusoidal:angular:pitch:$ANG_VELOCITY" \
    --traj "sinusoidal:angular:yaw:$ANG_VELOCITY" \
    --periods "$PERIODS" \
    --dt "$DT" \
    --end-dwell "$END_DWELL" \
    --home-hold "$HOME_HOLD" \
    "$@"
