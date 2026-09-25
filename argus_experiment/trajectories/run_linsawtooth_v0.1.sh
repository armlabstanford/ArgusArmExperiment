#!/usr/bin/env bash
#
# Linear sawtooth sweeps along x, y, z at 0.1 m/s.
#
# All motion runs in the link_6 CAMERA frame (camera_site, injected from the
# hand-eye result), same as run_trajectories.sh.
#
# Per axis:
#   home -> TRAJ_START_QPOS -> ramp to -edge -> 10 back-and-forth cycles,
#   pausing 5 s at each end -> return home -> hold 20 s before the next axis.
#
# Timing (LINEAR_AMPLITUDE = 0.1 m, so 0.2 m peak-to-peak at 0.1 m/s = 2 s/leg):
#   per axis  ~145 s of trajectory + ~8 s transit + 20 s home hold  = ~2.9 min
#   all three ~9 min total
#
# Usage:
#     bash argus_experiment/trajectories/run_linsawtooth_v0.1.sh TRIAL_NAME [--sim] [--dt S]
#
# Before EVERY new trajectory sequence, test in sim
#     bash argus_experiment/trajectories/run_linsawtooth_v0.1.sh TRIAL_NAME --sim
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 TRIAL_NAME [--sim] [--dt S] ..." >&2
    exit 1
fi
TRIAL_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VELOCITY=0.1    # m/s  — constant travel speed on every leg
DT=0.02       # s    — control timestep
PERIODS=10      # one period = one full back-and-forth
END_DWELL=5     # s    — hold at each end of travel before reversing
HOME_HOLD=20    # s    — hold at home between axes

python "$SCRIPT_DIR/run_trajectories.py" "$TRIAL_NAME" \
    --traj "sawtooth:linear:x:$VELOCITY" \
    --traj "sawtooth:linear:y:$VELOCITY" \
    --traj "sawtooth:linear:z:$VELOCITY" \
    --periods "$PERIODS" \
    --dt "$DT" \
    --end-dwell "$END_DWELL" \
    --home-hold "$HOME_HOLD" \
    "$@"
