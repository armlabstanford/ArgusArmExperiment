#!/usr/bin/env bash
#
# Run one experiment trial end to end: ARGUS recording, mocap recording and an
# arm trajectory, each in its own tmux pane ("terminal").
#
# Usage:
#     ./run_experiment.sh TRAJECTORY [TRIAL_NAME]
#
#     ./run_experiment.sh angsawtooth_v0.2
#     ./run_experiment.sh angsawtooth_v0.2 angsawtooth_v0.2_take2
#
# TRAJECTORY selects argus_experiment/trajectories/run_<TRAJECTORY>.sh.
# TRIAL_NAME (default: TRAJECTORY) names the ARGUS bag, the mocap bag and the
# trial output folder. Mocap won't overwrite an existing bag, so a repeat take
# of the same trajectory needs a new TRIAL_NAME.
#
# Panes:
#   1 ARGUS       ./argus_connect_tools/start_argus_recording.sh TRIAL_NAME
#   2 mocap       (ROS 2) ros2 launch mocap_connect/launch/record_mocap.launch.py bag_name:=TRIAL_NAME
#   3 trajectory  (conda yam_arms) bash argus_experiment/trajectories/run_TRAJECTORY.sh TRIAL_NAME
#
# Sequence:
#   start ARGUS -> wait 30 s -> start mocap -> wait for the mocap bag folder
#   -> run trajectory -> Ctrl-C mocap, Ctrl-C ARGUS -> wait for both to exit.
#   If ARGUS exits early or the mocap bag never appears, it stops both
#   recordings before the arm moves.
#
# Ctrl-C in pane 3 aborts the run, and both recordings are still stopped
# cleanly. Detaching (Ctrl-b d) leaves the run going. The session stays open
# afterwards for scrollback; close it with:  tmux kill-session -t exp_<TRIAL_NAME>
#
# Before running: arm powered, can0 up, Motive streaming (argus_experiment/README.md).
#
set -euo pipefail

ARGUS_LEAD=30       # s - ARGUS records alone this long before mocap starts
MOCAP_TIMEOUT=30    # s - abort if the mocap bag folder hasn't appeared by then
STOP_TIMEOUT=60     # s - how long to wait for both recordings to shut down
CONDA_ENV=yam_arms
ROS_SETUP=/opt/ros/humble/setup.bash
MOCAP_BAG_DIR="$HOME/ArgusArmExperiment/mocap_data"   # must match BAG_DIR in record_mocap.launch.py

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$REPO_DIR/$(basename "${BASH_SOURCE[0]}")"
ARGUS_SCRIPT="$REPO_DIR/argus_connect_tools/start_argus_recording.sh"
MOCAP_SETUP="$REPO_DIR/mocap_connect/install/setup.bash"
MOCAP_LAUNCH="$REPO_DIR/mocap_connect/launch/record_mocap.launch.py"
TRAJ_DIR="$REPO_DIR/argus_experiment/trajectories"

log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
die() { printf '[%s] ERROR: %s\n' "$(date +%T)" "$*" >&2; exit 1; }

available_trajs() {
    local f out=()
    for f in "$TRAJ_DIR"/run_*.sh; do
        f="${f##*/run_}"
        out+=("${f%.sh}")
    done
    echo "${out[*]}"
}

usage() {
    echo "Usage: $0 TRAJECTORY [TRIAL_NAME]     e.g. $0 angsawtooth_v0.2" >&2
    echo "Trajectories: $(available_trajs)" >&2
    exit 1
}

# Type a command into a pane and press Enter, as if typed by hand.
pane_run() { tmux send-keys -t "$1" -l -- "$2" && tmux send-keys -t "$1" Enter; }

# True when the pane's shell is back at its prompt (nothing in the foreground).
pane_idle() {
    local pid fg
    pid="$(tmux display-message -p -t "$1" '#{pane_pid}' 2>/dev/null)" || return 0
    fg="$(ps -o tpgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [ -z "$fg" ] || [ "$fg" = "$pid" ]
}

# ---------------------------------------------------------------------------
# Controller: runs inside pane 3, drives panes 1 and 2.
# ---------------------------------------------------------------------------
SESSION="" ARGUS_PANE="" MOCAP_PANE=""
ARGUS_STARTED=0 MOCAP_STARTED=0

wait_stopped() {  # PANE LABEL DEADLINE
    while ! pane_idle "$1"; do
        if (( SECONDS >= $3 )); then
            log "WARNING: $2 still running after ${STOP_TIMEOUT}s - check it and stop it by hand"
            return
        fi
        sleep 0.5
    done
    log "$2 stopped"
}

# EXIT trap: Ctrl-C whichever recordings were started (mocap first, so the
# ARGUS recording covers the whole mocap one), then wait for them to close.
stop_recordings() {
    trap - EXIT INT TERM
    set +e
    (( MOCAP_STARTED )) && { log "Stopping mocap (pane 2)"; tmux send-keys -t "$MOCAP_PANE" C-c; }
    (( ARGUS_STARTED )) && { log "Stopping ARGUS (pane 1)"; tmux send-keys -t "$ARGUS_PANE" C-c; }
    local deadline=$((SECONDS + STOP_TIMEOUT))
    (( MOCAP_STARTED )) && wait_stopped "$MOCAP_PANE" "mocap (pane 2)" "$deadline"
    (( ARGUS_STARTED )) && wait_stopped "$ARGUS_PANE" "ARGUS (pane 1)" "$deadline"
    log "Done. Check panes 1 and 2 for clean shutdowns."
    log "Ctrl-b d detaches; 'tmux kill-session -t $SESSION' closes all panes."
}

run_control() {  # TRAJECTORY TRIAL_NAME SESSION ARGUS_PANE MOCAP_PANE
    local traj="$1" trial="$2"
    SESSION="$3" ARGUS_PANE="$4" MOCAP_PANE="$5"
    trap stop_recordings EXIT
    trap 'echo; log "Interrupted"; exit 130' INT TERM

    log "[1/3] Starting ARGUS recording (pane 1)"
    pane_run "$ARGUS_PANE" "$(printf '%q %q' "$ARGUS_SCRIPT" "$trial")"
    ARGUS_STARTED=1

    local t
    for (( t = ARGUS_LEAD; t > 0; t-- )); do
        printf '\r           mocap starts in %2d s ' "$t"
        sleep 1
        # Give ssh a couple of seconds to start before checking on it.
        if (( ARGUS_LEAD - t >= 2 )) && pane_idle "$ARGUS_PANE"; then
            echo
            die "ARGUS recording exited early - see pane 1. The arm was not moved."
        fi
    done
    echo

    log "[2/3] Starting mocap recording (pane 2)"
    # ROS 2 must run without conda; deactivate up to 3 stacked envs.
    pane_run "$MOCAP_PANE" "for _ in 1 2 3; do conda deactivate 2>/dev/null; done; source $ROS_SETUP && source $(printf %q "$MOCAP_SETUP") && ros2 launch $(printf %q "$MOCAP_LAUNCH") bag_name:=$trial"
    MOCAP_STARTED=1

    local bag="$MOCAP_BAG_DIR/$trial" deadline=$((SECONDS + MOCAP_TIMEOUT))
    log "      waiting for mocap bag $bag"
    until [ -d "$bag" ]; do
        (( SECONDS < deadline )) || die "mocap bag not created after ${MOCAP_TIMEOUT}s - is Motive streaming? See pane 2. The arm was not moved."
        sleep 0.5
    done
    if pane_idle "$ARGUS_PANE"; then
        die "ARGUS recording exited - see pane 1. The arm was not moved."
    fi

    log "[3/3] Running trajectory: run_$traj.sh $trial"
    local status=0
    bash "$TRAJ_DIR/run_$traj.sh" "$trial" || status=$?
    log "Trajectory finished (exit $status)"
    exit "$status"
}

# ---------------------------------------------------------------------------
# Launcher: checks prerequisites, builds the tmux session, attaches to it.
# ---------------------------------------------------------------------------
run_main() {
    { [ $# -ge 1 ] && [ $# -le 2 ]; } || usage
    local traj="$1" trial="${2:-$1}" n
    for n in "$traj" "$trial"; do
        [[ "$n" =~ ^[A-Za-z0-9._-]+$ ]] || die "names may only contain letters, digits, '.', '_' and '-': '$n'"
    done

    command -v tmux >/dev/null || die "tmux is not installed (sudo apt install tmux)"
    [ -f "$TRAJ_DIR/run_$traj.sh" ] || die "no trajectory script $TRAJ_DIR/run_$traj.sh
Trajectories: $(available_trajs)"
    [ -f "$ARGUS_SCRIPT" ] || die "missing $ARGUS_SCRIPT"
    [ -f "$ROS_SETUP" ]    || die "missing $ROS_SETUP (ROS 2 Humble not installed?)"
    [ -f "$MOCAP_SETUP" ]  || die "missing $MOCAP_SETUP - build mocap_connect first (see mocap_connect/README.md)"
    [ ! -e "$MOCAP_BAG_DIR/$trial" ] || die "mocap bag $MOCAP_BAG_DIR/$trial already exists - give a new TRIAL_NAME, e.g.: $0 $traj ${trial}_take2"

    local session="exp_${trial//[.:]/_}"   # tmux session names can't contain . or :
    if tmux has-session -t "=$session" 2>/dev/null; then
        die "tmux session $session already exists (attach: tmux attach -t $session, close: tmux kill-session -t $session)"
    fi

    # Layout: ARGUS top-left, mocap top-right, trajectory full width below.
    local p1 p2 p3
    p1="$(tmux new-session -d -s "$session" -c "$REPO_DIR" -P -F '#{pane_id}')"
    p2="$(tmux split-window -h -t "$p1" -c "$REPO_DIR" -P -F '#{pane_id}')"
    p3="$(tmux split-window -v -f -t "$p1" -c "$REPO_DIR" -P -F '#{pane_id}')"
    {
        tmux set-option -t "$session" mouse on   # scroll / click between panes
        tmux set-option -w -t "$p1" pane-border-status top
        tmux set-option -w -t "$p1" pane-border-format ' #{@label} '
        tmux set-option -p -t "$p1" @label "1 ARGUS: $trial"
        tmux set-option -p -t "$p2" @label "2 mocap: $trial"
        tmux set-option -p -t "$p3" @label "3 trajectory: run_$traj.sh $trial"
    } 2>/dev/null || true

    pane_run "$p3" "conda activate $CONDA_ENV && bash $(printf %q "$SELF") --control $traj $trial $session $p1 $p2"
    tmux select-pane -t "$p3"

    echo "Started tmux session $session. If this terminal doesn't switch to it: tmux attach -t $session"
    if [ -n "${TMUX:-}" ]; then
        tmux switch-client -t "$session"
    else
        tmux attach-session -t "$session"
    fi
}

if [ "${1:-}" = "--control" ]; then
    shift
    [ $# -eq 5 ] || die "--control is internal; run: $0 TRAJECTORY [TRIAL_NAME]"
    run_control "$@"
else
    run_main "$@"
fi
