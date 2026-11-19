#!/usr/bin/env bash
#
# Still experiment, one round per run: the 10 random poses of
# run_still_random.sh, one separate recording each, held still for 60 s. Run it
# again for the next round (r2, r3, ...); every round visits the same poses in
# the same order. Same protocol as run_finalexp.sh; each recording gets its own
# ARGUS bag and mocap bag:
#
#   robot up -> check the path -> move to pose K -> start ARGUS, then mocap
#   -> hold still 60 s -> stop mocap and ARGUS -> arm home -> robot closed
#
# The arm moves home only after both recordings have closed. After every
# recording, pane 4 moves the finished ARGUS bags off the Pi into
# bags/SESSION_TAG while the next recording runs. The poses and the hold time
# are defined in argus_experiment/trajectories/run_stillexp.py.
#
# Usage:
#     ./run_stillexp.sh              # hardware: the next round, in tmux (~15 min)
#     ./run_stillexp.sh --sim [N]    # MuJoCo only, first N recordings of round 1 (default 10);
#                                    # no ARGUS, mocap or transfers (conda yam_arms active)
#     ./run_stillexp.sh --schedule   # print the next round's order (conda yam_arms active)
#
# To compare the poses with an earlier still_random run's trajectories.txt:
#     python argus_experiment/trajectories/run_stillexp.py poses
#
# Which round: the lowest-numbered round with a recording not yet in
# mocap_data/SESSION_TAG/stillexp_done.txt. A finished round moves the next run
# on to the next round; an interrupted one is resumed (finished recordings are
# skipped). Any failure (a recording that doesn't start or close, a robot error,
# Ctrl-C) stops the recordings, homes the arm and ends the run. The failed
# recording's partial bags are kept, and its retry is named NAME_a2 (then _a3, ...).
#
# Panes:
#   1 ARGUS     ./argus_connect_tools/start_argus_recording.sh NAME, once per recording
#   2 mocap     ros2 launch record_mocap.launch.py bag_name:=SESSION_TAG/NAME, once per recording
#   3 control   this script and run_stillexp.py (conda yam_arms)
#   4 transfer  ./run_stillexp.sh --transfer after each recording
#
# Data:
#   names  still_r<round>_pose<NN>, e.g. still_r1_pose07
#   mocap  ~/ArgusArmExperiment/mocap_data/SESSION_TAG/NAME/
#   ARGUS  ~/ArgusArmExperiment/bags/SESSION_TAG/NAME/   (moved off the Pi's ~/bags)
#   log    ~/ArgusArmExperiment/mocap_data/SESSION_TAG/stillexp_r<round>_<date_time>.log
#          (pane 3 output: each pose's joint angles, and EVENT lines with the
#          wall-clock times of recording / hold_end / stopped, on the same clock as
#          mocap header.stamp)
#   No robot arm data is saved.
#
# Ctrl-C in pane 3 stops the run cleanly (as above). Ctrl-b d detaches and leaves it
# running. Afterwards: tmux kill-session -t stillexp_SESSION_TAG. Its tmux session
# is separate from run_finalexp.sh's, but run only one of the two at a time.
#
# Before running: the same setup as run_experiment.sh (arm powered, can0 up,
# Pi reachable over the hotspot, Motive streaming).
#
set -euo pipefail

SESSION_TAG=11_17   # day folder for this session's mocap bags, ARGUS bags and log
PI=argus@192.168.50.10
ARGUS_TIMEOUT=30    # s - abort if the ARGUS bag hasn't appeared on the Pi by then
MOCAP_TIMEOUT=30    # s - abort if the mocap bag folder hasn't appeared by then
STOP_TIMEOUT=60     # s - how long to wait for both recordings to close
CONDA_ENV=yam_arms
ROS_SETUP=/opt/ros/humble/setup.bash
SSH_CHECK=(ssh -o BatchMode=yes -o ConnectTimeout=5)   # non-interactive checks on the Pi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$REPO_DIR/$(basename "${BASH_SOURCE[0]}")"
ARGUS_SCRIPT="$REPO_DIR/argus_connect_tools/start_argus_recording.sh"
MOCAP_SETUP="$REPO_DIR/mocap_connect/install/setup.bash"
MOCAP_LAUNCH="$REPO_DIR/mocap_connect/launch/record_mocap.launch.py"
TRAJ_PY="$REPO_DIR/argus_experiment/trajectories/run_stillexp.py"
MOCAP_DIR="$HOME/ArgusArmExperiment/mocap_data/$SESSION_TAG"   # BAG_DIR in record_mocap.launch.py + SESSION_TAG
ARGUS_DIR="$HOME/ArgusArmExperiment/bags/$SESSION_TAG"
DONE_FILE="$MOCAP_DIR/stillexp_done.txt"
SESSION="stillexp_$SESSION_TAG"
SEC_PER_RECORDING=90   # rough wall time per recording, for the estimate only

log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
die() { printf '[%s] ERROR: %s\n' "$(date +%T)" "$*" >&2; exit 1; }

# Type a command into a pane and press Enter, as if typed by hand.
pane_run() { tmux send-keys -t "$1" -l -- "$2" && tmux send-keys -t "$1" Enter; }

# True when the pane's shell is back at its prompt (nothing in the foreground).
pane_idle() {
    local pid fg
    pid="$(tmux display-message -p -t "$1" '#{pane_pid}' 2>/dev/null)" || return 0
    fg="$(ps -o tpgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [ -z "$fg" ] || [ "$fg" = "$pid" ]
}

pane_label() { tmux set-option -p -t "$1" @label "$2" 2>/dev/null || true; }

wait_idle() {  # PANE LABEL TIMEOUT -> 1 if still busy after TIMEOUT s
    local deadline=$((SECONDS + $3))
    while ! pane_idle "$1"; do
        if (( SECONDS >= deadline )); then
            log "WARNING: $2 still running after $3 s - check it and stop it by hand"
            return 1
        fi
        sleep 0.5
    done
}

is_done() { [ -f "$DONE_FILE" ] && awk -F'\t' -v b="$1" '$1 == b {f = 1} END {exit !f}' "$DONE_FILE"; }

# Sets ROUND and ROWS (that round's schedule, one TSV line per recording) to the
# lowest-numbered round that still has a recording not in DONE_FILE.
ROUND=0 ROWS=()
pick_round() {
    local r=1 row schedule
    while :; do
        schedule="$(python "$TRAJ_PY" schedule --round "$r")" || die "could not get the schedule from $TRAJ_PY"
        mapfile -t ROWS <<< "$schedule"
        for row in "${ROWS[@]}"; do
            is_done "${row%%$'\t'*}" || { ROUND=$r; return 0; }
        done
        r=$((r + 1))
    done
}

count_done() { local row n=0; for row in "${ROWS[@]}"; do is_done "${row%%$'\t'*}" && n=$((n + 1)); done; echo "$n"; }

# 0 if ~/bags/NAME exists on the Pi, 1 if not; exits if the Pi can't be reached.
pi_has() {
    local r
    r="$("${SSH_CHECK[@]}" "$PI" "[ -e bags/$1 ] && echo yes || echo no")" \
        || die "cannot reach the Pi ($PI) - is the hotspot up?"
    [ "$r" = yes ]
}

# ---------------------------------------------------------------------------
# --transfer: move every closed ARGUS bag off the Pi (runs in pane 4)
# ---------------------------------------------------------------------------
run_transfer() {
    local bags b
    mkdir -p "$ARGUS_DIR"
    bags=$(ssh "$PI" 'cd ~/bags && for d in */; do [ -f "$d/metadata.yaml" ] && echo "${d%/}"; done; exit 0')
    echo "Moving: ${bags:-none}"
    for b in $bags; do
        rsync -avP "$PI:bags/$b" "$ARGUS_DIR" && ssh "$PI" "rm -rf -- ~/bags/'$b'" && echo "Deleted from Pi: $b"
    done
}

# ---------------------------------------------------------------------------
# --hook start|stop NAME: called by run_stillexp.py at the pose and after the
# hold. Drives panes 1 and 2 (ids in FE_ARGUS_PANE / FE_MOCAP_PANE).
# ---------------------------------------------------------------------------
hook_start() {
    local name="$1" deadline
    pane_idle "$FE_ARGUS_PANE" || die "ARGUS pane is still busy from an earlier recording"
    pane_idle "$FE_MOCAP_PANE" || die "mocap pane is still busy from an earlier recording"

    log "Starting ARGUS recording (pane 1): $name"
    pane_label "$FE_ARGUS_PANE" "1 ARGUS: $name"
    pane_run "$FE_ARGUS_PANE" "$(printf '%q %q' "$ARGUS_SCRIPT" "$name")"
    deadline=$((SECONDS + ARGUS_TIMEOUT))
    sleep 2
    until "${SSH_CHECK[@]}" "$PI" "[ -d bags/$name ]" 2>/dev/null; do
        pane_idle "$FE_ARGUS_PANE" && die "ARGUS recording exited - see pane 1"
        (( SECONDS < deadline )) || die "ARGUS bag ~/bags/$name not on the Pi after ${ARGUS_TIMEOUT}s - see pane 1"
        sleep 1
    done
    log "ARGUS recording"

    log "Starting mocap recording (pane 2): $SESSION_TAG/$name"
    pane_label "$FE_MOCAP_PANE" "2 mocap: $SESSION_TAG/$name"
    # ROS 2 must run without conda; deactivate up to 3 stacked envs.
    pane_run "$FE_MOCAP_PANE" "for _ in 1 2 3; do conda deactivate 2>/dev/null; done; source $ROS_SETUP && source $(printf %q "$MOCAP_SETUP") && ros2 launch $(printf %q "$MOCAP_LAUNCH") bag_name:=$SESSION_TAG/$name"
    deadline=$((SECONDS + MOCAP_TIMEOUT))
    sleep 2
    until [ -d "$MOCAP_DIR/$name" ]; do
        pane_idle "$FE_MOCAP_PANE" && die "mocap recording exited - see pane 2"
        (( SECONDS < deadline )) || die "mocap bag not created after ${MOCAP_TIMEOUT}s - is Motive streaming? See pane 2"
        sleep 0.5
    done
    pane_idle "$FE_ARGUS_PANE" && die "ARGUS recording exited - see pane 1"
    log "mocap recording"
}

# Ctrl-C whichever recordings are running (mocap first, so the ARGUS recording
# covers the whole mocap one), then wait for them to close. Safe to repeat.
hook_stop() {
    local ok=0
    pane_idle "$FE_MOCAP_PANE" || { log "Stopping mocap (pane 2)"; tmux send-keys -t "$FE_MOCAP_PANE" C-c; }
    pane_idle "$FE_ARGUS_PANE" || { log "Stopping ARGUS (pane 1)"; tmux send-keys -t "$FE_ARGUS_PANE" C-c; }
    wait_idle "$FE_MOCAP_PANE" "mocap (pane 2)" "$STOP_TIMEOUT" || ok=1
    wait_idle "$FE_ARGUS_PANE" "ARGUS (pane 1)" "$STOP_TIMEOUT" || ok=1
    (( ok == 0 )) && log "Both recordings closed"
    return "$ok"
}

# ---------------------------------------------------------------------------
# --control: the recording loop, runs inside pane 3
# ---------------------------------------------------------------------------
TRANSFER_PANE=""

finish() {
    trap - EXIT INT TERM
    set +e
    hook_stop >/dev/null   # normally a no-op: run_stillexp.py already stopped them
    local n
    n="$(count_done)"
    if (( ROUND > 0 )); then
        log "Round $ROUND: $n of ${#ROWS[@]} recordings done (list: $DONE_FILE)"
        if (( n == ${#ROWS[@]} )); then
            log "Run ./run_stillexp.sh again for round $((ROUND + 1))."
        else
            log "Run ./run_stillexp.sh again to finish round $ROUND."
        fi
    fi
    pane_idle "$TRANSFER_PANE" || log "A transfer is still running in pane 4; let it finish."
    log "'tmux kill-session -t $SESSION' closes all panes."
}

start_transfer() {
    if pane_idle "$TRANSFER_PANE"; then
        pane_run "$TRANSFER_PANE" "bash $(printf %q "$SELF") --transfer"
    else
        log "Previous transfer still running; this bag goes with the next one"
    fi
}

run_control() {  # ARGUS_PANE MOCAP_PANE TRANSFER_PANE
    export FE_ARGUS_PANE="$1" FE_MOCAP_PANE="$2"
    TRANSFER_PANE="$3"
    mkdir -p "$MOCAP_DIR" "$ARGUS_DIR"
    pick_round
    # tee -i: tee is in pane 3's process group, so without it Ctrl-C would kill it
    # and break the pipe under run_stillexp.py while it is still homing the arm.
    exec > >(tee -i -a "$MOCAP_DIR/stillexp_r${ROUND}_$(date +%Y%m%d_%H%M%S).log") 2>&1
    trap finish EXIT
    trap 'echo; log "Interrupted"; exit 130' INT TERM

    local row base pose rnd name n_total n_left i=0 status
    n_total=${#ROWS[@]}
    n_left=$(( n_total - $(count_done) ))
    log "Still experiment $SESSION_TAG, round $ROUND: $n_left of $n_total recordings to go (~$(( n_left * SEC_PER_RECORDING / 60 )) min)"

    for row in "${ROWS[@]}"; do
        i=$((i + 1))
        IFS=$'\t' read -r base pose rnd <<< "$row"
        is_done "$base" && continue

        # A name already used by a failed attempt (locally or on the Pi) gets _a2, _a3, ...
        name="$base"
        local k=1
        while [ -e "$MOCAP_DIR/$name" ] || [ -e "$ARGUS_DIR/$name" ] || pi_has "$name"; do
            k=$((k + 1))
            name="${base}_a$k"
        done

        echo
        log "========== round $rnd [$i/$n_total] $name: still at pose $pose =========="
        status=0
        python -u "$TRAJ_PY" run "$name" --pose "$pose" \
            --on-ready "bash $(printf %q "$SELF") --hook start $name" \
            --on-done  "bash $(printf %q "$SELF") --hook stop $name" || status=$?
        (( status == 0 )) || die "$name failed (exit $status) - see above. Fix it, then rerun ./run_stillexp.sh to resume."

        [ -f "$MOCAP_DIR/$name/metadata.yaml" ] \
            || die "mocap bag $MOCAP_DIR/$name has no metadata.yaml (not closed cleanly)"
        "${SSH_CHECK[@]}" "$PI" "[ -f bags/$name/metadata.yaml ]" || [ -f "$ARGUS_DIR/$name/metadata.yaml" ] \
            || die "ARGUS bag $name has no metadata.yaml on the Pi (not closed cleanly, or the Pi is unreachable)"
        printf '%s\t%s\t%s\n' "$base" "$name" "$(date '+%F %T')" >> "$DONE_FILE"
        log "Done: $name"
        start_transfer
    done

    echo
    log "Round $ROUND complete. Moving the last ARGUS bags off the Pi (pane 4) ..."
    sleep 2   # let a just-typed transfer start before checking whether pane 4 is free
    wait_idle "$TRANSFER_PANE" "transfer (pane 4)" 3600 || true
    pane_run "$TRANSFER_PANE" "bash $(printf %q "$SELF") --transfer"
    sleep 2
    wait_idle "$TRANSFER_PANE" "transfer (pane 4)" 3600 || true
    log "Transfers finished - check pane 4. ARGUS bags: $ARGUS_DIR  mocap bags: $MOCAP_DIR"
}

# ---------------------------------------------------------------------------
# --sim [N]: the same loop in MuJoCo, no recordings
# ---------------------------------------------------------------------------
run_sim() {
    local limit="${1:-0}" schedule rows=() row base pose rnd i=0
    command -v python >/dev/null || die "no python on PATH - conda activate $CONDA_ENV first"
    [[ "$limit" =~ ^[0-9]+$ ]] || die "N must be a whole number: '$limit'"
    schedule="$(python "$TRAJ_PY" schedule --round 1)"
    mapfile -t rows <<< "$schedule"
    (( limit > 0 )) || limit=${#rows[@]}
    for row in "${rows[@]:0:$limit}"; do
        i=$((i + 1))
        IFS=$'\t' read -r base pose rnd <<< "$row"
        echo
        log "========== [sim $i/$limit] $base: still at pose $pose =========="
        python -u "$TRAJ_PY" run "$base" --pose "$pose" --sim \
            || die "$base failed in sim"
    done
    log "Sim run of $limit recordings complete."
}

# ---------------------------------------------------------------------------
# Launcher: checks prerequisites, builds the tmux session, attaches to it
# ---------------------------------------------------------------------------
run_main() {
    [ $# -eq 0 ] || die "unknown arguments: $* (see the header of $SELF)"
    command -v tmux >/dev/null || die "tmux is not installed (sudo apt install tmux)"
    [ -f "$TRAJ_PY" ]      || die "missing $TRAJ_PY"
    [ -f "$ARGUS_SCRIPT" ] || die "missing $ARGUS_SCRIPT"
    [ -f "$ROS_SETUP" ]    || die "missing $ROS_SETUP (ROS 2 Humble not installed?)"
    [ -f "$MOCAP_SETUP" ]  || die "missing $MOCAP_SETUP - build mocap_connect first (see mocap_connect/README.md)"
    "${SSH_CHECK[@]}" "$PI" true || die "cannot ssh to the Pi ($PI) without a password prompt - is the hotspot up?"
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        die "tmux session $SESSION already exists (attach: tmux attach -t $SESSION, close: tmux kill-session -t $SESSION)"
    fi

    # Layout: ARGUS top-left, mocap top-right, control bottom-left, transfer bottom-right.
    local p1 p2 p3 p4
    p1="$(tmux new-session -d -s "$SESSION" -c "$REPO_DIR" -P -F '#{pane_id}')"
    p2="$(tmux split-window -h -t "$p1" -c "$REPO_DIR" -P -F '#{pane_id}')"
    p3="$(tmux split-window -v -f -t "$p1" -c "$REPO_DIR" -P -F '#{pane_id}')"
    p4="$(tmux split-window -h -t "$p3" -l 35% -c "$REPO_DIR" -P -F '#{pane_id}' 2>/dev/null \
          || tmux split-window -h -t "$p3" -c "$REPO_DIR" -P -F '#{pane_id}')"
    {
        tmux set-option -t "$SESSION" mouse on   # scroll / click between panes
        tmux set-option -w -t "$p1" pane-border-status top
        tmux set-option -w -t "$p1" pane-border-format ' #{@label} '
    } 2>/dev/null || true
    pane_label "$p1" "1 ARGUS"
    pane_label "$p2" "2 mocap"
    pane_label "$p3" "3 control: still experiment $SESSION_TAG"
    pane_label "$p4" "4 transfer: Pi ~/bags -> bags/$SESSION_TAG"

    pane_run "$p3" "conda activate $CONDA_ENV && bash $(printf %q "$SELF") --control $p1 $p2 $p4"
    tmux select-pane -t "$p3"

    echo "Started tmux session $SESSION. If this terminal doesn't switch to it: tmux attach -t $SESSION"
    if [ -n "${TMUX:-}" ]; then
        tmux switch-client -t "$SESSION"
    else
        tmux attach-session -t "$SESSION"
    fi
}

case "${1:-}" in
    --control)  shift; [ $# -eq 3 ] || die "--control is internal; run: $0"; run_control "$@" ;;
    --hook)     shift; [ $# -eq 2 ] || die "usage: --hook start|stop NAME"
                case "$1" in start) hook_start "$2" ;; stop) hook_stop ;; *) die "--hook start|stop NAME" ;; esac ;;
    --transfer) run_transfer ;;
    --sim)      shift; run_sim "$@" ;;
    --schedule) pick_round
                echo "Next run: round $ROUND ($(count_done) of ${#ROWS[@]} already done)"
                printf '%s\n' "${ROWS[@]}" ;;
    *)          run_main "$@" ;;
esac
