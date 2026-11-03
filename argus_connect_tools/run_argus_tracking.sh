#!/bin/bash

REMOTE_USER="argus"
REMOTE_HOST="192.168.50.10"

gnome-terminal --tab --title="argus_tracking" -- bash -c "
ssh -t ${REMOTE_USER}@${REMOTE_HOST} './start_argus.sh' 
exec bash
"