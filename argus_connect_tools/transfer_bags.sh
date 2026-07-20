#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <local_bag_path>"
    exit 1
fi

PATH_NAME="$1"

REMOTE_USER="argus"
REMOTE_HOST="192.168.50.10"

rsync -avP "${REMOTE_USER}@${REMOTE_HOST}:~/bags/" "$PATH_NAME"