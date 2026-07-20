#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <bag_name>"
    exit 1
fi

BAG_NAME="$1"

REMOTE_USER="argus"
REMOTE_HOST="192.168.50.10"

ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "./record_data.sh '${BAG_NAME}'"