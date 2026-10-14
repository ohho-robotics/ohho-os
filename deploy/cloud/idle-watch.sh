#!/bin/sh
# Guest backstop for idle shutdown (OHH-216). 900 seconds = 15 minutes.
# The tested control plane is ohho.cloud.reap. This script is for the VM.
# This repository does not execute this script.
#
# An established connection to local rosbridge (9090) or web video (8080)
# refreshes /var/lib/ohho/heartbeat. If that file is older than 900 seconds
# the guest shuts down. Boot creates the file, so a new VM has 15 minutes
# to accept a client before this watch will halt it.

set -eu

HEARTBEAT=/var/lib/ohho/heartbeat
IDLE=900

mkdir -p /var/lib/ohho

active=0
for table in /proc/net/tcp /proc/net/tcp6; do
    if [ ! -f "$table" ]; then
        continue
    fi
    # 9090 = 0x2382, 8080 = 0x1F90. TCP state 01 is ESTABLISHED.
    if awk 'NR > 1 {
        split($2, local, ":")
        port = toupper(local[2])
        state = toupper($4)
        if ((port == "2382" || port == "1F90") && state == "01") {
            found = 1
        }
    }
    END { exit found ? 0 : 1 }' "$table"; then
        active=1
    fi
done

if [ "$active" -eq 1 ] || [ ! -f "$HEARTBEAT" ]; then
    touch "$HEARTBEAT"
    exit 0
fi

now=$(date +%s)
mtime=$(stat -c %Y "$HEARTBEAT")
age=$((now - mtime))
if [ "$age" -ge "$IDLE" ]; then
    echo "idle for ${age}s (limit ${IDLE}s); shutting down"
    shutdown -h now
fi
