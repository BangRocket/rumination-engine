#!/bin/bash
# Wrapper script for the Rumination Engine.
# Restarts on exit code 42 (intentional restart via restart_self tool).
# Any other exit code stops the loop.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MIN_RESTART_INTERVAL=30  # seconds

while true; do
    start_time=$(date +%s)

    echo "[run.sh] Starting ruminate.py..."
    poetry run python -u ruminate.py
    exit_code=$?

    if [ $exit_code -ne 42 ]; then
        echo "[run.sh] Process exited with code $exit_code. Stopping."
        exit $exit_code
    fi

    # Exit code 42 = intentional restart
    elapsed=$(( $(date +%s) - start_time ))
    if [ $elapsed -lt $MIN_RESTART_INTERVAL ]; then
        wait_time=$(( MIN_RESTART_INTERVAL - elapsed ))
        echo "[run.sh] Restart cooldown: waiting ${wait_time}s..."
        sleep $wait_time
    fi

    echo "[run.sh] Restarting..."
done
