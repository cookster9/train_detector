#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f detector.pid ]; then
    echo "No detector.pid found — is it running?"
    exit 1
fi

PID=$(cat detector.pid)

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Detector stopped (PID $PID)"
    rm detector.pid
else
    echo "Process $PID not found — cleaning up stale PID file"
    rm detector.pid
fi