#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
 
if [ -f detector.pid ] && kill -0 "$(cat detector.pid)" 2>/dev/null; then
    echo "Detector is already running (PID $(cat detector.pid))"
    exit 1
fi
 
nohup python train_detector.py > train_log.txt 2>&1 &
echo $! > detector.pid
echo "Detector started (PID $!). Logging to train_log.txt" 