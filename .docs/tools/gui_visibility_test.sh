#!/usr/bin/env bash
set -euo pipefail

# Simple visibility smoke test for CloudWhisper main.py
# Usage: run from repo root: tools/gui_visibility_test.sh

ARTIFACTS_DIR=investigation_artifacts
mkdir -p "$ARTIFACTS_DIR"

# Kill existing processes named main.py
pids=$(pgrep -f 'main.py' || true)
if [ -n "$pids" ]; then
  echo "Killing existing main.py processes: $pids"
  pkill -f 'main.py' || true
  sleep 1
fi

# Remove old screenshot
rm -f cw_internal_screenshot.png
rm -f "$ARTIFACTS_DIR/cw_internal_screenshot.png"
rm -f "$ARTIFACTS_DIR/run.log"

# Start the app in background and capture logs
QT_QPA_PLATFORM=xcb ./venv/bin/python main.py 2>&1 | tee "$ARTIFACTS_DIR/run.log" &
APP_PID=$!
echo "$APP_PID" > "$ARTIFACTS_DIR/app.pid"

echo "Started app pid=$APP_PID"

# Wait up to N seconds for internal screenshot to appear and be non-empty
TIMEOUT=30
i=0
while [ $i -lt $TIMEOUT ]; do
  if [ -s cw_internal_screenshot.png ]; then
    cp cw_internal_screenshot.png "$ARTIFACTS_DIR/"
    echo "Found cw_internal_screenshot.png"
    exit 0
  fi
  sleep 1
  i=$((i+1))
done

echo "Timeout waiting for cw_internal_screenshot.png"
# Optionally capture last lines of the log for debugging
if [ -f "$ARTIFACTS_DIR/run.log" ]; then
  tail -n 200 "$ARTIFACTS_DIR/run.log" > "$ARTIFACTS_DIR/run.log.tail"
fi
exit 1
