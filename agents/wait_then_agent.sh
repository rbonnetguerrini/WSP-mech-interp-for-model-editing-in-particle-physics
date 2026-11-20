#!/bin/bash
# Waits (no model tokens) until the run's evaluation tool is up, then starts the agent.
# Usage: wait_then_agent.sh <run name>   (started detached by launch.py go; needs AGENT_RUNS_DIR, AGENT_RECORDS_DIR)
n=$1
here=$(cd "$(dirname "$0")" && pwd)
until grep -q oracle_up "$AGENT_RUNS_DIR/$n/run/ledger.json" 2>/dev/null; do sleep 60; done
echo "$(date +%FT%T) evaluation tool up, starting agent"
"${LAUNCHER_PYTHON:-python3}" "$here/launch.py" agent --name "$n"
