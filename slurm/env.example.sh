#!/bin/bash
# Copy to slurm/env.sh and edit. Sourced by the job scripts in slurm/: loads the software environment and the
# path variables of config.sh. Nothing here is specific to one cluster except what you put in.

# module load cuda/12.4                         # if your cluster needs it
# source /path/to/venv/bin/activate             # environment with requirements.txt and `pip install -e .`
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO
# shellcheck disable=SC1091
source "$REPO/config.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$RESULTS_DIR"
echo "[env] python: $(command -v python) ($(python -V 2>&1))"
