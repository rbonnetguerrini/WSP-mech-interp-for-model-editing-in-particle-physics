#!/bin/bash
# Score one edit on the test jets (scripts/evaluation/score_edit.py). One GPU, about 10 min for the small model.
#   EDIT=/path/to/edit.npz SIZE=s OUT=$RESULTS_DIR/scores/name.json sbatch slurm/score_edit.sbatch
#   (EDIT empty or unset: the original model). Extra arguments for score_edit.py go in SCORE_ARGS.
# Set the account, partition and wall time for your cluster (here as placeholders, or on the sbatch command line).
#SBATCH --job-name=score-edit
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source slurm/env.sh
: "${OUT:?set OUT to the output JSON path}"
SIZE="${SIZE:-s}"
A=()
[ -n "${EDIT:-}" ] && A=(--edit "$EDIT")
# shellcheck disable=SC2086
python scripts/evaluation/score_edit.py --size "$SIZE" --out "$OUT" "${A[@]}" ${SCORE_ARGS:-}
