#!/bin/bash
# One-vs-rest tagger scores of the small model, original and selected WSP edit (Fig. taggers_score_distributions).
#SBATCH --job-name=evaluation-tagger-scores
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=%x.%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source slurm/env.sh
O="$RESULTS_DIR/figures/small/tagger_scores"
python scripts/evaluation/tagger_scores.py --size s --name original_model --out-dir "$O"
python scripts/evaluation/tagger_scores.py --size s --name agents_target_informed_run2_edit_large \
  --edit "$EDITS_DIR/agents/target_informed/small/run2/edit_large.npz" --out-dir "$O"
