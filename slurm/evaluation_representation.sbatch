#!/bin/bash
# Representation test (Sec. 4, Fig. latent): UMAP and 15-nearest-neighbour JetClass-II shares of the classifier input,
# original model and selected WSP edit (scripts/analysis/latent_umap.py; needs umap-learn).
#SBATCH --job-name=evaluation-representation
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
J="$JETCLASS2_DIR/JetClass2_9.h5"
O="$RESULTS_DIR/figures/small/representation"
python scripts/analysis/latent_umap.py --jetclass2 "$J" --n 1500 --size s --name original_model --out-dir "$O"
python scripts/analysis/latent_umap.py --jetclass2 "$J" --n 1500 --size s --name agents_target_informed_run2_edit_large \
  --edit "$EDITS_DIR/agents/target_informed/small/run2/edit_large.npz" --out-dir "$O"
python scripts/evaluation/representation.py
