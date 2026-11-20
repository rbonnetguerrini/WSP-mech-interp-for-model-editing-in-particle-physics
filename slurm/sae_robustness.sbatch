#!/bin/bash
# SAE robustness checks of App. B.2 on the small model: seed and dictionary-size summary of the b7-6 SAEs
# (scripts/sae/dictionary_sweep.py, CPU only) and single neurons / residual coordinates against the SAE latents
# (scripts/sae/neuron_comparison.py, a few minutes on one GPU).
#   sbatch slurm/sae_robustness.sbatch          (SEED: SAE seed for the neuron comparison, default 0)
# SAE checkpoints: SAE_DIR (default: the data record's sae/ folder next to $EDITS_DIR).
# Set the account, partition and wall time for your cluster (placeholders below, or on the sbatch command line).
#SBATCH --job-name=sae-robustness
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source slurm/env.sh
D=()
[ -n "${SAE_DIR:-}" ] && D=(--sae-dir "$SAE_DIR")
python scripts/sae/dictionary_sweep.py --size s --reader b7-6 "${D[@]}"
python scripts/sae/neuron_comparison.py --seed "${SEED:-0}" --device cuda "${D[@]}"
