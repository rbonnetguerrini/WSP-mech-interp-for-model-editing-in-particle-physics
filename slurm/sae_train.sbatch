#!/bin/bash
# TopK SAE at the input of one reader, latent identification and latent-subtraction test (scripts/sae/train_sae.py).
#   READER=b7-6 sbatch --array=0-2 slurm/sae_train.sbatch                 small model, seeds 0-2 (array index = seed)
#   READER=b4-0 sbatch --array=0-2 slurm/sae_train.sbatch;  READER=c1-2 ...
#   READER=b7-6 LATENTS=512 sbatch --array=0-1 slurm/sae_train.sbatch     dictionary-size sweep (512 ... 8192)
#   SIZE=m READER=b7-0 sbatch --time=01:00:00 slurm/sae_train.sbatch      medium readers: b7-0, b11-7, b11-0
#   SIZE=l READER=b19-11 sbatch --time=03:00:00 slurm/sae_train.sbatch    large readers: b19-11, b18-17, c0-2, c3-24
#   CHECKPOINT=/path/latents4096_top32_seed0.pt READER=b7-6 sbatch slurm/sae_train.sbatch   identification only
#   STEPS=200 NJETCLASS=50 NJETCLASS2=200 READER=b7-6 sbatch slurm/sae_train.sbatch   short test of the code path
# The published SAEs used the training-batch seed SEED + 1 (passed below). Time on one A100 64 GB: about 4 min
# (small), 18 min (medium), 70 min (large). Set the account, partition and wall time for your cluster.
#SBATCH --job-name=sae-train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=%x.%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source slurm/env.sh
SIZE="${SIZE:-s}"
: "${READER:?set READER, e.g. b7-6}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=(--size "$SIZE" --reader "$READER" --device cuda)
if [ -n "${CHECKPOINT:-}" ]; then
  A+=(--checkpoint "$CHECKPOINT")
else
  A+=(--seed "$SEED" --batch-seed $((SEED + 1)))
  [ -n "${LATENTS:-}" ] && A+=(--latents "$LATENTS")
  [ -n "${STEPS:-}" ] && A+=(--steps "$STEPS")
  [ -n "${NJETCLASS:-}" ] && A+=(--n-jetclass "$NJETCLASS")
  [ -n "${NJETCLASS2:-}" ] && A+=(--n-jetclass2 "$NJETCLASS2")
  [ -n "${OUT:-}" ] && A+=(--out "$OUT")
fi
python scripts/sae/train_sae.py "${A[@]}"
