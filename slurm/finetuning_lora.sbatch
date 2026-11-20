#!/bin/bash
# Fine-tuning repair (scripts/finetuning/finetune_lora.py), then its score (scripts/evaluation/score_edit.py).
#   sbatch --array=0-2 slurm/finetuning_lora.sbatch                       selected recipe, small model, seeds 0-2
#   SIZE=m sbatch --time=04:00:00 slurm/finetuning_lora.sbatch            medium model, seed 0
#   VARIANT=attention_only_photons_only sbatch slurm/finetuning_lora.sbatch
#   STEPS=100 SKIP_SCORE=1 sbatch slurm/finetuning_lora.sbatch            short test of the code path
# Variables: SIZE (s, m, l; default s), VARIANT (a name of scripts/finetuning/_common.py VARIANTS; default the
# selected recipe), SEED (default: the array index, else 0), STEPS, SKIP_SCORE=1, FT_ARGS / SCORE_ARGS (extra
# arguments). Training time on one A100 64 GB: about 26 min (small), 1.8 h (medium), 10 h (large); scoring adds
# about 10 min (small). Set the account, partition and wall time for your cluster (placeholders below, or on the
# sbatch command line).
#SBATCH --job-name=finetuning-lora
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=%x.%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source slurm/env.sh
SIZE="${SIZE:-s}"
VARIANT="${VARIANT:-attention_only_both_neutral_types}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
case "$SIZE" in s) SNAME=small ;; m) SNAME=medium ;; l) SNAME=large ;; *) echo "SIZE must be s, m or l"; exit 1 ;; esac
[ "$SIZE" = l ] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAME="${VARIANT}_seed${SEED}"
OUTDIR="$RESULTS_DIR/edits/finetuning/$SNAME"
# shellcheck disable=SC2086
python scripts/finetuning/finetune_lora.py --size "$SIZE" --variant "$VARIANT" --seed "$SEED" --name "$NAME" \
  ${STEPS:+--steps "$STEPS"} --out-dir "$OUTDIR" ${FT_ARGS:-}
[ "${SKIP_SCORE:-0}" = 1 ] && exit 0
# shellcheck disable=SC2086
python scripts/evaluation/score_edit.py --size "$SIZE" --edit "$OUTDIR/$NAME.npz" \
  --out "$RESULTS_DIR/scores/$SNAME/finetuning_$NAME.json" ${SCORE_ARGS:-}
