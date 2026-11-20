#!/bin/bash
# The published SAE-guided edits of one model size from the SAE checkpoints (scripts/sae/key_edit.py), optionally
# scored (scripts/evaluation/score_edit.py). The edit itself needs no GPU (a few seconds; the large model needs
# about 20 GB of memory to load); scoring does.
#   SIZE=s SEED=0 sbatch slurm/sae_edits.sbatch            one / two / three readers + half strength, SAE seed 0
#   SIZE=m sbatch slurm/sae_edits.sbatch;  SIZE=l sbatch slurm/sae_edits.sbatch
#   SCORE=1 SIZE=s sbatch --time=01:00:00 slurm/sae_edits.sbatch
# SAE checkpoints: SAE_DIR (default: the data record's sae/ folder next to $EDITS_DIR).
# Set the account, partition and wall time for your cluster (placeholders below, or on the sbatch command line).
#SBATCH --job-name=sae-edits
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
SIZE="${SIZE:-s}"
SEED="${SEED:-0}"
case "$SIZE" in s) SNAME=small ;; m) SNAME=medium ;; l) SNAME=large ;; *) echo "SIZE must be s, m or l"; exit 1 ;; esac
OUT="$RESULTS_DIR/edits/sae/$SNAME"
D=()
[ -n "${SAE_DIR:-}" ] && D=(--sae-dir "$SAE_DIR")
SUFFIX=""
[ "$SIZE" = s ] && SUFFIX="_sae_seed$SEED"
NAMES=$(python -c "import sys; sys.path.insert(0, 'scripts/sae'); import _common as C; print(' '.join(C.EDITS['$SIZE']))")
for E in $NAMES; do
  python scripts/sae/key_edit.py --size "$SIZE" --edit "$E" --seed "$SEED" "${D[@]}" --out "$OUT/$E$SUFFIX.npz"
done
if [ "$SIZE" = s ]; then  # half strength of the three-reader edit
  python scripts/sae/key_edit.py --size s --edit three_readers_b7-6_b4-0_c1-2 --seed "$SEED" "${D[@]}" --alpha 0.5 \
    --out "$OUT/three_readers_b7-6_b4-0_c1-2_half_strength$SUFFIX.npz"
fi
if [ "${SCORE:-0}" = 1 ]; then
  for F in "$OUT"/*"$SUFFIX".npz; do
    python scripts/evaluation/score_edit.py --size "$SIZE" --edit "$F" --out "$RESULTS_DIR/scores/$SNAME/sae_$(basename "$F" .npz).json"
  done
fi
