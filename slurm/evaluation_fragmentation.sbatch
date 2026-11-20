#!/bin/bash
# Response to the fragmentation change after the removal rule (Phys. resp. of the fine-tuning and SAE edits) for the
# edits of one size, relative to the original model.   SIZE=l sbatch slurm/evaluation_fragmentation.sbatch
#SBATCH --job-name=evaluation-fragmentation
#SBATCH --time=01:00:00
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
SIZE="${SIZE:-s}"
case "$SIZE" in s) Z=small ;; m) Z=medium ;; l) Z=large ;; *) echo "SIZE must be s, m or l"; exit 1 ;; esac
E="$EDITS_DIR"
# the edits of Tab. tests_sizes: original, fine-tuning (three seeds on the small model), SAE edit, WSP target informed
EDITS=(original_model)
if [ "$SIZE" = s ]; then
  for k in 0 1 2; do EDITS+=("finetuning_attention_only_both_neutral_types_seed$k=$E/finetuning/small/attention_only_both_neutral_types_seed$k.npz"); done
  EDITS+=("sae_three_readers_b7-6_b4-0_c1-2=$E/sae/small/three_readers_b7-6_b4-0_c1-2.npz")
  for k in 0 1 2; do EDITS+=("sae_three_readers_b7-6_b4-0_c1-2_sae_seed$k=$E/sae/small/three_readers_b7-6_b4-0_c1-2_sae_seed$k.npz"); done
else
  EDITS+=("finetuning_attention_only_both_neutral_types_seed0=$E/finetuning/$Z/attention_only_both_neutral_types_seed0.npz")
  [ "$SIZE" = m ] && EDITS+=("sae_three_readers_b7-0_b11-7_b11-0=$E/sae/medium/three_readers_b7-0_b11-7_b11-0.npz")
  [ "$SIZE" = l ] && EDITS+=("sae_four_readers_b19-11_b18-17_c0-2_c3-24=$E/sae/large/four_readers_b19-11_b18-17_c0-2_c3-24.npz")
fi
for k in 1 2 3; do EDITS+=("agents_target_informed_run${k}_edit_large=$E/agents/target_informed/$Z/run$k/edit_large.npz"); done
export SOFTREPAIR_MATH_SDP=1
[ "$SIZE" = l ] && export SOFTREPAIR_MAX_BATCH=${SOFTREPAIR_MAX_BATCH:-32}
C=(); for e in "${EDITS[@]:1}"; do C+=("$e"); done
python -m softrepair.scoring fragmentation --size "$SIZE" --tag "_$Z" --cands "${C[@]}"
