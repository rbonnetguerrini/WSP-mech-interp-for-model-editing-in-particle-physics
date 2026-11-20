#!/bin/bash
# Response to softer particles (Sec. 4, Fig. ir): one photon, neutral hadron or charged hadron at 1 MeV - 10 GeV added
# to 5000 W and QCD jets (two draws), original model and the three selected repairs (scripts/analysis/soft_particle_scan.py).
#SBATCH --job-name=evaluation-softer
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
O="$RESULTS_DIR/figures/small/ir_response"
E="$EDITS_DIR"
for e in original_model \
         "finetuning_attention_only_both_neutral_types_seed0=$E/finetuning/small/attention_only_both_neutral_types_seed0.npz" \
         "sae_three_readers_b7-6_b4-0_c1-2=$E/sae/small/three_readers_b7-6_b4-0_c1-2.npz" \
         "agents_target_informed_run2_edit_large=$E/agents/target_informed/small/run2/edit_large.npz"; do
  name="${e%%=*}"; A=(); [[ "$e" == *=* ]] && A=(--edit "${e#*=}")
  python scripts/analysis/soft_particle_scan.py --n 5000 --draws 2 --threads 8 --device cuda --pt-min 0.001 --pt-max 10 \
    --families photon "neutral hadron" "charged hadron" --name "$name" "${A[@]}" --out-dir "$O"
done
python scripts/evaluation/softer_particles.py
