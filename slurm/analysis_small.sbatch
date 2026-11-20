#!/bin/bash
# Sec. 2 and App. A analyses of OmniLearned-s (scripts/analysis), with the settings of the stored results.
# One GPU, about 1 h in total. Outputs under $RESULTS_DIR/figures/small and $RESULTS_DIR/analysis.
#   sbatch slurm/analysis_small.sbatch                 (all parts)
#   PARTS="response scan attention" sbatch slurm/analysis_small.sbatch
# Parts: response (W-tagger score with one photon / neutral hadron, Fig. 2 left), scan (pT scans per particle type
# and impact-parameter swaps, Figs. 2 right / app_types; with b7/6 attention, Fig. attention middle), attention
# (attention grids, Figs. 3 left / attention left), patching (Figs. 3 right / attention right), probes (Fig. probes),
# shift (JetClass / JetClass-II class shares), latent (UMAP, Fig. 4), template (W-fraction template fit).
# Set the account, partition and wall time for your cluster (placeholders here, or on the sbatch command line).
#SBATCH --job-name=analysis-small
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=%x.%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source slurm/env.sh
PARTS="${PARTS:-response scan attention patching probes shift latent template}"
A=scripts/analysis
JETCLASS2_FILE="$JETCLASS2_DIR/JetClass2_9.h5"
T=8
has() { [[ " $PARTS " == *" $1 "* ]]; }
if has response; then
  python $A/w_tagger_response.py --size s --device cuda --threads $T || echo "FAILED response photon"
  python $A/w_tagger_response.py --size s --device cuda --threads $T --soft-pid 7 --no-scan || echo "FAILED response neutral hadron"
fi
if has scan; then
  python $A/soft_particle_scan.py --n 5000 --draws 2 --threads $T --device cuda || echo "FAILED scan"
  python $A/soft_particle_scan.py --n 5000 --draws 2 --threads $T --device cuda --attn --no-swaps \
    --families "neutral hadron" photon --name original_model_head_b7-6_attention || echo "FAILED scan attention"
  python $A/head_attention_threshold.py \
    --npz "$RESULTS_DIR/figures/small/soft_particle_scans/original_model_head_b7-6_attention.npz" || echo "FAILED threshold"
fi
if has attention; then
  python $A/attention_heads.py --n 1000 --device cuda --threads $T --classifier || echo "FAILED attention photon"
  python $A/attention_heads.py --n 1000 --device cuda --threads $T --classifier --soft-pid 7 || echo "FAILED attention nh"
  python $A/attention_heads.py --n 1000 --device cuda --threads $T --soft-pid 7 || echo "FAILED attention nh body"
fi
if has patching; then
  python $A/activation_patching.py --size s --n 400 --batch 250 --device cuda --threads $T || echo "FAILED patching"
fi
# the next three run the model on the CPU (as for the stored results)
has probes && { python $A/probes.py --jetclass2 "$JETCLASS2_FILE" --n 2000 --threads $T || echo "FAILED probes"; }
has shift && { python $A/dataset_shift.py --n 500 --threads $T --jetclass2 "$JETCLASS2_FILE" || echo "FAILED shift"; }
has latent && { python $A/latent_umap.py --jetclass2 "$JETCLASS2_FILE" --n 1500 --threads $T || echo "FAILED latent"; }
if has template; then
  python $A/template_fit_inputs.py --device cuda --threads $T && python $A/template_fit.py || echo "FAILED template"
fi
echo finished
