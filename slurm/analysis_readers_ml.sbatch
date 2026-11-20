#!/bin/bash
# App. E readers of OmniLearned-m and -l: attention of the summary tokens on one added photon at f = 1e-3 for every
# head of the backbone and the classifier, and activation patching per head (scripts/analysis), with the settings
# of the stored results. One GPU; the large model needs about 1 h for the patching.
#   SIZES="m l" sbatch slurm/analysis_readers_ml.sbatch
# Set the account, partition and wall time for your cluster (placeholders here, or on the sbatch command line).
#SBATCH --job-name=analysis-readers
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source slurm/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=scripts/analysis
for SIZE in ${SIZES:-m l}; do
  if [ "$SIZE" = l ]; then N=50; B=32; else N=500; B=250; fi
  python $A/attention_heads.py --size "$SIZE" --n 1000 --device cuda --threads 8 --classifier \
    --name readers_attention || echo "FAILED attention $SIZE"
  python $A/activation_patching.py --size "$SIZE" --n $N --batch $B --device cuda --threads 8 \
    --compute-probes photon_0.3GeV neutral_hadron_0.5GeV photon_3GeV split2 || echo "FAILED patching $SIZE"
done
echo finished
