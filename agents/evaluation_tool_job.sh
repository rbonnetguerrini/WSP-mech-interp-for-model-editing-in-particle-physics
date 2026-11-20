#!/bin/bash
# Evaluation tool of one agent run, on one GPU: RUN=<run dir> SIZE=s|m|l [WALL=s] [IDLE=s].
# Submitted by launch.py (tool / go) with a neutral --job-name; account and partition come from
# SLURM_ACCOUNT / SLURM_PARTITION (see config.example.sh).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
set -euo pipefail
: "${RUN:?}"
SIZE=${SIZE:-s}
export MODEL_SIZE=$SIZE SOFTREPAIR_MATH_SDP=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $SIZE in
  s) export SOFTREPAIR_EVAL_BATCH=256 ;;
  m) export SOFTREPAIR_MAX_BATCH=128 SOFTREPAIR_EVAL_BATCH=128 ;;
  l) export SOFTREPAIR_MAX_BATCH=32 SOFTREPAIR_EVAL_BATCH=32 ;;
esac
# EVALUATION_TOOL (path of agents/evaluation_tool.py) is set by launch.py; the environment with softrepair
# installed is inherited from the submitting shell (--export=ALL).
python "${EVALUATION_TOOL:?}" --run "$RUN" --wall "${WALL:-13800}" --idle "${IDLE:-5400}"
