# Copy to config.sh, edit the paths, then: source config.sh
# Every script reads its inputs and writes its outputs through these variables (src/softrepair/paths.py).

# directory with best_model_pretrain_s.pt, best_model_pretrain_m.pt, best_model_pretrain_l.pt
# (OmniLearned pretrained checkpoints, https://portal.nersc.gov/cfs/m4567/checkpoints/)
export OMNILEARNED_CHECKPOINTS=/path/to/checkpoints
# directory with the JetClass validation files JetClass_<i>.h5
# (https://portal.nersc.gov/cfs/dasrepo/omnilearned/jetclass/val/JetClass_<i>.h5)
export JETCLASS_DIR=/path/to/jetclass/val
# directory with the JetClass-II files JetClass2_9.h5, JetClass2_99.h5, JetClass2_998.h5
export JETCLASS2_DIR=/path/to/jetclass2/val
# prepared jet sets (practice, test, train, practice_ir, test_ir): from the data record, or scripts/data/build_sets.py
export DATA_DIR=/path/to/sets
# published edits and SAE checkpoints (data record)
export EDITS_DIR=/path/to/edits
# outputs
export RESULTS_DIR=/path/to/results

# optional
export MODEL_SIZE=s                  # default model size: s, m or l
# export SOFTREPAIR_MATH_SDP=1       # math attention kernel (score_edit.py sets it itself)
# export SOFTREPAIR_MAX_BATCH=32     # cap forward batches (large model)
