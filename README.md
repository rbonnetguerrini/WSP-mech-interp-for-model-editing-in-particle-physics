# Weight-space programming and mechanistic interpretability for model editing in particle physics

Code and records to reproduce the paper. The W tagger of the OmniLearned jet foundation model changes its prediction when a single soft neutral particle is added. We locate this sensitivity with mechanistic interpretability and repair it in three ways: fine-tuning, SAE-guided edits and weight-space programming (WSP) by language-model agents.

Weights, edits, SAE checkpoints, jet sets and cached results are in the companion data record on Zenodo: [10.5281/zenodo.24377675](https://doi.org/10.5281/zenodo.24377675).

## Contents

| folder | paper | content |
|---|---|---|
| `src/softrepair/` | all | library: model loading, jets and modifications, removal rule, metrics and scoring |
| `scripts/data/` | App. B | build the practice, test and training jet sets from JetClass / JetClass-II |
| `scripts/analysis/` | Sec. 2, App. A, App. E | soft-particle scans, template fit, attention, activation patching, probes, UMAP |
| `scripts/finetuning/` | Sec. 3.1, App. B.1 | low-rank fine-tuning repair |
| `scripts/sae/` | Sec. 3.2, App. B.2 | SAE training, latent identification, rank-one key edits, robustness checks |
| `agents/` | Sec. 3.3, App. C, D | evaluation tool, launcher, task descriptions with checksums, audit tools |
| `runs/` | App. C, D | records of the 27 agent runs: notes, scores, requests, transcripts, access records |
| `scripts/evaluation/` | Sec. 3.4, 4 | scoring of any edit, taggers, representation, softer particles, generator head, regression head, WSP edit selection |
| `scripts/tables/` | Tabs. 1-8 | one script per table |
| `scripts/figures/` | all figures | one script per figure, named after the figure file |
| `slurm/` | | job scripts |

## Setup

```bash
pip install -e ".[umap]"
cp config.example.sh config.sh   # set the paths, then:
source config.sh
```

Inputs:
- OmniLearned pretrained checkpoints `best_model_pretrain_{s,m,l}.pt`: https://portal.nersc.gov/cfs/m4567/checkpoints/
- JetClass and JetClass-II validation files (OmniLearned format): https://portal.nersc.gov/cfs/dasrepo/omnilearned/
- The Zenodo data record. After downloading it, run `python scripts/data/link_zenodo_sets.py` to restore the shared files of the jet sets.

Exact package versions used are in `requirements.txt`. The figures use LaTeX with the Charter font; pass `--no-tex` to render without LaTeX.

## Reproducing the results

The tables and figures are built from the score files and cached results of the data record:

```bash
export RESULTS_DIR=/path/to/zenodo/results
python scripts/tables/comparison.py
python scripts/figures/repair_vs_size.py
```

To recompute them:

```bash
# score any edit (remaining response, W efficiency, accuracy, response to physical changes, changed parameters)
python scripts/evaluation/score_edit.py --edit $EDITS_DIR/finetuning/small/attention_only_both_neutral_types_seed0.npz

# fine-tuning repair (selected recipe)
python scripts/finetuning/finetune_lora.py --help

# SAE at a reader, latent identification and key edit
python scripts/sae/train_sae.py --help
python scripts/sae/key_edit.py --help

# select the WSP edit of each run
python scripts/evaluation/select_wsp_edits.py --help
```

The compute scripts document their options with `--help`. The `slurm/` files show the settings used for each part. GPU jobs ran on one A100 each.

## Agent runs

`agents/task_descriptions/` holds the three task descriptions (flaw informed, target informed, physics origin informed). `SHA256SUMS` gives the checksum of each text as frozen before launch and of the public copy. The texts given to the agents, the evaluation-tool replies and the transcripts are published exactly as the agents saw them, so they keep the names used during the study. `agents/launch.py` rebuilds the exact task description and launch message of each run. `runs/README.md` describes the run records.

## Citation

To be added.
