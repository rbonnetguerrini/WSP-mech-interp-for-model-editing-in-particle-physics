# Weight-space programming and mechanistic interpretability for model editing in particle physics

Code and records to reproduce the paper.

The jet taggers of the OmniLearned foundation model are not infrared safe. Adding a single soft neutral particle (a photon or neutral hadron below the JetClass reconstruction thresholds) changes their prediction substantially. Mechanistic interpretability traces this to a dataset-provenance shortcut: the model reads the soft particle with a few attention heads (`b7/6`, `b4/0`, `c1/2` in the small model) and reassigns the jet from JetClass to JetClass-II.

We repair the flaw in three ways and compare them:

- **Fine-tuning**: low-rank updates of the body's attention modules with a dedicated loss.
- **SAE-guided edits**: rank-one key edits that remove one sparse-autoencoder latent at each reader head.
- **Weight-space programming (WSP)**: language-model agents edit the weights directly, without gradients, with at most 50 queries to the model. They receive three levels of information.

The repairs are then tested on the representation, on the nine taggers, on the generator head and on a new regression head.

Weights, edits, SAE checkpoints, jet sets and cached results are in the companion data record on Zenodo: [10.5281/zenodo.24377675](https://doi.org/10.5281/zenodo.24377675).

## Contents

| what | where |
|---|---|
| Model loading and access to attention heads | `src/softrepair/adapters/`, `src/softrepair/tools.py` |
| Added soft particles, collinear splits, physical changes of a jet | `src/softrepair/core.py`, `src/softrepair/ops.py` |
| The removal rule and the target predictions | `src/softrepair/removal_rule.py` |
| The metrics: remaining response, W efficiency, physical response, changed parameters, remaining error | `src/softrepair/scoring.py` |
| Practice, test and training jet sets | `scripts/data/build_sets.py` |
| Soft-particle scans, template fit | `scripts/analysis/soft_particle_scan.py`, `w_tagger_response.py`, `template_fit.py` |
| Attention maps and activation patching | `scripts/analysis/attention_heads.py`, `activation_patching.py` |
| Probes, dataset shift, UMAP | `scripts/analysis/probes.py`, `dataset_shift.py`, `latent_umap.py` |
| Fine-tuning loss and training | `scripts/finetuning/finetune_lora.py` |
| SAE training and latent identification | `scripts/sae/train_sae.py`, `src/softrepair/sae_model.py` |
| Rank-one key edit | `scripts/sae/key_edit.py` |
| SAE robustness (seeds, dictionary size, single neurons) | `scripts/sae/dictionary_sweep.py`, `neuron_comparison.py` |
| Task descriptions given to the agents | `agents/task_descriptions/` |
| Evaluation tool queried by the agents | `agents/evaluation_tool.py` |
| Launch, isolation and access audit of the agent runs | `agents/launch.py`, `agents/audit/` |
| Notes, transcripts, scores and access records of the 27 runs | `runs/<variant>/<size>/run<k>/` |
| Scoring of any edit, selection of the agent edits | `scripts/evaluation/score_edit.py`, `select_wsp_edits.py` |
| Nine taggers, regression head, generator head, representation, softer particles | `scripts/evaluation/` |
| Paper tables and figures | `scripts/tables/`, `scripts/figures/` |
| Job scripts | `slurm/` |

### Edit format

Every repair is stored as an `.npz` partial state dict of full replacement float32 tensors. Tensors missing from the file keep their pretrained values, so an empty file is the unchanged model. Fine-tuning, SAE edits and agent edits all use this format, and `scripts/evaluation/score_edit.py` scores any of them.

## Setup

```bash
git clone https://github.com/rbonnetguerrini/WSP-mech-interp-for-model-editing-in-particle-physics.git
cd WSP-mech-interp-for-model-editing-in-particle-physics
pip install -r requirements.txt
pip install -e ".[umap]"
cp config.example.sh config.sh   # set the paths, then:
source config.sh
```

`requirements.txt` pins the environment used for the paper (Python 3.10, PyTorch 2.6.0 with CUDA 12.4). The figures use LaTeX with the Charter font; pass `--no-tex` to render without LaTeX.

### Inputs

| input | source | variable |
|---|---|---|
| OmniLearned pretrained checkpoints `best_model_pretrain_{s,m,l}.pt` | https://portal.nersc.gov/cfs/m4567/checkpoints/ (sha256 in the Zenodo README) | `OMNILEARNED_CHECKPOINTS` |
| JetClass validation files `JetClass_<i>.h5` (OmniLearned format) | https://portal.nersc.gov/cfs/dasrepo/omnilearned/jetclass/val/ | `JETCLASS_DIR` |
| JetClass-II files `JetClass2_{9,99,998}.h5` (OmniLearned format) | https://portal.nersc.gov/cfs/dasrepo/omnilearned/jetclass2/val/ | `JETCLASS2_DIR` |
| jet sets (`jet_sets/` of the data record) | Zenodo, or rebuilt with `scripts/data/build_sets.py` | `DATA_DIR` |
| edits and SAE checkpoints (`edits/`, `sae/` of the data record) | Zenodo | `EDITS_DIR` |
| cached results (`results/` of the data record), or a new output folder | Zenodo | `RESULTS_DIR` |

After downloading the jet sets, run `python scripts/data/link_zenodo_sets.py` once. It restores the files that the scoring sets share with the practice and test sets. Without the data record, `python scripts/data/build_sets.py` rebuilds the sets from the JetClass files with the same seeds.

### Data record layout

```
edits/finetuning/<size>/                fine-tuning edits (selected recipe: attention_only_both_neutral_types_seed<k>.npz)
edits/sae/<size>/                       SAE-guided edits (e.g. small/three_readers_b7-6_b4-0_c1-2.npz)
edits/agents/<variant>/<size>/run<k>/   every edit handed in by the agent runs
sae/<size>/<reader>/                    SAE checkpoints and latent-identification records (b7-6 = body block 7, head 6)
jet_sets/                               practice, test and scoring sets
results/                                cached scores and figure inputs
regression_heads/                       the jet mass regression heads of Sec. 4
```

The selected WSP edit of the small model, used throughout Sec. 4, is `edits/agents/target_informed/small/run2/edit_large.npz`.

## Reproducing the results

### From the cached results (no GPU)

The tables and figures are built from the score files and cached results of the data record:

```bash
export RESULTS_DIR=/path/to/zenodo/results
python scripts/tables/comparison.py           # comparison of the three methods, small model
python scripts/figures/repair_vs_size.py      # remaining response against changed parameters
python scripts/analysis/print_numbers.py      # in-text numbers of Sec. 2 and App. A
```

| paper item | script |
|---|---|
| WSP variants, small model | `scripts/tables/wsp_variants.py` |
| comparison of the methods, small model | `scripts/tables/comparison.py` |
| nine taggers | `scripts/tables/taggers.py` |
| fine-tuning variants | `scripts/tables/ft_variants.py` |
| SAE-guided edit variants | `scripts/tables/sae_variants.py` |
| WSP on the three model sizes | `scripts/tables/wsp_sizes.py` |
| comparison, medium and large models | `scripts/tables/comparison_sizes.py` |
| tests of Sec. 4 on the three model sizes | `scripts/tables/tests_sizes.py` |
| every figure `<name>.pdf` | `scripts/figures/<name>.py` |

### Recomputing (GPU)

All GPU jobs of the paper ran on one NVIDIA A100 each. The `slurm/` files show the settings and job sizes used for each part; replace `YOUR_ACCOUNT` and `YOUR_GPU_PARTITION` with the values for your cluster.

**1. The flaw (Sec. 2, App. A).** Run the soft-particle scans, attention maps, activation patching, probes, dataset shift, UMAP and template fit with the scripts in `scripts/analysis/`. `slurm/analysis_small.sbatch` runs them for the small model, and `slurm/analysis_readers_ml.sbatch` runs the reader search of the medium and large models.

```bash
python scripts/analysis/w_tagger_response.py --size s
python scripts/analysis/attention_heads.py --size s --classifier
python scripts/analysis/activation_patching.py --size s
python scripts/analysis/probes.py --jetclass2 $JETCLASS2_DIR/JetClass2_9.h5
```

**2. Fine-tuning (Sec. 3.1, App. B.1).** The defaults are the selected recipe: rank 8, lambda = 30, mu = 10, 3000 steps.

```bash
python scripts/finetuning/finetune_lora.py --size s --seed 0
```

**3. SAE-guided edits (Sec. 3.2, App. B.2).** Train one SAE per reader and identify its latent, then write the rank-one key edit.

```bash
python scripts/sae/train_sae.py --size s --reader b7-6         # likewise b4-0 and c1-2
python scripts/sae/key_edit.py --size s --readers b7-6 b4-0 c1-2 --out three_readers.npz
python scripts/sae/dictionary_sweep.py --reader b7-6           # seed and dictionary-size robustness
python scripts/sae/neuron_comparison.py                        # single neurons against SAE latents
```

**4. Scoring any edit (App. B).** `score_edit.py` computes the remaining response, W efficiency, accuracy change, response to physical changes, changed parameters and remaining error.

```bash
python scripts/evaluation/score_edit.py --size s \
    --edit $EDITS_DIR/finetuning/small/attention_only_both_neutral_types_seed0.npz --out score.json
python scripts/evaluation/select_wsp_edits.py     # selected edit of each agent run (App. C)
```

**5. Tests of the repaired models (Sec. 4, App. E).** The scripts in `scripts/evaluation/` cover:

| test | script |
|---|---|
| nine taggers and the regression head | `taggers_and_regression_head.py` |
| tagger score distributions | `tagger_scores.py` |
| W efficiency | `w_efficiency.py` |
| representation | `representation.py` |
| softer particles | `softer_particles.py` |
| generator head | `generator_test.py` |

Every compute script documents its options with `--help`.

## Agent runs

`agents/` contains everything needed to run the weight-space programming study again:

- `task_descriptions/` holds the three task descriptions (flaw informed, target informed, physics origin informed). `SHA256SUMS` gives the checksum of each text as frozen before launch and of the public copy.
- `launch.py` prepares an isolated run folder, makes the other run folders unreadable, starts the evaluation tool as a GPU job and then the agent. It also rebuilds the exact task description and launch message of every run of the paper (listed in `launch_messages.md`).
- `evaluation_tool.py` is the tool the agents query for model outputs, internal activations and the scores of an edit, with a budget of 50 queries per run.
- `model_code/` is the copy of the model code the agents received.
- `audit/` holds the access audit, the interpreter hook that records every file opened, and the cost and token counter.

All runs used Claude Opus 5.5 through Claude Code 2.1.316. A run takes one to six hours, most of it waiting for the evaluation tool.

`runs/` holds the records of the 27 runs: three variants, three model sizes and three runs each. They are stored as `runs/<variant>/<size>/run<k>/`, with `runs/index.csv` as the overview and `runs/README.md` describing every file. The texts given to the agents, the evaluation-tool replies and the transcripts are published exactly as the agents saw them, so they keep the names used during the study. Paths, user and host names are replaced by placeholders.

## Known limits of the reproduction

- The UMAP layouts are sensitive to floating-point noise of about 1e-7. Rerunning `latent_umap.py` gives the same neighbour shares but a different 2-d layout.
- The OmniLearn check of Sec. 2 (`scripts/analysis/omnilearn_check.py`) needs TensorFlow 2.14, the OmniLearn code and checkpoint, and the raw JetClass ROOT files. None of these are in `requirements.txt`.
- A different GPU, CPU or thread count can change the last digits. To match runs that were made on CPU, pass `--device cpu --fused-attention` where a script offers them.

## Citation

To be added.
