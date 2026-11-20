# Task: compile a detector rule into a physics model's weights (no gradients)

You are the "programmer" in a weight-space programming experiment. Read this whole file first.

## Background

`pretrain_l` is a 423M-parameter particle-physics transformer (OmniLearned PET2) pretrained on simulated
jets from seven datasets. We use its 10 JetClass outputs (logits 2..11 of its 210-way head) as a jet
classifier.

The model has a known flaw. Adding one very soft neutral particle (a photon or a neutral hadron) to a
JetClass jet changes its classification substantially, although a particle of vanishing energy should not
matter. The cause is a dataset shortcut. JetClass's detector simulation never reconstructs neutral particles
below certain thresholds. Another pretraining dataset, JetClass-II, uses lower thresholds. During
pretraining, a soft neutral below JetClass's thresholds therefore always meant "this is not a JetClass jet",
and the model learned that. On a JetClass jet, one such particle moves the model into a regime its JetClass
outputs were never trained for.

A physicist fixes this outside the model, with one preprocessing rule that makes every input consistent
with JetClass's reconstruction:

> **The rule.** Drop every particle that JetClass's detector never reconstructs:
> - a photon (pid 6) with pT < 0.80 GeV **and** E < 1.80 GeV;
> - a neutral hadron (pid 7) with E < 2.20 GeV.
>
> In the jet format below, that particle's row becomes padding: all nine columns 0, except pid = 3.
> Nothing else changes. No real JetClass particle lies below these thresholds, so the rule leaves every
> clean JetClass jet unchanged.

Call this preprocessing `reco(x)`. The reference implementation is `{INPUTS}/rule.py`.

## Goal

**Implement the rule inside the network.** Edit the weights so that the edited model on a raw jet x gives
the same outputs as the original model on reco(x):

  f_edited(x) ≈ f_original(reco(x)) for every jet x.

This is programming, not fitting. Work out from the code and the weights how the network would have to
compute "this token is a sub-threshold neutral", and how such a token can be made to have no influence.
Then write that circuit into the weights with closed-form constructions. Tokens the rule keeps (all
charged particles, and neutrals above their thresholds) must be processed exactly as before.

You are not asked to repair anything else. Other sensitivities of the model, for example to collinear
splits or to soft photons *above* the thresholds, are out of scope. Leave them as they are.

## What you may change

- Any tensor whose name starts with `body.` or `classifier.`. Any `generator.*` change voids the
  submission.
- There is no limit on how many tensors you change or on the rank of a change. But the smaller the edit,
  the better (see **Size** below).
- Edits are full replacement tensors in an `.npz` (state_dict keys, float32, original shapes). Include only
  the tensors you change.
- No gradients and no iterative optimisation. Not allowed: autograd, `.backward()`, `torch.optim`,
  `torch.func` gradients, and any iterative optimiser (gradient descent, L-BFGS, `scipy.optimize`,
  random-search loops), whatever it optimises.
- Allowed: any closed-form computation on the weights and on activations the oracle returns, such as SVD,
  eigenvectors, projections, least squares, statistics, or applying single weight matrices to captured
  activations. Designing weights by hand from the model's equations is the point of the task.
- You may not run the model, or any network module, on data yourself. All forward passes go through the
  oracle.

## How you are scored (on hidden test jets)

All quantities are relative to the original model. c is the 10 JetClass logits minus their mean, and S is
the RMS distance between two random clean jets under the original model.

- **Rule error (lower is better).** For each probe v, e_v = mean over jets ||c_edited(x_v) - c_original(reco(x_v))|| / S.
  The headline is **E_rule**, the mean over the rule probes of e_v(edited) / e_v(original): 1 for the
  original model, 0 for a perfect implementation. The visible rule probes are:
  - `soft1e-3`, `soft1e-4`: one photon (pid 6) at that fraction of the jet's scalar pT sum, uniform in
    dR < 0.4. The rule drops it in 88% of the `soft1e-3` jets (the rest lie above 0.80 GeV) and in all
    `soft1e-4` jets;
  - `P`: the jet's fragmentation made softer (exponent x1.105 at fixed jet pT). In 75% of these jets, some
    of the softened neutrals fall below the thresholds.

  The hidden test also has rule probes you have not seen.
- **Side effects (hard limits).** The submission fails if any of these breaks:
  - 10-class accuracy drop at most 0.010;
  - accuracy drop of any single class at most 0.020;
  - mean NLL rise at most 0.030;
  - for each real physical edit, the response ||c(op x) - c(x)|| must stay within a factor e^(+-0.15) of
    the original's:
    - `D`: all angles x1.105;
    - `J`: all pT and E x1.05;
    - `P_reco`: reco(P x), the softer fragmentation as JetClass's detector would record it.

    The response to the raw `P` is expected to change, because the rule changes it. It is reported but
    not a limit;
  - the generator head, which shares the body, must not be damaged. On this model its real-vs-generated AUC is reported but is not a limit (the original already reaches AUC 1.00, so a rise cannot be measured); keep `gen_drift` small instead.
- **Size: degrees of freedom (DoF).** For each changed 2-D tensor with shape (m, n) whose change has rank r
  (singular values above 1e-4 x the largest), DoF = min(number of changed entries, r (m + n - r)). For a
  changed 1-D tensor, DoF = the number of changed entries. The edit's DoF is the sum over tensors.
- **Reported for every submission, not limits:**
  - e_v for the inputs the rule leaves unchanged: clean jets, `split2` (the hardest neutral split into
    two collinear halves) and `soft3e-3` (one photon at 3e-3 of the jet pT, above the thresholds). For
    these, reco(x) = x, so e_v measures how much your edit changed the model where it should not;
  - the flaw ratios R_vis, R_in and R_sub (reported for reference);
  - the direction errors of the physics responses, and the relative change of the `body.norm` summary
    tokens on clean jets.
- **Ranking.** We compare all submissions that pass every limit on a curve of E_rule against DoF. An edit
  is better if no other passing edit has both lower E_rule and lower DoF. You may hand in **up to 3 final
  edits of different sizes** (see the deliverables).

## Machine and files

You work on a shared Linux login node. These paths are not set as variables: write the full paths in
every command and file.
- `{INPUTS}`: your inputs (read only);
- `{RUN}`: your run folder.

Python: run `cd {RUN}/work && OMP_NUM_THREADS=4 {PYTHON} script.py`. numpy, scipy and torch are
available. `import torch` takes about 2 minutes cold, so prefer numpy. The node is shared and has no GPU:
keep each computation to a few minutes and never touch other processes.

## Inputs

All inputs are in one folder, `{INPUTS}`:
- **Weights.** `{INPUTS}/weights_l.npz` (every state_dict tensor, float32), with
  `{INPUTS}/param_manifest.json`.
- **Code** (copies): `{INPUTS}/code/network.py`, `layers.py` and `diffusion.py` (the model), and
  `{INPUTS}/code/omnilearned_adapter.py` (preprocessing, padding mask, and the activation site names).
- **The rule:** `{INPUTS}/rule.py` (`reco(raw)`, numpy).
- **Practice jets.** `{INPUTS}/practice/`: 10240 jets, 1024 per JetClass class.
  - `clean.npy`, `labels.npy` (global index 2..11);
  - the probes `split2.npy`, `soft3e-3.npy`, `soft1e-3.npy`, `soft1e-4.npy`, each with an `.ok.npy`
    validity mask;
  - the physical edits `D.npy`, `J.npy`, `P.npy` and `P_reco.npy`.

  Row i is the same jet in every file.
- **Jet format.** One jet is a (150, 9) array.
  - Columns: 0 deta, 1 dphi, 2 log_pt, 3 log_E (natural logs, GeV), 4 pid, 5-8 impact parameters (d0,
    sigma_d0, dz, sigma_dz).
  - Padding rows are all 0 except pid = 3 (so log_pt == 0).
  - pid codes: 6 photon, 7 neutral hadron; charged particles use the codes 0, 1, 2, 4, 5 and 8. Neutral
    particles have all four impact-parameter columns equal to 0; charged particles never do.

## Rules on files

- Read only `{INPUTS}/` and `{RUN}/`. Never open, list or search anything else on the machine. **Every file
  you open and every command you run is logged, and any access outside these two folders is reported with
  your results.**
- Running the Python interpreter named above and importing numpy, scipy or torch is fine. Do not import or
  open the installed `omnilearned` or `hep_ggb` packages; use the copies in `{INPUTS}/code/`.
- Write only in `{RUN}/requests/`, `{RUN}/submissions/`, `{RUN}/work/`, and the file `{RUN}/STOP`.
- Never commit or push. If you need more access, stop and say so.
- Keep your own context lean: write scripts and results to `{RUN}/work/` and print short summaries.

## The oracle (your only way to run the model)

A GPU job watches `{RUN}`. It is up when `{RUN}/ledger.json` exists and contains "oracle_up".

To use it:
1. Write the JSON to `{RUN}/requests/<id>.json.tmp`, then `mv` it to `<id>.json` (ids 001, 002, ...).
2. Wait (a command may run for at most 10 minutes, so repeat the wait if needed):
   ```
   timeout 540 bash -c 'until [ -f {RUN}/responses/<id>.json ]; do sleep 5; done'; cat {RUN}/responses/<id>.json
   ```

Files named in a request (`weights`, `input`) must lie inside `{RUN}`.

The budget is **50 passes**. Malformed or incomplete requests are rejected at no cost. The oracle runs for
about 16 hours from when it comes up, and exits after 90 minutes without requests, so plan your passes
accordingly.

**Every request must contain:**
- `"hypothesis"`: one or two sentences on what you expect and why;
- `"prediction"`: a dict of numbers you commit to before seeing the result, for example
  `{"E_rule": 0.4, "d_acc": -0.002, "rho_P_reco": 1.0}`.

Both are logged. Your prediction accuracy is part of the record.

Kinds:
- `{"kind": "surgical", "weights": "<abs path .npz>" | null, ...}`, the practice-set version of the score (on the first 205 practice jets of each class, 2,050 jets, to keep a pass short):
  - `rule`: e_v and e_v of the original for each visible probe, P and clean jets, and E_rule;
  - clean d_acc, worst per-class drop, d_nll and KL;
  - physics response ratios `rho` for D, J, P and P_reco, and `gate_rule_no_g2` (whether the limits other
    than the generator pass);
  - **generator check (proxy):** `gen_drift`, the relative change of the generator head's velocity
    prediction on 512 fixed noised W jets at diffusion times t = 0.2, 0.5 and 0.85 (0 = unchanged).
    Use it as your guide to keep the generator unharmed;
  - R_vis, R_in, R_sub and the direction errors (`dir_err`);
  - the size of your edit: its DoF, and the rank and changed entries of each tensor.
- `{"kind": "capture", "weights": ..., "input": SRC, "indices": [...], "sites": [...]}`:
  - logits plus activations at up to 4 sites, at most 2048 jets;
  - SRC is `clean`, any practice file name (`split2`, `soft1e-3`, `P_reco`, ...), or an abs path to your
    own `.npz` with key `data` (n <= 2048, 150, 9) under `{RUN}/work/`. Your own inputs are how you test
    a circuit on tokens you construct, for example neutral hadrons or particles just above and below a
    threshold;
  - sites include `embed`, `body.block0`..`body.block27`, `body.blockN.mlp_pre`, `body.blockN.mlp_post`,
    `body.norm`, `classifier.block0`, `classifier.block1`, `classifier.fc`;
  - body token order: [4 summary tokens][1 time token][150 particle slots].
- `{"kind": "logits", ...}`: the same, logits only.

## Required order and deliverables

1. **Zero passes first.** Before any request, write:
   - `{RUN}/submissions/zero_pass_notes.md`: from the code and the weights, how you will implement the
     rule (where the network can detect a sub-threshold neutral, and how its influence can be removed),
     and your predictions;
   - `{RUN}/submissions/zero_pass.npz`: your best edit from reading alone.

   The oracle serves nothing until `zero_pass.npz` exists.
2. **Iterate**, with up to 50 passes. Keep your edits and notes in `{RUN}/work/`.
3. **Final.**
   - Up to 3 final edits of different sizes: `{RUN}/submissions/final_small.npz`, `final_mid.npz` and
     `final_large.npz`. Hand in at least one; any you leave out are simply not scored. Each must satisfy
     the tensor rules and pass the practice-set limits.
   - `{RUN}/submissions/final_notes.md`, which must give:
     - for each final edit: the tensors changed, their ranks, the changed entries and the DoF;
     - the circuit in words and formulas: how the network detects a sub-threshold neutral, how it removes
       its influence, and why tokens the rule keeps are unaffected;
     - the final practice metrics;
     - a table of your predictions against the outcomes;
     - the ideas that failed;
     - the passes used.
4. `touch {RUN}/STOP`, then finish with a short summary and a status: DONE, PARTIAL, BLOCKED or FAILED.
