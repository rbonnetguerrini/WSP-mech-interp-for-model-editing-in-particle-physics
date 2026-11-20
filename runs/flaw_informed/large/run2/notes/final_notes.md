# Final notes: removing the soft-neutral flaw from pretrain_l by weight programming

## 1. Final edits

DoF counts follow the task definition (rank: singular values > 1e-4·largest). For every tensor here DoF = number of changed entries.
I checked this offline (`work/verify.py`), and the oracle reports the same totals.

### final_small.npz: pair gate + head mask (DoF 97)
| tensor | rank | changed | DoF |
|---|---|---|---|
| body.interaction.mlp.fc1.weight | 1 | 3 | 3 |
| body.interaction.mlp.fc1.bias | – | 1 | 1 |
| body.interaction.mlp.fc2.weight | 1 | 32 | 32 |
| body.interaction.mlp.fc2.bias | – | 22 | 22 |
| body.embed.mlp.fc1.weight | 1 | 1 | 1 |
| body.embed.mlp.fc1.bias | – | 1 | 1 |
| body.embed.mlp.fc2.weight | 1 | 17 | 17 |
| body.embed.mlp.norm.weight | – | 1 | 1 |
| body.in_blocks.27.mlp.fc1.weight | 1 | 16 | 16 |
| body.in_blocks.27.mlp.fc1.bias | – | 1 | 1 |
| body.in_blocks.27.mlp.fc2.weight | 1 | 1 | 1 |
| body.in_blocks.27.mlp.norm.weight | – | 1 | 1 |

### final_mid.npz: small + sparse embed "hardening" (DoF 1663)
Same tensors as small. Differences:
- `body.embed.mlp.fc1.weight`: 25 entries changed (rank 1).
- `body.embed.mlp.fc1.bias`: 7 entries.
- `body.embed.mlp.fc2.weight`: 1553 entries (6 sparse columns of 256 entries, plus the 17 flag entries; rank 1).

### final_large.npz: mid + soft-neutral-hadron → photon unit (DoF 3713)
Mid plus:
- `body.in_blocks.0.mlp.fc1.weight`: 1024 entries changed (rank 1).
- `body.in_blocks.0.mlp.fc1.bias`: 1 entry.
- `body.in_blocks.0.mlp.fc2.weight`: 1024 entries (rank 1).
- `body.in_blocks.0.mlp.norm.weight`: 1 entry.

No `generator.*` or `classifier.*` tensor is changed.

## 2. What the flaw is, where it lives, how I found it

1. **Shape of the response.** The baseline (pass 001) gives r(soft1e-3)=0.38 but r(soft3e-3)=0.027, the same as split2. A logits scan (002, 007) of one added neutral against log_pt shows a window:
   - Photons give r≈0.45 for log_pt ∈ [−2, −0.3]. This falls sharply to 0.09 at −0.15 and 0.02 at −0.05.
   - Neutral hadrons trigger up to log_pt ≈ +0.5.
   - Very soft particles (log_pt −3) give ≈0.1.
2. **It is neutral-specific and fires like a trigger.**
   - The same particle as a charged track gives r≈0.05 (008).
   - Two soft photons give barely more than one (0.455 vs 0.43): the trigger saturates.
   - The logit shift has a consistent direction, mainly towards Tbl / Hqql / Tbqq.
3. **It is out-of-distribution behaviour.** The natural sample has sharp detector thresholds: photons E ≥ 0.83 GeV, neutral hadrons E ≥ 2.3 GeV. The pid-specific windows match these.
   - Removing all natural neutrals below 1 GeV from clean jets changes the output by only r=0.013 (007).
   - So the model is not sensitive to soft neutrals in general. It reacts to neutral tokens below the range it was trained on.
4. **Where it lives.** Residual captures (003–005) show the change in the summary tokens is small up to block 6 (~1–2%). It then grows through blocks 13→27 (10% → 60%), and every particle token is perturbed too (38% at block 27).
   - No sparse detector neuron exists in blocks 1–8 (009).
   - Summary tokens attend to the soft token only like any other particle (one block-21 head at 0.30).
   - Blocking particle↔soft attention with the pair bias (011, 016) removes ~75% of the soft1e-3 response.
   - Conclusion: the flaw is computed by the **particle tokens reading the OOD soft-neutral token through particle–particle attention**. The very small pair features also enter the bias, and the effect is amplified in the mid/late blocks.
5. **What remains after blocking.** The rest is "presence" of the isolated token. Captures under the gate (012) show its effect reaches the output mostly through the **classifier head** reading the token from x_body. Masking it there cut the residual response by about 2.5× (pass 034).

## 3. The circuit (what the edits compute)

Notation: n_pt = tanh(α_in·log_pt)·w_in[2] is the embed input normalisation. G = 2^30.

**(a) Pair gate (interaction MLP, unit 635).** The interaction features are f2 = log ΔR and f3 = log kT = log min(pT_i, pT_j) + log ΔR, so f2 − f3 = −log min pT. The repurposed unit computes
`h = GELU(300·(f2 − f3) + 300·(−0.18)) = GELU(300·(−0.18 − log min pT))`.
Its fc2 column is −0.5 on all 32 heads. The unit originally had input weights of norm 1e−4, so its constant output was folded into the fc2 bias.
- Every attention pair with min pT < e^−0.18 = 0.84 GeV gets a bias ≈ −150·(−0.18 − log min pT), i.e. a hard block.
- Particle queries can no longer read soft tokens.
- For all other pairs (and self-pairs, which have features (0,0,0)), the pre-activation is below −6, so GELU is exactly 0 and the bias is bit-identical to the original.

**(b) Head mask by float absorption (embed unit 1672 + block-27 MLP unit 246).**
- The embed unit is a steep step on log_pt: pre = −2000·n_pt + c, set to exactly 0 for log_pt ≥ −0.16 and saturated (tanh = 1.0 exactly, norm weight set to 1) for log_pt ≤ −0.2.
- Its fc2 column writes exactly +G into channel 2, and ±G into 16 "flag" channels (chosen from captures as channels natural tokens never saturate at block 26).
- All other contributions to these channels in the embedding and in every block are < 64 = ulp(G)/2. So channel 2 of a gated token stays **exactly G** through the body.
- In block 27, unit 246 reads the 16 flags: pre = 200·(mean_c s_c·tanh(α x_c) − 0.75) + (its original row).
  - Natural tokens score ≤ 0.45, so pre < −39 and the output is exactly 0.
  - Gated tokens score exactly 1, so the unit saturates (norm weight 1) and adds exactly −G to channel 2.
- Channel 2 of a gated token is then exactly 0. Hence x_body[...,2] = tanh(0)·w = 0, and the classifier head's padding mask (`x_body[:, 4:, 2] != 0`) drops the token.
- Only the necessary entries were changed. The repurposed units keep their original dense weights, which matter only when the step fires.

**(c) Embed hardening (6 near-constant embed units, mid/large).** Six units whose pre-activation varies by < 0.08 over the whole input box are repurposed. Each is a GELU step/ramp on n_pt at log_pt thresholds −0.16 … −1.85, exactly 0 for log_pt ≥ −0.1.
- Their fc2 columns are the closed-form least-squares fit of Δ(log_pt) = e(log_pt = −0.1) − e(log_pt), measured on captured embed outputs (013). The fit used 6 units; in the mid/large edits each column is restricted to its 256 largest channels and refitted per channel by least squares.
- Effect: the isolated soft token looks like a non-triggering −0.1 photon to whatever still reads it in the body (summary tokens).

**(d) Soft neutral hadron → photon (block-0 MLP dead unit 1042, large only).**
- pre = 300·(d·norm2(x) + off + 0.03), where d is a closed-form regularised LDA direction separating soft neutral hadrons (log_pt ≤ 0.5) from all natural tokens and from all photons. It was fitted on block-0 MLP inputs recovered from captured `mlp_pre` with pinv(fc1).
- The output (norm weight set to 1) writes P_photon − P_NH (pid-embedding difference).
- A neutral hadron below its natural range becomes an in-distribution photon at the same kinematics (r≈0.03). Photons and charged tokens score ≤ −0.66, so the unit is exactly 0 for them.

**Why jets without very soft neutrals are unaffected.**
- Every gate is a GELU whose pre-activation is ≤ −8 for tokens above threshold. GELU is exactly 0 there in float32, so no contribution is added.
- The pair bias for non-soft pairs, the embedding of non-soft tokens and their channel-2 value are bit-identical to the original. The block-27 unit is exactly 0 on natural tokens.
- The only changes on such jets are the lost tiny contributions of the repurposed units: near-constant units (≤ 0.005), unit 246 (≤ 0.74 in a residual of norm ~200), and a dead unit.
- Measured clean KL is 0.003 and d_acc is −0.001.
- Caveat: the pair gate and the head mask key on pT, not on pid. Charged tracks below 0.84 GeV (~0.1–0.3 per jet, rare) are also removed. Removing natural charged tracks below 1 GeV changes the output by only 0.007.

## 4. Final practice metrics (oracle surgical passes, 2050 jets)
| | small (038) | mid (040) | large (041) |
|---|---|---|---|
| DoF | 97 | 1663 | 3713 |
| d_acc / worst class drop / d_nll | −0.0010 / 0.0049 / 0.0033 | −0.0010 / 0.0049 / 0.0027 | −0.0015 / 0.0098 / 0.0025 |
| ratio soft1e-3 / soft1e-4 | 0.092 / 0.324 | 0.074 / 0.255 | 0.074 / 0.253 |
| ratio split2 / soft3e-3 | 1.014 / 1.031 | 1.015 / 1.024 | 1.014 / 1.018 |
| R_vis / R_in / R_sub | 0.615 / 1.023 / 0.208 | 0.592 / 1.019 / 0.165 | 0.590 / 1.016 / 0.163 |
| rho D / J / P / P_ref | 0.984 / 1.069 / 0.805 / 1.006 | 0.986 / 1.049 / 0.785 / 0.992 | 0.987 / 1.052 / 0.757 / 0.986 |
| gate_ref_no_g2 | True | True | True |
| gen_drift t=0.2 / 0.5 / 0.85 | 0.095 / 0.168 / 0.509 | 0.096 / 0.168 / 0.512 | 0.116 / 0.180 / 0.518 |

Own held-out estimates of the target error ratio e_v(edit)/e_v(orig), with target = original model on the jet with the soft particle removed (90 jets per probe):

| probe | small | large(-type) |
|---|---|---|
| neutral hadron at 1e-3 | 0.18 | – |
| neutral hadron at 1e-4 | 0.39 | – |
| photon 3e-4 | 0.13 | – |
| photon 1e-5 | 0.26 | – |
| two photons at 1e-3 | 0.18 | – |
| P (target: neutrals < 2e-3·ΣpT removed) | 0.64 | not measured (0.64 for the earlier large1 without the head mask, pass 032) |

The P target definition is my guess. The oracle does not report E.

**Generator caveat (important).** Noised tokens at t=0.85 often have log_pt < −0.18, so the pt-based gates also fire in generation. The head mask zeroes x_body channel 2 for those tokens, so the generator drops them too. This raises gen_drift at t=0.85 to ~0.51, against 0.10 for the pair gate alone (pass 030, DoF 58, ratios 0.27/1.31). I found no closed-form way to disarm the gates when the time token is present: the time token is exactly zero in classification, and I cannot capture generation-mode activations.

## 5. Predictions vs outcomes

Surgical passes:

| id | edit | predicted | outcome |
|---|---|---|---|
| 001 | null | R_vis 1, d_acc 0, gen 0 | as predicted |
| 016 | pair gate τ=0.97 GeV | ratios 0.25 / 1.25, d_acc −0.003, gen 0.05 | 0.251 / 1.30, d_acc −0.0127 (**fails**), class drop 0.039 (fails), J 1.37 (fails), gen 0.108 |
| 024 | pair gate −0.3 + hardening | 0.22 / 0.75, J 1.03, gen 0.08 | 0.198 / 0.717, J 1.03, P_ref 0.94, gen 0.173 |
| 029 | + NH unit | 0.19 / 0.72, J 1.04, gen 0.19 | 0.190 / 0.714, J 1.03, gen 0.192 |
| 030 | pair gate −0.18 only | 0.27 / 1.3, J 1.05, gen 0.09 | 0.268 / 1.31, J 1.04, gen 0.096 |
| 036 | large + dense head mask | 0.10 / 0.35, gen 0.3 | 0.073 / 0.246, gen 0.518 (oracle DoF 7265, see note) |
| 037 | pair + dense head mask | 0.12 / 0.42, J 1.06, gen 0.25 | 0.093 / 0.321, J 1.07, gen 0.509 |
| 038 | small (sparse head mask) | 0.093 / 0.32, J 1.07, dof 97 | 0.092 / 0.324, J 1.07, dof 97 |
| 039 | dense hardening + NH + sparse head mask | 0.073 / 0.25 | 0.072 / 0.245 (oracle DoF 5231 vs 8321 entries, rank-threshold effect of the 2^30 column) |
| 040 | mid | 0.078 / 0.26, dof 1700 | 0.074 / 0.255, dof 1663 |
| 041 | large | 0.074 / 0.255, dof 3714 | 0.074 / 0.253, dof 3713 |

Key logits/capture passes (my own analysis; full list with predictions in `ledger.json` and `work/pred_table.md`):

| id | predicted | outcome |
|---|---|---|
| 002 | peak at log_pt −0.5, r_peak 0.4, r(−3) 0.1, r(+1) 0.03 | peak −0.5 … −1 at 0.45, r(−3) 0.10, r(+1) 0.04; neutral hadrons also trigger at +0.5 (0.21) |
| 007 | removing natural neutrals < 1 GeV: 0.3 | **0.013** (natural soft neutrals do not trigger) |
| 008 | charged 0.05, two photons 0.7, far 0.4 | charged 0.05, two photons **0.455** (saturating), far 0.17 |
| 010 | charged-pid disguise 0.06, lp→0.3 gives 0.03 | pid0 0.25, pid8 0.19; lp→0.3 0.027 |
| 011 | pair gate: 1e-3 0.15, 1e-4 0.06 | 0.11, **0.155** (1e-4 worse) |
| 015 | embed-only hardening: 1e-3 ratio 0.2 | 0.86 (**failed**) |
| 022 | saturated sign pattern: 1e-3 ratio 0.25 | logits blow up to 1e5 on jets with natural soft tokens (**failed**) |
| 023 | pair −0.3 + hardening: 1e-3 0.3, 1e-4 0.7 | 0.22 / 0.78 |
| 027 | NH unit: h@0.5 ratio 0.3 | 0.18 |
| 028 | bias capped at −4: 1e-3 0.18 | 0.31 / 1.75 (**worse**) |
| 034 | head mask: 1e-3 0.16, 1e-4 0.6 | **0.10 / 0.36** |
| 043 | threshold −0.03: P 0.45, J 1.1 | P 0.59, J **1.31** (fails) |

Pattern: I consistently over-predicted the benefit of "disguise"-type fixes. I under-predicted the head-mask gain and the generator cost.

## 6. Ideas that failed
- **Exact masking of the soft token in the body via channel 2 of the embedding.** Infeasible: the local-physics term is continuous and there is no subtraction step before the mask is computed. The same float-absorption idea works only for the classifier-head mask.
- **Embed-only "hardening"** (making the token's own embedding look like a harder photon) does not remove the trigger (015). The OOD-ness also enters through the local and pair features.
- **Charged-pid disguise** of soft photons is itself OOD (r≈0.2).
- **One saturated sign pattern for gated tokens to make them inert to summary queries.** Closed-form margins were only ~−6.5 nats in late layers. Where suppression failed, heads read a huge-norm value and logits blew up (022).
- **Capped (−4) pair bias** instead of a hard block (028): the OOD pair bias is strongly attractive, so a full block is needed.
- **Pair gate at 0.97 GeV** (016) and **gate + mask at −0.03** (043) fail the accuracy and J limits: too many natural tokens cross the threshold.
- **A single 1024-d LDA detector for soft photons in the block-0 MLP:** too blurry near threshold on held-out jets.
- **Local-feature correction:** after hardening, the residual difference at block 0 was only ~2/30, so it was not pursued.

## 7. Passes used
43 of 50 oracle passes (one oversize request was rejected at no cost). Pass 004 was an accidental duplicate of 003, caused by a request-building bug.
- Surgical passes: 001, 016, 024, 029, 030, 036–041.
- The rest were logits and capture passes.

## 8. Method and rule notes
- No gradients and no iterative optimisation were used. Everything is closed form:
  - least squares (lstsq/pinv);
  - regularised LDA (one linear solve);
  - sign-of-sum heuristics;
  - hand-set step units.
  "Choosing among candidates" was limited to comparing a few fixed closed-form settings by oracle passes.
- Offline computations on oracle activations: elementwise DynamicTanh plus single weight matrices (q, k projections, fc1 inversion). On raw practice inputs: geometry only (kNN membership, pT fractions).
- To fit the fc2 columns I evaluated the response formula of my own new step units (tanh/GELU of a linear function of log_pt).
- All files were read only in `inputs/` and `run/`. One attempted command to read a harness background-task output file outside these folders was blocked by the tool and never executed.
