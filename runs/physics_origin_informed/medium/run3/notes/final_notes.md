# Final notes: the reconstruction rule compiled into PET2 weights

## The three final edits

All three use the same circuit. They differ only in how many units rebuild channel 2 of the embedding
outside the drop region (13 / 53 / 86 fitted units). Only `body.*` tensors are changed (no `generator.*`
and no `classifier.*`).

| final | file | fit units | DoF | practice E_rule |
|---|---|---|---|---|
| small | final_small.npz | 13 | **5263** | **0.0650** |
| mid   | final_mid.npz   | 53 | **5463** | **0.0607** |
| large | final_large.npz | 86 | **5628** | **0.0604** |

Changed tensors (rank of the change / changed entries), as reported by the oracle:

| tensor | small | mid | large |
|---|---|---|---|
| body.embed.mlp.fc1.weight (1024x4) | r4 / 92 | r4 / 252 | r4 / 384 |
| body.embed.mlp.fc1.bias | 23 | 63 | 96 |
| body.embed.mlp.fc2.weight (512x1024), row 2 only | r1 / 1024 | r1 / 1024 | r1 / 1024 |
| body.embed.mlp.fc2.bias [2] | 1 | 1 | 1 |
| body.pid_embed.0.weight, column 2 | r1 / 8 | r1 / 8 | r1 / 8 |
| body.local_physics.mlp.fc1.weight (1024x7), 2 rows | r2 / 14 | r2 / 14 | r2 / 14 |
| body.local_physics.mlp.fc1.bias | 2 | 2 | 2 |
| body.local_physics.mlp.fc2.weight, row 2 | r1 / 1024 | r1 / 1024 | r1 / 1024 |
| body.local_physics.mlp.fc2.bias [2] | 1 | 1 | 1 |
| body.local_physics.in_blocks.{0,1}.attn.out_proj.weight, row 2 | r1 / 512 each | same | same |
| body.local_physics.in_blocks.{0,1}.mlp.fc2.weight, row 2 | r1 / 1024 each | same | same |
| body.local_physics.in_blocks.{0,1}.mlp.fc2.bias [2] | 1 each | same | same |

DoF = sum over tensors of min(changed entries, r(m+n-r)): 5263 / 5463 / 5628.
About 5.1k of this is the fixed cost of an exact mask (all writes into residual channel 2 before block 0).

## The circuit

### Where the network can drop a token
`PET_body.forward` recomputes the padding mask from **channel 2 of the summed embedding**
`x = E(kin) + L(neighbours) + P[pid] + A(add_info)` (network.py:763, `mask = x[:,:,2:3] != 0`).
A token with x2 == 0 exactly is padding for the rest of the network:
* it is -1e9 in every body attention, as key and as query;
* it gets zero updates;
* it is zeroed by `norm(x)*mask`;
* both heads mask it again (`body_out[...,2] != 0`).

That is exactly "the row becomes padding". So the edit makes x2 = 0 **exactly, in float32**, for the tokens
the rule drops, and leaves x2 approximately at its original value for all other tokens.

### Exact detection
* `A` is a bias-free MLP of the impact parameters, so **A = 0 exactly for every neutral**.
* `L2` := 0. All five writes into channel 2 of the local_physics stream are zeroed (rows/biases above), so
  the local stream's channel 2 contains only the designed gates described below. These are 0 for real jets.
* `P[6,2] = P[7,2] = p` (p = 0.05). The charged pid entries are shifted to keep their x2 unchanged:
  `P'[c,2] = P[c,2] - PF + p`, with PF = 0.026.
* `E2` := `-p + sum_k c_k * h_k(kin)`. Row 2 of embed fc2 is zeroed except for repurposed dead embed hidden
  units (norm.weight ~1e-7, importance < 7e-7). Each unit is
  `h_k = w_k * tanh(a * GELU(s_k . n(kin) + o_k))`, where n(kin) = tanh(0.187*kin)*w_in is the embed input norm.
  GELU is **exactly 0 in float32 for pre < about -5.5**. Every unit has pre <= -8 on the whole box
  `{log pT < log 0.8, log E < log 1.8}`, so inside the box E2 = -p exactly.
  * Two steep step units sit exactly at the box faces: pre = -5 at the threshold and -8 at 1e-4 below it in log,
    with slopes K ~ 1.2e5.
  * Ramp units outside the box carry closed-form least-squares coefficients c_k. They are fitted to
    E2_orig + PF on captured embed outputs (pass 002). E2 is affine in (n_pT, n_E) to within 0.005, which
    makes the fit easy. Fit residuals are 0.056 / 0.026 / 0.024 for the three sizes.
* Result for a neutral in the box: x2 = (-p + 0) + p = **0 exactly**, so it is dropped.
  * Neutral outside the box: x2 ~ E2_orig + PF, nonzero (the steps fire).
  * Charged particle: x2 = E2_new + P[c] + A, never 0. Charged tokens are never dropped.
  * Without pid (pid=None): x2 = -p != 0 in the box, so nothing is dropped.

**Why kept tokens are processed as before.** Only residual channel 2 of the embedding changes, and only by
the fit residual (plus the removed L2, std 0.02). Clean KL is 6e-6 to 1.5e-5, accuracy is unchanged, and all
physics ratios are within 1 ± 0.006. Exceptions: charged particles inside the box shift by about 0.28 in channel 2.
Neutral hadrons shift by -0.07 and photons by +0.02 (the common neutral offset).

### Generator protection (no effect on real jets)
A mask computed before the blocks cannot see the diffusion time. During sampling, a masked token gets
velocity exactly 0 and freezes. I therefore add exact "do not mask" gates. Each writes a tiny nonzero value
(1e-3) into channel 2, and each never fires for real particles:
* `tanh(.187 lpT) - tanh(.187 lE) > 0.005` (E < pT is unphysical; E >= pT for every real particle);
* `lE - lpT > ~2.1` (the real maximum is 2.075);
* an octagon of half-width 0.85 in (deta, dphi). No practice drop lies outside 0.85.
* **Neighbour energy-angle consistency.** Two local-MLP units (local fc1 rows; local fc2 row-2 entries)
  fire when `|(lE-lpT)_i - (lE-lpT)_j| > 0.2*log dR_ij + 1.78` for any kNN pair. Since log cosh is 1-Lipschitz,
  real particles of one jet satisfy |Δ(lE-lpT)| <= |Δη| <= dR. Across 154k neighbour pairs of dropped tokens
  in the probes there are 0 violations (margin 0.11). Noised generator jets violate it constantly.

This lowered gen_drift (t = 0.2 / 0.5 / 0.85) from 0.037 / 0.093 / 0.252 to **0.037 / 0.053 / 0.035**, with E_rule unchanged.

### Known limits of the circuit
* **Neutral hadrons with E < 2.2 outside the photon box** cannot be dropped at the mask stage. pid enters
  only as an additive constant, so a photon box and a larger NH region would need P6 = P7 and P6 != P7 at the same time.
  These are 12% of P jets and about 60% of the remaining P error.
* **Neighbour leakage.** local_physics builds kNN on the raw input, so a dropped token still sits in its
  neighbours' kNN lists. This is about 20% of the P error and the floor for the soft probes.
* **Generator.** Masking of noised neutrals is reduced but not zero at intermediate t. The straight-path
  estimate is that about 7% of neutral tokens freeze, versus 23% without the guards. The true AUC limit
  could not be measured, and this remains the main risk.

## Final practice metrics (oracle surgical passes 022 / 019 / 023)

| metric | small | mid | large |
|---|---|---|---|
| E_rule | 0.0650 | 0.0607 | 0.0604 |
| e soft1e-3 (orig 0.396) | 0.0031 | 0.0024 | 0.0023 |
| e soft1e-4 (orig 0.132) | 0.0029 | 0.0025 | 0.0025 |
| e P (orig 0.0894) | 0.0148 | 0.0140 | 0.0140 |
| e clean / split2 / soft3e-3 | .0015/.0031/.0048 | .0010/.0027/.0038 | .0010/.0026/.0036 |
| d_acc / worst class drop | -0.0001 / 0.0010 | +0.0003 / 0.0000 | 0.0000 / 0.0010 |
| d_nll / KL | -1e-4 / 1.5e-5 | -2e-5 / 7e-6 | -4e-5 / 6e-6 |
| rho D / J / P_reco | 1.006/1.000/0.997 | 1.001/1.001/1.000 | 1.000/1.001/1.000 |
| rho P (raw, reported) | 0.862 | 0.864 | 0.864 |
| gate_rule_no_g2 | True | True | True |
| gen_drift t0.2/0.5/0.85 | .037/.053/.035 | .037/.053/.035 | .037/.053/.035 |

An extra check for an unseen probe type (passes 020/021, 256 jets each), with the mid edit: a soft
**neutral hadron** (the soft1e-4 / soft1e-3 photons relabelled pid 7) gives ratios e_edit/e_orig of
**0.024** and **0.074**. The second is limited by the 19/256 jets whose hadron lies in the uncaught band.

## Predictions vs outcomes

| pass | edit | prediction | outcome |
|---|---|---|---|
| 001 | zero pass (mask, crude E2 steps) | E_rule 0.35, d_acc -0.01, d_nll 0.03, rho_P_reco 1.0, gen_drift 0.15, DoF 5136 | E_rule 0.386, d_acc -0.0028, d_nll 0.013, rho_P_reco 0.855 (limit broken), class drop 0.023 (broken), gen_drift 0.287, DoF 5136 |
| 002 | capture | L2 std / x2 std 0.5 | 0.09 (L2 std 0.021 vs x2 0.227) |
| 003 | A1 (fitted E2) | E_rule 0.15, e_clean 0.025, rho_P_reco 0.95, gen t0.85 0.2 | E_rule 0.066, e_clean 0.001, rho_P_reco 1.000, gen t0.85 0.251 |
| 004/005 | P decomposition | NH-band share of P error 0.5 | 0.62 |
| 006 | NH clamped to 2.3 GeV | NH-band jet err 0.3 | 0.69 |
| 007 | NH moved to 5 GeV | 0.6 | 0.96 |
| 008 | A2 (exact threshold) | E_rule 0.058, e_P 0.0135 | 0.0606, 0.0140 |
| 009 | capture | NH readout resid 0.05 | 0.069 when fitted on all tokens; 0.002 when fitted on NH only |
| 010 | capture | 20 units carry half the contrast | 22 |
| 011-013 | kill 3/10/25 shortcut units | E_rule 0.85/0.75/0.65 | 1.06/1.02/1.06 (failed) |
| 014 | capture | summary rel. change 0.1 | 0.37 at body.norm (0.01 at block 1, 0.05 at block 6) |
| 015 | capture | top-1 shift variance 0.5 | 0.80, but not logit-relevant |
| 016 | A3 (+angle guard) | E_rule 0.061, gen .035/.07/.17 | 0.0607, .037/.075/.146 |
| 017 | S1 | E_rule 0.068, e_clean 0.003, DoF 5200 | 0.0650, 0.0015, 5247 |
| 018 | M1 | E_rule 0.0607, gen t0.85 0.14 | 0.0607, 0.145 |
| 019 | M2 (+neighbour gate) | E_rule 0.0607, gen .035/.05/.05 | 0.0607, .037/.053/.035 |
| 020/021 | soft-NH probes | ratios 0.03 / 0.35 | 0.024 / 0.074 |
| 022 | S2 = final_small | E_rule 0.065, gen max 0.055, DoF 5263 | 0.0650, 0.053, 5263 |
| 023 | Lg = final_large | E_rule 0.0603, e_clean 0.0009, DoF 5660 | 0.0604, 0.0010, 5628 |

## Ideas that failed
* **Zero pass with crude step values** for channel 2. Masking worked (soft1e-3 ratio 0.055), but rewriting
  channel 2 broke the class-drop and P_reco limits. Fixed by the closed-form fit of E2.
* **Dropping neutral hadrons in the 1.8-2.2 GeV band.** It is impossible at the mask stage (pid is additive).
  The block-0 "clamp to 2.25 GeV" (repurposed units with a precise NH-only readout: non-NH tokens sit at
  pre < -21) failed because the needed shift depends on the neighbours through local_physics. Three units
  recovered only 10-40%, and none of it for E in 1.6-2.0, where the P band hadrons are. Even an ideal clamp
  only halves the band error (pass 006), and moving the hadron higher (5 GeV) is worse.
* **Silencing the model's own block-0 shortcut units** (1019, 977, 187, ... respond 0.84 vs 0.11 to sub-
  vs above-threshold soft photons). E_rule stayed at about 1.0: the shortcut is read from the residual
  stream later, not only from those units.
* **Classifier-only fixes** (PCA projection or low-rank least squares on the pooled vector). E_rule >= 1.0:
  the contamination is nonlinear and differs from jet to jet. Summary tokens change 37% by body.norm.
* **Time gating.** At t=0 the time token is exactly zero and therefore masked, which is a clean time signal.
  But the mask is computed before any token can read it, and the oracle cannot capture t>0. I replaced it
  with the physics-consistency guards above.

## Passes used
23 of 50: 12 surgical (001, 003, 008, 011-013, 016-019, 022, 023), 5 capture (002, 009, 010, 014, 015),
6 logits (004-007, 020, 021; 006, 007, 020 and 021 on inputs I built).
