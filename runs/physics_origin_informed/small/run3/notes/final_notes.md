# Final notes: the JetClass reconstruction rule compiled into PET2 weights

## 1. Final edits

All three use the same circuit: an exact detector, a saturated "inert corner" write, and a block-1 rewrite.
They differ in the suppression pattern (sigma) and in whether the scratch channels are folded out of the
existing block-0 MLP units. Sizes are the oracle's DoF counts.

| file | E_rule (practice) | DoF | changed entries | gen_drift max (t=.2/.5/.85) | note |
|---|---|---|---|---|---|
| `final_small.npz` (v8)  | **0.1005** | **482**  | 482   | 0.066 / 0.141 / 0.206 | lowest generator drift |
| `final_mid.npz` (v12)   | **0.0838** | **491**  | 491   | 0.165 / 0.309 / 0.476 | particle-weighted blocks-2..7 pattern; higher generator drift (risk) |
| `final_large.npz` (v8f) | **0.0927** | **3510** | 16344 | 0.067 / 0.142 / 0.205 | v8 plus a linear fold of the scratch channels: smallest side effects (clean e 0.0007) |

Per-tensor changes (rank = oracle rank; "changed" = changed entries):

| tensor | small (v8) | mid (v12) | large (v8f) |
|---|---|---|---|
| body.embed.mlp.fc1.weight | r2, 3 | r2, 3 | r2, 3 |
| body.embed.mlp.fc1.bias | 3 | 3 | 3 |
| body.embed.mlp.fc2.weight | r3, 3 | r3, 3 | r3, 3 |
| body.in_blocks.0.norm1.weight | 3 | 3 | 3 |
| body.in_blocks.0.mlp.fc1.weight | r5, 134 | r5, 134 | r8, 15842 (DoF 3008 by rank) |
| body.in_blocks.0.mlp.fc1.bias | 5 | 5 | 159 |
| body.in_blocks.0.mlp.fc2.weight | r4, 259 | r4, 259 | r4, 259 |
| body.in_blocks.1.mlp.fc1.weight | r1, 40 | r1, 40 | r1, 40 |
| body.in_blocks.1.mlp.fc1.bias | 1 | 1 | 1 |
| body.in_blocks.1.mlp.fc2.weight | r1, 31 | r1, 40 | r1, 31 |
| **DoF** | **482** | **491** | **3510** |

No `generator.*` or `classifier.*` tensor is touched. The files hold only the changed tensors, as float32
in their original shapes.

## 2. The circuit

### 2.1 Facts used (from code and weights)
* The rule needs pid AND (pT, E). Before block 0 these never meet in one module. The embed MLP sees
  `n = tanh(0.125 x) * w0` of (deta, dphi, log pT, log E); `pid_embed` sees only pid; `add_embed` sees only
  the impact parameters; local/interaction blocks see only pair kinematics.
* A photon's condition is a quadrant in (log pT, log E). One layer of ridge units cannot produce a quadrant
  indicator exactly near the corner, so the AND needs two non-linear stages.
* Exact gates are possible. GELU(erf) returns exactly 0 for pre-activation below about -5.6. tanh returns
  exactly 1 above about 9. So a hidden unit can output exactly 0 or exactly `nw`.
* The model has dead capacity: 38 dead embed-MLP units, 102 dead block-0 MLP units and 3 dead block-1 units
  (norm weight below 5e-3). Block 0's `norm1` ignores 15 residual channels (weight below 1e-3).
* The body has 8 heads of 16 dims (the interaction MLP has 8 outputs). The body padding mask is computed
  once from the embedded channel 2. It cannot be reached exactly, because the pid-blind local term is
  continuous.
* The time token is exactly 0 at t=0 (`MPFourier` multiplies by t), so it is masked during classification.

### 2.2 Stage A, embed MLP: exact kinematic steps
Three dead embed units become step units, each with a single fc1 entry (K' = 1e5):

    s_pt  = [log pT < ln 0.8]  : pre = -K' n_2 + K' tanh(0.125 ln0.8) w0_2 + 0.65
    s_E18 = [log E  < ln 1.8]  : same on n_3
    s_E22 = [log E  < ln 2.2]  : same on n_3

Because n_k is a monotone function of the raw feature, each step is an exact threshold, with a transition
width of about 1e-3 in log units. Each unit writes `+V` (V = 128) into its own scratch channel: 122, 87
and 4. These channels are ignored by block-0 attention (`norm1` weight set to exactly 0). The unit keeps its
old tiny norm weight `nw`, and the single fc2 entry is V/nw, so the write is exactly V.

These three channels were chosen because:
* the existing block-0 MLP reads them least (summed |fc1·w2| = 21, versus 49 for the first choice);
* their original value never reaches tanh > 0.87, so "+V written" (tanh = 1) is separable from the original
  content;
* the time token carries little weight in them (unlike channel 116).

### 2.3 Stage B, block-0 MLP: AND, restore, flag
Five dead units are reused. Write r_c = n2_c / w2_c for the scratch read in block 0's norm2 output.
* **Restore** (×3): `pre = K2 (r_c - thr_c)`, with thr_c halfway between the largest original read and 1.
  The unit fires exactly when the step was written and writes exactly `-V` back. Channels 122/87/4 are
  therefore exactly restored (up to ulp(128) ≈ 1e-5) for every token from block 1 on.
* **Photon flag**: `pre = K [ a6 (r_122 + r_87) + s6(n2) + b6 ]`, where s6 is a 64-channel linear pid-6
  readout fit by least squares on 1280 jets (separation ≥ 0.77 vs ≤ -0.53). It is positive only when both
  steps are written (r = 1 exactly) and pid = 6.
* **Neutral-hadron flag**: `pre = K [ a7 r_4 + s7(n2) + b7 ]`, the same construction with pid 7.

The thresholds put the margin equally on the three failure cases. In the offline check over 1280 jets
(916 dropped tokens) the flags were exact: on ≥ 16.4, off ≤ -16.8. Oracle checks were exact as well:
681/681 on custom soft photons and neutral hadrons, and 399/399 on P jets, with no false flags.

A flag unit writes `L·sigma1` (L = 1000) into the whole residual of the flagged token. Every later
`tanh` norm then reads the token as the fixed corner `w ⊙ sigma`.

### 2.4 Stage C, block 1: rewrite the corner
One dead block-1 unit detects the corner: `pre = Kr (sigma1⊙w2)_top40 · n2 / |w2_top40|² - 0.8 Kr`.
Kept tokens reach at most -22.6; the flagged token gives +15.5. The unit writes `L (sigma2 - sigma1)`, which
flips the flagged token to a second pattern, sigma2.

### 2.5 Why the flagged token loses its influence
Its key in every later head is `k_f = W_k (w1 ⊙ sigma)`. sigma1 (block 1) and sigma2 (blocks 2-7 plus the
two classifier token-attention blocks) are chosen in closed form:

    sigma = -sign( sum_heads  mu_h / |mu_h|² )

Here `mu_h` is the mean over captured kept-token queries of `(W_k^T q_i) ⊙ w1 / sqrt(16)`. Summary-token
and particle-token query groups are normalised separately. For the mid edit, the summary group of blocks
2-7 and the classifier has weight 0.3. The result is a flagged-key logit of about -4 to -10 in most heads, so
the flagged token takes almost no attention. In the classifier its body output is the corner
`w_f ⊙ sigma2`, whose classifier-key logits are about -8 to -12.

### 2.6 Why kept tokens are unaffected
* Every new unit is off for kept tokens with exactly zero output (pre-activation ≤ -16). The exceptions are
  the restore units, which cancel their own write exactly.
* The only change a kept token sees is in block 0's MLP: its existing units read the saturated scratch
  channel for tokens whose step fired (soft charged particles). In `final_large`, that read is replaced by a
  least-squares prediction from the other channels (R² ≥ 0.98), which removes the effect.
* Measured clean e_v: 0.0022 (small/mid) and 0.0007 (large).

### 2.7 Leaks that cannot be removed
* The soft particle is still a key in **block-0 attention**, because the flag only exists after block 0's
  MLP.
* It still sits in the **kNN neighbourhoods** of the local embedding, which use the raw mask and are
  pid-blind.

These leaks are about 1% (soft photon) to 2% (P) in block-0 activations. They are most of the remaining
E_rule.

## 3. Final practice metrics (surgical passes 026, 032, 037)

| metric | small v8 | mid v12 | large v8f |
|---|---|---|---|
| E_rule | 0.1005 | 0.0838 | 0.0927 |
| e soft1e-3 / orig | 0.0104 / 0.2486 | 0.0084 | 0.0095 |
| e soft1e-4 / orig | 0.0105 / 0.2646 | 0.0084 | 0.0101 |
| e P / orig | 0.0223 / 0.1016 | 0.0189 | 0.0205 |
| e clean / split2 / soft3e-3 | 0.0022 / 0.0023 / 0.0026 | same | 0.0007 / 0.0017 / 0.0007 |
| d_acc / worst class drop / d_nll | +0.0003 / 0.000 / +0.0001 | same | +0.0002 / 0.001 / -0.00003 |
| rho D / J / P_reco (limits e^±0.15) | 0.996 / 0.999 / 0.993 | 0.996 / 0.999 / 0.993 | 1.000 / 1.000 / 1.000 |
| rho P (raw, not a limit) | 0.72 | 0.72 | 0.72 |
| R_vis / R_in / R_sub | 0.525 / 0.999 / 0.052 | 0.522 / 0.999 / 0.045 | 0.525 / 0.998 / 0.052 |
| gate_rule_no_g2 | True | True | True |
| gen_drift t=.2 / .5 / .85 | 0.066 / 0.141 / 0.206 | 0.165 / 0.309 / 0.476 | 0.067 / 0.142 / 0.205 |

Rule probes not in the practice set, built by me on 512 clean jets each and scored with my estimate of S
(18.5). Each figure is the edited error as a fraction of the original's:

| probe | small | mid | large |
|---|---|---|---|
| one soft neutral hadron (E 0.3-2.2 GeV) | 0.187 | 0.164 | 0.187 |
| three soft photons at ~3e-4 | 0.142 | 0.143 | 0.142 |

**Generator caveat (not resolved).** The circuit is not time-gated, so at t > 0 noised neutrals that fall
below the thresholds are also flagged and made inert. Control experiments isolated the causes:
* restore units alone give zero drift;
* the original fold on channel 116 (where the time token is large) gave 0.08 drift, so the final edits
  avoid it;
* the rest of the drift scales with how strongly sigma2 suppresses.

I could not test the real-vs-generated AUC limit. `final_mid` has about twice the drift of the other two and
is the riskiest.

## 4. Predictions against outcomes (surgical passes)

| pass | edit | predicted | observed |
|---|---|---|---|
| 004 | v1, shared sigma | E 0.45, d_acc -0.002, drift 0.05 | E 0.452, d_acc +0.0007, drift 0.56 |
| 007 | v2, block-1 rewrite | E 0.38, drift 0.55 | E 0.131, drift 0.40 |
| 011 | v3, lE<lpt veto via fold read | E 0.135, drift .27 | E 0.494 (soft1e-4 broke), drift 0.34 |
| 014 | v4, refit readouts | E 0.13, drift 0.40, DoF 4500 | E 0.132, drift 0.40, DoF 4475 |
| 015 | v4s, no fold | E 0.14, e_clean 0.01, DoF 1950 | E 0.224, e_clean 0.0096, DoF 1953 |
| 016 | v5, 8-head sigma | E 0.125, drift 0.40 | E 0.098, drift 0.21 |
| 017 | v4m, mean fold | E 0.17, e_clean 0.006 | E 0.458, d_acc -0.017 (fails) |
| 018 | v6a, particle-weighted sigma | E 0.10, drift 0.21 | E 0.082, drift 0.48 |
| 019 | v6b, summary-weighted | E 0.11, drift 0.25 | E 0.119, drift 0.215 |
| 020 | v5s, no fold | E 0.19, rho_P_reco 0.92 | E 0.197, rho_P_reco 0.916 |
| 021 | control, L=0 | E 0.98, drift 0.01 | E 1.002, drift 0.081 |
| 022 | restores only | drift 0.03 | drift 0.000 |
| 023 | fold only | drift 0.04, e_clean .003 | drift 0.081, e_clean .0026 |
| 024 | v7, time-light scratch + fold | E 0.10, drift 0.15 | E 0.095, drift 0.21 |
| 025 | v7s, cheap channels, no fold | E 0.15, rho_P_reco .95 | E 0.1005, rho_P_reco .993 |
| 026 | **v8**, minimal entries | E 0.10, DoF 480, drift .21 | **E 0.1005, DoF 482, drift .206** |
| 029 | v9, ws0.3 minimal | E 0.085, drift .45 | E 0.0846, drift .477 |
| 030 | v10, + generator term | E 0.10, drift .16 | E 0.195, drift .204 |
| 031 | v11, mix sigma1 ws0.3 | E 0.09 | E 0.102 |
| 032 | **v12**, mix sigma2 ws0.3 | E 0.095, drift .35 | **E 0.0838, drift .476** |
| 033 | v13 | E 0.088 | E 0.093 |
| 034 | v14, summary 0.1 | E 0.08 | E 0.636 |
| 035 | v15, plausibility AND | E 0.11, rho_P_reco .98 | E 0.278, rho_P_reco .845 (fails) |
| 036 | v15f | E 0.10 | E 0.274 (fails) |
| 037 | **v8f**, fold time-light | E 0.095, DoF 3000 | **E 0.0927, DoF 3510** |

Capture checks: pass 013 predicted 3 flag mismatches and found 0 (681/681); pass 027 predicted 0 and found 0
(399/399 on P).

## 5. Ideas that failed
* **Exact masking through channel 2** (body or head mask). The pid-blind local term is continuous, and
  float absorption needs an "add -L" that no free unit can provide in block 7.
* **A shared corner for blocks 1-7** (v1, E 0.45). Block 1 needs its own pattern, which is why the block-1
  rewrite exists.
* **Linear key/query edits along a "constant" direction.** The kept-token spread (s/G ≈ 0.002-0.013) gives
  logit errors of about 0.1.
* **Per-block rewrites in blocks 2-7.** No always-off units exist there.
* **Block-0 key suppression via an AND-coded channel.** It would also suppress the time token at t > 0 and
  break the generator.
* **Implausibility veto (log E < log pT).** First as an AND-NOT read of a folded scratch channel: noise and
  extrapolation broke soft1e-4. Then as a positive "plausible" AND input: it forced an expensive channel
  (116) and broke P_reco, while drift only went 0.21 → 0.19.
* **Mean fold** (catastrophic).
* **The original fold on channel 116**, the time token's large channel: +0.08 generator drift.
* **Adding generator-attention suppression to the sigma objective**: E_rule doubled.
* **The particle-heavy sigma (summary weight 0.1)**: E_rule 0.64.
* **General lesson:** my closed-form sigma objective is a weak proxy for E_rule. I therefore only compared a
  handful of principled candidates.

## 6. Passes used
42 of 50: 25 surgical, 17 capture/logits (001-042). The final files are byte-identical copies of the edits
evaluated in passes 026 (small), 032 (mid) and 037 (large). Notes, scripts and all intermediate edits are in
`run/work/`.
