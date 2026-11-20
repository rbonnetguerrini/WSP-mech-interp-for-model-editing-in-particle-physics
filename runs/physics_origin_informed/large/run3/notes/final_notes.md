# Final notes: the JetClass reconstruction rule compiled into `pretrain_l`

All three final edits use closed-form constructions only (least squares, covariance solves, eigenvectors, sign rounding of an eigenvector). There is no gradient and no iterative optimiser. No `generator.*` tensor is touched, and all tensors are float32 in their original shapes.

## 1. Final edits

| file | E_rule (practice) | DoF | soft1e-3 ratio | soft1e-4 ratio | P ratio | clean e | pass |
|---|---|---|---|---|---|---|---|
| `final_small.npz` | 0.2568 | 98,266 | 0.101 | 0.452 | 0.238 | 0.0014 | 33 |
| `final_mid.npz` | 0.1932 | 245,650 | 0.091 | 0.278 | 0.194 | 0.0112 | 31 |
| `final_large.npz` | 0.1866 | 2,320,341 | 0.091 | 0.278 | 0.187 | 0.0089 | 32 |

P ratio = e_P / e_orig,P with e_orig,P = 0.151. Original-model e: soft1e-3 0.373, soft1e-4 0.105, P 0.151.

### Tensors changed (rank / changed entries)

**Common detector** (in all three; about 8.2k DoF):
- `body.in_blocks.0.mlp.fc1.weight`: rank 2, 2048 entries (rows 1042 and 683).
- `body.in_blocks.0.mlp.fc1.bias`: 2.
- `body.in_blocks.0.mlp.norm.weight`: 2.
- `body.in_blocks.0.mlp.fc2.weight`: rank 2, 2048 (columns).
- `body.in_blocks.0.mlp.fc2.bias`: 1024.
- `body.in_blocks.1.mlp.fc1.weight`: rank 1, 1024 (row 1475).
- `body.in_blocks.1.mlp.fc1.bias`: 1.
- `body.in_blocks.1.mlp.norm.weight`: 1.
- `body.in_blocks.1.mlp.fc2.weight`: rank 1, 1024.
- `body.in_blocks.1.mlp.fc2.bias`: 1024.

**Removal (key/query flag terms), all in `attn.in_proj_weight` (3072x1024). Each one changes about 2.1M entries.**
- **small:** body blocks 1-9 plus classifier blocks 0-1, each rank 2 (DoF 8188 each).
- **mid:** body blocks 1-27 plus classifier blocks 0-1, each rank 2.
- **large:** rank 2 in body blocks 1-13. Rank 34 in body blocks 14-27 and classifier block 1, rank 33 in classifier block 0 (about 138k DoF each).

DoF totals are the oracle's numbers: 98,266 / 245,650 / 2,320,341.

## 2. The circuit

### 2.1 Why the true padding mask cannot be reproduced

The body's padding mask is read once, from channel 2 of the summed embedding `E(kin) + L(kNN) + P[pid] + A(add_info)`. Pid and kinematics meet only additively before block 0. An additive function cannot vanish on every kept token (soft charged particles, photons with pT>0.8, ...) while being non-zero on the dropped ones. So "neutral AND below threshold" needs a nonlinearity, and the first one is block 0. The mask itself is therefore emulated, not reproduced.

### 2.2 Detection: exact-zero "inverted" units

DynamicTanh `w*tanh(a*x)` saturates to exactly 1.0 in float32 for `a*x > 9.01`, and GELU(z) is exactly 0 for z < -5.6. A repurposed dead hidden unit (norm weight about 1e-6, set to 1) gets the pre-activation

`z = 34.5-ish - s * (readout - threshold) - M * gate`, so that `h = tanh(a*GELU(z))` is exactly 1 for every kept token.

Its fc2 column is `-Delta` and `Delta` is added to fc2.bias. The unit's net contribution is therefore exactly 0 when h = 1 (kept tokens), and `+Delta` when z <= -5.6 (dropped tokens). The partial band only ever adds more than 0, never a negative push.

- **Block 0, unit 683 (neutral hadron):** readout of `log 2.2 - log E` from `norm2(x)`, fitted by weighted least squares on captured activations. Near-threshold rmse is 0.008. The gate is a Fisher-type pid-7 readout (about 1e-4 noise on pid 7, at most -1 on everything else), with s = 8000 and M = 3e4. It writes `Delta = 512*s` (s is a fixed ±1 sign vector).
- **Block 0, unit 1042 (photon, pT part):** readout of `log 0.8 - log pT` (rmse 0.0016 near threshold), pid-6 gate. It writes `Delta = -256 e_648`, a single residual channel. That channel is >= 0 for every kept token, so it is a clean binary flag one block later.
- **Block 1, unit 1475 (photon AND E<1.8 GeV):** `z = 14 - 8000*Ehat + K*(1 - f)`. Here `f = -norm2(x)_648 / w_648` is exactly 1 for pT-flagged photons and at most 0.07 for every other token. `Ehat` is the log-E readout fitted on pT-flagged photons (rmse 0.0024). K = 1.3 times the largest value needed to keep every non-flagged token's z above 34.5. It writes `Delta = 512*s`.

So a dropped token's residual becomes `x + 512*s`. From block 2 on (block 1 for neutral hadrons) every DyT norm in the body, and body.norm, sees it at the exact vertex `w ⊙ s`. Kept tokens are unchanged, apart from float rounding of the bias/column cancellation (at most about 1e-4) and the measured few tokens within about 0.002 log units of a threshold.

The sign vector s is the sign of the lowest eigenvector of `sum_b M_b / tr M_b`, where `M_b = sum_h diag(w_b) W_k^T E[qq^T] W_k diag(w_b)`. This minimises the vertex's raw attention scores: they fell from 95x to 0.04x of a random sign vector at block 1.

### 2.3 Removal: a flag-sensitive "extra dimension" in every attention

Key-only suppression is impossible: in every head, some queries point opposite to the mean query. So both projections get a term (shared readouts rho and r; per head h, vectors a_h and b_h):

- `q_i += a_h (rho·n_i)`, where `rho = argmin Var(rho·n)` subject to `rho·mu = 1`. Then `rho·n_i ≈ 1` for every kept query (min 0.65-0.98).
- `k_j += b_h (r·n_j)`, where `r = (Sigma_kept + eps I)^-1 (n* - mu)` is normalised so `r·(n* - mu) = 1`. Its spread on kept tokens is 2e-5 (block 1) to 6.5e-3 (block 27).

The added logit is `(q_i·b_h)(r·n_j) + (rho·n_i)(a_h·k_j) + (a_h·b_h)(rho·n_i)(r·n_j)`:
- Choose `a_h ∝ e_h` and `b_h ∝ f_h` (with `e_h·f_h = 1`), from the generalised eigenproblem `Cov(k) e = λ E[qq^T]^-1 e`. Then the cross terms are nearly constant across keys for kept tokens, and softmax ignores constant shifts.
- Set `a_h·b_h/√32 = -T_h`, where T_h = T0 + the 99.9% quantile of "vertex raw score minus best kept score".
- Choose `|b_h|` so the uncontrolled `(q·f)` term is at most T/ub.

The flagged vertex is then at least about 4-10 nats below the best kept key for practically every query. Two settings are used:
- **Blocks 1-13:** T0 = 12, ub = 4.
- **Blocks 14-27 and the classifier (mid):** T0 = 6, ub = 2. This halves the kept-logit perturbation there.

Because rho and r are shared by all heads, each block's change is rank 2.

**Large variant ("sacrifice a head dimension"), blocks 14-27 and the classifier.** In each head, the key component along a unit direction e_h is removed and replaced by the flag:

`k' = k - e_h (e_h·k) + β f_h (r·n)`, `q' = q + α e_h (rho·n)`.

This removes the `(rho·n)(a·k)` cross term exactly. The price is the variance of `(q·e_h)(k·e_h)`, with e_h chosen among eigenvectors to minimise it. It costs rank 34 per block, but the late-block kept perturbation falls from about 0.05-0.08 to about 0.01-0.03 nats (T0 = 5).

All readouts were fitted on 300 practice jets that are not in the surgical subset. The classifier head uses the same construction, with summary tokens as queries and body outputs as keys (vertex `w_bodynorm ⊙ s`).

### 2.4 Why kept tokens are unaffected, and what is not removed

- Kept tokens are bit-identical through the detector (the GELU/tanh saturations are exact). In attention, their logits move only by the residual spread of the readouts: std 3e-4 to 0.08 nats per block. That is the clean e: 0.0014 / 0.011 / 0.009.
- **Not removable with weights:**
  1. Block-0 attention (it runs before any nonlinearity can see pid and kinematics together).
  2. The kNN local embedding of the dropped particle's neighbours (it has no pid input, and the neighbour substitution cannot be emulated).
  3. Particles within about 0.002-0.005 log units of a threshold.

  Pass 29 showed that removing block-0 particle-particle attention to ultra-soft particles changes nothing, so (2) is most likely the main residual for soft1e-4.
- **Generator:** `gen_drift` is about 0.265 for every variant. It is caused by noised (t>0) neutral tokens that look sub-threshold being flagged; the q/k edits do not affect it (pass 13 vs 12). It cannot be gated on t with the available data, because the time token is masked at t=0 and the oracle only captures at t=0.

## 3. Final practice metrics (surgical passes 33 / 31 / 32)

| | small | mid | large |
|---|---|---|---|
| E_rule | 0.2568 | 0.1932 | 0.1866 |
| d_acc | 0.0000 | +0.0015 | +0.0015 |
| worst class drop | 0.000 | 0.0049 | 0.0049 |
| d_nll | -0.0001 | +0.0007 | +0.0005 |
| rho D / J / P_reco | 1.000 / 1.000 / 0.999 | 1.024 / 0.993 / 0.998 | 1.020 / 0.990 / 1.000 |
| rho P (raw, expected to change) | 0.69 | 0.71 | 0.71 |
| e split2 / soft3e-3 / clean | 0.0014 / 0.0015 / 0.0014 | 0.0134 / 0.0114 / 0.0112 | 0.0108 / 0.0090 / 0.0089 |
| gate_rule_no_g2 | true | true | true |
| gen_drift max | 0.265 | 0.266 | 0.265 |

**Unseen rule probes I built myself** (300 non-surgical jets, logits passes 34-38, S estimated from the 300 clean jets):
- one soft neutral hadron (E 0.5-2.1 GeV): e_orig = 0.222; ratios small 0.183, mid 0.174, large 0.169.
- one photon at 3e-4 of the jet pT: e_orig = 0.155; ratios small 0.257, mid 0.229, large 0.221.

## 4. Predictions vs outcomes (surgical and key captures)

| pass | what | predicted | outcome |
|---|---|---|---|
| 1 | block-0 readouts | rmse log pT 0.05, pid margin large | rmse 0.0016-0.012, pid perfectly separable |
| 2 | stage-1 exactness | kept change < 1e-3, flags 99% | kept unchanged except 2 tokens within 0.0004 of threshold; flags correct |
| 3 | block-1 E readout | 0.005 | 0.0024 |
| 4-10 | vertex saturation | 98% | 100% from block 2 |
| 12 | full rank-2 v2 | E 0.35, clean 0.06, gen 0.05 | E 0.234, clean 0.022, gen 0.26 |
| 13 | blocks 1-13 + classifier | E 0.45, clean 0.01 | E 0.238, clean 0.0016 |
| 16 | v3 (block-1 photon suppression) | E 0.20 | E 0.252 (worse), gen 0.68 |
| 17 | refit stage 1 | E 0.235 | E 0.2357 (gen 0.69, weak gate) |
| 18 | blocks 1-6 + classifier | E 0.35 | E 0.408 |
| 19 | large v5 | E 0.20, soft1e-4 0.33 | E 0.198, 0.277 |
| 20 | strong gate | E 0.236, gen 0.27 | E 0.2361, gen 0.265 |
| 21 | re-push at 13 (live unit) | E 0.215 | E 0.226 |
| 22 | small with re-push at 6 | E 0.32 | E 0.852 (fail) |
| 23 | stronger early margin | E 0.225 | E 0.237 |
| 24 | large v8 | E 0.185, clean 0.009 | E 0.1866, clean 0.0092 |
| 25 | no classifier suppression | E 0.30 | E 0.255 |
| 26 | blocks 1-9 + classifier | E 0.31 | E 0.257 |
| 27 | rank-2 everywhere, low late margin | E 0.21, clean 0.012 | E 0.194, clean 0.0117 |
| 29 | + ultra-soft x_int unit | E 0.17, soft1e-4 0.20 | E 0.194, 0.277 (no effect) |
| 30 | large v12 | E 0.19 | E 0.1865 |
| 31 | mid refit | E 0.192, clean 0.011 | E 0.1932, clean 0.0112 |
| 32 | large refit | E 0.185, clean 0.009 | E 0.1866, clean 0.0089 |
| 33 | small refit | E 0.255 | E 0.2568 |
| 35-38 | unseen probes | ratios nh 0.3 / ph 0.2 (mid) | nh 0.174 / ph 0.229 |

## 5. Ideas that failed

- **Making the mask exact (channel 2 = 0):** impossible, because the AND needs a nonlinearity (section 2.1).
- **Key-only (rank-1) suppression:** queries are not in a half-space; up to 50% have the opposite sign.
- **Sign vector aligned with the mean residual:** the vertex became an attention magnet (raw scores up to 4000). Fixed by the eigenvector sign choice.
- **Photon pT flag written on 256 channels, block-1 photon suppression, un-push unit (pass 16):** no gain, worse E readout, and gen_drift 0.68.
- **Weak pid gate (M = 3000):** gen_drift 0.69. M = 3e4 restores 0.26.
- **Re-pushing flagged tokens to a "quiet" vertex in unsuppressed blocks:** about 3x lower mean attention weight, but much worse results (pass 22: E 0.85). Dead units exist only in blocks 0-7, and the pass-21 variant had to overwrite a live unit.
- **Hybrid sacrifice (8 worst heads only):** kept perturbation got worse (0.067 vs 0.022).
- **Lowering T0 to 4 in rank-2 blocks:** little gain, and the flagged vertex stops being suppressed for a few queries.
- **Ultra-soft (pT < 0.12 GeV) suppression through a repurposed interaction-MLP unit (36 DoF, pass 29):** no measurable effect, so it was dropped.
- **Exact flags in the input embedding:** the InputBlock MLP has no dead units.

## 6. Passes used

38 of 50:
- 14 captures/diagnostics (1-11, 14, 15, 28)
- 19 surgical (12, 13, 16-27, 29-33)
- 5 logits checks on unseen probes (34-38)
