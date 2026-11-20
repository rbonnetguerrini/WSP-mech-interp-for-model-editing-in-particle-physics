# Final notes: implementing the reference inside PET2 (pretrain_s)

## 1. What the reference does, and how I found it

**Finding: f_ref(x) = f_orig(x with "sub-threshold neutral" particles removed).** The removal rule is:

- **photon (pid 6):** drop iff pT < 0.8 GeV **and** E < 1.8 GeV;
- **neutral hadron (pid 7):** drop iff E < 2.2 GeV (the boundary lies in [2.126, 2.215] GeV);
- **charged particles (pid 0,1,2,4,5,8):** never dropped.

Clean jets contain no such particles. In the practice data every photon with pT < 0.8 GeV has E ≥ 1.85 GeV, and every neutral hadron has E ≥ 2.26 GeV. That is why f_ref = f_orig exactly on clean jets.

How I found it (passes 001–004):

1. **Pass 001 (surgical on the original).** e_orig is 0 on clean and soft3e-3, 0.249 on soft1e-3, 0.265 on soft1e-4 and 0.102 on P. For soft1e-4, e_orig equals the original's own soft-photon response r to every digit. This means f_ref(soft1e-4) = f_orig(clean).
2. **Passes 002–003 (reference and original logits on 512 jets × {clean, soft1e-3, soft1e-4, P}).**
   - f_ref(soft1e-4 x) equals f_orig(x) bitwise for all 512 jets.
   - For soft1e-3, each jet is exactly one of two cases: 462 jets give f_ref = f_orig(clean) bitwise, and 50 jets give f_ref = f_orig(probe) bitwise.
   - So the reference is the original model run on a filtered jet, with a hard per-particle rule. The dropped and kept soft1e-3 photons separate exactly in (pT, E) at pT = 0.8 GeV and E ≈ 1.8 GeV.
3. **Pass 004 (reference on 2048 constructed jets).** Each jet adds one particle to a clean jet, scanning pid, pT, E and position. "Dropped" means f_ref equals the stored f_orig(clean) bitwise. The rule above has **zero mismatches**:
   - 640 + 128 photons on a fine grid and at very soft pT;
   - 512 neutral hadrons;
   - 512 charged particles, none ever dropped;
   - 256 photons at dR between 0.4 and 1.0, all dropped (position does not matter).

The original model reacts strongly to these out-of-distribution soft neutrals: one 1e-4 photon moves the summary tokens by about 40%. A photon just above the threshold moves the logits about ten times less.

## 2. The circuit (all closed-form; no gradients, no fitting loops)

Exact removal through the padding mask is impossible with weight edits. The body mask is `embedded_channel_2 != 0`, an exact float test on a sum of four pid-blind and pid-aware terms. So the edit makes flagged particles **invisible to attention** in every body and classifier attention layer. It uses one residual channel (c = 70) as a "pin/flag" channel.

**Why channel 70.** It is already a near-bias channel. In the original model it is strongly negative for every token: the mean of tanh(αx)/1 for norm1 is −0.72 (block 0), −0.94 (block 1) and about −0.99 in blocks 2–6. Mean-ablating it has by far the lowest estimated read cost of all 128 channels (2.6 versus a median of 9.9).

### 2a. Flag: AND of pid and kinematics by tanh saturation

The flag uses the 8 dead hidden units of `body.embed.mlp`: units 237, 200, 245, 74, 133, 155, 163 and 76, whose norm weights are about 1e-5. Each unit gets a step-function pre-activation on one input feature:

```
pre_u = s * (t_u - n_f(x)),   s = 2e4,   n_f = tanh(0.1253 * x_f) * w_f   (the embed input norm)
```

The thresholds t_u are, for the photon-specific conditions:

- pT < 0.8;
- E < 1.8;

and for the common conditions:

- E < 2.2;
- E ≥ pT (that is, n_E/w_E − n_pT/w_pT + 1e-3 > 0);
- |Δη| < 0.8 and |Δφ| < 0.8, using two units per angle.

GELU followed by tanh(1.305·) saturates each unit to exactly 0 or exactly 1 × norm weight. fc2[70, u] = F_u / normweight_u turns each unit into a clean F_u·k_u written into channel 70. Other channels receive at most about 1e-5 × the original column, which is negligible.

With B = 400, the weights are F = B for the two photon-specific units and F = 10B for the six common units. The offsets are:

- embed fc2.bias[70] += −61.5B;
- pid_embed[:,70] += −B for charged pids, +5.5B for pid 7, and 0 for pid 6.

The channel-70 residual of a particle token is then:

```
x_70 = orig + B * ( k_pT + k_E18 + 10*(k_E22 + k_phys + k_a1..k_a4) - 61.5 + {0 photon, +5.5 NH, -1 charged} )
     = +0.5B or more   for flagged photons / neutral hadrons
     = -0.5B or less   for every other particle
```

Summary tokens get `body.token[:,:,70] = −5B`. The time token, which is only present in generator mode, gets −5B through dead `time_embed` unit 181, which reads the Fourier feature with the smallest frequency (MPFourier index 92). Without this pin, `gen_drift` is 0.66 (pass 041).

Every norm (DyTanh) therefore sees channel 70 saturated:

- −w for all normal tokens (a constant);
- +w for flagged tokens.

### 2b. Key suppression: rank-1 per attention layer

In each body attention block b, the K-rows of column 70 of `in_proj_weight` are replaced by a per-head vector U_h. Since normal keys all have norm1_70 = −w, their logits shift by a constant per query, and the softmax is unchanged. A flagged key gets an extra logit

```
Δ = 2w (q_i · U_h) / 4.
```

U_h = −λ_h Σ_q⁻¹ μ_q is the whitened mean query direction of head h, computed in closed form from captured activations. λ_h is set so that Δ = −30 at the 0.5% weakest query. In nearly all heads, all queries (summary and particle tokens) lie on one side of this direction.

The classifier TokenAttBlocks get the same treatment. Their keys read the body output directly, where channel 70 is tanh(0.188x)·0.607, saturated at ±0.607.

### 2c. Undoing the cost of pinning channel 70 for normal tokens

Every other read of channel 70 (Q and V rows, MLP fc1) now sees the constant −w instead of the original value. Two closed-form fixes:

- **mean-ablation:** scale the column, W[:,70] ← W[:,70] · m/(−w), so the read equals the data mean m. Used in final_small, in blocks 0 and 1 and the classifier.
- **least-squares reconstruction:** n_70 ≈ β0 + Σ_c β_c n_c is fitted on captured norm outputs (256 clean jets, summary tokens weighted ×20). Then W[:,c] += W[:,70]·β_c, and W[:,70] ← W[:,70]·β0/(−w). This is a rank-1 update to every reading matrix, or rank 2 together with the K column. The residual RMS is 0.017 versus 0.63 for mean-ablation in block 0, and 0.016 versus 0.35 in block 1.

**Why inputs where the reference equals the original are unaffected (up to the pin floor).**

- No normal token is ever flagged: charged pids carry −B, and above-threshold neutrals miss at least one unit.
- A normal token's only change is the pinned channel 70, which reads as a constant.
- A constant key component does not change the softmax.
- The remaining reads of channel 70 are restored by mean or regression.

The residual "floor" is e_clean = 0.021 (small), 0.012 (mid) and 0.010 (large).

**What is not implemented exactly.** A flagged particle still enters its neighbours' kNN local features (K = 15 in (η, φ, log pT)). That path is pid-blind and has no weights that could remove it. This is the residual above the floor on P (P has about 1.5 flagged particles per jet).

## 3. Final edits

| final | file | DoF | E_ref (practice) | what is in it |
|---|---|---|---|---|
| small | final_small.npz | 2238 | 0.160 | flag + K suppression in 8 body and 2 classifier blocks; mean-ablation of channel 70 Q/V in blocks 0 and 1 and the classifier; everything is a single column 70 |
| mid | final_mid.npz | 3893 | 0.095 | small + regression reconstruction in block 0 and 1 attention (rank 2) and the block-1 MLP (rank 1) |
| large | final_large.npz | 10267 | 0.080 | regression in attention of blocks 0, 1, 2, 3 and 7, MLP of blocks 1–7, and both classifier attentions |

DoF per tensor, as min(changed, r(m+n−r)):

- column-only edits of `in_proj_weight`: 128 or 384 entries;
- rank-2 `in_proj_weight` (384×128): 1020;
- rank-1 `mlp.fc1.weight` (256×128): 383;
- rank-2 `classifier...in_proj_weight`: 1020;
- embed fc1 rows (rank 4, 32 entries): 32;
- embed fc1.bias: 8; fc2.weight (8 entries of row 70): 8; fc2.bias: 1; body.token: 4; pid_embed column 70: 7; time_embed fc1 and fc2: 1 entry each.

## 4. Final practice metrics

| metric | original | final_small (018) | final_mid (031) | final_large (023) |
|---|---|---|---|---|
| E_ref | 1.0000 | 0.1598 | 0.0953 | 0.0800 |
| e[soft1e-3] | 0.2486 | 0.0228 | 0.0135 | 0.0117 |
| e[soft1e-4] | 0.2646 | 0.0222 | 0.0140 | 0.0123 |
| e[P] | 0.1016 | 0.0309 | 0.0182 | 0.0149 |
| e[clean] | 0.0000 | 0.0208 | 0.0119 | 0.0101 |
| e[split2] | 0.0001 | 0.0239 | 0.0121 | 0.0113 |
| e[soft3e-3] | 0.0000 | 0.0275 | 0.0130 | 0.0109 |
| DoF | 0 | 2238 | 3893 | 10267 |
| acc / d_acc | 0.8357 / +0.0000 | 0.8360 / +0.0003 | 0.8363 / +0.0006 | 0.8359 / +0.0002 |
| worst class drop | 0.0000 | 0.0059 | 0.0029 | 0.0029 |
| d_nll | +0.0000 | +0.0015 | +0.0015 | +0.0005 |
| KL | 0.0000 | 0.0019 | 0.0009 | 0.0006 |
| rho D | 1.000 | 1.025 | 0.981 | 1.007 |
| rho J | 1.000 | 1.002 | 0.987 | 1.006 |
| rho P | 1.000 | 0.751 | 0.733 | 0.732 |
| rho P_ref | 1.000 | 1.027 | 1.003 | 1.000 |
| gen_drift t0.2/0.5/0.85 | 0.000/0.000/0.000 | 0.038/0.047/0.074 | 0.037/0.044/0.075 | 0.039/0.051/0.074 |
| R_vis / R_in / R_sub | 1.000 / 1.000 / 1.000 | 0.575 / 1.116 / 0.034 | 0.520 / 1.008 / 0.031 | 0.521 / 1.012 / 0.031 |
| dir_err D/J/P | 0.000/0.000/0.000 | 0.186/0.097/0.342 | 0.065/0.060/0.339 | 0.050/0.046/0.338 |
| gate_ref_no_g2 | True | True | True | True |
Size per tensor:

#### final_small (pass 018): DoF 2238, changed entries 2238

| tensor | rank | changed entries |
|---|---|---|
| `body.token` | 1-D/3-D | 4 |
| `body.embed.mlp.fc1.weight` | 4 | 32 |
| `body.embed.mlp.fc1.bias` | 1-D/3-D | 8 |
| `body.embed.mlp.fc2.weight` | 1 | 8 |
| `body.embed.mlp.fc2.bias` | 1-D/3-D | 1 |
| `body.time_embed.fc1.weight` | 1 | 1 |
| `body.time_embed.fc2.weight` | 1 | 1 |
| `body.pid_embed.0.weight` | 1 | 7 |
| `body.in_blocks.0.attn.in_proj_weight` | 1 | 384 |
| `body.in_blocks.1.attn.in_proj_weight` | 1 | 384 |
| `body.in_blocks.2.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.3.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.4.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.5.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.6.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.7.attn.in_proj_weight` | 1 | 128 |
| `classifier.in_blocks.0.attn.in_proj_weight` | 1 | 384 |
| `classifier.in_blocks.1.attn.in_proj_weight` | 1 | 256 |

#### final_mid (pass 031): DoF 3893, changed entries 132542

| tensor | rank | changed entries |
|---|---|---|
| `body.token` | 1-D/3-D | 4 |
| `body.embed.mlp.fc1.weight` | 4 | 32 |
| `body.embed.mlp.fc1.bias` | 1-D/3-D | 8 |
| `body.embed.mlp.fc2.weight` | 1 | 8 |
| `body.embed.mlp.fc2.bias` | 1-D/3-D | 1 |
| `body.time_embed.fc1.weight` | 1 | 1 |
| `body.time_embed.fc2.weight` | 1 | 1 |
| `body.pid_embed.0.weight` | 1 | 7 |
| `body.in_blocks.0.attn.in_proj_weight` | 2 | 49152 |
| `body.in_blocks.1.attn.in_proj_weight` | 2 | 49152 |
| `body.in_blocks.1.mlp.fc1.weight` | 1 | 32768 |
| `body.in_blocks.2.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.3.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.4.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.5.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.6.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.7.attn.in_proj_weight` | 1 | 128 |
| `classifier.in_blocks.0.attn.in_proj_weight` | 1 | 384 |
| `classifier.in_blocks.1.attn.in_proj_weight` | 1 | 256 |

#### final_large (pass 023): DoF 10267, changed entries 573883

| tensor | rank | changed entries |
|---|---|---|
| `body.token` | 1-D/3-D | 4 |
| `body.embed.mlp.fc1.weight` | 4 | 32 |
| `body.embed.mlp.fc1.bias` | 1-D/3-D | 8 |
| `body.embed.mlp.fc2.weight` | 1 | 8 |
| `body.embed.mlp.fc2.bias` | 1-D/3-D | 1 |
| `body.time_embed.fc1.weight` | 1 | 1 |
| `body.time_embed.fc2.weight` | 1 | 1 |
| `body.pid_embed.0.weight` | 1 | 7 |
| `body.in_blocks.0.attn.in_proj_weight` | 2 | 49152 |
| `body.in_blocks.1.attn.in_proj_weight` | 2 | 49152 |
| `body.in_blocks.1.mlp.fc1.weight` | 1 | 32768 |
| `body.in_blocks.2.attn.in_proj_weight` | 2 | 49152 |
| `body.in_blocks.2.mlp.fc1.weight` | 1 | 32768 |
| `body.in_blocks.3.attn.in_proj_weight` | 2 | 49152 |
| `body.in_blocks.3.mlp.fc1.weight` | 1 | 32768 |
| `body.in_blocks.4.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.4.mlp.fc1.weight` | 1 | 32768 |
| `body.in_blocks.5.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.5.mlp.fc1.weight` | 1 | 32768 |
| `body.in_blocks.6.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.6.mlp.fc1.weight` | 1 | 32768 |
| `body.in_blocks.7.attn.in_proj_weight` | 2 | 49152 |
| `body.in_blocks.7.mlp.fc1.weight` | 1 | 32768 |
| `classifier.in_blocks.0.attn.in_proj_weight` | 2 | 49149 |
| `classifier.in_blocks.1.attn.in_proj_weight` | 2 | 49152 |

### Predictions vs outcomes (every pass)

| pass | kind | weights | prediction | outcome |
|---|---|---|---|---|
| 001 | surgical |  | {"E_ref": 1.0, "d_acc": 0.0, "e_clean": 0.0} | E_ref=1.0000, e_clean=0.0000, d_acc=+0.0000, gen_drift=0.000, DoF=0, gate=True |
| 002 | reference | setA.npz | {"ref_soft1e-4_vs_orig_clean_dist": 0.0, "ref_clean_vs_orig_clean": 0.0} | data returned (see analysis) |
| 003 | capture | setA.npz | {"orig_soft1e-3_response": 0.25} | data returned (see analysis) |
| 004 | reference | setB.npz | {"frac_photon_grid_dropped": 0.35, "frac_charged_dropped": 0.0} | data returned (see analysis) |
| 005 | capture | setA.npz | {"n_channels_saturated_all_tokens_block7": 2} | data returned (see analysis) |
| 006 | capture | setA.npz | {"ch70_mean_block5": -5.0} | data returned (see analysis) |
| 007 | capture | setA.npz | {"ch70_bodynorm_mean": -0.8} | data returned (see analysis) |
| 008 | surgical | e1.npz | {"E_ref": 0.2, "d_acc": -0.003, "d_nll": 0.01, "gen_drift_max": 0.05, "rho_P_ref": 1.0} | E_ref=0.1630, e_clean=0.0217, d_acc=+0.0008, gen_drift=0.182, DoF=5792, gate=True |
| 009 | surgical | e2.npz | {"E_ref": 1.0, "e_clean": 0.022, "gen_drift_max": 0.08} | E_ref=1.0672, e_clean=0.0217, d_acc=+0.0008, gen_drift=0.030, DoF=5767, gate=True |
| 010 | surgical | e3.npz | {"E_ref": 0.165, "gen_drift_max": 0.07} | E_ref=0.1630, e_clean=0.0217, d_acc=+0.0008, gen_drift=0.083, DoF=5822, gate=True |
| 011 | surgical | e3b.npz | {"E_ref": 0.165, "gen_drift_max": 0.1} | E_ref=0.1630, e_clean=0.0217, d_acc=+0.0008, gen_drift=0.119, DoF=5798, gate=True |
| 012 | surgical | e4.npz | {"E_ref": 0.17, "e_clean": 0.023, "gen_drift_max": 0.07, "dof": 3200} | E_ref=0.1605, e_clean=0.0211, d_acc=+0.0004, gen_drift=0.075, DoF=3518, gate=True |
| 013 | surgical | e5.npz | {"E_ref": 0.22, "e_clean": 0.035, "gen_drift_max": 0.08, "dof": 1400} | E_ref=0.4179, e_clean=0.1002, d_acc=-0.0091, gen_drift=0.084, DoF=1342, gate=False |
| 014 | surgical | e6.npz | {"E_ref": 0.14, "e_clean": 0.015, "dof": 4900} | E_ref=0.1571, e_clean=0.0209, d_acc=+0.0008, gen_drift=0.075, DoF=4670, gate=True |
| 015 | surgical | e7.npz | {"E_ref": 0.08, "e_clean": 0.008, "gen_drift_max": 0.06, "dof": 10000} | E_ref=0.0800, e_clean=0.0101, d_acc=+0.0002, gen_drift=0.074, DoF=10523, gate=True |
| 016 | surgical | e8.npz | {"E_ref": 0.1, "e_clean": 0.013, "dof": 6500} | E_ref=0.0877, e_clean=0.0099, d_acc=+0.0004, gen_drift=0.075, DoF=5807, gate=True |
| 017 | surgical | e9.npz | {"E_ref": 0.075, "e_clean": 0.0095, "dof": 13000} | E_ref=0.0802, e_clean=0.0102, d_acc=+0.0001, gen_drift=0.075, DoF=13199, gate=True |
| 018 | surgical | e10.npz | {"E_ref": 0.2, "e_clean": 0.03, "dof": 2500} | E_ref=0.1598, e_clean=0.0208, d_acc=+0.0003, gen_drift=0.074, DoF=2238, gate=True |
| 019 | surgical | e11.npz | {"E_ref": 0.1, "e_clean": 0.012, "dof": 3600} | E_ref=0.1083, e_clean=0.0142, d_acc=+0.0015, gen_drift=0.074, DoF=4146, gate=True |
| 020 | logits | e8.npz | {"median_err_dropped_over_S": 0.012, "median_err_charged_over_S": 0.012} | data returned (see analysis) |
| 021 | surgical | e12.npz | {"E_ref": 0.085, "e_soft1e-4": 0.0118, "dof": 5807} | E_ref=0.0888, e_clean=0.0099, d_acc=+0.0004, gen_drift=0.081, DoF=5807, gate=True |
| 022 | surgical | f_mid.npz | {"E_ref": 0.0877, "dof": 5551} | E_ref=0.0877, e_clean=0.0099, d_acc=+0.0004, gen_drift=0.075, DoF=5551, gate=True |
| 023 | surgical | f_large.npz | {"E_ref": 0.08, "dof": 10267} | E_ref=0.0800, e_clean=0.0101, d_acc=+0.0002, gen_drift=0.074, DoF=10267, gate=True |
| 024 | surgical | f_large2.npz | {"E_ref": 0.072, "e_clean": 0.008, "dof": 12300} | E_ref=0.0952, e_clean=0.0144, d_acc=+0.0001, gen_drift=0.074, DoF=12313, gate=True |
| 025 | surgical | La.npz | {"E_ref": 0.078, "e_clean": 0.0097} | E_ref=0.0808, e_clean=0.0104, d_acc=+0.0000, gen_drift=0.074, DoF=10650, gate=True |
| 026 | surgical | Lb.npz | {"E_ref": 0.085, "e_clean": 0.011} | E_ref=0.0787, e_clean=0.0101, d_acc=+0.0005, gen_drift=0.074, DoF=11291, gate=True |
| 027 | surgical | Sa.npz | {"E_ref": 0.14, "e_clean": 0.018, "dof": 2900} | E_ref=0.1586, e_clean=0.0207, d_acc=-0.0001, gen_drift=0.074, DoF=2874, gate=True |
| 028 | surgical | Sb.npz | {"E_ref": 0.13, "e_clean": 0.017, "dof": 2900} | E_ref=0.1197, e_clean=0.0160, d_acc=+0.0008, gen_drift=0.073, DoF=2874, gate=True |
| 029 | surgical | M1.npz | {"E_ref": 0.089, "dof": 4900} | E_ref=0.0899, e_clean=0.0103, d_acc=+0.0009, gen_drift=0.075, DoF=4915, gate=True |
| 030 | surgical | M2.npz | {"E_ref": 0.095, "dof": 4000} | E_ref=0.1201, e_clean=0.0158, d_acc=+0.0001, gen_drift=0.074, DoF=4023, gate=True |
| 031 | surgical | M3.npz | {"E_ref": 0.1, "dof": 3900} | E_ref=0.0953, e_clean=0.0119, d_acc=+0.0006, gen_drift=0.075, DoF=3893, gate=True |
| 032 | surgical | L2.npz | {"E_ref": 0.081, "dof": 9000} | E_ref=0.0856, e_clean=0.0111, d_acc=-0.0001, gen_drift=0.074, DoF=8867, gate=True |
| 033 | surgical | N1.npz | {"E_ref": 0.105, "dof": 3500} | E_ref=0.1101, e_clean=0.0145, d_acc=+0.0015, gen_drift=0.074, DoF=3510, gate=True |
| 034 | surgical | S6.npz | {"E_ref": 0.25, "dof": 1982} | E_ref=0.2689, e_clean=0.0208, d_acc=+0.0003, gen_drift=0.069, DoF=1982, gate=True |
| 035 | surgical | S0.npz | {"E_ref": 0.18, "dof": 2110} | E_ref=0.5498, e_clean=0.1007, d_acc=-0.0083, gen_drift=0.078, DoF=1854, gate=False |
| 036 | capture | cap_b0id.npz | {"ch70_mean_over_w_block0_input": -0.72} | data returned (see analysis) |
| 037 | surgical | v2_small.npz | {"E_ref": 0.155, "dof": 2238} | E_ref=0.1723, e_clean=0.0259, d_acc=+0.0004, gen_drift=0.073, DoF=2238, gate=True |
| 038 | surgical | v2_Sb.npz | {"E_ref": 0.11, "dof": 2874} | E_ref=0.1194, e_clean=0.0160, d_acc=+0.0005, gen_drift=0.079, DoF=2874, gate=True |
| 039 | surgical | v2_M3.npz | {"E_ref": 0.09, "dof": 3893} | E_ref=0.0941, e_clean=0.0120, d_acc=+0.0007, gen_drift=0.082, DoF=3893, gate=True |
| 040 | surgical | v2_large.npz | {"E_ref": 0.075, "dof": 10267} | E_ref=0.0779, e_clean=0.0099, d_acc=+0.0000, gen_drift=0.080, DoF=10267, gate=True |
| 041 | surgical | M3nt.npz | {"E_ref": 0.0953, "gen_drift_max": 0.09} | E_ref=0.0953, e_clean=0.0119, d_acc=+0.0006, gen_drift=0.664, DoF=3891, gate=True |
| 042 | surgical | M3r6.npz | {"E_ref": 0.1, "gen_drift_max": 0.065} | E_ref=0.1185, e_clean=0.0119, d_acc=+0.0006, gen_drift=0.060, DoF=3893, gate=True |
## 5. Failed ideas and negative results

- **Exact masking through embedded channel 2 = 0.** This is impossible. The mask is an exact float test on a sum of four terms that include pid-blind kNN features. Float absorption tricks can zero channel 2, but they also destroy the information of normal tokens: pid offsets are unconditional, and the flag needs AND(pid, kinematics) before any nonlinearity.
- **A purely kinematic flag** would also hide soft charged particles. About 500 charged particles in the clean practice set have pT < 0.8 and E < 1.8 GeV. Solved by the pid offsets combined with saturation.
- **Pin without mean-ablation** (pass 013): clean e = 0.10, and the gates fail (class drop 0.056, d_acc −0.009).
- **The first flag without plausibility conditions** (pass 008) gave `gen_drift` 0.18 at t = 0.85, because noised generator inputs fall into the flagged region. Adding E ≥ pT and the |Δη|, |Δφ| < 0.8 box brought it to 0.075 with no loss in E_ref. A 0.6 box gives 0.060 but worsens E_ref (0.095 → 0.119, pass 042).
- **No time-token pin** (pass 041): `gen_drift` 0.66.
- **Stronger suppression** (T = 100, pass 021): no gain. Leakage is already saturated, and the residual is the kNN path.
- **No key suppression in blocks 6–7** (pass 034): E_ref 0.27. Suppression is needed in every block.
- **Regression in every block** (pass 017): no gain over the large edit.
- **Classifier MLP/fc compensation** (passes 024–026): no gain or worse, except a sliver from the fc1 bias (0.080 → 0.079 for +1024 DoF; not used).
- **Refitting block 0 on the true block-0 input** (identity-block capture, pass 036; passes 037–040): E_ref improved by only 0.001–0.002 while `gen_drift` rose from 0.074 to 0.080, and mean-ablation got worse. I kept the original fit.

## 6. Risks

- The generator limit (real-vs-generated AUC) cannot be measured here. All three finals have `gen_drift` of about 0.04 (t = 0.2) to 0.075 (t = 0.85). About 0.03 of that comes from the pinned channel alone (pass 009); the rest comes from flagged noised particles at high t.
- The neutral-hadron E threshold is known only to lie in [2.13, 2.21] GeV; I used 2.2.
- Flagged particles outside |Δη|, |Δφ| < 0.8 are not hidden (0.06% of P's flagged particles).
- The kNN path is not removed.

## 7. Passes used

42 of 50:

- 34 surgical (001 baseline, 008–019, 021–035, 037–042);
- 2 reference (002, 004);
- 5 capture (003, 005, 006, 007, and 036 with a capture-only identity edit of block 0);
- 1 logits (020: e8 on the 2048 constructed probes).

Full list in section 4.

The e8 check on the constructed probes (pass 020, the mid-size predecessor), as mean error / S:

- kept charged particles and above-threshold neutrals: 0.012–0.013, the pin floor;
- dropped photons and neutral hadrons: median 0.014–0.016, with a p95 tail of 0.07–0.11 from the kNN path;
- far photons beyond the 0.8 box: p95 0.22, by design.
