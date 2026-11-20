# Final notes: the reco rule compiled into PET2 `pretrain_s` (closed form, no gradients)

## Final edits (practice-set metrics from the oracle)

| file | flag channel f | DoF | E_rule | e_clean | e_soft1e-3 / 1e-4 / P | d_acc | worst class drop | d_nll | rho D / J / P_reco | gen_drift (max) | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `final_small.npz` | 51 | 2251 | **0.191** | 0.0312 | 0.0318 / 0.0319 / 0.0331 | -0.0037 | 0.0137 | +0.0085 | 1.006 / 1.007 / 0.990 | 0.047 | pass |
| `final_mid.npz` | 51 | 5315 | **0.0994** | 0.0148 | 0.0157 / 0.0157 / 0.0178 | -0.0013 | 0.0049 | +0.0032 | 0.998 / 0.993 / 0.986 | 0.061 | pass |
| `final_large.npz` | 70 | 14321 | **0.0761** | 0.0121 | 0.0129 / 0.0132 / 0.0128 | +0.0008 | 0.0029 | +0.0004 | 0.994 / 0.988 / 1.000 | 0.164 | pass |

Original model: e_v(soft1e-3)=0.2486, e_v(soft1e-4)=0.2646, e_v(P)=0.1016, so E_rule=1 by definition.
Other reported values: split2 e = 0.032 / 0.017 / 0.014, soft3e-3 e = 0.032 / 0.015 / 0.012; R_vis 0.51 / 0.51 / 0.53,
R_in 1.00 / 0.99 / 1.03, R_sub 0.026 for all three; dir_err D 0.096 / 0.063 / 0.059, J 0.104 / 0.069 / 0.069.
**Generator risk:** `final_large` uses channel 70, which the generator head reads heavily. Its gen_drift is 0.164, against
0.047 and 0.061 for the channel-51 edits. I could not check the real-vs-generated AUC. If one edit fails the generator
limit, it will most likely be `final_large`.

Extra check on a probe the practice set doesn't have: one soft **neutral hadron** (pid 7, E in [0.3, 2.15] GeV,
uniform in dR<0.4) added to 1024 clean jets. Error against orig(reco x)=orig(clean), with my S=17.97:
original 0.203, `final_mid` 0.0293, `final_large` 0.0281 (collateral floor on those jets: 0.0144). With a dropped
**photon** at the same positions, `final_mid` also gives 0.0292. So the excess over the floor is kinematic: it comes
from kNN local features (see the limits section). It is not specific to pid 7.

## The circuit

### 1. Detection: an exact flag at the embedding (the only place that sees raw kinematics and pid before any attention)
The embedding sum is x = E(kin) + L(local) + P(pid) + A(add). A(add) is exactly 0 for neutrals, because add_embed has
no bias. The embed MLP has near-dead hidden units (|DyT weight|·|fc2 column| ~ 1e-7). Three of them become exact step
functions on the DyT-normalised inputs ñ_pt = 1.27·tanh(0.1253·log pT) and ñ_E:

  z_u = K(ñ − ñ_thr) + 10,  K = 2·10^5,  h_u = w_u·tanh(1.305·GELU(z_u)) ∈ {0, w_u} (exactly 0 below, saturated above)

giving t_pt=[pT≥0.8], t_E18=[E≥1.8], t_E22=[E≥2.2]. In the flag channel f:

  x_f / M = −1 + p(pid) − (t_pt + t_E18 + 3·t_E22),  p(γ=6)=1.5, p(n=7)=3.5, p(other)=0 (unchanged, ~1e-4), M=400

The values are:
- dropped photon (0,0,0): +0.5;
- kept photons: ≤ −0.5;
- dropped neutral hadron (t_E22=0): ≥ +0.5;
- kept neutral hadrons: ≤ −1.5;
- charged particles, and generation without pid: ≤ −1.

Summary tokens get x_f = −M (`body.token`). The time token is only nonzero at t>0 (generator). It gets
−10^5·h_6(t) through time_embed unit 6, whose pre-activation is >0 for all t∈(0,1]; I checked this from the MPFourier
buffers and fc1. Every later DyT reads channel f saturated: n_f = s_k (kept) or −s_k (dropped). A few O(1) block writes
cannot flip a value of ±200 or more.

Lean wiring: the step units keep their tiny DyT weight w_u and their fc2 column. Only fc2[f,u] = −coef·M/w_u, fc1 row u
and fc1.bias[u] change (fc1 rank 2, 12 entries; fc2 1 entry per unit).

### 2. Removal: a key-side logit penalty (attention is the only way a token influences the others)
The padding mask is fixed before block 0 (channel 2 of the embedding sum). Exact masking of a dropped token is
therefore impossible: kept and dropped photons share the same pid row, and L is continuous. Instead, every attention
that can read a dropped token gets a closed-form change in **column f of in_proj only**: 8 body blocks and the 2
classifier token-attention blocks, 8 heads each, head dim 16.
- All kept queries (particles and summary tokens) read the same constant n_f = s_k. Writing g/s_k into Wq[:,f] adds
  a constant query offset g to the head.
- Writing −λu/(s_d − s_k) into Wk[:,f] gives a dropped key the extra component −λu relative to a kept key.
- The logit change for a dropped key is −λ(q̃_i·u + g·u)/4. All kept keys get the same per-query constant, and
  softmax ignores that.
- Per head, u minimises Var(q̃·u) subject to mean(q̃·u)=1 (closed-form Lagrange solve on captured queries). The
  margin is τ = max(q_{1e-4}, 0.15·median). Then g = β·C_k^{-1}u/(uᵀC_k^{-1}u) with β = τ − q_{1e-4} (zero when the
  queries already have a margin), and λ = 4L/τ with L = 30.
- Every kept query therefore loses ≥30 logits (attention weight ×e^-30) on a dropped key. The only side effect on kept
  pairs is the "bad term" g·k̃_j: on the data it averages 0.005 logits (max 0.12).
- The classifier uses the same construction, with keys = x_body (channel f saturated by body.norm) and queries = summary
  tokens.

### 3. Keeping the rule's kept tokens as before
Channel f's content is sacrificed: every reader now sees a constant. Two closed-form repairs:
- **Level = mean.** Each DyT weight on f (norm1, norm2 of every block, and body.norm) is set to −a0. a0 is the channel
  mean, or the regression intercept, so a saturated kept token reads exactly the value the readers expect on average.
  This costs one entry per norm instead of rewriting columns. It also matters for the generator, which reads x_body.
- **Regression restoration (mid, large).** Least squares n_f ≈ a0 + βᵀn_{-f} on captured activations, folded into
  the readers as R' = R + R[:,f]βᵀ (rank 1 per reader).
  - Mid: only the MLP readers (fc1). The fit uses the exact norm2 outputs, recovered by inverting fc1 on captured
    mlp_pre, from clean plus P_reco jets.
  - Large (channel 70): also in_proj and the classifier keys and queries.
  - Attention-side regression on channel 51 gave nothing (0.191 → 0.191). The MLP-side regression gave
    0.191 → 0.107 → 0.099.

Tokens the rule keeps go through exactly the original weights, except for (a) the loss of channel f's information
beyond what the regression restores, and (b) the bad term. This is the whole collateral floor (e_clean). The rule
probes sit almost exactly on that floor.

### Known limits
A dropped particle still enters other particles' kNN local features: the local block has no pid, and a soft photon
is kinematically identical to a soft charged hadron there. This is the small excess on split2/P/NH probes.
Particles within ~1e-4 (relative) of a threshold can be partly flagged.

## Tensors, ranks, changed entries

### final_small

| tensor | shape | rank | changed |
|---|---|---|---|
| `body.token` | (1, 4, 128) | - | 4 |
| `body.embed.mlp.fc1.weight` | (256, 4) | 2 | 12 |
| `body.embed.mlp.fc1.bias` | (256,) | - | 3 |
| `body.embed.mlp.fc2.weight` | (128, 256) | 1 | 3 |
| `body.embed.mlp.fc2.bias` | (128,) | - | 1 |
| `body.time_embed.fc2.weight` | (128, 256) | 1 | 1 |
| `body.pid_embed.0.weight` | (9, 128) | 1 | 2 |
| `body.in_blocks.0.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.0.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.0.attn.in_proj_weight` | (384, 128) | 1 | 384 |
| `body.in_blocks.1.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.1.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.1.attn.in_proj_weight` | (384, 128) | 1 | 384 |
| `body.in_blocks.2.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.2.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.2.attn.in_proj_weight` | (384, 128) | 1 | 128 |
| `body.in_blocks.3.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.3.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.3.attn.in_proj_weight` | (384, 128) | 1 | 128 |
| `body.in_blocks.4.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.4.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.4.attn.in_proj_weight` | (384, 128) | 1 | 144 |
| `body.in_blocks.5.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.5.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.5.attn.in_proj_weight` | (384, 128) | 1 | 160 |
| `body.in_blocks.6.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.6.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.6.attn.in_proj_weight` | (384, 128) | 1 | 160 |
| `body.in_blocks.7.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.7.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.7.attn.in_proj_weight` | (384, 128) | 1 | 208 |
| `body.norm.weight` | (128,) | - | 1 |
| `classifier.in_blocks.0.attn.in_proj_weight` | (384, 128) | 1 | 256 |
| `classifier.in_blocks.1.attn.in_proj_weight` | (384, 128) | 1 | 256 |

### final_mid

| tensor | shape | rank | changed |
|---|---|---|---|
| `body.token` | (1, 4, 128) | - | 4 |
| `body.embed.mlp.fc1.weight` | (256, 4) | 2 | 12 |
| `body.embed.mlp.fc1.bias` | (256,) | - | 3 |
| `body.embed.mlp.fc2.weight` | (128, 256) | 1 | 3 |
| `body.embed.mlp.fc2.bias` | (128,) | - | 1 |
| `body.time_embed.fc2.weight` | (128, 256) | 1 | 1 |
| `body.pid_embed.0.weight` | (9, 128) | 1 | 2 |
| `body.in_blocks.0.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.0.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.0.attn.in_proj_weight` | (384, 128) | 1 | 384 |
| `body.in_blocks.0.mlp.fc1.weight` | (256, 128) | 1 | 32509 |
| `body.in_blocks.1.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.1.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.1.attn.in_proj_weight` | (384, 128) | 1 | 384 |
| `body.in_blocks.1.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.2.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.2.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.2.attn.in_proj_weight` | (384, 128) | 1 | 128 |
| `body.in_blocks.2.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.3.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.3.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.3.attn.in_proj_weight` | (384, 128) | 1 | 128 |
| `body.in_blocks.3.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.4.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.4.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.4.attn.in_proj_weight` | (384, 128) | 1 | 144 |
| `body.in_blocks.4.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.5.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.5.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.5.attn.in_proj_weight` | (384, 128) | 1 | 160 |
| `body.in_blocks.5.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.6.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.6.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.6.attn.in_proj_weight` | (384, 128) | 1 | 160 |
| `body.in_blocks.6.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.7.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.7.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.7.attn.in_proj_weight` | (384, 128) | 1 | 208 |
| `body.in_blocks.7.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.norm.weight` | (128,) | - | 1 |
| `classifier.in_blocks.0.attn.in_proj_weight` | (384, 128) | 1 | 256 |
| `classifier.in_blocks.1.attn.in_proj_weight` | (384, 128) | 1 | 256 |

### final_large

| tensor | shape | rank | changed |
|---|---|---|---|
| `body.token` | (1, 4, 128) | - | 4 |
| `body.embed.mlp.fc1.weight` | (256, 4) | 2 | 12 |
| `body.embed.mlp.fc1.bias` | (256,) | - | 3 |
| `body.embed.mlp.fc2.weight` | (128, 256) | 1 | 3 |
| `body.embed.mlp.fc2.bias` | (128,) | - | 1 |
| `body.time_embed.fc2.weight` | (128, 256) | 1 | 1 |
| `body.pid_embed.0.weight` | (9, 128) | 1 | 2 |
| `body.in_blocks.0.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.0.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.0.attn.in_proj_weight` | (384, 128) | 2 | 48883 |
| `body.in_blocks.0.mlp.fc1.weight` | (256, 128) | 1 | 31990 |
| `body.in_blocks.1.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.1.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.1.attn.in_proj_weight` | (384, 128) | 2 | 48912 |
| `body.in_blocks.1.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.2.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.2.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.2.attn.in_proj_weight` | (384, 128) | 2 | 48896 |
| `body.in_blocks.2.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.3.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.3.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.3.attn.in_proj_weight` | (384, 128) | 2 | 48894 |
| `body.in_blocks.3.mlp.fc1.weight` | (256, 128) | 1 | 32511 |
| `body.in_blocks.4.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.4.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.4.attn.in_proj_weight` | (384, 128) | 2 | 48911 |
| `body.in_blocks.4.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.5.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.5.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.5.attn.in_proj_weight` | (384, 128) | 2 | 48928 |
| `body.in_blocks.5.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.6.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.6.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.6.attn.in_proj_weight` | (384, 128) | 2 | 48912 |
| `body.in_blocks.6.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.in_blocks.7.norm1.weight` | (128,) | - | 1 |
| `body.in_blocks.7.norm2.weight` | (128,) | - | 1 |
| `body.in_blocks.7.attn.in_proj_weight` | (384, 128) | 2 | 48960 |
| `body.in_blocks.7.mlp.fc1.weight` | (256, 128) | 1 | 32512 |
| `body.norm.weight` | (128,) | - | 1 |
| `classifier.in_blocks.0.attn.in_proj_weight` | (384, 128) | 3 | 49020 |
| `classifier.in_blocks.1.attn.in_proj_weight` | (384, 128) | 3 | 49024 |

DoF (oracle): small 2251, mid 5315, large 14321.

## Predictions vs outcomes (all 43 passes)
| id | kind | weights | prediction | outcome |
|---|---|---|---|---|
| 001 | surgical | zero_pass.npz | E_rule 0.4, d_acc -0.006, d_nll 0.02, rho_D 1, rho_J 1, rho_P_reco 1, e_clean 0.15, gen_drift 0.15 | E_rule 2.896, d_acc -0.4616, d_nll 1.417, e_clean 0.5704, gen_drift 0.24, dof 6297 |
| 002 | capture | orig | acc 0.8 | see notes |
| 003 | capture | orig | acc 0.8 | see notes |
| 004 | capture | orig | acc 0.8 | see notes |
| 005 | capture | id0.npz | acc 0.2 | see notes |
| 006 | surgical | v2a.npz | E_rule 0.3, d_acc -0.008, d_nll 0.02, e_clean 0.08, rho_P_reco 1, gen_drift 0.15 | E_rule 0.2394, d_acc -0.004492, d_nll 0.01194, e_clean 0.04023, gen_drift 0.07071, dof 7833 |
| 007 | surgical | v2abl.npz | E_rule 1.05, d_acc -0.004, d_nll 0.01, e_clean 0.05, gen_drift 0.1 | E_rule 0.7318, d_acc -0.004297, d_nll 0.006979, e_clean 0.03129, gen_drift 0.05239, dof 7833 |
| 008 | surgical | v3a.npz | E_rule 0.17, d_acc -0.003, d_nll 0.008, e_clean 0.028, gen_drift 0.06 | E_rule 0.1968, d_acc -0.000293, d_nll 0.005344, e_clean 0.03164, gen_drift 0.2114, dof 7831 |
| 009 | surgical | v3b.npz | E_rule 0.09, d_acc -0.001, d_nll 0.003, e_clean 0.015, gen_drift 0.05 | E_rule 0.1372, d_acc -0.0006836, d_nll 0.005513, e_clean 0.02383, gen_drift 0.1871, dof 1.622e+04 |
| 010 | surgical | v3b0.npz | E_rule 0.75, d_acc -0.0005, d_nll 0.002, e_clean 0.01, gen_drift 0.15 | E_rule 0.6926, d_acc 0.0005859, d_nll 0.0006198, e_clean 0.009751, gen_drift 0.1191, dof 1.114e+04 |
| 011 | surgical | v3c.npz | E_rule 0.11, d_acc -0.0005, d_nll 0.004, e_clean 0.018, gen_drift 0.18 | E_rule 0.08313, d_acc 0.0008789, d_nll 0.001776, e_clean 0.01344, gen_drift 0.1787, dof 1.622e+04 |
| 012 | surgical | v4s.npz | E_rule 0.13, d_acc 0, d_nll 0.004, e_clean 0.02, gen_drift 0.09, dof 3200 | E_rule 0.1683, d_acc 9.766e-05, d_nll 0.002416, e_clean 0.02698, gen_drift 0.1933, dof 2106 |
| 013 | surgical | v4l.npz | E_rule 0.08, d_acc 0, d_nll 0.002, e_clean 0.012, gen_drift 0.08, dof 12000 | E_rule 0.0876, d_acc 0.001172, d_nll 0.001684, e_clean 0.01447, gen_drift 0.1654, dof 4.132e+04 |
| 014 | surgical | v4l0t.npz | E_rule 0.75, e_clean 0.01, gen_drift 0.06 | E_rule 0.6914, d_acc 0.0003906, d_nll 0.0005004, e_clean 0.01098, gen_drift 0.1141, dof 9234 |
| 015 | surgical | v4l51.npz | E_rule 0.12, d_acc -0.001, e_clean 0.02, gen_drift 0.07 | E_rule 0.1172, d_acc -0.0008789, d_nll 0.004188, e_clean 0.01764, gen_drift 0.06115, dof 1.369e+04 |
| 016 | surgical | v5l51_0.npz | E_rule 0.72, e_clean 0.012, gen_drift 0.05 | E_rule 0.8643, d_acc -0.001465, d_nll 0.003416, e_clean 0.017, gen_drift 0.02877, dof 9618 |
| 017 | surgical | v5l51.npz | E_rule 0.11, d_acc -0.0005, e_clean 0.016, gen_drift 0.06 | E_rule 0.1172, d_acc -0.0008789, d_nll 0.004188, e_clean 0.01764, gen_drift 0.06115, dof 1.369e+04 |
| 018 | surgical | v6l51.npz | E_rule 0.1, d_acc -0.0005, e_clean 0.015, gen_drift 0.06 | E_rule 0.1177, d_acc -0.001562, d_nll 0.004026, e_clean 0.01761, gen_drift 0.05962, dof 1.369e+04 |
| 019 | surgical | v6l70.npz | E_rule 0.075, d_acc 0, e_clean 0.012, gen_drift 0.16 | E_rule 0.08519, d_acc 0.0009766, d_nll 0.00126, e_clean 0.01394, gen_drift 0.1721, dof 1.47e+04 |
| 020 | capture | orig | acc 0.835 | see notes |
| 021 | surgical | v7l51.npz | E_rule 0.08, d_acc 0, e_clean 0.012, gen_drift 0.06 | E_rule 0.1297, d_acc -0.001758, d_nll 0.004877, e_clean 0.01949, gen_drift 0.06114, dof 2.212e+04 |
| 022 | surgical | v5l47.npz | E_rule 0.1, d_acc 0, e_clean 0.015, gen_drift 0.08 | E_rule 0.1759, d_acc 0.001367, d_nll 0.002057, e_clean 0.03068, gen_drift 0.1026, dof 1.42e+04 |
| 023 | surgical | v5l108.npz | E_rule 0.11, d_acc 0, e_clean 0.016, gen_drift 0.06 | E_rule 0.1551, d_acc 0.0009766, d_nll -0.002468, e_clean 0.01913, gen_drift 0.06376, dof 1.47e+04 |
| 024 | surgical | s51.npz | E_rule 0.2, d_acc -0.003, e_clean 0.03, gen_drift 0.06, dof 2100 | E_rule 0.1927, d_acc -0.003906, d_nll 0.008235, e_clean 0.0313, gen_drift 0.04716, dof 2634 |
| 025 | surgical | m51a.npz | E_rule 0.15, d_acc -0.002, e_clean 0.023, gen_drift 0.06, dof 7000 | E_rule 0.1913, d_acc -0.003906, d_nll 0.008144, e_clean 0.03102, gen_drift 0.05049, dof 1.062e+04 |
| 026 | surgical | m51b.npz | E_rule 0.16, d_acc -0.002, e_clean 0.025, gen_drift 0.06, dof 5200 | E_rule 0.1178, d_acc -0.001562, d_nll 0.004014, e_clean 0.01779, gen_drift 0.05941, dof 5698 |
| 027 | capture | orig | acc 0.835 | see notes |
| 028 | capture | orig | acc 0.835 | see notes |
| 029 | surgical | m51c.npz | E_rule 0.1, d_acc -0.001, e_clean 0.015, gen_drift 0.06, dof 5700 | E_rule 0.1072, d_acc -0.001367, d_nll 0.003232, e_clean 0.01482, gen_drift 0.06209, dof 5698 |
| 030 | surgical | m70c.npz | E_rule 0.075, d_acc 0, e_clean 0.012, gen_drift 0.17, dof 5700 | E_rule 0.1672, d_acc -0.0004883, d_nll 0.002616, e_clean 0.02736, gen_drift 0.1778, dof 5170 |
| 031 | surgical | fs51.npz | E_rule 0.19, d_acc -0.004, e_clean 0.031, gen_drift 0.05, dof 1900 | E_rule 0.1913, d_acc -0.003711, d_nll 0.008492, e_clean 0.03124, gen_drift 0.04713, dof 2251 |
| 032 | surgical | fm51.npz | E_rule 0.107, d_acc -0.0014, e_clean 0.0148, gen_drift 0.062, dof 5000 | E_rule 0.1072, d_acc -0.001367, d_nll 0.003232, e_clean 0.01482, gen_drift 0.06209, dof 5315 |
| 033 | surgical | fl51.npz | E_rule 0.104, d_acc -0.001, e_clean 0.0145, gen_drift 0.062, dof 13000 | E_rule 0.109, d_acc -0.001465, d_nll 0.003618, e_clean 0.01524, gen_drift 0.06461, dof 1.33e+04 |
| 034 | surgical | fl70.npz | E_rule 0.075, d_acc 0, e_clean 0.012, gen_drift 0.17, dof 13000 | E_rule 0.07741, d_acc 0.0004883, d_nll 0.0003306, e_clean 0.01222, gen_drift 0.1643, dof 1.432e+04 |
| 035 | capture | orig | acc 0.8 | see notes |
| 036 | capture | orig | acc 0.8 | see notes |
| 037 | surgical | fm51p.npz | E_rule 0.095, d_acc -0.0014, e_clean 0.015, e_P 0.017, gen_drift 0.062, dof 5315 | E_rule 0.09939, d_acc -0.00127, d_nll 0.003234, e_clean 0.01484, gen_drift 0.06133, dof 5315, e_P 0.0178 |
| 038 | surgical | fl70p.npz | E_rule 0.075, d_acc 0, e_clean 0.0122, gen_drift 0.165, dof 14321 | E_rule 0.07606, d_acc 0.0007813, d_nll 0.000356, e_clean 0.01214, gen_drift 0.1643, dof 1.432e+04 |
| 039 | logits | orig | e_nh_orig 0.25 | see notes |
| 040 | logits | fm51p.npz | e_nh_edit 0.016 | see notes |
| 041 | logits | fl70p.npz | e_nh_edit 0.013 | see notes |
| 042 | logits | fm51p.npz | e_clean_mine 0.015 | see notes |
| 043 | logits | fm51p.npz | e_ph_edit 0.02 | see notes |
Capture/logits outcomes:
- Clean-jet accuracy (predicted 0.80): 002–004 0.831; 020 0.831; 027/028 0.838.
- 005, block 0 disabled (predicted 0.2): 0.746.
- 035/036 P_reco (predicted 0.8): 0.778.
- NH probe (error vs orig(clean), my normalisation):
  - 039 original: predicted 0.25, got 0.203;
  - 040 mid: predicted 0.016, got 0.0293;
  - 041 large: predicted 0.013, got 0.0281.
- 042, floor on the same jets: predicted 0.015, got 0.0144.
- 043, photon control: predicted 0.02, got 0.0292.

My predictions were too optimistic on collateral error early on (zero pass) and on the NH probe. Later they were within
about 10–20 % on E_rule. gen_drift was underpredicted for channel 70.

## Ideas that failed or were dropped
- **Exact masking through channel 2** (the padding-mask channel): float cancellation is impossible. Kept and dropped
  photons share pid rows, and the local term is continuous. Absorption tricks with huge constants also fail.
- **Zero pass** (weights-only head directions, zero-ablation): E_rule 2.90, accuracy 0.37. The bad terms from guessed
  query statistics destroyed the model. Data-driven directions fixed it (0.239).
- **Channel 70 for the small/mean variant:** E_rule 0.168 at gen_drift 0.19. Channel 70 is near-constant at t=0, but
  the generator relies on it at t>0. Not saturating the time token did not help (0.114 without penalty). Setting the
  body.norm level to the mean did not help either.
- Channels 47 and 108 (E_rule 0.176, 0.155), both worse than 51.
- Summary-weighted regression (no change). Regression restoration of the classifier MLPs and classifier.fc (slightly
  worse: 0.130). Attention-side regression on channel 51 (no gain, so the "large" channel-51 variant, 0.109 @ 13.3k,
  is dominated by mid).
- Approximating the MLP input by the mean of block input and output. The exact fc1-inverted inputs were better
  (0.118 → 0.107).
- Kinematic kNN contamination: no pid-blind fix exists in the local block, so it was left alone.

## Passes used
43 of 50:
- 29 surgical;
- 9 captures: 002–005, 020, 027, 028, 035, 036;
- 5 logits: 039–043.

Scripts: `work/build2.py`, `build3.py`, `build4.py` (final builder), `cache*.py`, `send.py`. Final configs:
- small: `{"f":51,"lean":true,"exactn2":true,"L":30,"tfrac":0.15,"umode":"one","qlow":1e-4}`;
- mid: small + `"regmlp":true,"mixP":true`;
- large: `{"f":70, ..., "reg":true,"regmlp":true,"regcls":true,"mixP":true}`.
