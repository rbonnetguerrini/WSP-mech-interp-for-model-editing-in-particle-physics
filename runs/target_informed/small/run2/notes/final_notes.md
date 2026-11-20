# Final notes — weight-space implementation of the reference

## 1. What the reference does (and how I found it)

**f_ref(x) = f_orig(x with all sub-threshold neutral particles removed)**, where a constituent is removed iff

* photon (pid 6) with **pT < 0.8 GeV and E < 1.8 GeV**, or
* neutral hadron (pid 7) with **E < 2.2 GeV**.

The rule is per particle and absolute (it doesn't depend on the jet pT fraction, the position (tested up to dR 0.8), the slot order or the impact-parameter columns; it depends on pid). Clean practice jets contain no such particles (they are the detector acceptance edge of the clean data), so the reference equals the original exactly on clean jets.

How it was found (passes 1–4, 14, 35):
* pass 1 (null surgical): e_orig = 0 on clean and soft3e-3; for soft1e-4 e_orig equals the response r exactly → the reference equals orig(clean) there.
* passes 2+3 (reference + original logits on 16 constructed variants × 128 jets): soft photon / neutral hadron at 1e-4…1e-3 → ref = orig(clean jet) exactly; charged at 1e-3, photon at 3e-3, shuffled order, sorted insertion → ref = orig(x). The removed vs kept photons overlapped in pT and in E, but not in the (pT, E) plane.
* pass 4 (grid scan of one added particle over (log pT, log E), pid 6/7/charged, with and without IP, dR 0–0.8): sharp, jet-independent boundaries. pid decides, not the IP columns.
* pass 14 (fine scan + P): thresholds pinned to pT 0.8 / E 1.8 (photon) and E 2.2 (NH). On 896 P jets, ref(P) == ref(P minus my flagged set) == orig(P minus flagged) **for every jet**, so the rule is exact.
* pass 35: the rule also holds for neutral hadrons at 3e-4/1e-3, two soft photons, collinear soft photons, soft charged particles (kept) and P with exponents 1.05 and 1.2.

## 2. The circuit (all closed-form)

Notation: A = 200, L = 500, K = 1e4, K0 = 1000, T = 5. The flag channel is residual channel **c = 70**. This channel already sits at a large negative value (near-saturated in the DynamicTanh norms) in every block, so fixing it to a constant costs little. Its exact block-0 input (diagnostic capture, pass 22) has std 0.24.

**(a) Kinematic steps in the embedding MLP** (`body.embed.mlp`). Three hidden units that were dead in the trained model (163, 155, 133; |fc1| ~1e-4, norm weight ~1e-6) become steep step functions of the normalised inputs t_k = tanh(α_in x_k)·w_in,k:
  h_j = tanh(α·GELU(K(T_j − t_k)))·wn_j ≈ wn_j · [x_k < θ_j], with θ = ln 0.8 on log pT, and ln 1.8 / ln 2.2 on log E.
  `fc2[70, j] = (A, A, 3A)/wn_j` writes these steps into channel 70 only (the other fc2 entries of these units are multiplied by wn_j ≈ 1e-6 and are negligible).

**(b) AND with the pid** through constants: `fc2.bias[70] −= 10A`, `pid_embed[6,70] += 5.5A`, `pid_embed[7,70] += 8A`. Then
  x_70 = orig + A(s_pT + s_E1.8) + 3A s_E2.2 + 5.5A[pid=6] + 8A[pid=7] − 10A,
which is ≥ +0.5A exactly for (photon ∧ pT<0.8 ∧ E<1.8) ∨ (NH ∧ E<2.2), and ≤ −0.5A otherwise. Every real particle is saturated in channel 70: −1 for normal particles, +1 for flagged ones.

**(c) Block 0 sees the original mean.** `in_blocks.0.norm1.weight[70] = −mean(n_70 over particles)` (exact block-0 input from pass 22), so every normal particle presents the original mean in channel 70. The summary tokens are untouched by (a)/(b). Their `body.token[:,70]` is re-solved, tanh(α·x*)·w' = original value, so they read exactly what they read before. (The time token is exactly zero at t = 0 and therefore masked.)

**(d) Exact flag reader in block 0** (`in_blocks.0.mlp`). Dead hidden unit 52 gets `fc1[52,70] = K0/w2_70` and `fc1.bias[52] = −0.9K0`, so it fires only for flagged tokens. `fc2[70,52] = 2L/wn_52` and `fc2.bias[70] −= L`. After block 0 *every* real token (particles and summary tokens) has x_70 ≤ −L, and flagged tokens have x_70 ≥ +L. So in blocks 1–7 and at `body.norm`, channel 70 equals −w'_b for all normal tokens and +w'_b for flagged ones, and the norm weights w'_b are set to minus the original (particle) mean.

**(e) Key gating** in body blocks 1–7 and both classifier token-attention blocks:
  `in_proj_weight[128+32h : 160+32h, 70] += −G_bh d_bh`.
Here d_bh is the Fisher direction Σ⁻¹μ of the edited model's queries for head h (from captured activations), and G_bh = T√32 / (2 w'_b p_bh), with p the 0.05-percentile of q·d (every query has q·d > 0).
  * Normal keys all shift by the same vector +w'G d, so each query's softmax is unchanged (it is invariant to a key-independent score offset).
  * The flagged key's score drops by 2w'G(q·d)/√32 ≥ T relative to every other key, so the flagged particle is removed from all attention after block 0, including the summary-token readout in the classifier.

**(f) Compensation (mid/large only).** Channel 70's lost variation is predicted linearly from the other channels at each norm: n_70 ≈ β0 + β·n_rest (least squares on captured clean activations). The readers get the rank-1 update `M += outer(M[:,70], β)` (in_proj for norm1, mlp.fc1 for norm2) and `weight[70] = −β0`. Mid: block 1 only. Large: blocks 1–7.

**Why inputs where the reference equals the original are (almost) unaffected.** For a jet without sub-threshold neutrals nothing is flagged. The only change is that channel 70 becomes a constant equal to its original mean, and that channel was already nearly constant (block-0 std 0.08 after tanh, ≤ 0.4 later; partly restored by (f)). The gating adds a key-independent offset, which is exactly invariant. Measured clean change e_clean = 0.016 / 0.012 / 0.011 (small/mid/large, in units of S), d_acc ≈ 0.

**What is not removed (the remaining leak, R_sub ≈ 0.055).** (i) Block-0 attention still sees the flagged token: gating there is impossible without moving the summary tokens' keys (shown in pass 33). (ii) The kNN/local-feature module uses raw coordinates and no pid, so a soft neutral still changes its neighbours' local features and displaces a neighbour from their kNN set. No weight edit can undo this exactly.

## 3. Final edits: tensors, ranks, changed entries, DoF
(From the oracle's size report. A 1-D tensor counts its changed entries. A 2-D tensor counts min(changed, r(m+n−r)).)

#### final_small.npz (pass 044) — total DoF 1197

| tensor | rank | changed entries |
|---|---|---|
| `body.token` | None | 4 |
| `body.embed.mlp.fc1.weight` | 2 | 12 |
| `body.embed.mlp.fc1.bias` | None | 3 |
| `body.embed.mlp.fc2.weight` | 1 | 3 |
| `body.embed.mlp.fc2.bias` | None | 1 |
| `body.pid_embed.0.weight` | 1 | 2 |
| `body.in_blocks.0.norm1.weight` | None | 1 |
| `body.in_blocks.0.mlp.fc1.weight` | 1 | 1 |
| `body.in_blocks.0.mlp.fc1.bias` | None | 1 |
| `body.in_blocks.0.mlp.fc2.weight` | 1 | 1 |
| `body.in_blocks.0.mlp.fc2.bias` | None | 1 |
| `body.in_blocks.1.norm1.weight` | None | 1 |
| `body.in_blocks.1.norm2.weight` | None | 1 |
| `body.in_blocks.1.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.2.norm1.weight` | None | 1 |
| `body.in_blocks.2.norm2.weight` | None | 1 |
| `body.in_blocks.2.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.3.norm1.weight` | None | 1 |
| `body.in_blocks.3.norm2.weight` | None | 1 |
| `body.in_blocks.3.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.4.norm1.weight` | None | 1 |
| `body.in_blocks.4.norm2.weight` | None | 1 |
| `body.in_blocks.4.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.5.norm1.weight` | None | 1 |
| `body.in_blocks.5.norm2.weight` | None | 1 |
| `body.in_blocks.5.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.6.norm1.weight` | None | 1 |
| `body.in_blocks.6.norm2.weight` | None | 1 |
| `body.in_blocks.6.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.7.norm1.weight` | None | 1 |
| `body.in_blocks.7.norm2.weight` | None | 1 |
| `body.in_blocks.7.attn.in_proj_weight` | 1 | 128 |
| `body.norm.weight` | None | 1 |
| `classifier.in_blocks.0.attn.in_proj_weight` | 1 | 128 |
| `classifier.in_blocks.1.attn.in_proj_weight` | 1 | 128 |

#### final_mid.npz (pass 045) — total DoF 2472

| tensor | rank | changed entries |
|---|---|---|
| `body.token` | None | 4 |
| `body.embed.mlp.fc1.weight` | 2 | 12 |
| `body.embed.mlp.fc1.bias` | None | 3 |
| `body.embed.mlp.fc2.weight` | 1 | 3 |
| `body.embed.mlp.fc2.bias` | None | 1 |
| `body.pid_embed.0.weight` | 1 | 2 |
| `body.in_blocks.0.norm1.weight` | None | 1 |
| `body.in_blocks.0.mlp.fc1.weight` | 1 | 1 |
| `body.in_blocks.0.mlp.fc1.bias` | None | 1 |
| `body.in_blocks.0.mlp.fc2.weight` | 1 | 1 |
| `body.in_blocks.0.mlp.fc2.bias` | None | 1 |
| `body.in_blocks.1.norm1.weight` | None | 1 |
| `body.in_blocks.1.norm2.weight` | None | 1 |
| `body.in_blocks.1.attn.in_proj_weight` | 2 | 48896 |
| `body.in_blocks.1.mlp.fc1.weight` | 1 | 32512 |
| `body.in_blocks.2.norm1.weight` | None | 1 |
| `body.in_blocks.2.norm2.weight` | None | 1 |
| `body.in_blocks.2.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.3.norm1.weight` | None | 1 |
| `body.in_blocks.3.norm2.weight` | None | 1 |
| `body.in_blocks.3.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.4.norm1.weight` | None | 1 |
| `body.in_blocks.4.norm2.weight` | None | 1 |
| `body.in_blocks.4.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.5.norm1.weight` | None | 1 |
| `body.in_blocks.5.norm2.weight` | None | 1 |
| `body.in_blocks.5.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.6.norm1.weight` | None | 1 |
| `body.in_blocks.6.norm2.weight` | None | 1 |
| `body.in_blocks.6.attn.in_proj_weight` | 1 | 128 |
| `body.in_blocks.7.norm1.weight` | None | 1 |
| `body.in_blocks.7.norm2.weight` | None | 1 |
| `body.in_blocks.7.attn.in_proj_weight` | 1 | 128 |
| `body.norm.weight` | None | 1 |
| `classifier.in_blocks.0.attn.in_proj_weight` | 1 | 128 |
| `classifier.in_blocks.1.attn.in_proj_weight` | 1 | 128 |

#### final_large.npz (pass 046) — total DoF 10122

| tensor | rank | changed entries |
|---|---|---|
| `body.token` | None | 4 |
| `body.embed.mlp.fc1.weight` | 2 | 12 |
| `body.embed.mlp.fc1.bias` | None | 3 |
| `body.embed.mlp.fc2.weight` | 1 | 3 |
| `body.embed.mlp.fc2.bias` | None | 1 |
| `body.pid_embed.0.weight` | 1 | 2 |
| `body.in_blocks.0.norm1.weight` | None | 1 |
| `body.in_blocks.0.mlp.fc1.weight` | 1 | 1 |
| `body.in_blocks.0.mlp.fc1.bias` | None | 1 |
| `body.in_blocks.0.mlp.fc2.weight` | 1 | 1 |
| `body.in_blocks.0.mlp.fc2.bias` | None | 1 |
| `body.in_blocks.1.norm1.weight` | None | 1 |
| `body.in_blocks.1.norm2.weight` | None | 1 |
| `body.in_blocks.1.attn.in_proj_weight` | 2 | 48896 |
| `body.in_blocks.1.mlp.fc1.weight` | 1 | 32512 |
| `body.in_blocks.2.norm1.weight` | None | 1 |
| `body.in_blocks.2.norm2.weight` | None | 1 |
| `body.in_blocks.2.attn.in_proj_weight` | 2 | 48896 |
| `body.in_blocks.2.mlp.fc1.weight` | 1 | 32512 |
| `body.in_blocks.3.norm1.weight` | None | 1 |
| `body.in_blocks.3.norm2.weight` | None | 1 |
| `body.in_blocks.3.attn.in_proj_weight` | 2 | 48894 |
| `body.in_blocks.3.mlp.fc1.weight` | 1 | 32512 |
| `body.in_blocks.4.norm1.weight` | None | 1 |
| `body.in_blocks.4.norm2.weight` | None | 1 |
| `body.in_blocks.4.attn.in_proj_weight` | 2 | 48891 |
| `body.in_blocks.4.mlp.fc1.weight` | 1 | 32512 |
| `body.in_blocks.5.norm1.weight` | None | 1 |
| `body.in_blocks.5.norm2.weight` | None | 1 |
| `body.in_blocks.5.attn.in_proj_weight` | 2 | 48896 |
| `body.in_blocks.5.mlp.fc1.weight` | 1 | 32512 |
| `body.in_blocks.6.norm1.weight` | None | 1 |
| `body.in_blocks.6.norm2.weight` | None | 1 |
| `body.in_blocks.6.attn.in_proj_weight` | 2 | 48896 |
| `body.in_blocks.6.mlp.fc1.weight` | 1 | 32512 |
| `body.in_blocks.7.norm1.weight` | None | 1 |
| `body.in_blocks.7.norm2.weight` | None | 1 |
| `body.in_blocks.7.attn.in_proj_weight` | 2 | 48896 |
| `body.in_blocks.7.mlp.fc1.weight` | 1 | 32512 |
| `body.norm.weight` | None | 1 |
| `classifier.in_blocks.0.attn.in_proj_weight` | 1 | 128 |
| `classifier.in_blocks.1.attn.in_proj_weight` | 1 | 128 |
## 4. Final practice metrics (surgical passes 44 / 45 / 46)

| metric | final_small (044) | final_mid (045) | final_large (046) | original |
|---|---|---|---|---|
| DoF | 1197 | 2472 | 10122 | 0 |
| **E_ref** | **0.139** | **0.1205** | **0.1162** | 1 |
| e/e_orig soft1e-3 | 0.080 | 0.065 | 0.066 | 1 |
| e/e_orig soft1e-4 | 0.070 | 0.058 | 0.057 | 1 |
| e/e_orig P | 0.267 | 0.238 | 0.225 | 1 |
| e_clean (units of S) | 0.0157 | 0.0118 | 0.0109 | 0 |
| d_acc | +0.0010 | −0.0003 | +0.0004 | 0 |
| worst class drop (≤0.020) | 0.0098 | 0.0039 | 0.0039 | 0 |
| d_nll (≤0.030) | 0.0011 | 0.0012 | 0.0012 | 0 |
| rho D / J / P_ref (e^±0.15) | 0.997 / 0.996 / 1.017 | 0.997 / 0.994 / 0.998 | 1.008 / 1.000 / 1.000 | 1 |
| rho P (not a limit) | 0.742 | 0.729 | 0.732 | 1 |
| gate_ref_no_g2 | True | True | True | True |
| gen_drift t=0.2/0.5/0.85 | 0.033/0.070/0.132 | 0.032/0.071/0.133 | 0.032/0.071/0.130 | 0 |
| R_vis / R_in / R_sub | 0.563/1.070/0.055 | 0.556/1.057/0.055 | 0.558/1.059/0.056 | 1 |

Validation on probe types I hadn't seen (passes 35–39, mid-type edit at T = 5): error ratios are 0.12–0.17 for a soft neutral hadron, a photon at 3e-4 and two soft photons, 0.33 for a collinear soft photon, and 0.25–0.37 for P with exponents 1.05/1.2. A soft charged particle (which the reference keeps) gets only the clean-level error. Hidden probes are therefore likely to score somewhat worse than the visible ones.

**Generator caveat.** The real limit (real-vs-generated AUC) can't be measured with the oracle. The proxy gen_drift is 0.03 / 0.07 / 0.13 at t = 0.2 / 0.5 / 0.85. It mostly comes from fixing channel 70 to a constant (0.138 even with gating switched off, pass 17). The rest comes from the gate correctly firing on noised neutrals that fall below threshold. This is the main residual risk of all three submissions.

## 5. Ideas that failed
* **e1 (pass 8):** gating in all blocks including block 0, using a query-shift trick and saturating the summary tokens' channel 70. Soft-probe removal was good, but the clean change was huge (d_acc −0.036, gen_drift 0.39).
* **Black-hole representation** for flagged tokens (saturated sign-pattern keys using the existing W_k): offline margins were too weak, so it was never submitted.
* **Block-0 compensation fitted on a reconstructed block-0 input** (x0+attn0 from mlp_pre). Block-0 attention writes a lot into channel 70, so the reconstruction was badly wrong (std 4.75 vs a true 0.24) and the "compensation" added spurious terms (e5, e7). Fixed by a diagnostic capture with block-0 out_proj zeroed and near-linear norm2 (pass 22), which gave the exact x0. This halved the clean change.
* **Sparse gating directions** (4–32 coordinates per head, e6M/e6S): more leak.
* **Channel 108 as flag channel** (lower block-0 cost): its block-0 intercept is tiny, so the summary-token targets become unreachable. Not submitted.
* **Block-0 key gating** (pass 33): it moves the summary-token keys; class drop 0.11.
* **Strong gating** T = 15/25 leaks *more* than T = 5–8 (passes 26–30, 40–41; the effect is in the body blocks). T = 3 under-suppresses. I adopted the empirical optimum T = 5, which also holds on the validation probes. I believe the small residual attention to the flagged token partly cancels the kNN/local leak, so part of this gain may not carry over to every probe type.
* **Weighting summary tokens in the channel-70 norm weights** (wsum 0.8): class drop 0.0215, fails. Particle-mean weights (wsum 0) were best.
* **Block-0 β-compensation on exact data** (e9L): smaller clean change but more leak.

## 6. Passes used
46 of 50 (0 rejected). Requests 001–046 are in `requests/`. Scripts and analyses are in `work/` (builder chain build3 → build4 → build5 → build6 → build9 → build11 → build14; the finals come from build15 (small) and build16 (mid/large)).

## 7. Predictions vs outcomes

| pass | kind | weights/input | prediction | outcome |
|---|---|---|---|---|
| 001 | surgical | null | {"E_ref": 1.0, "d_acc": 0.0, "e_clean_orig": 0.0, "rho_P_ref": 1.0} | {"E_ref": 1.0, "d_acc": 0.0, "class_drop": 0.0, "d_nll": 0.0, "e_clean": 0.0, "rho_P_ref": 1.0, "gen_max": 0.0, "R_sub": 1.0, "dof": 0, "gate": true} |
| 002 | reference | v1.npz | {"ref_eq_orig_clean": 1.0, "ref_eq_orig_shuffled": 1.0} | {"kind": "reference", "see": "work/ analysis"} |
| 003 | logits | v1.npz | {"shuffled_equals_clean": 1.0} | {"kind": "logits", "see": "work/ analysis"} |
| 004 | reference | v2.npz | {"photon_threshold_logpt": -0.2, "photon_threshold_logE": 0.6, "nh_threshold_logE": 0.8, "charged_dropped_frac": 0.0} | {"kind": "reference", "see": "work/ analysis"} |
| 005 | capture | clean | {"time_token_zero_frac": 1.0} | {"kind": "capture", "see": "work/ analysis"} |
| 006 | capture | clean | {"time_token_zero_frac": 1.0} | {"kind": "capture", "see": "work/ analysis"} |
| 007 | capture | clean | {"time_token_zero_frac": 1.0} | {"kind": "capture", "see": "work/ analysis"} |
| 008 | surgical | e1.npz | {"E_ref": 0.4, "e_soft1e-4_ratio": 0.3, "e_soft1e-3_ratio": 0.35, "e_P_ratio": 0.5, "d_acc": -0.003, "d_nll": 0.01, "rho_D": 1.0, "rho_P_ref": 1.0, "gen_drift_max": 0.1} | {"E_ref": 0.812, "d_acc": -0.0359, "class_drop": 0.166, "d_nll": 0.0832, "e_clean": 0.2133, "rho_P_ref": 0.419, "gen_max": 0.393, "R_sub": 0.072, "dof": 1450, "gate": false} |
| 009 | capture | e1.npz | {"rel_change_block0": 0.1, "rel_change_body_norm": 0.2} | {"kind": "capture", "see": "work/ analysis"} |
| 010 | surgical | e3.npz | {"E_ref": 0.35, "d_acc": -0.008, "d_nll": 0.02, "e_clean": 0.06, "rho_P_ref": 0.95, "gen_drift_max": 0.15} | {"E_ref": 0.263, "d_acc": -0.0021, "class_drop": 0.0205, "d_nll": 0.0026, "e_clean": 0.041, "rho_P_ref": 0.875, "gen_max": 0.164, "R_sub": 0.106, "dof": 1193, "gate": false} |
| 011 | capture | e3.npz | {"rel_change_block0_particles": 0.03, "rel_change_body_norm_summary": 0.08} | {"kind": "capture", "see": "work/ analysis"} |
| 012 | surgical | e4.npz | {"E_ref": 0.2, "d_acc": -0.001, "class_drop": 0.012, "e_clean": 0.025, "rho_P_ref": 0.92, "gen_drift_max": 0.14} | {"E_ref": 0.231, "d_acc": -0.0006, "class_drop": 0.0059, "d_nll": 0.0032, "e_clean": 0.0314, "rho_P_ref": 0.863, "gen_max": 0.163, "R_sub": 0.103, "dof": 10118, "gate": true} |
| 013 | surgical | e5.npz | {"E_ref": 0.21, "class_drop": 0.006, "e_clean": 0.027, "rho_P_ref": 0.9, "rho_J": 0.97, "gen_drift_max": 0.15} | {"E_ref": 0.204, "d_acc": -0.0005, "class_drop": 0.0029, "d_nll": 0.0018, "e_clean": 0.0184, "rho_P_ref": 0.949, "gen_max": 0.176, "R_sub": 0.103, "dof": 10633, "gate": true} |
| 014 | reference | v3.npz | {"frac_ref_P_equal_P_minus_flagged": 1.0, "nh_threshold_E": 2.2} | {"kind": "reference", "see": "work/ analysis"} |
| 015 | logits | v3.npz | {"dummy": 0} | {"kind": "logits", "see": "work/ analysis"} |
| 016 | logits | e5.npz | {"leak_over_total": 0.6} | {"kind": "logits", "see": "work/ analysis"} |
| 017 | surgical | e5ng.npz | {"E_ref": 0.95, "gen_drift_max": 0.06, "e_clean": 0.018} | {"E_ref": 0.736, "d_acc": -0.0005, "class_drop": 0.0029, "d_nll": 0.0018, "e_clean": 0.0184, "rho_P_ref": 0.949, "gen_max": 0.138, "R_sub": 0.584, "dof": 6814, "gate": true} |
| 018 | surgical | e6M.npz | {"E_ref": 0.23, "class_drop": 0.01, "rho_P_ref": 0.93, "e_clean": 0.028, "dof": 1000} | {"E_ref": 0.401, "d_acc": -0.0002, "class_drop": 0.0195, "d_nll": 0.0011, "e_clean": 0.0277, "rho_P_ref": 0.963, "gen_max": 0.205, "R_sub": 0.246, "dof": 921, "gate": true} |
| 019 | surgical | e6S.npz | {"E_ref": 0.26, "class_drop": 0.018, "rho_P_ref": 0.88, "e_clean": 0.038, "dof": 480} | {"E_ref": 0.424, "d_acc": -0.0021, "class_drop": 0.0205, "d_nll": 0.0027, "e_clean": 0.0418, "rho_P_ref": 0.873, "gen_max": 0.21, "R_sub": 0.252, "dof": 410, "gate": false} |
| 020 | surgical | e7S.npz | {"E_ref": 0.23, "class_drop": 0.009, "rho_P_ref": 0.95, "e_clean": 0.022, "gen_drift_max": 0.18} | {"E_ref": 0.211, "d_acc": -0.0002, "class_drop": 0.0039, "d_nll": 0.0014, "e_clean": 0.0191, "rho_P_ref": 0.953, "gen_max": 0.163, "R_sub": 0.104, "dof": 2983, "gate": true} |
| 021 | surgical | e7M.npz | {"E_ref": 0.215, "class_drop": 0.006, "rho_P_ref": 0.95, "e_clean": 0.02, "gen_drift_max": 0.18} | {"E_ref": 0.206, "d_acc": -0.0002, "class_drop": 0.0029, "d_nll": 0.002, "e_clean": 0.0186, "rho_P_ref": 0.947, "gen_max": 0.178, "R_sub": 0.103, "dof": 6808, "gate": true} |
| 022 | capture | cap_x0.npz | {"x0_ch70_particle_std": 4.5} | {"kind": "capture", "see": "work/ analysis"} |
| 023 | surgical | e9L.npz | {"E_ref": 0.18, "e_clean": 0.012, "class_drop": 0.003, "rho_P_ref": 0.97, "gen_drift_max": 0.16} | {"E_ref": 0.177, "d_acc": -0.0005, "class_drop": 0.0059, "d_nll": 0.0008, "e_clean": 0.0085, "rho_P_ref": 1.001, "gen_max": 0.162, "R_sub": 0.109, "dof": 10633, "gate": true} |
| 024 | surgical | e9S.npz | {"E_ref": 0.2, "e_clean": 0.016, "class_drop": 0.004, "rho_P_ref": 0.96, "dof": 2470} | {"E_ref": 0.162, "d_acc": 0.0001, "class_drop": 0.002, "d_nll": 0.0007, "e_clean": 0.0119, "rho_P_ref": 1.004, "gen_max": 0.157, "R_sub": 0.091, "dof": 2472, "gate": true} |
| 025 | surgical | e10L.npz | {"E_ref": 0.155, "e_clean": 0.009, "class_drop": 0.003, "rho_P_ref": 1.0, "R_sub": 0.091} | {"E_ref": 0.152, "d_acc": 0.0001, "class_drop": 0.0059, "d_nll": 0.001, "e_clean": 0.0108, "rho_P_ref": 0.999, "gen_max": 0.162, "R_sub": 0.09, "dof": 10122, "gate": true} |
| 026 | surgical | e10S25.npz | {"E_ref": 0.161, "R_sub": 0.09, "gen_drift_max": 0.16} | {"E_ref": 0.22, "d_acc": 0.0001, "class_drop": 0.002, "d_nll": 0.0007, "e_clean": 0.0119, "rho_P_ref": 1.004, "gen_max": 0.164, "R_sub": 0.146, "dof": 2472, "gate": true} |
| 027 | surgical | e11S15.npz | {"E_ref": 0.15, "R_sub": 0.08, "e_clean": 0.012, "class_drop": 0.002} | {"E_ref": 0.164, "d_acc": 0.0001, "class_drop": 0.002, "d_nll": 0.0007, "e_clean": 0.0119, "rho_P_ref": 1.004, "gen_max": 0.16, "R_sub": 0.093, "dof": 2472, "gate": true} |
| 028 | surgical | e11S25.npz | {"E_ref": 0.15, "R_sub": 0.08} | {"E_ref": 0.222, "d_acc": 0.0001, "class_drop": 0.002, "d_nll": 0.0007, "e_clean": 0.0119, "rho_P_ref": 1.004, "gen_max": 0.167, "R_sub": 0.148, "dof": 2472, "gate": true} |
| 029 | surgical | e11S8.npz | {"E_ref": 0.15, "R_sub": 0.07} | {"E_ref": 0.125, "d_acc": 0.0001, "class_drop": 0.002, "d_nll": 0.0007, "e_clean": 0.0119, "rho_P_ref": 1.004, "gen_max": 0.146, "R_sub": 0.056, "dof": 2472, "gate": true} |
| 030 | surgical | e11S5.npz | {"E_ref": 0.17, "R_sub": 0.09} | {"E_ref": 0.124, "d_acc": 0.0001, "class_drop": 0.002, "d_nll": 0.0007, "e_clean": 0.0119, "rho_P_ref": 1.004, "gen_max": 0.135, "R_sub": 0.055, "dof": 2472, "gate": true} |
| 031 | surgical | e12L.npz | {"E_ref": 0.118, "e_clean": 0.0108, "class_drop": 0.004, "R_sub": 0.055, "gen_drift_max": 0.14} | {"E_ref": 0.116, "d_acc": 0.0001, "class_drop": 0.0059, "d_nll": 0.001, "e_clean": 0.0108, "rho_P_ref": 0.999, "gen_max": 0.137, "R_sub": 0.056, "dof": 10122, "gate": true} |
| 032 | surgical | e12XS.npz | {"E_ref": 0.15, "e_clean": 0.02, "class_drop": 0.015, "rho_P_ref": 0.99, "dof": 1200} | {"E_ref": 0.161, "d_acc": -0.0008, "class_drop": 0.0156, "d_nll": 0.0, "e_clean": 0.0199, "rho_P_ref": 1.02, "gen_max": 0.139, "R_sub": 0.056, "dof": 1197, "gate": true} |
| 033 | surgical | e13b0.npz | {"R_sub": 0.035, "e_clean": 0.02, "E_ref": 0.13} | {"E_ref": 0.525, "d_acc": -0.0136, "class_drop": 0.1133, "d_nll": 0.0316, "e_clean": 0.0993, "rho_P_ref": 0.976, "gen_max": 0.128, "R_sub": 0.088, "dof": 2600, "gate": false} |
| 034 | surgical | e13T3.npz | {"E_ref": 0.13, "R_sub": 0.065} | {"E_ref": 0.157, "d_acc": 0.0001, "class_drop": 0.002, "d_nll": 0.0007, "e_clean": 0.0119, "rho_P_ref": 1.004, "gen_max": 0.127, "R_sub": 0.08, "dof": 2472, "gate": true} |
| 035 | reference | v5.npz | {"ref_eq_orig_ch1e-4": 1.0, "ref_removes_nh1e-3_frac": 0.95} | {"kind": "reference", "see": "work/ analysis"} |
| 036 | logits | v5.npz | {"dummy": 0} | {"kind": "logits", "see": "work/ analysis"} |
| 037 | logits | e11S5.npz | {"ratio_soft_neutral": 0.08, "ratio_P": 0.25} | {"kind": "logits", "see": "work/ analysis"} |
| 038 | logits | e11S15.npz | {"ratio_g3e-4": 0.12} | {"kind": "logits", "see": "work/ analysis"} |
| 039 | logits | e11S8.npz | {"ratio_g3e-4": 0.12} | {"kind": "logits", "see": "work/ analysis"} |
| 040 | surgical | e14b5c15.npz | {"E_ref": 0.13, "R_sub": 0.06} | {"E_ref": 0.124, "d_acc": 0.0001, "class_drop": 0.002, "d_nll": 0.0007, "e_clean": 0.0119, "rho_P_ref": 1.004, "gen_max": 0.135, "R_sub": 0.055, "dof": 2472, "gate": true} |
| 041 | surgical | e14b15c5.npz | {"E_ref": 0.155, "R_sub": 0.085} | {"E_ref": 0.163, "d_acc": 0.0001, "class_drop": 0.002, "d_nll": 0.0007, "e_clean": 0.0119, "rho_P_ref": 1.004, "gen_max": 0.16, "R_sub": 0.093, "dof": 2472, "gate": true} |
| 042 | surgical | e15XS8.npz | {"E_ref": 0.155, "class_drop": 0.012, "e_clean": 0.019} | {"E_ref": 0.19, "d_acc": -0.001, "class_drop": 0.0215, "d_nll": 0.0002, "e_clean": 0.0256, "rho_P_ref": 1.021, "gen_max": 0.143, "R_sub": 0.058, "dof": 1197, "gate": false} |
| 043 | surgical | e15XS2.npz | {"E_ref": 0.15, "class_drop": 0.012, "e_clean": 0.017} | {"E_ref": 0.143, "d_acc": 0.0, "class_drop": 0.0127, "d_nll": 0.0005, "e_clean": 0.0163, "rho_P_ref": 1.018, "gen_max": 0.135, "R_sub": 0.056, "dof": 1197, "gate": true} |
| 044 | surgical | e15XS0.npz | {"E_ref": 0.138, "class_drop": 0.011, "e_clean": 0.015} | {"E_ref": 0.139, "d_acc": 0.001, "class_drop": 0.0098, "d_nll": 0.0011, "e_clean": 0.0157, "rho_P_ref": 1.017, "gen_max": 0.132, "R_sub": 0.055, "dof": 1197, "gate": true} |
| 045 | surgical | e16S.npz | {"E_ref": 0.115, "class_drop": 0.003, "e_clean": 0.01} | {"E_ref": 0.121, "d_acc": -0.0003, "class_drop": 0.0039, "d_nll": 0.0012, "e_clean": 0.0118, "rho_P_ref": 0.998, "gen_max": 0.133, "R_sub": 0.055, "dof": 2472, "gate": true} |
| 046 | surgical | e16L.npz | {"E_ref": 0.11, "class_drop": 0.005, "e_clean": 0.009} | {"E_ref": 0.116, "d_acc": 0.0004, "class_drop": 0.0039, "d_nll": 0.0012, "e_clean": 0.0109, "rho_P_ref": 1.0, "gen_max": 0.13, "R_sub": 0.056, "dof": 10122, "gate": true} |
**Prediction accuracy summary.**
* Too pessimistic early: e3 predicted 0.35 → got 0.263. e9S predicted 0.20 → 0.1615. T = 8 predicted 0.15 → 0.125.
* Systematically wrong about gating strength: I predicted that stronger suppression (T = 25) would not hurt; E_ref rose to 0.22.
* Underestimated the clean-side damage of saturating the summary tokens (e1) and of block-0 gating (e13b0).
* Overestimated the class drop of the XS variants with particle-mean norm weights.
