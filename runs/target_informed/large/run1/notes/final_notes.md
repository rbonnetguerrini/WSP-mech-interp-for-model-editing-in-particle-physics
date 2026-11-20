# Final notes — implementing the reference in PET2-L by weight edits

## 1. Final edits

| file | DoF (oracle) | E_ref (practice) | gate (limits) |
|---|---|---|---|
| `final_small.npz` (XS) | 27,666 | 0.242 | pass (worst class drop 0.0195: close to the 0.020 limit) |
| `final_mid.npz` (S3) | 34,830 | 0.171 | pass |
| `final_large.npz` (S) | 41,994 | 0.147 | pass |

All three are built from the same circuit. They differ only in how many least-squares restoration stages they carry: XS has block 2; S3 has blocks 2 and 3; S has blocks 2, 3 and 4. Every changed tensor is `body.*` (nothing in `generator.*` or `classifier.*`), float32, with the original shape.

Tensors in `final_large` are listed below. `final_mid` has the same set without the block-4 rows; `final_small` additionally drops the block-3 rows. "changed" = changed entries; DoF = min(changed, r(m+n-r)), my own count; the oracle's totals are in the table above.

```
final_large 24 valid True
   body.embed.mlp.fc1.bias                          rank -   changed 2        dof 2
   body.embed.mlp.fc1.weight                        rank 2   changed 8        dof 8
   body.embed.mlp.fc2.bias                          rank -   changed 1        dof 1
   body.embed.mlp.fc2.weight                        rank 1   changed 4094     dof 3071
   body.embed.mlp.norm.weight                       rank -   changed 2        dof 2
   body.in_blocks.0.attn.in_proj_weight             rank 1   changed 3145728  dof 4095
   body.in_blocks.0.mlp.fc1.bias                    rank -   changed 1        dof 1
   body.in_blocks.0.mlp.fc1.weight                  rank 1   changed 1024     dof 1024
   body.in_blocks.0.mlp.fc2.weight                  rank 1   changed 1024     dof 1024
   body.in_blocks.0.mlp.norm.weight                 rank -   changed 1        dof 1
   body.in_blocks.1.attn.in_proj_weight             rank 1   changed 3145696  dof 4095
   body.in_blocks.1.attn.out_proj.weight            rank 1   changed 1024     dof 1024
   body.in_blocks.1.mlp.fc1.bias                    rank -   changed 1        dof 1
   body.in_blocks.1.mlp.fc1.weight                  rank 1   changed 1024     dof 1024
   body.in_blocks.1.mlp.fc2.weight                  rank 1   changed 1024     dof 1024
   body.in_blocks.1.mlp.norm.weight                 rank -   changed 1        dof 1
   body.in_blocks.2.attn.in_proj_weight             rank 1   changed 3145728  dof 4095
   body.in_blocks.2.mlp.fc2.bias                    rank -   changed 1024     dof 1024
   body.in_blocks.2.mlp.fc2.weight                  rank 2   changed 2096994  dof 6140
   body.in_blocks.3.mlp.fc2.bias                    rank -   changed 1024     dof 1024
   body.in_blocks.3.mlp.fc2.weight                  rank 2   changed 2096996  dof 6140
   body.in_blocks.4.mlp.fc2.bias                    rank -   changed 1024     dof 1024
   body.in_blocks.4.mlp.fc2.weight                  rank 2   changed 2096943  dof 6140
   body.pid_embed.0.weight                          rank 1   changed 9        dof 9
```

## 2. What the reference does, and how I found it
- **Pass 1** (baseline surgical): e_v(original) = 0 on clean, split2 and soft3e-3, and > 0 on soft1e-3, soft1e-4 and P. So the reference equals the original model on clean jets.
- **Passes 2–3** (reference and original logits on 1,950 constructed jets: clean jets plus one added particle at pT fraction 1e-5…1e-2, as photon, neutral hadron or charged): ref(x + particle) was exactly orig(x) whenever the particle was a soft photon or neutral hadron, and exactly orig(x + particle) otherwise. Charged particles are never removed. Removing a clean jet's softest particle is not done by the reference.
- **Pass 4** (reference on 1,950 jets with one random neutral, pT 0.27–1.5 GeV, random position): the decision is binary and absolute (it does not depend on jet pT):
  - **photon (pid 6) is removed iff pT < 0.80 GeV and E < 1.77 GeV** (data bounds: pT 0.796–0.805, E 1.735–1.805);
  - **neutral hadron (pid 7) is removed iff E < 2.19 GeV** (2.174–2.206).
- No clean practice jet contains such a particle, which is why the reference equals the original on clean jets. The reference therefore re-applies detector-style neutral thresholds: **f_ref(x) = f_orig(x with sub-threshold neutrals deleted)**.

## 3. The circuit
**Key lever.** `PET_body` derives its padding mask once, from channel 2 of the summed particle embedding (`x_embed + local + pid_embed + add_embed`). A particle whose channel 2 is exactly 0 is masked:
- it is removed from all 28 body attention layers (keys and queries);
- its body output row is zeroed;
- it is masked as a key in the classifier.

That is the same as deleting it, except for the kNN local features of its neighbours. The edit makes channel 2 exactly 0 for box neutrals and repairs the information this destroys for every other particle.

1. **Gates (embed MLP, two repurposed near-dead hidden units 1636 and 1970).** Unit 1636 is fed `A·(n_pT − t_pT)` and unit 1970 `A·(n_E − t_E)`, with A = 2·10⁴, on the DynamicTanh-normalised log pT and log E, at thresholds 0.80 GeV and 1.77 GeV. GELU underflows to exactly 0 below threshold, and tanh saturates to exactly 1 above it (norm weight set to 1). Embed `fc2` row 2 = 2³³·(u_pT + u_E) with bias −2³².
2. **Float32 absorption at the body-input sum.** `pid_embed[:,2] = +2³²` for every pid. The body computes `((emb + local) + pid) + add` in fp32:
   - for a box particle: (−2³² + loc2) = −2³² (loc2 is absorbed), then + 2³² = 0, then + add2 = 0 for neutrals. **Channel 2 is exactly 0, so the particle is masked.**
   - for any kept particle: channel 2 = 2³³ or 2³⁴ (one or both gates open). Soft charged tracks in the box get channel 2 = add2 ≠ 0 and are kept.
   - **Design limit:** the gate cannot see pid, so the photon box is applied to all neutrals. Neutral hadrons with E < 2.19 GeV outside the photon box are not removed (the remaining error on P).
3. **Bringing channel 2 to exactly 0 after the mask is fixed.** One repurposed MLP hidden unit in each of body blocks 0 and 1 acts as a detector: it reads n₂ and fires exactly when channel 2 is huge, and writes −2³³ (fp32 absorption makes this exact). Block 1's attention write to channel 2 is zeroed. After block 1, channel 2 = 0 for every kept particle; summary tokens are untouched.
4. **Read compensation (rank 1 on `in_proj` of blocks 0, 1, 2).** Each block's read of the wrong channel 2 is replaced by β·n(other channels). β is a ridge least-squares regression of the original normalised channel 2 on the other normalised channels from oracle captures (R² 0.998). q/k/v error drops from 0.82 / 0.50 / 0.06 to 0.007 / 0.004 / 0.007.
5. **Restoration (rank-2 updates of MLP `fc2` in blocks 2, 3 and 4).** These write the original channel-2 content (and the few perturbed channels) back. They are reduced-rank least squares from each block's captured MLP hidden activations to (original − edited) block output, weighted by the next block's DynamicTanh slope. They are fitted on practice jets 300–339 of each class, which are disjoint from the 205-per-class evaluation set.

**Why inputs without box neutrals are (almost) unaffected.**
- Every channel other than 2 at the body input is bit-identical to the original (verified by capture).
- The only change is the channel-2 value, and it is brought to 0 and then restored by closed-form fits.
- The local block, the interaction bias, the classifier and the generator weights are untouched.
- What remains is the restoration error: clean e_v 0.017 for S, versus 0.10–0.37 for the original's errors on the probes.

**Removal quality.** The probe response on soft photons collapses: r_soft1e-4 goes from 0.105 to 0.002 and r_soft1e-3 from 0.383 to 0.012. The leftover comes from kNN neighbours and restoration error.

## 4. Final practice metrics (surgical, 2,050 jets)
**final_small (XS)** (pass 048): E_ref 0.2423, DoF 27666; clean d_acc -0.0068, worst class drop 0.0195, d_nll 0.0105, KL 0.0107; rho D 1.009, J 1.024, P 0.721, P_ref 0.999; gen_drift 0.2692; R_vis 0.598, R_in 1.168, R_sub 0.028; gate_ref_no_g2 True

| probe | e_v edited | e_v original | ratio |
|---|---|---|---|
| soft1e-3 | 0.0345 | 0.3734 | 0.092 |
| soft1e-4 | 0.0338 | 0.1055 | 0.321 |
| P | 0.0474 | 0.1511 | 0.314 |
| clean | 0.0335 | 0.0000 | - |
| split2 | 0.0382 | 0.0000 | - |
| soft3e-3 | 0.0336 | 0.0000 | - |

**final_mid (S3)** (pass 049): E_ref 0.1706, DoF 34830; clean d_acc -0.0029, worst class drop 0.0098, d_nll 0.0050, KL 0.0068; rho D 1.005, J 1.012, P 0.722, P_ref 0.999; gen_drift 0.2692; R_vis 0.559, R_in 1.093, R_sub 0.025; gate_ref_no_g2 True

| probe | e_v edited | e_v original | ratio |
|---|---|---|---|
| soft1e-3 | 0.0222 | 0.3734 | 0.060 |
| soft1e-4 | 0.0212 | 0.1055 | 0.201 |
| P | 0.0379 | 0.1511 | 0.251 |
| clean | 0.0211 | 0.0000 | - |
| split2 | 0.0265 | 0.0000 | - |
| soft3e-3 | 0.0213 | 0.0000 | - |

**final_large (S)** (pass 047): E_ref 0.1466, DoF 41994; clean d_acc -0.0029, worst class drop 0.0146, d_nll 0.0034, KL 0.0047; rho D 1.006, J 1.010, P 0.721, P_ref 0.998; gen_drift 0.2692; R_vis 0.544, R_in 1.064, R_sub 0.024; gate_ref_no_g2 True

| probe | e_v edited | e_v original | ratio |
|---|---|---|---|
| soft1e-3 | 0.0184 | 0.3734 | 0.049 |
| soft1e-4 | 0.0174 | 0.1055 | 0.165 |
| P | 0.0341 | 0.1511 | 0.225 |
| clean | 0.0172 | 0.0000 | - |
| split2 | 0.0218 | 0.0000 | - |
| soft3e-3 | 0.0174 | 0.0000 | - |

`gen_drift` is ~0.27 for all variants. The generator shares the body: noised particles that fall in the box are masked, and channel 2 is restored only approximately. It is not a limit, but it is large. The generator AUC is not measurable on this model.

## 5. Predictions vs outcomes (all requests)
| pass | kind | weights | prediction | outcome |
|---|---|---|---|---|
| 001 | surgical | orig | E_ref=1.0, d_acc=0.0, gen_drift=0.0, rho_D=1.0, rho_J=1.0, rho_P_ref=1.0, e_clean_orig=0.05 | E_ref=1, d_acc=0, gen_drift=0, dof=0, gate=True |
| 002 | reference | - | ref_minus_origclean_rel_at_1e-4=0.05, ref_minus_origclean_rel_at_1e-2=0.5 | (activations/logits returned) |
| 003 | logits | orig | orig_soft1e-4_shift_rel=0.3 | (activations/logits returned) |
| 004 | reference | - | frac_dropped=0.45, linear_rule_accuracy_pt_E=0.95 | (activations/logits returned) |
| 005 | capture | orig | embed_ch2_std_over_median_channel_std=1.0 | (activations/logits returned) |
| 006 | surgical | edit1.npz | E_ref=0.35, e_soft1e-4_ratio=0.25, e_soft1e-3_ratio=0.3, e_P_ratio=0.5, d_acc=-0.005, gen_drift=0.05, rho_D=1.0, rho_P_ref=0.9 | E_ref=1.476, d_acc=-0.03512, gen_drift=0.02136, dof=12302, gate=False |
| 007 | capture | diagA.npz | loc2_std=0.3 | (activations/logits returned) |
| 008 | surgical | diagB.npz | d_acc=-0.01, kl=0.03, e_clean=0.08 | E_ref=1.463, e_clean=0.1922, d_acc=-0.02976, kl=0.1049, dof=8195, gate=False |
| 009 | surgical | diagC.npz | d_acc=-0.02, kl=0.06, e_clean=0.15 | E_ref=1.117, e_clean=0.04511, d_acc=-0.005854, kl=0.02044, dof=4107, gate=False |
| 010 | surgical | edit2.npz | E_ref=0.8, d_acc=-0.015, kl=0.05, e_clean=0.1, r_soft1e-4=0.03, gen_drift=0.03 | E_ref=1.376, e_clean=0.1448, d_acc=-0.01756, kl=0.06704, r_soft1e-4=0.1126, gen_drift=0.02098, dof=4109, gate=False |
| 011 | capture | edit2.npz | photon_c2_zero_frac=1.0 | (activations/logits returned) |
| 012 | surgical | edit3.npz | E_ref=0.9, r_soft1e-4=0.04, d_acc=-0.018, kl=0.065, e_clean=0.14 | E_ref=1.376, e_clean=0.1448, d_acc=-0.01756, kl=0.06704, r_soft1e-4=0.1126, dof=4109, gate=False |
| 013 | capture | diag3A.npz | photon_c2_in_abs_max=0.0 | (activations/logits returned) |
| 014 | capture | diag1A.npz | photon_c2_in_abs_max=0.0, photon_norm_row=0.0 | (activations/logits returned) |
| 015 | surgical | edit4.npz | E_ref=0.7, r_soft1e-4=0.03, d_acc=-0.012, kl=0.04, e_clean=0.1 | E_ref=0.6834, e_clean=0.1054, d_acc=-0.01073, kl=0.0422, r_soft1e-4=0.003022, dof=4109, gate=False |
| 016 | capture | diag4A.npz | n_tokens=16000 | (activations/logits returned) |
| 017 | capture | diagA.npz | c2_in_std=4.2 | (activations/logits returned) |
| 018 | surgical | edit5.npz | E_ref=0.35, e_clean=0.04, d_acc=-0.004, kl=0.012, r_soft1e-4=0.003, gen_drift=0.25 | E_ref=0.3645, e_clean=0.04496, d_acc=-0.005366, kl=0.01227, r_soft1e-4=0.002553, gen_drift=0.2695, dof=6157, gate=False |
| 019 | capture | edit5.npz | rel_err_block0_other_channels=0.02, rel_err_block0_ch2=0.2 | (activations/logits returned) |
| 020 | capture | orig | n=400 | (activations/logits returned) |
| 021 | surgical | edit6.npz | E_ref=0.15, e_clean=0.015, d_acc=-0.001, class_drop=0.01, kl=0.002, r_soft1e-4=0.003, gen_drift=0.25 | E_ref=0.3303, e_clean=0.0487, d_acc=-0.01024, class_drop=0.03415, kl=0.02141, r_soft1e-4=0.003125, gen_drift=0.2695, dof=14340, gate=False |
| 022 | capture | edit5.npz | rel_err_norm=0.05 | (activations/logits returned) |
| 023 | capture | orig | n=200 | (activations/logits returned) |
| 024 | capture | diag4Z.npz | n=400 | (activations/logits returned) |
| 025 | capture | diagZ.npz | n=400 | (activations/logits returned) |
| 026 | capture | diag7Z.npz | S2_mean_err_excl2=0.002, qkv_err=0.12 | (activations/logits returned) |
| 027 | capture | edit7b.npz | block0_rel_err=0.05 | (activations/logits returned) |
| 028 | surgical | edit8.npz | E_ref=0.15, e_clean=0.015, d_acc=-0.002, class_drop=0.012, d_nll=0.004, kl=0.003, r_soft1e-4=0.003, gen_drift=0.25 | E_ref=0.3075, e_clean=0.04446, d_acc=-0.008293, class_drop=0.02927, d_nll=0.01711, kl=0.01921, r_soft1e-4=0.003092, gen_drift=0.2699, dof=33742, gate=False |
| 029 | capture | edit8.npz | block1_mlp_pre_particle_err=0.03 | (activations/logits returned) |
| 030 | capture | edit8.npz | growth_per_block=1.25 | (activations/logits returned) |
| 031 | capture | orig | growth_per_block=1.25 | (activations/logits returned) |
| 032 | capture | edit8.npz | ch2_share_weighted_err=0.4 | (activations/logits returned) |
| 033 | surgical | edit9.npz | E_ref=0.25, e_clean=0.025, d_acc=-0.004, class_drop=0.018, d_nll=0.009, kl=0.008 | E_ref=0.2832, e_clean=0.04082, d_acc=-0.007805, class_drop=0.02927, d_nll=0.01316, kl=0.01634, dof=59278, gate=False |
| 034 | capture | B0Z.npz | other_channel_err=0.0, neutral_c2_values=[512, 1024] | (activations/logits returned) |
| 035 | capture | B1.npz | ch2_share_weighted_err=0.95 | (activations/logits returned) |
| 036 | capture | D1.npz | block0_ch2_hard=0.0, other_channels_rel_err_block0=0.01 | (activations/logits returned) |
| 037 | capture | D2.npz | block1_ch2_kept_particles=0.0 | (activations/logits returned) |
| 038 | capture | orig | n=400 | (activations/logits returned) |
| 039 | surgical | D3.npz | E_ref=0.2, e_clean=0.025, d_acc=-0.004, class_drop=0.018, d_nll=0.008, kl=0.008, r_soft1e-4=0.003 | E_ref=0.2421, e_clean=0.03346, d_acc=-0.006829, class_drop=0.01951, d_nll=0.01052, kl=0.01071, r_soft1e-4=0.002636, dof=46038, gate=True |
| 040 | surgical | D4.npz | E_ref=0.16, e_clean=0.018, d_acc=-0.003, class_drop=0.012, d_nll=0.004, kl=0.004, r_soft1e-4=0.003 | E_ref=0.2421, e_clean=0.03346, d_acc=-0.006829, class_drop=0.01951, d_nll=0.01052, kl=0.01071, r_soft1e-4=0.002636, dof=46038, gate=True |
| 041 | capture | D3.npz | block2_weighted_err_vs_orig=0.003 | (activations/logits returned) |
| 042 | capture | orig | n=400 | (activations/logits returned) |
| 043 | surgical | D5b.npz | E_ref=0.2, e_clean=0.026, d_acc=-0.005, class_drop=0.015, d_nll=0.007, kl=0.007 | E_ref=0.1707, e_clean=0.0211, d_acc=-0.002927, class_drop=0.01463, d_nll=0.0049, kl=0.006869, dof=59334, gate=True |
| 044 | capture | D5b.npz | weighted_err=0.0005 | (activations/logits returned) |
| 045 | capture | orig | n=400 | (activations/logits returned) |
| 046 | surgical | D6.npz | E_ref=0.16, e_clean=0.019, d_acc=-0.003, class_drop=0.013, d_nll=0.004, kl=0.006 | E_ref=0.1464, e_clean=0.01718, d_acc=-0.002927, class_drop=0.01463, d_nll=0.00326, kl=0.004615, dof=66498, gate=True |
| 047 | surgical | S.npz | E_ref=0.155, e_clean=0.018, d_acc=-0.003, class_drop=0.015, d_nll=0.004, dof=42000 | E_ref=0.1466, e_clean=0.01722, d_acc=-0.002927, class_drop=0.01463, d_nll=0.003397, dof=41994, gate=True |
| 048 | surgical | XS.npz | E_ref=0.245, e_clean=0.034, d_acc=-0.007, class_drop=0.02, d_nll=0.011, dof=27650 | E_ref=0.2423, e_clean=0.03345, d_acc=-0.006829, class_drop=0.01951, d_nll=0.01046, dof=27666, gate=True |
| 049 | surgical | S3.npz | E_ref=0.172, e_clean=0.021, d_acc=-0.003, class_drop=0.015, d_nll=0.005, dof=33800 | E_ref=0.1706, e_clean=0.02106, d_acc=-0.002927, class_drop=0.009756, d_nll=0.004973, dof=34830, gate=True |
My predictions were generally too optimistic about clean damage until design D. From then on (passes 39–49) they were within about ±0.03 on E_ref.

## 6. Ideas that failed
- **Zeroing the local block's channel-2 chain (edit1, diag B).** I first assumed the local block had 2 blocks, but it has 4. Zeroing those writes also starves the local block's internal reads, giving d_acc −0.03.
- **Absorption inside the local block (+M in local block-2 MLP, −M in block 3).** This works exactly (edit4: r_soft1e-4 = 0.003). But the saturated block-3 reads distort other local channels (375, 928, …), and every particle loses loc2. Even with block-0/1 repairs this stalled at E_ref 0.28 with worst class drop 0.029 (fail).
- **Raw-MSE low-rank repair at block 0 (edit6).** It fixed the huge, saturated channels 928/347, which do not matter downstream. A DynamicTanh-slope weighting was needed.
- **Design B** (gate values 512/1024 after absorption). The bimodal channel-2 value could not be restored by least squares (median residual 14).
- **Design C** (local-block soft-centre flag; drop all neutrals with pT < 0.8). It cannot express the E condition, and 6% of clean jets contain soft forward photons the reference keeps.
- **General limit.** An exact 2-D (pT AND E) box with fully preserved channel 2 is impossible with these additive stages, so channel 2 must be rebuilt for every particle.

## 7. Notes on method and compliance
- No gradients and no iterative optimisers. All fits are closed-form ridge least squares, SVD rank truncation, or hand-set constants.
- When fitting on oracle-captured activations, I applied single weight matrices and the elementwise DynamicTanh (tanh(α·x)·w) of the next norm.
- The kNN-leakage estimate was computed from raw input coordinates only.
- I never ran the network myself.

## 8. Passes used
49 of 50. The zero-pass deliverables were written before any request.
