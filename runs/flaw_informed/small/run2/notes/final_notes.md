# Final notes: removing the soft-neutral flaw from pretrain_s by weight editing

## 1. Final edits

Each tensor's DoF = min(changed entries, r(m+n−r)); a 1-D tensor counts its changed entries.

Shared part (all three edits), DoF 1421:

| tensor | rank | changed entries | DoF | role |
|---|---|---|---|---|
| body.in_blocks.0.mlp.fc1.weight | 4 | 768 (rows 52,122,8,162,185,247) | 768 | detector + 5 veto units |
| body.in_blocks.0.mlp.fc1.bias | – | 6 | 6 | thresholds |
| body.in_blocks.0.mlp.norm.weight | – | 6 | 6 | dead units switched on (0 → 1) |
| body.in_blocks.0.mlp.fc2.weight | 1 | 768 | 383 | mark / anti-mark along one direction p |
| body.in_blocks.1.mlp.fc1.weight | 1 | 128 (row 17) | 128 | confirm unit |
| body.in_blocks.1.mlp.fc1.bias | – | 1 | 1 | |
| body.in_blocks.1.mlp.norm.weight | – | 1 | 1 | dead unit switched on |
| body.in_blocks.1.mlp.fc2.weight | 1 | 128 (col 17) | 128 | writes the saturating flag F·s |

Key gates: each is a rank-1 change to the K rows of an `attn.in_proj_weight` (384×128), DoF 511 each.

| final | gated attention tensors | DoF |
|---|---|---|
| `final_large.npz` | body.in_blocks.{2,3,4,5,6,7}, classifier.in_blocks.{0,1} (8 gates) | **5509** |
| `final_mid.npz` | body.in_blocks.{2,3,5}, classifier.in_blocks.{0,1} (5 gates) | **3976** |
| `final_small.npz` | body.in_blocks.{2,5}, classifier.in_blocks.{0,1} (4 gates) | **3465** |

No `generator.*` tensor is changed.

## 2. What the flaw is and where it lives

- **Size of the effect.** Adding one photon at 1e-4 or 1e-3 of the jet pT moves the centred JetClass logits by e ≈ 0.26 S. Clean accuracy drops from 84.4% to 79.0% on the soft1e-4 jets (passes 1–2).
- **No per-particle down-weighting.** The added token enters the jet output only as a key/value in the 8 body and 2 classifier attention layers, plus marginally through kNN local features. Attention is not pT-weighted, so a 0.05 GeV photon is read like any other particle.
- **Measured spread.** The summary tokens move 7–10% of the jet-to-jet spread already after block 0, 15% after block 5, and about 100% at the body output. The soft token itself is not special: soft1e-3 photons lie inside the natural neutral distribution and still cause the same e ≈ 0.26. So the flaw is a multiplicity/IRC response: every attention layer counts the extra token.
- **The token is identifiable per token.** In the block-0 MLP input the log relative pT (log z) of a neutral is linearly decodable with error 0.08. Neutral vs charged separates with a huge margin, because add_info is exactly 0 for neutrals and the pid embedding differs. Natural neutrals almost never have z < 1e-3 (0.24% of them).
- **Free capacity.** The block-0 and block-1 MLPs have dead hidden units (norm weight ≈ 0): 8 in block 0 and 3 in block 1. These can be reused without touching anything else.

## 3. The circuit

Notation: n = DynamicTanh output; X0, X1 = fc1 outputs (mlp_pre) of blocks 0 and 1.

1. **Stage A, detector (block 0, unit 52).** pre = β(θ − f(X0)) with β = 1000 and θ = log(1.5e-3).
   - f = (decoded log z) + 20·g, where g is a neutral-vs-other discriminant: ≈0 on neutrals, ≥1 on charged and summary tokens.
   - The unit's output tanh(α·GELU(pre)) is a sharp 0/1 step at z = 1.5e-3 for neutrals only.
   - It writes a small **mark** m·p: p is the lowest-variance residual direction (σ = 0.008) and m = 100σ = 0.82.
   - β = 1000 makes GELU's negative lobe cover only 0.005 in log z (see failures).
2. **Vetoes (block 0, units 122, 8, 162, 185, 247).** Sharp units fire on *unphysical* tokens: decoded log_pt − log_E > 0.45, |Δη| > 1, |Δφ| > 1. Each writes an **anti-mark** −1.3·m·p.
   - None of them fires on any clean or soft-probe token (max pre ≤ −4).
   - They fire on about 45% of generator-noised tokens (t ≥ 0.5). There they cancel the mark of noised "candidates", so the edit stays inert in the generator pass.
3. **Stage B, confirm and flag (block 1, unit 17).** pre = 50·(mark readout − 0.5) − 4·φ.
   - mark readout: LDA of X1, normalised to 0 for unmarked and 1 for marked tokens. Measured min 0.83 on candidates, max 0.41 on others.
   - φ: a time-token signal (see 5.). At t>0 the time token is unmasked, while at t = 0 it is exactly zero and masked.
   - The unit writes a **flag** F·s (F = 40) along a sign pattern s ∈ {±1}^128. The flag drives every later DynamicTanh of that token to the corner w⊙s of the tanh cube, a fixed point far from all normal tokens: Mahalanobis distance 50–350 in every layer.
4. **Key gates (rank 1 per layer).** ΔW_K = −(G_h b_h) vᵀ / Δ, where:
   - v is the LDA direction separating the corner from normal key inputs (std 1 on normal tokens);
   - Δ = v·(corner − mean);
   - b_h is the unit mean-query direction of head h;
   - G_h is chosen so that 82% of queries put the flagged key ≥ 3.5 logits below their best normal key (capped at 18).

   Result: the flagged token gets ~e^-3.5…e^-20 relative attention, i.e. it is removed from the softmax. This is done in body layers ≥ 2 (large: all of them) and in both classifier token-attention layers.

   Formula: logit(i→j) += −G_h (q_i·b_h)(v·n_j − c)/(4Δ). This is ≈ −G_h(q_i·b_h)/4 for flagged j and ≈ 0 (a per-query constant plus small noise) for normal j.

**Why jets without very soft neutrals are unaffected.**
- Stage A is exactly 0 for tokens with z > 1.5e-3·e^0.005 and for all charged, summary and time tokens. Their pre-activations are ≤ −30, so GELU underflows. Only the tiny fc2 contributions remain: the dead units' original columns times 0.
- Stage B is ≤ −4 for every unmarked clean token.
- The remaining effect on clean jets is (a) the small noise the rank-1 key gates add to normal keys, and (b) removal of natural neutrals with z < 1.5e-3, which is the intended behaviour: about 2.8% of natural neutrals, in about 30% of jets.

## 4. Final practice metrics (passes 47–49)

| | large (5509) | mid (3976) | small (3465) |
|---|---|---|---|
| ratio soft1e-4 | 0.084 | 0.149 | 0.177 |
| ratio soft1e-3 | 0.064 | 0.130 | 0.165 |
| ratio soft3e-3 / split2 (reported) | 1.113 / 1.077 | 1.078 / 1.073 | 1.151 / 1.108 |
| R_vis / R_in / R_sub | 0.585 / 1.095 / 0.074 | 0.608 / 1.076 / 0.139 | 0.650 / 1.129 / 0.171 |
| d_acc | −0.0015 | −0.0010 | −0.0021 |
| worst class drop | 0.0088 | 0.0137 | 0.0176 |
| d_nll | 0.0048 | 0.0057 | 0.0086 |
| KL | 0.0044 | 0.0057 | 0.0091 |
| rho D / J / P_ref | 1.056 / 1.052 / 0.941 | 1.028 / 1.042 / 0.956 | 1.029 / 1.044 / 0.960 |
| rho P (not a limit) | 0.734 | 0.785 | 0.811 |
| dir_err D / J / P | 0.152 / 0.168 / 0.368 | 0.136 / 0.174 / 0.334 | 0.175 / 0.197 / 0.340 |
| gen_drift t=0.2 / 0.5 / 0.85 | 0.035 / 0.055 / 0.053 | 0.034 / 0.057 / 0.056 | 0.032 / 0.057 / 0.056 |
| practice limits (gate_ref_no_g2) | pass | pass | pass |

Caveats:
- The generator limit (AUC rise ≤ 0.03) could not be measured. gen_drift ≈ 0.055 is close to that of the bare rank-1 gates without any detector (0.06, pass 14).
- final_small's worst-class drop (0.0176) is close to the 0.02 limit and may not pass on the hidden jets.
- The visible probes all add photons. Hidden probes with neutral hadrons should be covered, since the detector was fitted on natural pid-7 tokens too.
- The threshold z = 1.5e-3 is a guess at what "very soft" means: above 1e-3 and below 3e-3.

## 5. How it was found and built
- **Passes 1–10:** captures of the residual stream (clean, soft1e-4, soft1e-3, P) and of the block-0 MLP. These located the effect (every layer, beginning at block 0), the dead units, and the linear decodability of log z and neutral-ness.
- **Block-0 input:** captured once by editing block 0 into an identity (pass 16).
- **Noised-input behaviour:** captured on my own generator-style noised jets (pass 28).
- **Time-token signal:** estimated from the weights. MPFourier(t) → time_embed gives the time token at t>0; combined with block-0 attention on the captured block-0 inputs, ignoring the interaction bias.
- **All constructions are closed-form:** least squares, LDA (C⁻¹Δμ), eigenvectors, quantiles. The flag pattern s is the sign of the top eigenvector of Σ_L diag(w_L) C_L⁻¹ diag(w_L), chosen from 2 closed-form candidates.

**Disclosures:**
- I evaluated DynamicTanh (tanh(αx)·w), softmax attention logits, and the bias-free time-embedding MLP elementwise on captured activations and on weights. I treated these as single-layer closed-form computations on captured activations, but they go a bit further than "single weight matrices".
- My first few numpy scripts ran with `python3` from the shell PATH. That is a different interpreter from the prescribed one: a venv whose files live outside the two allowed folders. I switched to the prescribed interpreter as soon as I noticed. No gradients, optimisers or model forward passes were run locally.

## 6. Ideas that failed
- **Pure linear LDA key gate in every layer, no detector** (zero-pass plan): a linear score cannot implement a threshold. Noise on normal keys was 0.5–1 logits.
- **Strong gates (|gap| ≥ 15–20 for 98% of queries; passes 11–12):** clean accuracy collapsed (−45%, −32%).
- **β = 12 detector:** GELU's negative lobe wrote a −0.13·F flag onto every token just above threshold (~7% of neutrals). This was the real cause of the collapse; fixed with β = 1000 (pass 15).
- **Flag written directly in block 0 (pass 15–18):** the generator drift was ~1.0, because noised generator tokens were flagged. Hence the two-stage mark→confirm design.
- **Linear time veto in stage B alone:** fine for t ≤ 0.5, but drift 0.34 at t = 0.85 (pass 25, 27).
- **One-sided time-veto unit writing an anti-mark (pass 29):** it fired on most noised tokens, so drift rose to 0.4. Making it AND candidate didn't help (pass 30); the weight-based time-signal estimate was too optimistic.
- **"Self-gating" flag pattern** (sign pattern chosen to make existing keys negative; pass 37): worse class drop.
- **Extra layer-1 gate reading the mark (pass 35):** too noisy (class drop 0.03, drift 0.29).
- **Gating only the classifier, or only layer 4 (passes 40, 42):** class drop 0.06, because ungated corner tokens disturb the body.
- **Gating only layer 2 (pass 41):** class drop 0.032.

## 7. Passes used
49 of 50: 21 captures, 28 surgical evaluations. Zero pass: empty edit.

## 8. Predictions vs outcomes
| pass | kind | weights | prediction | outcome |
|---|---|---|---|---|
| 001 | capture | orig | {"mean_logit_shift_soft1e-4_over_S": 0.5} | capture clean body.block0,body.block2,body.block5,body.norm |
| 002 | capture | orig | {"mean_logit_shift_soft1e-4_over_S": 0.5} | capture soft1e-4 body.block0,body.block2,body.block5,body.norm |
| 003 | capture | orig | {"none": 0} | capture clean body.block1,body.block3,body.block4,body.block6 |
| 004 | capture | orig | {"none": 0} | capture soft1e-4 body.block1,body.block3,body.block4,body.block6 |
| 005 | capture | orig | {"e_soft1e-3_vs_clean": 0.15} | capture soft1e-3 body.block0,body.block2,body.block5,body.norm |
| 006 | capture | orig | {"e_soft1e-3_vs_clean": 0.15} | capture soft1e-3 body.block1,body.block3,body.block4,body.block6 |
| 007 | capture | orig | {"none": 0} | capture clean embed,body.block0.mlp_pre,body.block0.mlp_post,classifier.block0 |
| 008 | capture | orig | {"none": 0} | capture soft1e-4 embed,body.block0.mlp_pre,body.block0.mlp_post,classifier.block0 |
| 009 | capture | orig | {"none": 0} | capture soft1e-3 embed,body.block0.mlp_pre,body.block0.mlp_post,classifier.block0 |
| 010 | capture | orig | {"none": 0} | capture P embed,body.block0.mlp_pre,body.block0.mlp_post,classifier.block0 |
| 011 | surgical | e1.npz | {"ratio_soft1e-4": 0.3, "ratio_soft1e-3": 0.35, "d_acc": -0.003, "d_nll": 0.01, "rho_D": 1.0, "rho_J": 1.0, "rho_P_ref": 0.9, "gen_drift": 0.02, "DoF": 4857} | ratio s1e-4 1.185, s1e-3 1.195; d_acc -0.4496; class_drop 0.6143; d_nll 3.6067; rho J 5.22 D 1.49 P_ref 2.48; gen_drift 1.005; pass=False; DoF 4857 |
| 012 | surgical | e5.npz | {"ratio_soft1e-4": 0.4, "ratio_soft1e-3": 0.45, "d_acc": -0.01, "d_nll": 0.03, "gen_drift": 0.2} | ratio s1e-4 0.530, s1e-3 0.531; d_acc -0.3175; class_drop 0.5254; d_nll 1.0167; rho J 2.71 D 0.94 P_ref 1.65; gen_drift 1.009; pass=False; DoF 4857 |
| 013 | capture | e5.npz | {"frac_tokens_unit52_gt_0.5": 0.012} | capture clean body.block0.mlp_pre,body.block0.mlp_post,body.block0,body.block1 |
| 014 | surgical | e5_gatesonly.npz | {"d_acc": -0.25, "gen_drift": 0.8, "ratio_soft1e-4": 1.0} | ratio s1e-4 1.056, s1e-3 1.080; d_acc -0.0031; class_drop 0.0391; d_nll 0.0080; rho J 0.99 D 1.11 P_ref 1.06; gen_drift 0.065; pass=False; DoF 4599 |
| 015 | surgical | e6.npz | {"ratio_soft1e-4": 0.4, "ratio_soft1e-3": 0.45, "d_acc": -0.006, "d_nll": 0.015, "gen_drift": 0.1} | ratio s1e-4 0.122, s1e-3 0.117; d_acc -0.0094; class_drop 0.0459; d_nll 0.0234; rho J 1.19 D 1.15 P_ref 0.98; gen_drift 1.067; pass=False; DoF 4857 |
| 016 | capture | probe_b0id.npz | {"none": 0} | capture clean body.block0 |
| 017 | capture | probe_b0id.npz | {"none": 0} | capture soft1e-4 body.block0 |
| 018 | surgical | e7.npz | {"ratio_soft1e-4": 0.2, "ratio_soft1e-3": 0.2, "d_acc": -0.006, "class_drop": 0.03, "rho_J": 1.12, "gen_drift": 1.0} | ratio s1e-4 0.074, s1e-3 0.070; d_acc -0.0038; class_drop 0.0195; d_nll 0.0114; rho J 1.13 D 1.09 P_ref 0.95; gen_drift 1.061; pass=True; DoF 4857 |
| 019 | capture | stageA.npz | {"mark_readout_separation_sigma": 20} | capture clean body.block1.mlp_pre,body.block1 |
| 020 | capture | stageA.npz | {"mark_readout_separation_sigma": 20} | capture soft1e-4 body.block1.mlp_pre,body.block1 |
| 021 | capture | stageA.npz | {"mark_readout_separation_sigma": 20} | capture soft1e-3 body.block1.mlp_pre,body.block1 |
| 022 | capture | stageA.npz | {"mark_neg_std": 0.05, "mark_pos_std": 0.05} | capture clean body.block1.mlp_pre |
| 023 | capture | stageA.npz | {"mark_neg_std": 0.05, "mark_pos_std": 0.05} | capture soft1e-4 body.block1.mlp_pre |
| 024 | capture | stageA.npz | {"mark_neg_std": 0.05, "mark_pos_std": 0.05} | capture soft1e-3 body.block1.mlp_pre |
| 025 | surgical | e8.npz | {"ratio_soft1e-4": 0.12, "ratio_soft1e-3": 0.12, "d_acc": -0.004, "class_drop": 0.02, "gen_drift_max": 0.3} | ratio s1e-4 0.090, s1e-3 0.058; d_acc -0.0032; class_drop 0.0234; d_nll 0.0103; rho J 1.03 D 1.07 P_ref 0.93; gen_drift 0.322; pass=False; DoF 4604 |
| 026 | surgical | stageA.npz | {"gen_drift_t0.85": 0.1, "d_acc": 0.0, "ratio_soft1e-4": 0.98} | ratio s1e-4 1.011, s1e-3 0.986; d_acc 0.0002; class_drop 0.0010; d_nll -0.0000; rho J 1.00 D 1.00 P_ref 0.99; gen_drift 0.010; pass=True; DoF 258 |
| 027 | surgical | e9.npz | {"ratio_soft1e-4": 0.12, "ratio_soft1e-3": 0.09, "class_drop": 0.018, "d_acc": -0.003, "gen_drift_max": 0.3} | ratio s1e-4 0.082, s1e-3 0.057; d_acc -0.0024; class_drop 0.0176; d_nll 0.0055; rho J 1.04 D 1.06 P_ref 0.93; gen_drift 0.344; pass=True; DoF 4604 |
| 028 | capture | stageA.npz | {"frac_tokens_candidate_t0.85": 0.2} | capture noised.npz body.block0.mlp_pre,body.block1.mlp_pre |
| 029 | surgical | e10_1.0.npz | {"ratio_soft1e-4": 0.08, "ratio_soft1e-3": 0.06, "class_drop": 0.018, "gen_drift_t0.85": 0.1, "gen_drift_t0.2": 0.03} | ratio s1e-4 0.051, s1e-3 0.047; d_acc -0.0016; class_drop 0.0166; d_nll 0.0038; rho J 1.01 D 1.06 P_ref 0.93; gen_drift 0.460; pass=True; DoF 4862 |
| 030 | surgical | e11.npz | {"ratio_soft1e-4": 0.06, "ratio_soft1e-3": 0.05, "class_drop": 0.017, "gen_drift_t0.2": 0.03, "gen_drift_t0.85": 0.15} | ratio s1e-4 0.053, s1e-3 0.047; d_acc -0.0017; class_drop 0.0166; d_nll 0.0040; rho J 1.02 D 1.05 P_ref 0.93; gen_drift 0.440; pass=True; DoF 4862 |
| 031 | surgical | e12.npz | {"ratio_soft1e-4": 0.08, "ratio_soft1e-3": 0.057, "class_drop": 0.0176, "gen_drift_t0.85": 0.2, "gen_drift_t0.5": 0.025} | ratio s1e-4 0.082, s1e-3 0.059; d_acc -0.0024; class_drop 0.0117; d_nll 0.0066; rho J 1.07 D 1.06 P_ref 0.94; gen_drift 0.077; pass=True; DoF 5894 |
| 032 | surgical | e13.npz | {"ratio_soft1e-4": 0.08, "ratio_soft1e-3": 0.059, "class_drop": 0.012, "gen_drift_max": 0.06} | ratio s1e-4 0.082, s1e-3 0.058; d_acc -0.0021; class_drop 0.0176; d_nll 0.0060; rho J 1.06 D 1.06 P_ref 0.93; gen_drift 0.055; pass=True; DoF 5894 |
| 033 | surgical | e13s.npz | {"ratio_soft1e-4": 0.2, "ratio_soft1e-3": 0.18, "class_drop": 0.012, "gen_drift_max": 0.05} | ratio s1e-4 0.177, s1e-3 0.165; d_acc -0.0013; class_drop 0.0176; d_nll 0.0102; rho J 1.05 D 1.02 P_ref 0.93; gen_drift 0.058; pass=True; DoF 4361 |
| 034 | capture | e13.npz | {"mark_mahalanobis_layer1": 30} | capture mix.npz body.block0,body.block0.mlp_post |
| 035 | surgical | e14.npz | {"ratio_soft1e-4": 0.05, "ratio_soft1e-3": 0.04, "class_drop": 0.019, "gen_drift_max": 0.06} | ratio s1e-4 0.091, s1e-3 0.065; d_acc -0.0038; class_drop 0.0303; d_nll 0.0155; rho J 1.18 D 1.08 P_ref 0.94; gen_drift 0.294; pass=False; DoF 6405 |
| 036 | surgical | e15.npz | {"ratio_soft1e-4": 0.08, "ratio_soft1e-3": 0.06, "class_drop": 0.012, "gen_drift_max": 0.05} | ratio s1e-4 0.097, s1e-3 0.080; d_acc -0.0005; class_drop 0.0068; d_nll 0.0040; rho J 1.05 D 1.05 P_ref 0.95; gen_drift 0.055; pass=True; DoF 5894 |
| 037 | surgical | e16.npz | {"ratio_soft1e-4": 0.07, "ratio_soft1e-3": 0.055, "class_drop": 0.012, "gen_drift_max": 0.05} | ratio s1e-4 0.117, s1e-3 0.066; d_acc -0.0048; class_drop 0.0244; d_nll 0.0207; rho J 1.02 D 1.08 P_ref 0.93; gen_drift 0.089; pass=False; DoF 5894 |
| 038 | surgical | e15s.npz | {"ratio_soft1e-4": 0.18, "ratio_soft1e-3": 0.17, "class_drop": 0.01} | ratio s1e-4 0.189, s1e-3 0.185; d_acc -0.0019; class_drop 0.0186; d_nll 0.0106; rho J 1.04 D 1.02 P_ref 0.94; gen_drift 0.057; pass=True; DoF 4361 |
| 039 | surgical | e15m2.npz | {"ratio_soft1e-4": 0.25, "ratio_soft1e-3": 0.24, "class_drop": 0.01} | ratio s1e-4 0.183, s1e-3 0.175; d_acc -0.0021; class_drop 0.0146; d_nll 0.0096; rho J 1.04 D 1.03 P_ref 0.96; gen_drift 0.057; pass=True; DoF 3850 |
| 040 | surgical | e15c.npz | {"ratio_soft1e-4": 0.6, "ratio_soft1e-3": 0.6, "class_drop": 0.015} | ratio s1e-4 0.495, s1e-3 0.517; d_acc -0.0147; class_drop 0.0586; d_nll 0.0409; rho J 1.05 D 0.96 P_ref 0.95; gen_drift 0.056; pass=False; DoF 2828 |
| 041 | surgical | e15L2.npz | {"ratio_soft1e-4": 0.3, "ratio_soft1e-3": 0.3, "class_drop": 0.015} | ratio s1e-4 0.208, s1e-3 0.204; d_acc -0.0032; class_drop 0.0322; d_nll 0.0135; rho J 1.05 D 1.02 P_ref 0.96; gen_drift 0.058; pass=False; DoF 3339 |
| 042 | surgical | e15L4.npz | {"ratio_soft1e-4": 0.3, "ratio_soft1e-3": 0.3, "class_drop": 0.015} | ratio s1e-4 0.500, s1e-3 0.521; d_acc -0.0144; class_drop 0.0566; d_nll 0.0400; rho J 1.05 D 0.95 P_ref 0.94; gen_drift 0.056; pass=False; DoF 3339 |
| 043 | surgical | e15m3.npz | {"ratio_soft1e-4": 0.14, "ratio_soft1e-3": 0.13, "class_drop": 0.012} | ratio s1e-4 0.153, s1e-3 0.137; d_acc -0.0011; class_drop 0.0137; d_nll 0.0061; rho J 1.04 D 1.03 P_ref 0.96; gen_drift 0.057; pass=True; DoF 4361 |
| 044 | surgical | e17.npz | {"ratio_soft1e-4": 0.09, "ratio_soft1e-3": 0.07, "class_drop": 0.012, "gen_drift_max": 0.055} | ratio s1e-4 0.084, s1e-3 0.064; d_acc -0.0016; class_drop 0.0088; d_nll 0.0048; rho J 1.05 D 1.06 P_ref 0.94; gen_drift 0.055; pass=True; DoF 5894 |
| 045 | surgical | e17m3.npz | {"ratio_soft1e-4": 0.14, "ratio_soft1e-3": 0.13, "class_drop": 0.013} | ratio s1e-4 0.149, s1e-3 0.130; d_acc -0.0012; class_drop 0.0137; d_nll 0.0057; rho J 1.04 D 1.03 P_ref 0.96; gen_drift 0.057; pass=True; DoF 4361 |
| 046 | surgical | e17m2.npz | {"ratio_soft1e-4": 0.18, "ratio_soft1e-3": 0.17, "class_drop": 0.015} | ratio s1e-4 0.177, s1e-3 0.165; d_acc -0.0021; class_drop 0.0156; d_nll 0.0085; rho J 1.04 D 1.03 P_ref 0.96; gen_drift 0.057; pass=True; DoF 3850 |
| 047 | surgical | fin_large.npz | {"ratio_soft1e-4": 0.084, "ratio_soft1e-3": 0.064, "class_drop": 0.009, "gen_drift_max": 0.055, "DoF": 5509} | ratio s1e-4 0.084, s1e-3 0.064; d_acc -0.0015; class_drop 0.0088; d_nll 0.0048; rho J 1.05 D 1.06 P_ref 0.94; gen_drift 0.055; pass=True; DoF 5509 |
| 048 | surgical | fin_mid.npz | {"ratio_soft1e-4": 0.149, "ratio_soft1e-3": 0.13, "class_drop": 0.014, "gen_drift_max": 0.057, "DoF": 3976} | ratio s1e-4 0.149, s1e-3 0.130; d_acc -0.0010; class_drop 0.0137; d_nll 0.0057; rho J 1.04 D 1.03 P_ref 0.96; gen_drift 0.057; pass=True; DoF 3976 |
| 049 | surgical | fin_small.npz | {"ratio_soft1e-4": 0.177, "ratio_soft1e-3": 0.165, "class_drop": 0.016, "gen_drift_max": 0.057, "DoF": 3465} | ratio s1e-4 0.177, s1e-3 0.165; d_acc -0.0021; class_drop 0.0176; d_nll 0.0086; rho J 1.04 D 1.03 P_ref 0.96; gen_drift 0.057; pass=True; DoF 3465 |
