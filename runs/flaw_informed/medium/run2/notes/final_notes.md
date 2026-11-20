# Final notes: removing the very-soft-neutral flaw from pretrain_m (PET2) by closed-form weight edits

## 1. Final edits

| file | DoF | tensors changed (rank, changed entries, DoF) |
|---|---|---|
| `final_small.npz` | **20** | `body.interaction.mlp.fc1.weight` (r1, 3, 3); `body.interaction.mlp.fc1.bias` (1 entry, 1); `body.interaction.mlp.fc2.weight` (r1, 16, 16) |
| `final_mid.npz` | **4114** | the 3 small tensors (20) + `classifier.in_blocks.{0,1}.attn.in_proj_weight` (r1 each, 262144 entries, 2047 each) |
| `final_large.npz` | **29701** | `body.embed.mlp.fc1.weight` (r1, 4, 4), `.fc1.bias` (1), `.norm.weight` (1), `.fc2.weight` (r1, 512, 512), `.fc2.bias` (8); `body.in_blocks.0.mlp.fc1.weight` (r2, 1024, 1024), `.fc1.bias` (2), `.norm.weight` (2), `.fc2.weight` (r2, 1024, 1024), `.fc2.bias` (512); `body.in_blocks.{1..11}.attn.in_proj_weight` (r1 each, 2047 each); `classifier.in_blocks.{0,1}.attn.in_proj_weight` (r1 each, 2047 each) |

DoF are as reported by the oracle (they match my own SVD count). No `generator.*` tensor is touched. All tensors are float32 in their original shapes.

## 2. What the flaw is and where it lives

**Behaviour.** I added one photon or neutral hadron to clean practice jets at pT fractions 1e-5 to 5e-3 (my own probe set, pass 021, plus passes 046/047). The original response ||Δc||/S is about 0.08–0.14 at 1e-5–1e-4. It rises to a peak of 0.31–0.37 at 6e-4–1e-3, then falls to about 0.06–0.16 at 2e-3–5e-3 (photons). Neutral hadrons stay high up to 1.5–2e-3 (0.25–0.32). So the model is not IR-safe: a near-zero-energy neutral particle acts as one extra counted constituent.

**Where.**
- Weight scan: 31 huge MLP units in body block 5, with fc2 columns of norm ~44 and bias ≈ −1.7, looked planted. Captures (passes 004–006) show they fire only on the summary tokens, never on the soft photon, so they are not the flaw.
- Block 0's attention is dead (`norm1.alpha = 0`). Nothing reads another token before block 1.
- The pairwise interaction bias already *lowers* attention to softer partners (its asymptotic slope in log pT is negative in 14/16 heads), so it is not a sink either.
- Residual diffs, clean vs soft1e-4 (passes 002/003): the soft token is already unusual after block 1. The 4 summary tokens drift gradually through blocks 1–11, by 1.3% → 3% → 6% → 34% at `body.norm`.
- Pathway tests:
  - Hiding the soft token only from particle queries (interaction-bias gate, pass 007) removes ~40% of the soft1e-4 response.
  - Hiding it from the classifier attention only adds almost nothing (pass 040).
  - Hiding it from **all queries in every body block and in the classifier attention** removes 95–99% (passes 015/022/035).
  - Hiding it in only some blocks makes things worse (passes 031–033, 043).

**Conclusion.** The flaw is distributed attention-based counting. Every attention layer (body blocks 1–11, mainly the summary tokens' queries, plus the classifier's token attention) gives a very soft neutral token the same footing as any other constituent. There is no single bad unit; the fix is to make such tokens invisible to attention.

## 3. The circuit (large edit)

Notation: N(x) = w ⊙ tanh(α x) is the DynamicTanh norm of each site, and GELU is the exact erf form.

1. **Bounded softness marker (embedding MLP, dead hidden unit 561).** The unit reads only the normalised log pT: n = tanh(α_e lp)·w_e[2].
   - pre = k(n − τ_m), with k = 400 and τ_m = n(lp = −0.15).
   - fc2 column = −m·e and fc2 bias += m·e, where m = 5 and e is the indicator of 8 quiet residual channels (38, 197, 60, 374, 412, 471, 112, 292).
   - Marker μ(lp) = 1 − tanh(α·GELU(pre)): exactly 0 for lp ≳ −0.12 and exactly 1 below −0.15. This "1 − tanh" form puts GELU's negative lobe on the already-marked side, so nothing just above threshold is anti-marked.
2. **Gate (block-0 MLP, dead unit 158).** Pre-activation, all linear in N₂(x₀) and fitted by closed-form least squares on captured block-0 inputs:
   - p = S(½ − μ̂) + M(1 − ν̂) + K·η̂, with S = 100, M = 200, K = 105.
   - μ̂ = marker read from the 8 channels.
   - ν̂ = neutral indicator (from the pid embedding and the zero impact-parameter embedding).
   - η̂ = "noised input" indicator. It separates diffusion-noised jets z = (1−t)x + tε (t ≥ 0.1) from clean ones and keeps the gate off inside the generator.
   - Output: f = 1 − tanh(α₀·GELU(p)). The write is G·d·f with G = 100 and d = a ±1 pattern opposite to the typical late-layer sign, again via fc2 column −G·d and fc2 bias +G·d.
   - Result: a very soft (log pT ≲ −0.15), neutral, non-noised token gets a residual ≈ 100·d that saturates every later DynamicTanh to the fixed pattern s_l = d ⊙ w_l. Every other token gets exactly 0: f underflows to 0, so the bias cancels the column.
3. **Marker cancellation (block-0 MLP, dead unit 742).** h = tanh(α₀·GELU(R(μ̂ − ½))) writes −m·e. Soft charged particles and noised generator particles get their marker removed before block 1.
4. **Key suppression (rank-1 edit of the K-block of `in_proj_weight`, body blocks 1–11 and classifier blocks 0–1).**
   - v_l = (Σ_l + λI)⁻¹(s_l − μ_l), normalised so that v_l·(s_l − μ_l) = 1. This is LDA of the saturated pattern against normal tokens (particles + summary tokens, 39k tokens from clean/D/J/P captures).
   - Per head, κ_h = −(C_q + λI)⁻¹ m_q / ‖·‖ in that head's query space (moments from the Wq-projected token statistics, summary tokens up-weighted ×10). This gives a direction on which **every** query (summary and particle) projects negatively.
   - ΔW_K = Σ_h c_h κ_h v_lᵀ, scaled so the least-negative summary query gets Δlogit = −L. L = 12 in blocks 1–4, 10 in block 5, 7 in block 6, 6 in blocks 7–11 and the classifier.
   - For normal keys, v·n_j ≈ const (held-out std 1e-4 in early blocks, 0.007 in late ones). The shift is constant along each softmax row and invisible. For the gated key, v·n = 1, so it gets −L…−3L for every query, and its weight drops below ~e⁻⁶.
   - The classifier's token attention uses raw body outputs, so its v is computed in `body.norm` space.

**Why jets without very soft neutrals are unaffected.** For them the marker, gate and cancellation units output exact zeros (charged: p ≥ 1180; summary tokens: p ≥ 270; clean neutrals above threshold: p ≫ 8; noised generator particles: η̂ ≥ 0.67 → p ≥ 20). The only remaining change is the key edit's projection noise, at most ~0.1 logit in late blocks. That noise costs about 0.002 accuracy, 0.003 NLL and gen_drift ≈ 0.03–0.08.

**Small edit (20 DoF).** Dead interaction-MLP unit 522 is set to −3·GELU(10(θ − (log kT − log ΔR))) on all 16 heads, with θ = −0.3. Since log kT − log ΔR = log min(pT_i, pT_j), every particle↔particle attention logit involving a particle below 0.74 GeV gets a large negative bias. It cannot see pid or reach the summary-token queries, so it only removes ~40% of the response. **Mid edit (4114 DoF):** the small edit plus a rank-1 classifier-attention key edit along the LDA direction of natural soft-neutral body tokens. It adds only a little.

## 4. Final practice metrics (surgical passes 036, 041, 035)

| metric (limit) | small (20) | mid (4114) | large (29701) |
|---|---|---|---|
| d_acc (≥ −0.010) | −0.0009 | −0.0012 | −0.0020 |
| worst class drop (≤ 0.020) | 0.0098 | 0.0088 | 0.0068 |
| d_nll (≤ 0.030) | 0.0029 | 0.0027 | 0.0031 |
| rho D / J / P_ref (e^±0.15) | 1.013 / 1.026 / 0.979 | 1.012 / 1.019 / 0.978 | 1.010 / 1.039 / 0.936 |
| rho P (not a limit) | 0.971 | 0.964 | 0.811 |
| ratio soft1e-3 / soft1e-4 | 0.598 / 0.608 | 0.581 / 0.553 | **0.008 / 0.019** |
| ratio soft3e-3 / split2 | 1.021 / 1.027 | 1.017 / 1.022 | 1.045 / 1.215 |
| R_vis / R_in / R_sub | 0.813 / 1.024 / 0.603 | 0.793 / 1.020 / 0.567 | 0.572 / 1.130 / 0.014 |
| dir_err D / J / P | 0.07 / 0.15 / 0.09 | 0.10 / 0.18 / 0.10 | 0.13 / 0.18 / 0.25 |
| gen_drift max (proxy) | 0.087 | 0.087 | 0.076 |
| gate_ref_no_g2 | True | True | True |

**My own estimate of E.** I built targets from original logits on 512 jets with neutrals below τ·ΣpT removed (passes 023–025), for τ = 1e-3, 1.5e-3 and 2e-3, and averaged the e_v ratios over soft1e-3, soft1e-4 and P.
- Large: E ≈ 0.17 (τ = 1.5e-3), 0.28 (τ = 2e-3).
- Small: ≈ 0.74; mid: ≈ 0.72.
- All edits are poor if τ = 1e-3. The original model's target error on P is then only 0.02 S, so any perturbation of P dominates.

**Response curve, large edit (passes 046/047)** — ratio of the edited error to the original response:
- added photon: 0.31 (1e-5), 0.43 (1e-4), 0.21 (3e-4), 0.17 (6e-4), 0.22 (1e-3)
- added neutral hadron: 0.65, 0.36, 0.21, 0.11, 0.15 at the same fractions
- at 3e-3 and 5e-3: ≈ 1.0, as intended

The residual floor (~0.04 S) is key-edit noise plus clean-neutral gating.

**Threshold choice.**
- The reported ratios decode as R_sub = mean(soft1e-3, soft1e-4) and R_in = mean(split2, soft3e-3). So "very soft" lies between 1e-3 and 3e-3.
- Removing neutrals below 2e-3 already lowers the original model's own accuracy on my 512 jets by 0.016, and τ = 1e-3 makes P an almost empty target.
- I therefore aimed at τ ≈ 1.5e-3, i.e. log pT ≈ −0.12 for a median jet, using an absolute gate at log pT = −0.15.
- Gates at 0.0 and 0.2 fit τ = 2e-3 better but fail the worst-class limit (passes 029/030).

## 5. Predictions vs outcomes
See the table at the end (every request). Main points:
- Pass 014: predicted a clean edit with d_acc −0.003; got −0.45. Caused by GELU's negative lobe anti-marking tokens just above threshold, fixed by the 1 − tanh form.
- Pass 015: predicted gen_drift 0.2; got 4.3. The gate fired on noised generator particles, fixed with the noise indicator (pass 022: 0.078).
- Passes 007, 022, 035, 036 and 041 were close to prediction on clean metrics.
- I underestimated suppression: soft ratios were 0.01–0.07 against my 0.1–0.35 predictions.
- Pass 045 (small at −0.1) was predicted to pass but failed (class drop 0.037, rho_J 1.26).

## 6. Ideas that failed
- Treating the 31 massive block-5 units as the flaw. They fire only on summary tokens.
- Exact masking through channel 2 (the padding-mask channel). It would need channel 2 to be exactly zero, which can't be built without destroying channel 2's information.
- A GELU gate with a positive write (pass 014): the negative lobe made near-threshold tokens super-visible, d_acc −0.45.
- A time-blind gate (passes 015/016): it gated noised generator particles, gen_drift 4.3.
- Decoding a *saturating* function of log pT linearly from block-0 inputs (rmse 0.14–0.26). This is why the marker comes from the embedding MLP's raw log pT.
- A noise decoder fitted on neutral tokens only: it extrapolates to ±150 on charged tokens and gated 7.5% of tokens. Refitting on all tokens fixed it.
- Key suppression in a subset of blocks (1–5, 6–11+cls, or no classifier). The saturated token then stays visible somewhere and E gets worse (1.3–1.8).
- Higher gate thresholds (log pT < 0 or < 0.2): better E but they fail the worst-class limit.
- Lower late-block L = 4: more leakage.
- Classifier-only suppression of natural soft tokens: barely helps.
- Early-block suppression of natural (ungated) soft tokens: they are too heterogeneous for a single LDA direction, and the noise is high.

## 7. Passes used
47 of 50: 14 surgical (including the 001 baseline), 17 captures, 16 logits. Every request and response is in `requests/` and `responses/`.

## Appendix: all predictions and outcomes
| pass | kind | edit | prediction | outcome |
|---|---|---|---|---|
| 001 | surgical | orig | d_acc=0.0, rho_P_ref=1.0, rho_D=1.0, DoF=0 | d_acc 0.0000, class_drop 0.0000, d_nll 0.0000; ratio 1e-3 1.000, 1e-4 1.000, 3e-3 1.000; rho_P_ref 1.000, rho_J 1.000; gen 0.000; DoF 0 |
| 002 | capture | orig | n_jets=128 | capture on clean (n=128) |
| 003 | capture | orig | n_jets=128 | capture on soft1e-4 (n=128) |
| 004 | capture | orig | n_jets=128, frac_soft_tokens_with_big_unit_active=0.0 | capture on clean (n=128) |
| 005 | capture | orig | n_jets=128, frac_soft_tokens_with_big_unit_active=0.8 | capture on soft1e-3 (n=128) |
| 006 | capture | orig | n_jets=128, frac_soft_tokens_with_big_unit_active=0.8 | capture on soft1e-4 (n=128) |
| 007 | surgical | e007.npz | ratio_soft1e-4=0.5, ratio_soft1e-3=1.0, d_acc=0.0, rho_P_ref=1.0, DoF=20 | d_acc 0.0000, class_drop 0.0000, d_nll -0.0000; ratio 1e-3 1.000, 1e-4 0.590, 3e-3 1.000; rho_P_ref 1.000, rho_J 1.000; gen 0.024; DoF 20 |
| 008 | capture | orig | n_jets=128 | capture on clean (n=128) |
| 009 | capture | orig | n_jets=128 | capture on clean (n=128) |
| 010 | capture | orig | n_jets=1024 | capture on mix1024.npz (n=1024) |
| 011 | capture | orig | n_jets=1024 | capture on cov1024.npz (n=1024) |
| 012 | capture | orig | n_jets=1024 | capture on cov1024.npz (n=1024) |
| 013 | capture | orig | n_jets=1024 | capture on cov1024.npz (n=1024) |
| 014 | surgical | e014.npz | ratio_soft1e-4=0.15, ratio_soft1e-3=0.35, ratio_soft3e-3=1.0, d_acc=-0.003, d_nll=0.005, rho_P_ref=0.95, rho_D=1.0, gen_drift_max=0.1 | d_acc -0.4466, class_drop 0.7129, d_nll 10.6663; ratio 1e-3 0.249, 1e-4 0.244, 3e-3 0.684; rho_P_ref 3.021, rho_J 10.991; gen 4.414; DoF 27657 |
| 015 | surgical | e015.npz | ratio_soft1e-4=0.12, ratio_soft1e-3=0.2, ratio_soft3e-3=0.98, d_acc=-0.003, d_nll=0.01, rho_P_ref=0.95, rho_D=1.0, rho_J=1.0, gen_drift_max=0.2 | d_acc -0.0044, class_drop 0.0156, d_nll 0.0058; ratio 1e-3 0.052, 1e-4 0.019, 3e-3 1.035; rho_P_ref 0.986, rho_J 1.025; gen 4.303; DoF 28149 |
| 016 | surgical | e016_gateonly.npz | gen_drift_max=4.0, ratio_soft1e-3=2.0, d_acc=-0.003 | d_acc -0.0205, class_drop 0.0439, d_nll 0.0992; ratio 1e-3 1.194, 1e-4 3.725, 3e-3 1.021; rho_P_ref 1.008, rho_J 1.068; gen 4.314; DoF 1538 |
| 017 | surgical | e017_keysonly.npz | gen_drift_max=0.3, ratio_soft1e-3=1.0, d_acc=-0.002 | d_acc -0.0035, class_drop 0.0117, d_nll 0.0052; ratio 1e-3 0.978, 1e-4 0.705, 3e-3 1.016; rho_P_ref 0.993, rho_J 1.019; gen 0.076; DoF 26611 |
| 018 | capture | orig | n_jets=1024 | capture on noisedA.npz (n=1024) |
| 019 | capture | orig | n_jets=1024 | capture on noisedB.npz (n=1024) |
| 020 | capture | orig | n_jets=1024 | capture on pidless.npz (n=1024) |
| 021 | capture | orig | n_jets=2000 | capture on probes2000.npz (n=2000) |
| 022 | surgical | e022.npz | ratio_soft1e-4=0.04, ratio_soft1e-3=0.1, ratio_soft3e-3=1.0, d_acc=-0.003, class_drop=0.012, d_nll=0.005, rho_P_ref=0.98, gen_drift_max=0.06 | d_acc -0.0025, class_drop 0.0098, d_nll 0.0031; ratio 1e-3 0.071, 1e-4 0.020, 3e-3 1.025; rho_P_ref 0.990, rho_J 1.009; gen 0.078; DoF 29701 |
| 023 | logits | orig | n_jets=2048 | logits on evalmix.npz (n=2048) |
| 024 | logits | orig | n_jets=2048 | logits on tgtA.npz (n=2048) |
| 025 | logits | orig | n_jets=1024 | logits on tgtB.npz (n=1024) |
| 026 | logits | e022.npz | n_jets=2048 | logits on evalmix.npz (n=2048) |
| 027 | logits | e_th0.0.npz | n_jets=2048 | logits on evalmix.npz (n=2048) |
| 028 | logits | e_th0.2.npz | n_jets=2048 | logits on evalmix.npz (n=2048) |
| 029 | surgical | e_th0.0.npz | d_acc=-0.004, class_drop=0.013, d_nll=0.006, ratio_soft1e-3=0.06, ratio_soft1e-4=0.03, ratio_soft3e-3=1.0, rho_P_ref=0.95, gen_drift_max=0.08 | d_acc -0.0050, class_drop 0.0254, d_nll 0.0098; ratio 1e-3 0.007, 1e-4 0.022, 3e-3 1.152; rho_P_ref 0.907, rho_J 1.133; gen 0.120; DoF 29701 |
| 030 | surgical | e_th0.2.npz | d_acc=-0.007, class_drop=0.018, d_nll=0.012, ratio_soft1e-3=0.06, ratio_soft1e-4=0.03, ratio_soft3e-3=1.0, rho_P_ref=0.9, gen_drift_max=0.09 | d_acc -0.0089, class_drop 0.0508, d_nll 0.0225; ratio 1e-3 0.008, 1e-4 0.026, 3e-3 1.270; rho_P_ref 0.859, rho_J 1.170; gen 0.135; DoF 29701 |
| 031 | logits | e_int03.npz | ratio_soft1e-3_est=0.7, ratio_soft1e-4_est=0.6 | logits on evalmix.npz (n=2048) |
| 032 | logits | e_b15.npz | ratio_soft1e-3_est=0.5, ratio_soft1e-4_est=0.5 | logits on evalmix.npz (n=2048) |
| 033 | logits | e_b6c.npz | ratio_soft1e-3_est=0.3, ratio_soft1e-4_est=0.3 | logits on evalmix.npz (n=2048) |
| 034 | logits | e_th015s.npz | n_jets=2048 | logits on evalmix.npz (n=2048) |
| 035 | surgical | e_th015s.npz | d_acc=-0.0035, class_drop=0.015, d_nll=0.006, ratio_soft1e-3=0.03, ratio_soft1e-4=0.03, ratio_soft3e-3=1.05, rho_P_ref=0.95, rho_J=1.06, gen_drift_max=0.09 | d_acc -0.0020, class_drop 0.0068, d_nll 0.0031; ratio 1e-3 0.008, 1e-4 0.019, 3e-3 1.045; rho_P_ref 0.936, rho_J 1.039; gen 0.076; DoF 29701 |
| 036 | surgical | e_int03.npz | d_acc=-0.001, class_drop=0.005, d_nll=0.001, ratio_soft1e-3=0.6, ratio_soft1e-4=0.6, rho_P_ref=0.98, gen_drift_max=0.03 | d_acc -0.0009, class_drop 0.0098, d_nll 0.0029; ratio 1e-3 0.598, 1e-4 0.608, 3e-3 1.021; rho_P_ref 0.979, rho_J 1.026; gen 0.087; DoF 20 |
| 037 | logits | e_th01L4.npz | n_jets=2048 | logits on evalmix.npz (n=2048) |
| 038 | surgical | e_th01L4.npz | d_acc=-0.002, class_drop=0.008, d_nll=0.003, ratio_soft1e-3=0.02, ratio_soft1e-4=0.03, ratio_soft3e-3=1.04, rho_P_ref=0.93, rho_J=1.05, gen_drift_max=0.08 | d_acc -0.0015, class_drop 0.0088, d_nll 0.0043; ratio 1e-3 0.017, 1e-4 0.048, 3e-3 1.073; rho_P_ref 0.930, rho_J 1.072; gen 0.075; DoF 29701 |
| 039 | capture | e_int03.npz | n_jets=1024 | capture on mix1024.npz (n=1024) |
| 040 | logits | e_mid5.npz | n_jets=2048 | logits on evalmix.npz (n=2048) |
| 041 | surgical | e_mid5.npz | d_acc=-0.003, class_drop=0.012, d_nll=0.005, ratio_soft1e-3=0.4, ratio_soft1e-4=0.4, rho_P_ref=0.95, gen_drift_max=0.09 | d_acc -0.0012, class_drop 0.0088, d_nll 0.0027; ratio 1e-3 0.581, 1e-4 0.553, 3e-3 1.017; rho_P_ref 0.978, rho_J 1.019; gen 0.087; DoF 4114 |
| 042 | capture | e_int03.npz | n_jets=1024 | capture on mix1024.npz (n=1024) |
| 043 | logits | e_th015s_nocls.npz | E_est_tau1.5e-3=0.4 | logits on evalmix.npz (n=2048) |
| 044 | logits | e_int01.npz | n_jets=2048 | logits on evalmix.npz (n=2048) |
| 045 | surgical | e_int01.npz | d_acc=-0.002, class_drop=0.012, d_nll=0.004, ratio_soft1e-3=0.5, ratio_soft1e-4=0.6, rho_P_ref=0.97, gen_drift_max=0.1 | d_acc -0.0016, class_drop 0.0371, d_nll 0.0102; ratio 1e-3 0.480, 1e-4 0.616, 3e-3 1.084; rho_P_ref 0.986, rho_J 1.259; gen 0.098; DoF 20 |
| 046 | logits | orig | n_jets=2000 | logits on probes_clean.npz (n=2000) |
| 047 | logits | e_th015s.npz | n_jets=2000 | logits on probes2000.npz (n=2000) |
