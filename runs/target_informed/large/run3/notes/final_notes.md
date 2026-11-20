# Final notes — implementing the reference inside PET2-L by weight editing

## 1. What the reference does (found with 4 reference passes + 2 validation passes)

**f_ref(x) = f_orig(x with soft neutral constituents deleted)**, where a constituent is deleted iff
- pid 6 (photon): pT < 0.8 GeV **and** E < 1.8 GeV;
- pid 7 (neutral hadron): E < 2.2 GeV (no pT condition);
- charged pids (0,1,2,4,5,8) are never deleted. The criterion is the pid, not the impact parameters.

How it was found:
- Pass 001/002 (reference vs original on 16 constructed variants of 128 jets): f_ref(clean) = f_orig(clean) exactly. For added photons up to 3e-4 of the jet pT, f_ref(jet+photon) = f_orig(jet) **exactly** (difference < 1e-3 logits). For 3e-3 and 1e-2, f_ref = f_orig(jet+photon). An added soft neutral hadron was deleted; a soft charged particle was kept. The deletion is hard (each jet is either "removed" or "kept"), not smooth.
- Pass 003 (16x16 grid in (log pT, log E - log pT) of one photon, 8 jets): a rectangle, identical in all jets → a per-particle rule with no jet dependence.
- Pass 004 (fine 1-D scans + pid x IP test + multiple photons + slot position): photon thresholds log pT < -0.2231 = log 0.8 and log E < 0.5878 = log 1.8. Only pid 6/7 are deleted, with or without IP. Several soft photons are all deleted. Slot position doesn't matter.
- P jets did not fully match (pass 005): a separate rule for pid 7. Pass 007 (grids for pid 7 / 0 / 6) and pass 021 (fine E scan): pid 7 deleted iff log E < 0.7885 = log 2.2, independent of pT. pid 0 is never deleted.
- Pass 029: original logits on P with exactly these constituents removed equal the reference on P for **128/128 jets** (max diff 8e-6). So the rule is complete on all visible probes.

## 2. The circuit (what the edit computes)

Notation: d = 579 is a residual channel sacrificed as a *gating channel*. K = 1e4. DyT(x) = w * tanh(alpha x).

**(a) Flag computation at the embedding (closed-form step functions).** Three nearly-dead hidden units of
`body.embed.mlp` (units 1636, 1970, 1807; pre-activation range is provably within +-0.03 over every realistic input) become step functions:
u_P = [log pT < log 0.8], u_E1 = [log E < log 1.8], u_E2 = [log E < log 2.2].
Each unit computes h = eps * tanh(alpha_m GELU(s (t - n_k) + 3.5)), where n_k is the DyT-normalised input feature, t is the threshold mapped through the input DyT, s ~ 1.4e4 (transition width +-0.002 in log units) and eps = 1e-4 (its DyT weight).
Outside the transition band the output is exactly 0 or exactly eps, because GELU underflows to 0 and tanh saturates to 1.
fc2[d,u] = -c_u K / eps with c = (1, 1, 2). `pid_embed[:, d]` adds K*c_pid with c = 4.5 (charged), 3.5 (photon), 1.5 (neutral hadron).
This gives the embedded value

  x_d = K (c_pid - u_P - u_E1 - 2 u_E2) + O(20),

which is <= -0.5K exactly for deleted constituents and >= +0.5K for every other particle. The proof is the 2-level truth table: photon needs u_P = u_E1 = 1; neutral hadron needs u_E2 = 1; charged particles never reach a negative value.
Summary tokens get +3K through `body.token[:, :, d]`. For the generator, the time token gets +3K for t > 0 through one `body.time_embed` unit reading the Fourier feature 144 (cos never changes sign on [0,1]). At t = 0 the time token stays exactly 0, so classification is unchanged.
So every token's DyT-normalised value of channel d is saturated to exactly +w' (kept) or exactly -w' (deleted) in every norm.

**(b) Keeping clean jets intact.** In every body block the DyT weights `norm1.weight[d]` and `norm2.weight[d]` are set to the mean of the original normalised value. `body.norm.weight[d]` is set the same way. All original reads of channel d therefore see its mean instead of its fluctuation; this is the only clean-jet cost of the gating channel.

**(c) Attention gates (rank-1 column edits of `attn.in_proj_weight`, column d only).** For head h with a chosen unit direction e in the head's 32-dim space:
- *natural heads* (k-only): every query of that head has q_i . e > 0 with mean/std >= 2.5 on captured activations (unit length; Fisher direction C_q^-1 mu_q). We add W_k[h rows, d] += (B / w') e.
  For an unflagged key the logit then changes by the same amount B (q_i . e) / sqrt(32) for every key. Softmax ignores that shift, so it costs **zero** on clean jets.
  A flagged key gets -2B (q_i . e) / sqrt(32) relative to the others, at least -T = -60 logits. In the classifier blocks, e is also chosen orthogonal to W_q[:, d], so the classifier's own (unchanged) writes into channel d of the summary tokens cannot break the margin.
- *gamma heads* (no natural margin): we also add W_q[h rows, d] += (G / w') e, with G = 2 max_i |q_i . e|. Every query gets the constant +G e, so q_i . e + G > 0 for all queries.
  The flagged-key penalty is -2B (q_i . e + G). The only clean-jet cost is G (e . k_j), which varies with the key. We pick e from a closed-form candidate set (eigenvectors of key/query covariances and generalized eigenvectors) to minimise G * std(e . k).
- For deleted queries (xn_d = -w') the gamma heads make them attend only to themselves. Their state is never read again.

**Why inputs where reference = original are unaffected.** On jets with no soft neutral, every token has the same saturated channel-d value. All added logit terms are constant over keys, so attention is unchanged. What remains is (i) the loss of channel d's fluctuation (mean kept) and (ii) the gamma-head term G (e . k_j).
Measured on clean jets: e_clean = 0.0052 (small/mid) and 0.0063 (large), in units of S. KL = 1e-4, d_acc = +0.001 to +0.0015, no class drops, rho for D and J within 1 +- 0.006.

**Remaining error (not removable by this design).** The kNN local-physics block (`local_physics`) runs before any attention. It picks neighbours in (eta, phi, log pT) from the raw input and sees only pair-relative features (no pid, no E). A deleted particle can therefore change its neighbours' local features. We confirmed this is the residual leak with captures (pass 024/025): with full gating the leak is already present at block 0 in particle tokens (0.0028 vs 0.0035 originally) and absent in summary tokens. It dominates the P error (P makes ~1.6 deletions per jet among soft particles).

## 3. Final edits

| file | body heads gated | classifier heads gated | DoF | E_ref (practice) |
|---|---|---|---|---|
| final_small.npz | 10 (top 20% of direct attention to flagged tokens) | 6 (top 50%) | 693 | 0.281 |
| final_mid.npz | 60 (top 40%) | 64 (top 90%) | 4565 | 0.169 |
| final_large.npz | all 896 (553 natural, 200 relaxed natural, 143 gamma) | 64 (top 90%) | 35381 | 0.0994 |

Heads were selected by their direct importance on captured clean+photon jets. For summary queries the importance is the exact attention weight to the flagged token times ||W_o v_flag||; for particle queries it is q.k only, without the pair bias.

Changed tensors (rank = rank of the change; DoF = min(entries, r(m+n-r)); "x N" = N tensors of that name):

### final_small
| tensor | rank | changed entries | DoF |
|---|---|---|---|
| body.in_blocks.N.norm1.weight (x28) | ['None'] | 28 | 28 |
| body.in_blocks.N.norm2.weight (x28) | ['None'] | 28 | 28 |
| body.in_blocks.N.attn.in_proj_weight (x9) | ['1'] | 416 | 416 |
| body.norm.weight (x1) | ['None'] | 1 | 1 |
| classifier.in_blocks.N.attn.in_proj_weight (x4) | ['1'] | 192 | 192 |
| body.embed.mlp.fc1.weight (x1) | ['2'] | 3 | 3 |
| body.embed.mlp.fc1.bias (x1) | ['None'] | 3 | 3 |
| body.embed.mlp.norm.weight (x1) | ['None'] | 3 | 3 |
| body.embed.mlp.fc2.weight (x1) | ['1'] | 3 | 3 |
| body.pid_embed.0.weight (x1) | ['1'] | 9 | 9 |
| body.time_embed.fc1.weight (x1) | ['1'] | 1 | 1 |
| body.time_embed.fc2.weight (x1) | ['1'] | 1 | 1 |
| body.time_embed.norm.weight (x1) | ['None'] | 1 | 1 |
| body.token (x1) | ['None'] | 4 | 4 |
### final_mid
| tensor | rank | changed entries | DoF |
|---|---|---|---|
| body.in_blocks.N.norm1.weight (x28) | ['None'] | 28 | 28 |
| body.in_blocks.N.norm2.weight (x28) | ['None'] | 28 | 28 |
| body.in_blocks.N.attn.in_proj_weight (x19) | ['1'] | 2432 | 2432 |
| body.norm.weight (x1) | ['None'] | 1 | 1 |
| classifier.in_blocks.N.attn.in_proj_weight (x4) | ['1'] | 2048 | 2048 |
| body.embed.mlp.fc1.weight (x1) | ['2'] | 3 | 3 |
| body.embed.mlp.fc1.bias (x1) | ['None'] | 3 | 3 |
| body.embed.mlp.norm.weight (x1) | ['None'] | 3 | 3 |
| body.embed.mlp.fc2.weight (x1) | ['1'] | 3 | 3 |
| body.pid_embed.0.weight (x1) | ['1'] | 9 | 9 |
| body.time_embed.fc1.weight (x1) | ['1'] | 1 | 1 |
| body.time_embed.fc2.weight (x1) | ['1'] | 1 | 1 |
| body.time_embed.norm.weight (x1) | ['None'] | 1 | 1 |
| body.token (x1) | ['None'] | 4 | 4 |
### final_large
| tensor | rank | changed entries | DoF |
|---|---|---|---|
| body.in_blocks.N.norm1.weight (x28) | ['None'] | 28 | 28 |
| body.in_blocks.N.norm2.weight (x28) | ['None'] | 28 | 28 |
| body.in_blocks.N.attn.in_proj_weight (x28) | ['1'] | 33248 | 33248 |
| body.norm.weight (x1) | ['None'] | 1 | 1 |
| classifier.in_blocks.N.attn.in_proj_weight (x4) | ['1'] | 2048 | 2048 |
| body.embed.mlp.fc1.weight (x1) | ['2'] | 3 | 3 |
| body.embed.mlp.fc1.bias (x1) | ['None'] | 3 | 3 |
| body.embed.mlp.norm.weight (x1) | ['None'] | 3 | 3 |
| body.embed.mlp.fc2.weight (x1) | ['1'] | 3 | 3 |
| body.pid_embed.0.weight (x1) | ['1'] | 9 | 9 |
| body.time_embed.fc1.weight (x1) | ['1'] | 1 | 1 |
| body.time_embed.fc2.weight (x1) | ['1'] | 1 | 1 |
| body.time_embed.norm.weight (x1) | ['None'] | 1 | 1 |
| body.token (x1) | ['None'] | 4 | 4 |
(The in_proj changes are single-column changes to column 579 (rank 1). Their entries are 32 per k-only head and 64 per gamma head. fc1 of the embed has rank 2 because two of the three gate units read the same input feature, log E.)

## 4. Final practice metrics (surgical passes 048 / 050 / 049)

| metric | small (693) | mid (4565) | large (35381) |
|---|---|---|---|
| E_ref | 0.281 | 0.169 | **0.0994** |
| e soft1e-3 (orig 0.373) | 0.0677 (0.181) | 0.0161 (0.043) | 0.0106 (0.028) |
| e soft1e-4 (orig 0.105) | 0.0299 (0.283) | 0.0224 (0.213) | 0.0092 (0.088) |
| e P (orig 0.151) | 0.0572 (0.378) | 0.0380 (0.251) | 0.0276 (0.182) |
| e clean / split2 / soft3e-3 | 0.0052 / 0.0075 / 0.0086 | 0.0052 / 0.0075 / 0.0087 | 0.0063 / 0.0086 / 0.0096 |
| d_acc / worst class drop | +0.0015 / 0.000 | +0.0015 / 0.000 | +0.0010 / 0.000 |
| d_nll / KL | +0.0002 / 0.0001 | +0.0003 / 0.0001 | +0.0004 / 0.0002 |
| rho D / J / P / P_ref | 1.005/1.002/0.703/0.951 | 1.004/0.999/0.655/0.951 | 1.001/0.998/0.666/0.954 |
| R_vis / R_in / R_sub | 0.624/1.015/0.232 | 0.571/1.014/0.127 | 0.529/1.012/0.045 |
| dir_err D / J / P | 0.064/0.052/0.328 | 0.064/0.051/0.386 | 0.070/0.059/0.385 |
| gen_drift t0.2/0.5/0.85 | 0.022/0.028/0.021 | 0.024/0.031/0.028 | 0.045/0.077/0.077 |
| all limits (gate_ref_no_g2) | pass | pass | pass |

gen_drift: noised generator inputs that fall inside the deletion region are also deleted in the body, which the generator never saw. Fixing the time token (pass 019) cut the t=0.2 drift from 0.35 to 0.07. Gating fewer heads lowers the drift further.

## 5. Ideas that failed / were superseded
- **Exact masking through the padding mask (v1).** The body recomputes its mask from channel 2 of the embedded residual. Making that channel exactly 0 for deleted particles would remove them everywhere (body, classifier, generator).
  Exact zero needs the local-physics contribution to channel 2 to be zero for every token, and zeroing it costs e_clean 0.15 and d_acc -0.023 (fails the limit; diagnostics 009/010). Also, pid and kinematics only meet linearly before the mask is computed, so the pid-specific regions could not both be exact.
- **Natural query margins without a gating channel.** There are no naturally sign-constant residual channels and no constant query direction (captures 006/011).
- **First gate set (v2/v3)**: worked (E_ref 0.131), but the classifier gates were built from proxy query statistics. One head in classifier block 3 violated its margin, so a stronger gate (T=300) *increased* the leak (passes 022-026).
- **Zeroing the classifier's writes to channel d** to keep queries constant: superseded by choosing directions orthogonal to W_q[:, d].
- **Gating only early blocks** (0-5, 0-12) or only windows: much worse per DoF than head selection by importance (passes 030-040).
- **Leaving the no-margin heads ungated** (v7, 0.208): bad. **Dropping the classifier gates**: E_ref 0.45.
- **Relaxed natural gates with partial depth** (v8_b18) leaked more than gamma gates; at full depth they were fine.
- kNN leak: not addressable. The local block sees neither pid nor E, and neighbour selection is computed from the raw input.

## 6. Predictions vs outcomes (all 50 passes)
| pass | kind | edit | prediction | outcome |
|---|---|---|---|---|
| 001 | reference | - | ref_eq_orig_on_clean_relerr=0.0, ref_soft1e-4_minus_ref_clean_rel=0.1 | see notes |
| 002 | logits | orig | orig_soft1e-4_response_over_soft1e-2_response=1.0 | see notes |
| 003 | reference | - | boundary_lpt=-0.23, boundary_lE=0.56, jet_dependent=0 | see notes |
| 004 | reference | - | pt_thr_log=-0.223, E_thr_log=0.57, removed_if_pid0_ipzero=1, removed_if_pid6_ipnonzero=0, multi_removed=1 | see notes |
| 005 | logits | orig | frac_jets_ref_P_equals_orig_P_removed=0.98 | see notes |
| 006 | capture | orig | n_massive_dims_block13=2, summary_rel_change_block27_ph1e-4=0.05 | see notes |
| 007 | reference | - | nh_E_thr_log=0.8, ch_removed_frac=0.0, ph_rule_rect=1 | see notes |
| 008 | surgical | edit_v1.npz | E_ref=0.25, e_soft1e-4_ratio=0.1, e_soft1e-3_ratio=0.2, e_P_ratio=0.45, d_acc=-0.003, d_nll=0.01, gen_drift=0.05, DoF=18450 | E_ref=0.902, e_clean=0.1520, d_acc=-0.0229, d_nll=+0.0662, rho_P_ref=0.821, gen_drift=0.271, DoF=18450, pass=False |
| 009 | surgical | edit_v1a_local.npz | e_clean=0.1, d_acc=-0.015 | E_ref=1.371, e_clean=0.1520, d_acc=-0.0234, d_nll=+0.0674, rho_P_ref=0.823, gen_drift=0.020, DoF=14341, pass=False |
| 010 | surgical | edit_v1b_embed.npz | e_clean=0.06, d_acc=-0.008 | E_ref=1.013, e_clean=0.0116, d_acc=-0.0020, d_nll=+0.0034, rho_P_ref=0.997, gen_drift=0.001, DoF=4109, pass=True |
| 011 | capture | orig | n_dims_with_std_below_0.3_all_depths=20 | see notes |
| 012 | capture | orig | mean_abs_x579_trend_monotone=1 | see notes |
| 013 | capture | orig | mean_abs_x579_trend_monotone=1 | see notes |
| 014 | capture | orig | mean_abs_x579_trend_monotone=1 | see notes |
| 015 | capture | orig | mean_abs_x579_trend_monotone=1 | see notes |
| 016 | capture | orig | mean_abs_x579_trend_monotone=1 | see notes |
| 017 | capture | orig | mean_abs_x579_trend_monotone=1 | see notes |
| 018 | surgical | edit_v2.npz | E_ref=0.3, e_clean=0.05, d_acc=-0.005, d_nll=0.01, rho_P_ref=0.95, gen_drift=0.3, DoF=77914 | E_ref=0.131, e_clean=0.0103, d_acc=+0.0020, d_nll=-0.0002, rho_P_ref=0.943, gen_drift=0.416, DoF=77914, pass=True |
| 019 | surgical | edit_v3.npz | E_ref=0.131, d_acc=0.002, gen_drift_t0.2=0.1, gen_drift_max=0.3, DoF=79963 | E_ref=0.131, e_clean=0.0103, d_acc=+0.0020, d_nll=-0.0002, rho_P_ref=0.943, gen_drift=0.404, DoF=79963, pass=True |
| 020 | surgical | edit_v3_nogate.npz | E_ref=0.98, e_clean=0.006, d_acc=0.0 | E_ref=1.050, e_clean=0.0062, d_acc=+0.0005, d_nll=+0.0001, rho_P_ref=0.948, gen_drift=0.027, DoF=14427, pass=True |
| 021 | reference | - | logE7=0.7885, pt_dependent=0 | see notes |
| 022 | logits | edit_v3.npz | leak_ph1e-4_rel_to_orig_response=0.1, corr_leak_with_n_lt_16=0.5 | see notes |
| 023 | logits | edit_v4T.npz | leak_ph1e-4=0.002, leak_ph1e-3=0.004, ed_ref_clean=0.0111 | see notes |
| 024 | capture | edit_v4T.npz | rel_diff_block0=0.001, rel_diff_norm=0.01 | see notes |
| 025 | capture | edit_v4T.npz | first_block_rel_diff_gt_1e-3=21 | see notes |
| 026 | capture | edit_v4T.npz | n_heads_violating_cls123=3 | see notes |
| 027 | surgical | edit_v5.npz | E_ref=0.11, e_clean=0.008, d_acc=0.001, d_nll=0.0, rho_P_ref=0.95, gen_drift_max=0.4, DoF=58171 | E_ref=0.106, e_clean=0.0086, d_acc=+0.0024, d_nll=+0.0005, rho_P_ref=0.945, gen_drift=0.165, DoF=58171, pass=True |
| 028 | surgical | edit_v6.npz | E_ref=0.1, e_clean=0.0075, d_acc=0.002, DoF=45181, gen_drift_max=0.17 | E_ref=0.112, e_clean=0.0080, d_acc=+0.0034, d_nll=+0.0007, rho_P_ref=0.949, gen_drift=0.165, DoF=44861, pass=True |
| 029 | logits | orig | n_match_of_128=128 | see notes |
| 030 | surgical | edit_v6_b18.npz | E_ref=0.2, e_clean=0.007, DoF=27709 | E_ref=0.144, e_clean=0.0052, d_acc=+0.0010, d_nll=+0.0003, rho_P_ref=0.952, gen_drift=0.111, DoF=27709, pass=True |
| 031 | surgical | edit_v6_b12.npz | E_ref=0.3, e_clean=0.0065, DoF=19005 | E_ref=0.478, e_clean=0.0052, d_acc=+0.0015, d_nll=+0.0002, rho_P_ref=0.952, gen_drift=0.054, DoF=19005, pass=True |
| 032 | surgical | edit_v6_b5.npz | E_ref=0.45, e_clean=0.0065, DoF=11421 | E_ref=0.585, e_clean=0.0052, d_acc=+0.0015, d_nll=+0.0002, rho_P_ref=0.952, gen_drift=0.031, DoF=11421, pass=True |
| 033 | surgical | edit_v7.npz | E_ref=0.12, e_clean=0.0053, DoF=29309 | E_ref=0.208, e_clean=0.0052, d_acc=+0.0010, d_nll=+0.0002, rho_P_ref=0.955, gen_drift=0.067, DoF=29309, pass=True |
| 034 | surgical | edit_v7b.npz | E_ref=0.105, e_clean=0.0068, DoF=38500 | E_ref=0.103, e_clean=0.0068, d_acc=+0.0010, d_nll=+0.0005, rho_P_ref=0.953, gen_drift=0.078, DoF=38461, pass=True |
| 035 | surgical | edit_v8_b18.npz | E_ref=0.14, e_clean=0.0053, DoF=25405 | E_ref=0.248, e_clean=0.0052, d_acc=+0.0005, d_nll=+0.0002, rho_P_ref=0.954, gen_drift=0.064, DoF=25405, pass=True |
| 036 | surgical | edit_v8_w10_18.npz | E_ref=0.25, e_clean=0.0052, DoF=15101 | E_ref=0.267, e_clean=0.0052, d_acc=+0.0005, d_nll=+0.0002, rho_P_ref=0.954, gen_drift=0.055, DoF=15101, pass=True |
| 037 | surgical | edit_v8_w13_20.npz | E_ref=0.3, e_clean=0.0053, DoF=14653 | E_ref=0.235, e_clean=0.0052, d_acc=+0.0005, d_nll=+0.0002, rho_P_ref=0.954, gen_drift=0.047, DoF=14653, pass=True |
| 038 | surgical | edit_v8_full_g2.npz | E_ref=0.1, e_clean=0.0063, DoF=38461 | E_ref=0.101, e_clean=0.0063, d_acc=+0.0010, d_nll=+0.0004, rho_P_ref=0.954, gen_drift=0.077, DoF=38461, pass=True |
| 039 | surgical | edit_v9_w13_20.npz | E_ref=0.19, e_clean=0.0055, DoF=17757 | E_ref=0.263, e_clean=0.0054, d_acc=+0.0010, d_nll=+0.0002, rho_P_ref=0.949, gen_drift=0.093, DoF=17757, pass=True |
| 040 | surgical | edit_v9_w10_22.npz | E_ref=0.14, e_clean=0.006, DoF=24893 | E_ref=0.155, e_clean=0.0059, d_acc=+0.0010, d_nll=+0.0004, rho_P_ref=0.951, gen_drift=0.131, DoF=24893, pass=True |
| 041 | surgical | edit_v10_full_nocls.npz | E_ref=0.2, e_clean=0.0063, DoF=34365 | E_ref=0.453, e_clean=0.0063, d_acc=+0.0010, d_nll=+0.0004, rho_P_ref=0.954, gen_drift=0.077, DoF=34365, pass=True |
| 042 | surgical | edit_v11_hm95.npz | E_ref=0.13, e_clean=0.006, DoF=28000 | E_ref=0.154, e_clean=0.0063, d_acc=+0.0010, d_nll=+0.0005, rho_P_ref=0.951, gen_drift=0.068, DoF=26717, pass=True |
| 043 | surgical | edit_v11_hm90.npz | E_ref=0.16, e_clean=0.006, DoF=25000 | E_ref=0.153, e_clean=0.0062, d_acc=+0.0010, d_nll=+0.0005, rho_P_ref=0.950, gen_drift=0.063, DoF=23453, pass=True |
| 044 | surgical | edit_v11_hm75.npz | E_ref=0.2, e_clean=0.006, DoF=16349 | E_ref=0.160, e_clean=0.0056, d_acc=+0.0010, d_nll=+0.0002, rho_P_ref=0.950, gen_drift=0.047, DoF=16349, pass=True |
| 045 | surgical | edit_v11_hm60.npz | E_ref=0.27, e_clean=0.0058, DoF=11837 | E_ref=0.169, e_clean=0.0054, d_acc=+0.0010, d_nll=+0.0003, rho_P_ref=0.951, gen_drift=0.040, DoF=11837, pass=True |
| 046 | surgical | edit_v12_hm60_c90.npz | E_ref=0.18, e_clean=0.0055, DoF=9789 | E_ref=0.168, e_clean=0.0054, d_acc=+0.0010, d_nll=+0.0003, rho_P_ref=0.951, gen_drift=0.040, DoF=9789, pass=True |
| 047 | surgical | edit_v12_hm40_c90.npz | E_ref=0.23, e_clean=0.0054, DoF=5597 | E_ref=0.169, e_clean=0.0052, d_acc=+0.0015, d_nll=+0.0003, rho_P_ref=0.951, gen_drift=0.031, DoF=5597, pass=True |
| 048 | surgical | edit_v13_hm20_c50.npz | E_ref=0.4, e_clean=0.0052, DoF=693 | E_ref=0.281, e_clean=0.0052, d_acc=+0.0015, d_nll=+0.0002, rho_P_ref=0.951, gen_drift=0.028, DoF=693, pass=True |
| 049 | surgical | edit_v13_full_c90.npz | E_ref=0.103, e_clean=0.0062, DoF=35381 | E_ref=0.099, e_clean=0.0063, d_acc=+0.0010, d_nll=+0.0004, rho_P_ref=0.954, gen_drift=0.077, DoF=35381, pass=True |
| 050 | surgical | edit_v13_hm40_c90.npz | E_ref=0.169, e_clean=0.0052, d_acc=0.0015, DoF=4549 | E_ref=0.169, e_clean=0.0052, d_acc=+0.0015, d_nll=+0.0003, rho_P_ref=0.951, gen_drift=0.031, DoF=4565, pass=True |
Notes on non-surgical passes:
- 001-005, 007, 021, 029: rule discovery and validation (see section 1).
- 006, 011-017: original activations used for statistics.
- 022/023: v3 leak about 0.01 S at any photon pT; T=300 leaked more (0.013).
- 024-026: localisation. The leak is present at block 0 in particles (kNN). Classifier block-3 head violating its margin found.
- Prediction quality: rule thresholds were predicted to within 0.01 in log units. The first circuit's clean damage was badly underestimated (v1). Surgical E_ref predictions were typically within +-0.05, except for the window/partial-depth variants.

## 7. Passes used
50 of 50 (reference: 001, 003, 004, 007, 021; original logits: 002, 005, 029; captures: 006, 011-017, 024-026; edited logits: 022, 023; surgical: 008-010, 018-020, 027, 028, 030-050).
