# Final notes: the JetClass reconstruction rule compiled into pretrain_l

Passes used: **47 of 50** (36 surgical/logits/capture passes after the zero pass; see table). No gradients and no
iterative optimisers were used. All weights were built in closed form: least squares/ridge on captured activations,
pinv, eigenvectors and sign patterns, and rank-1 algebra on single weight matrices.

## The three final edits

Original reference e_orig: soft1e-3 0.3734, soft1e-4 0.1055, P 0.1511. All three edits pass every practice limit
(`gate_rule_no_g2 = true`).

| file | idea | DoF | E_rule | e soft1e-3 / soft1e-4 / P | e clean / split2 / soft3e-3 | d_acc | worst class drop | d_nll | rho D / J / P_reco | gen_drift (max) |
|---|---|---|---|---|---|---|---|---|---|---|
| final_small.npz | relabel photons only | 2,050 | 0.2557 | 0.0358 / 0.0268 / 0.0630 | 0.0002 / 0.0002 / 0.0002 | 0.000 | 0.000 | 0.00000 | 1.000 / 1.000 / 0.980 | 0.208 |
| final_mid.npz | relabel photons + neutral hadrons | 4,100 | 0.2355 | 0.0358 / 0.0268 / 0.0539 | 0.0003 / 0.0002 / 0.0002 | 0.000 | 0.000 | 0.00001 | 1.000 / 1.000 / 0.980 | 0.293 |
| final_large.npz | removal (saturate + zero value + suppress key) | 122,855 | 0.1840 | 0.0199 / 0.0226 / 0.0430 | 0.0018 / 0.0017 / 0.0019 | 0.000 | 0.005 | -0.0006 | 1.008 / 1.011 / 0.965 | 0.268 |

R_vis / R_in / R_sub: large 0.576 / 1.016 / 0.136; mid 0.592 / 1.002 / 0.181; small 0.591 / 1.002 / 0.181.

### Tensors changed (rank, changed entries, DoF)
- **final_small**: `body.in_blocks.0.mlp.fc1.weight` (rank 1, 1024 entries, DoF 1024), `...fc1.bias` (1),
  `...fc2.weight` (rank 1, 1024, DoF 1024), `...mlp.norm.weight` (1). Total **2,050**.
- **final_mid**: the same four tensors with two units: fc1.weight (rank 2, 2048, DoF 2048), fc1.bias (2),
  fc2.weight (rank 2, 2048, DoF 2048), norm.weight (2). Total **4,100**.
- **final_large**:
  - block-0 MLP: fc1.weight (rank 2, 2048), fc1.bias (2), fc2.weight (rank 1, 2048), norm.weight (2), for 4,100 DoF.
  - `body.in_blocks.{1..27}.attn.in_proj_weight` and `classifier.in_blocks.{0,1}.attn.in_proj_weight`: each rank 1
    (all entries of the K/V rows change), DoF 3072+1024-1 = 4,095 each, 29 × 4,095 = 118,755.
  - Total **122,855**.
- No `generator.*` tensor is touched.

## The circuit

### Where the network can see "sub-threshold neutral"
- The body input token is `x0 = embed(kin) + local(kNN kin) + pid_embed[pid] + add_embed(ip)`. Only `pid_embed`
  knows the pid; embed, local and interaction see only kinematics.
- Exact masking (channel 2 → 0) is impossible:
  - The padding mask is `x0[:,:,2] != 0`, a sum that is separable in (kinematics, pid).
  - A photon and a neutral hadron with the same (pt<0.8, 1.8<E<2.2) kinematics differ only by a constant pid row,
    yet the rule drops one and keeps the other.
  - Soft charged particles share the kinematic term.
- The first nonlinearity that sees pid and kinematics together is **block 0's MLP**. It has ~211 effectively dead
  hidden units (`mlp.norm.weight` < 1e-3); I reuse units 1042 (photon) and 683 (neutral hadron).
- **Detector.** `u = norm2(x)` is recovered exactly from captured `mlp_pre` via `u = pinv(fc1)(pre - b1)`. On 51k
  tokens (constructed jets with photons/NH/charged inserted around the thresholds, plus P jets), I fit by weighted
  ridge least squares a readout `m̂(u)` of the signed log-margin:
  - photons: `min(log 0.8 - log pt, log 1.8 - log E)`; NH: `log 2.2 - log E`;
  - every other token gets the target −1.5, which encodes the pid gate.

  The unit is then `z = 300 (m̂(u) − bias)`; the bias (−0.027 for photons, −0.056 for NH) is the mean readout error
  near threshold. Held-out precision near threshold is σ ≈ 0.05 in log-margin. No non-member token fires: z < −360 for
  all charged/kept tokens, so GELU(z) is exactly 0 and those tokens are bit-identical after block 0.

### How the influence is removed (final_large)
1. **Saturating flag.** A firing unit writes `F s` (F = 1e4, s a fixed ±1 pattern) into the residual. From block 1
   on, every norm of that token is exactly `n*_l = w_l ⊙ s` (tanh saturates). So its key and value in every later
   attention are known constants `k*_l = W_k n*_l` and `v*_l = W_v n*_l`. This also holds for `body.norm`:
   `x* = w_bn ⊙ s`.
2. **Read direction.** `b_l = (M_l + λI)^{-1} n*_l / (n*_l^T (M_l+λI)^{-1} n*_l)`, where `M_l` is the second moment of
   normal tokens' `norm1` input, computed from captures. Then `b·n* = 1`, and `b·n` for normal tokens is 1e-4 (block
   1) to 6e-3 (block 27) held out.
3. **Zero value**, in all 27 body blocks and both classifier token-attention blocks:
   `ΔW_v = −v*_l b_l^T`. The flagged token contributes no content.
4. **Suppress key**, in blocks 1–14 only, where leakage is small:
   `ΔW_k = (k_new − k*_l) b_l^T`, with `k_new,h = −κ_h g_h`.
   - `g_h` is the LDA direction of head h's queries, so `q_i · g_h > 0` for >99.9% of queries.
   - κ_h shifts the 1%-quantile query score by −10 (blocks 1–8) or −5 (blocks 9–14).
   - This removes the renormalisation error. Without it the saturated token was an attention sink: mean weight
     0.43 per head in block 1, measured with a known probe value.
5. **Why kept tokens are unaffected.** The detector output is exactly 0 for them. The in_proj edits act on them only
   through the leak `b·n_j` (≈1e-4–1e-3 in the edited blocks). Result: clean e = 0.0018, d_acc 0.000.

### Relabel circuit (final_small / final_mid)
- Same detector units. A firing unit writes `1.5 (P_8 − P_6)` (photon) or `1.5 (P_8 − P_7)` (neutral hadron) into
  the residual: the sub-threshold neutral is turned into a soft charged hadron (pid code 8).
- The shortcut is neutral-specific: soft charged hadrons are normal in JetClass, so the model processes the relabelled
  token almost as if it were absent.
- Kept tokens see exactly 0 from these units, so clean e is only 0.0002–0.0003.
- Target and scale were chosen by tests:

  | target | E_rule |
  |---|---|
  | code 8, ×1.0 | 0.251 |
  | code 8, ×1.5 | **0.236** |
  | code 8, ×2.0 | 0.264 |
  | code 0 | 0.50 |
  | code 1 | 0.55 |

### What cannot be fixed (the floor)
- The flag exists only after block 0's MLP, so others still see the dropped particle in block-0 attention.
- It also stays in the kNN lists of `local_physics`, which is pid-blind.
- In P-jets, per-jet logit analysis gives a floor of ≈0.16 of the original error for clearly dropped particles. The
  rest comes from detector errors within ~±0.05 of the thresholds.

## Prediction vs outcome
See the table below (copied from the ledger and responses). Summary: zero-pass and two "obvious" improvements
(lower threshold, new s pattern) failed, and I underpredicted how good relabelling would be (predicted 0.6, got
0.25).

| pass | kind | weights | prediction | outcome |
|---|---|---|---|---|
| 001 | surgical | zero_pass.npz | E_rule=0.9, d_acc=-0.003, d_nll=0.01, rho_P_reco=1.0, gen_drift=0.02 | E_rule=2.461, e_clean=0.4457, e_P=0.2951, e_s1e-3=0.4437, e_s1e-4=0.4475, d_acc=-0.2502, rho_P_reco=0.223, gen_drift=1.153, gate=False |
| 002 | capture | orig | lstsq_detector_error_logpt=0.1, n_dead_units_block0=211 | capture/logits ok |
| 003 | capture | orig | fisher_leak_rms=0.0005, frac_queries_outside_cone=0.001 | capture/logits ok |
| 004 | capture | orig | fisher_leak_rms=0.0005, frac_queries_outside_cone=0.001 | capture/logits ok |
| 005 | capture | orig | fisher_leak_rms=0.0005, frac_queries_outside_cone=0.001 | capture/logits ok |
| 006 | capture | orig | fisher_leak_rms=0.0005, frac_queries_outside_cone=0.001 | capture/logits ok |
| 007 | capture | orig | fisher_leak_rms=0.0005, frac_queries_outside_cone=0.001 | capture/logits ok |
| 008 | capture | orig | fisher_leak_rms=0.0005, frac_queries_outside_cone=0.001 | capture/logits ok |
| 009 | capture | orig | fisher_leak_rms=0.0005, frac_queries_outside_cone=0.001 | capture/logits ok |
| 010 | surgical | v1_VK.npz | E_rule=0.35, d_acc=-0.003, d_nll=0.01, e_clean=0.03, rho_P_reco=1.0, gen_drift=0.3 | E_rule=0.705, e_clean=0.0937, e_P=0.1372, e_s1e-3=0.1135, e_s1e-4=0.0951, d_acc=-0.0327, rho_P_reco=1.056, gen_drift=0.273, gate=False |
| 011 | surgical | v1_V.npz | E_rule=0.5, d_acc=-0.002, d_nll=0.005, e_clean=0.02, rho_P_reco=1.0, gen_drift=0.3 | E_rule=0.257, e_clean=0.0012, e_P=0.0632, e_s1e-3=0.0364, e_s1e-4=0.0269, d_acc=-0.0005, rho_P_reco=0.994, gen_drift=0.231, gate=True |
| 012 | surgical | v2_K12.npz | E_rule=0.22, d_acc=-0.002, e_clean=0.004, rho_P_reco=1.0, gen_drift=0.25 | E_rule=0.261, e_clean=0.0019, e_P=0.0693, e_s1e-3=0.0330, e_s1e-4=0.0248, d_acc=-0.0005, rho_P_reco=0.991, gen_drift=0.231, gate=True |
| 013 | surgical | v2_K12c.npz | E_rule=0.22, d_acc=-0.002, e_clean=0.004, rho_P_reco=1.0, gen_drift=0.25 | E_rule=0.248, e_clean=0.0023, e_P=0.0678, e_s1e-3=0.0313, e_s1e-4=0.0224, d_acc=-0.0005, rho_P_reco=0.991, gen_drift=0.231, gate=True |
| 014 | capture | orig | heldout_wrong_frac_margin_gt_0.05=0.003 | capture/logits ok |
| 015 | capture | orig | heldout_wrong_frac_margin_gt_0.05=0.003 | capture/logits ok |
| 016 | surgical | v3_V.npz | E_rule=0.23, d_acc=-0.001, e_clean=0.0015, e_P=0.055, rho_P_reco=0.995, gen_drift=0.25 | E_rule=0.209, e_clean=0.0011, e_P=0.0464, e_s1e-3=0.0252, e_s1e-4=0.0267, d_acc=-0.0005, rho_P_reco=0.964, gen_drift=0.268, gate=True |
| 017 | logits | orig | n=512 | capture/logits ok |
| 018 | logits | v3_V.npz | mean_err_ratio_vs_orig=0.4 | capture/logits ok |
| 019 | logits | orig | n=512 | capture/logits ok |
| 020 | surgical | v4_d04.npz | E_rule=0.19, d_acc=-0.001, e_clean=0.002, e_P=0.04, rho_P_reco=0.95, gen_drift=0.28 | E_rule=0.231, e_clean=0.0023, e_P=0.0542, e_s1e-3=0.0274, e_s1e-4=0.0275, d_acc=-0.0010, rho_P_reco=0.924, gen_drift=0.274, gate=True |
| 021 | surgical | v4_d04c.npz | E_rule=0.18, d_acc=-0.001, e_clean=0.0025, e_P=0.041, rho_P_reco=0.95, gen_drift=0.28 | E_rule=0.231, e_clean=0.0039, e_P=0.0561, e_s1e-3=0.0272, e_s1e-4=0.0264, d_acc=-0.0015, rho_P_reco=0.930, gen_drift=0.274, gate=True |
| 022 | surgical | v4_dp03.npz | E_rule=0.205, d_acc=-0.0005, e_clean=0.0009, e_P=0.045, rho_P_reco=0.98, gen_drift=0.26 | E_rule=0.214, e_clean=0.0009, e_P=0.0477, e_s1e-3=0.0272, e_s1e-4=0.0266, d_acc=-0.0005, rho_P_reco=0.988, gen_drift=0.263, gate=True |
| 023 | capture | orig | top5_pc_variance_fraction_of_delta=0.6 | capture/logits ok |
| 024 | capture | orig | top5_pc_variance_fraction_of_delta=0.6 | capture/logits ok |
| 025 | capture | orig | n=384 | capture/logits ok |
| 026 | capture | v3_V.npz | rel_delta_summary=0.015 | capture/logits ok |
| 027 | capture | v5_probe1.npz | mean_weight=0.02 | capture/logits ok |
| 028 | surgical | v6_corr1.npz | E_rule=0.16, e_soft1e-4=0.017, e_soft1e-3=0.02, e_P=0.04, d_acc=-0.0005, e_clean=0.0012, rho_P_reco=0.965, gen_drift=0.3 | E_rule=0.209, e_clean=0.0011, e_P=0.0463, e_s1e-3=0.0252, e_s1e-4=0.0267, d_acc=-0.0005, rho_P_reco=0.965, gen_drift=0.268, gate=True |
| 029 | surgical | v8_K8.npz | E_rule=0.19, e_soft1e-4=0.022, e_P=0.045, d_acc=-0.0005, e_clean=0.0015, rho_P_reco=0.96, gen_drift=0.27 | E_rule=0.197, e_clean=0.0017, e_P=0.0445, e_s1e-3=0.0208, e_s1e-4=0.0253, d_acc=-0.0010, rho_P_reco=0.965, gen_drift=0.258, gate=True |
| 030 | surgical | v7_relabel.npz | E_rule=0.6, d_acc=-0.0005, e_clean=0.0005, rho_P_reco=0.98, gen_drift=0.05 | E_rule=0.251, e_clean=0.0002, e_P=0.0525, e_s1e-3=0.0379, e_s1e-4=0.0320, d_acc=0.0000, rho_P_reco=0.983, gen_drift=0.313, gate=True |
| 031 | surgical | v9_V.npz | E_rule=0.18, e_soft1e-3=0.02, e_soft1e-4=0.022, e_P=0.042, d_acc=-0.0005, e_clean=0.0012, rho_P_reco=0.965, gen_drift=0.27 | E_rule=0.430, e_clean=0.0040, e_P=0.0966, e_s1e-3=0.0556, e_s1e-4=0.0528, d_acc=-0.0010, rho_P_reco=0.966, gen_drift=0.675, gate=True |
| 032 | surgical | v9_K8.npz | E_rule=0.18, e_clean=0.0017, d_acc=-0.0007, rho_P_reco=0.965, gen_drift=0.27 | E_rule=0.402, e_clean=0.0040, e_P=0.0931, e_s1e-3=0.0519, e_s1e-4=0.0476, d_acc=-0.0005, rho_P_reco=0.968, gen_drift=0.674, gate=True |
| 033 | surgical | v7_rel0.npz | E_rule=0.25, e_clean=0.0002, d_acc=0.0, rho_P_reco=0.983, gen_drift=0.3 | E_rule=0.501, e_clean=0.0002, e_P=0.0925, e_s1e-3=0.1142, e_s1e-4=0.0615, d_acc=0.0000, rho_P_reco=0.987, gen_drift=0.307, gate=True |
| 034 | surgical | v7_rel1.npz | E_rule=0.3, e_clean=0.0002, d_acc=0.0, rho_P_reco=0.98, gen_drift=0.3 | E_rule=0.553, e_clean=0.0009, e_P=0.1150, e_s1e-3=0.1142, e_s1e-4=0.0625, d_acc=0.0000, rho_P_reco=0.990, gen_drift=0.172, gate=True |
| 035 | surgical | v7_rel8_dm04.npz | E_rule=0.23, e_clean=0.0004, d_acc=0.0, rho_P_reco=0.97, gen_drift=0.32 | E_rule=0.256, e_clean=0.0005, e_P=0.0549, e_s1e-3=0.0371, e_s1e-4=0.0321, d_acc=0.0000, rho_P_reco=0.947, gen_drift=0.319, gate=True |
| 036 | surgical | v10_V13K8.npz | E_rule=0.24, e_clean=0.0015, d_acc=-0.001, rho_P_reco=0.96, gen_drift=0.3 | E_rule=0.540, e_clean=0.0015, e_P=0.0469, e_s1e-3=0.0230, e_s1e-4=0.1315, d_acc=-0.0010, rho_P_reco=0.966, gen_drift=0.259, gate=True |
| 037 | surgical | v7_rel8_ph.npz | E_rule=0.28, e_soft1e-3=0.025, e_P=0.065, e_clean=0.0001, d_acc=0.0, rho_P_reco=0.99, gen_drift=0.2 | E_rule=0.271, e_clean=0.0002, e_P=0.0619, e_s1e-3=0.0379, e_s1e-4=0.0320, d_acc=0.0000, rho_P_reco=0.983, gen_drift=0.213, gate=True |
| 038 | surgical | v11_K14.npz | E_rule=0.19, e_clean=0.002, d_acc=-0.001, rho_P_reco=0.965, gen_drift=0.26 | E_rule=0.184, e_clean=0.0018, e_P=0.0430, e_s1e-3=0.0199, e_s1e-4=0.0226, d_acc=0.0000, rho_P_reco=0.965, gen_drift=0.268, gate=True |
| 039 | surgical | v12_K27.npz | E_rule=0.18, e_clean=0.0025, d_acc=-0.001, rho_P_reco=0.96, gen_drift=0.27 | E_rule=0.209, e_clean=0.0146, e_P=0.0528, e_s1e-3=0.0281, e_s1e-4=0.0213, d_acc=-0.0005, rho_P_reco=0.973, gen_drift=0.274, gate=True |
| 040 | capture | orig | heldout_rmse_near_photon=0.035, heldout_rmse_near_nh=0.05 | capture/logits ok |
| 041 | capture | orig | heldout_rmse_near_photon=0.035, heldout_rmse_near_nh=0.05 | capture/logits ok |
| 042 | capture | v13_base.npz | mean_weight_particles=0.05, explained_particle_gap=0.3 | capture/logits ok |
| 043 | capture | v13_probe.npz | mean_weight_particles=0.05, explained_particle_gap=0.3 | capture/logits ok |
| 044 | surgical | v14_K14s10.npz | E_rule=0.18, e_clean=0.002, d_acc=-0.0005, rho_P_reco=0.965, gen_drift=0.27 | E_rule=0.191, e_clean=0.0026, e_P=0.0450, e_s1e-3=0.0215, e_s1e-4=0.0229, d_acc=-0.0015, rho_P_reco=0.975, gen_drift=0.311, gate=True |
| 045 | surgical | v7_rel8_s15.npz | E_rule=0.26, e_clean=0.0003, d_acc=0.0, rho_P_reco=0.98, gen_drift=0.32 | E_rule=0.235, e_clean=0.0003, e_P=0.0539, e_s1e-3=0.0358, e_s1e-4=0.0268, d_acc=0.0000, rho_P_reco=0.980, gen_drift=0.293, gate=True |
| 046 | surgical | v7_rel8_s20.npz | E_rule=0.235, e_clean=0.0003, d_acc=0.0, rho_P_reco=0.975, gen_drift=0.3 | E_rule=0.264, e_clean=0.0004, e_P=0.0643, e_s1e-3=0.0373, e_s1e-4=0.0282, d_acc=0.0000, rho_P_reco=0.979, gen_drift=0.291, gate=True |
| 047 | surgical | v7_rel8_ph_s15.npz | E_rule=0.255, e_clean=0.0002, d_acc=0.0, rho_P_reco=0.985, gen_drift=0.2 | E_rule=0.256, e_clean=0.0002, e_P=0.0630, e_s1e-3=0.0358, e_s1e-4=0.0268, d_acc=0.0000, rho_P_reco=0.980, gen_drift=0.208, gate=True |

## Ideas that failed
- **Zero pass** (readouts from a weights-only linearisation of the embed): it fired on many kept tokens
  (E_rule 2.46, d_acc −0.25).
- **Exact masking via channel 2**: impossible by the separability argument above.
- **Key suppression in late blocks** (15–27) or with a larger shift in 9–14: leakage through `b·n` changes normal
  attention (clean e 0.015, E_rule 0.209 / 0.191). Key suppression from the zero-pass era with F=1e3 was also harmful
  (clean e 0.094).
- **Threshold shifts**: δ = −0.04 → 0.231 and δ = +0.03 → 0.214, against 0.209 at δ = 0. False flags of kept
  near-threshold neutrals cost more than estimated.
- **Choosing s to make the flagged key naturally unattractive** (sign of summed score functionals): the mean query's
  z-score was −20 in early blocks, but leakage rose and anti-saturated states got attractive keys. E_rule 0.43.
- **A constant correction value broadcast by the flagged token in block 1**, to cancel the block-0 perturbation:
  - it explained 73% of the summary-token gap but had zero effect on the output (0.2090 vs 0.2094);
  - with key = 0 it explained 0.2% of the particle-token gap.
- **Value zeroing only in blocks 1–13**: soft1e-4 blows up (e = 0.13), so every block is needed.
- **Second-stage detector at block 1**: only 15–20% sharper (σ 0.040 vs 0.049), not worth losing block-1 protection.
- **Relabel to pid codes 0 or 1**; relabel scale 2.0.

## Notes
- `gen_drift` is 0.21–0.29. Noised generator inputs contain many soft-looking neutrals, which the time-blind
  detector flags. At t = 0 the time token is exactly zero and masked, so I could not measure a time signal with
  captures (they run at t = 0) to gate on.
- Candidate and intermediate edits are in `run/work/` and `run/submissions/cand_*.npz`. The builders are
  `work/build3.py` / `build4.py` (removal), `work/mkrel2.py` / `mkrel3b.py` (relabel) and `work/det2.py` (detector fit).
