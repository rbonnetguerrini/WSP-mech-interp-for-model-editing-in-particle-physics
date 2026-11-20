# Final notes: the JetClass reconstruction rule compiled into PET2 (`pretrain_s`)

## Summary

| final | scorer DoF | DoF, rank at abs. threshold 1e-3 | E_rule (practice) | e_clean | d_acc | worst class drop | d_nll | gen_drift max |
|---|---|---|---|---|---|---|---|---|
| final_small | 1985 | 3759 | 0.1060 | 0.0143 | +0.0011 | 0.0088 | +0.0004 | 0.028 |
| final_mid | 6063 | 7837 | 0.0711 | 0.0085 | -0.0002 | 0.0029 | +0.0012 | 0.030 |
| final_large | 13456 | 15230 | 0.0578 | 0.0062 | +0.0008 | 0.0020 | +0.0005 | 0.029 |

All three pass every practice limit (`gate_rule_no_g2 = True`). All rho values are within [0.98, 1.02] for D, J and P_reco. The generator AUC itself was not measurable here; see "Generator" below.

**DoF caveat.** The scorer counts singular values above 1e-4 x the largest one. The time-gate entries in block-0 `in_proj`/`out_proj` are very large (about 1e6), so the other changes in those two tensors fall below that relative threshold. The scorer counts rank 1 for `in_proj` (functionally rank 4: two gate heads, the key column and the regression) and rank 1 for `out_proj` (functionally rank 2). I did not enlarge any entry to hide rank. These magnitudes are what the gate needs, because block-0 `norm1.weight[4]` is 8e-5. The middle column gives the DoF with an absolute threshold, for a fair comparison.

## The circuit

Notation: C = 70 is the repurposed residual channel ("flag"). g = 4 is the time-indicator channel. M = 2000.

### 1. Detection, exact, at the embedding (pid table + 3 idle embedding units)
- A token is a sum: embed-MLP(kinematics) + local(kNN) + pid_embed[pid] + add_embed(IP). No layer before body block 0 combines pid and kinematics nonlinearly. A sum of a kinematic term and a pid term cannot form an AND without changing kept tokens.
- I therefore give up one residual channel (C = 70). Of all 128 channels it carries the least information that later layers read: its normed value is nearly saturated, with a lost read-variance score of 2.0 against a median of 8.8. In this channel every token gets a large signed value, and the AND is formed by sign/saturation.
- Three embedding-MLP hidden units were dead (fc1 = 0, norm weight about 1e-6): 76, 74 and 200. I rewrote each as an exact step function of one kinematic input. `z = G (n_thr - n) - 8`, where n = 1.27 tanh(0.125 log_pt) or 1.28 tanh(0.125 log_E). GELU is exactly 0 at or above threshold, and tanh(1.3 GELU) is exactly 1 at threshold - 0.002 in log. Outputs:
  u1 = [pT < 0.8], u2 = [E < 1.8], u3 = [E < 2.2]. Each unit writes only into channel C, with weights M, M and 2M (norm weight 1e-4, fc2 = a/1e-4, so leakage into other channels is below 1e-6).
- pid_embed[:, C] holds a per-pid base: charged (0,1,2,4,5,8) -4.5M, photon -3.5M, neutral hadron -1.5M. The summary tokens are set to -M/2.
- Result: x_C = base + M u1 + M u2 + 2M u3.
  - Photon: drop iff u1 = u2 = 1, giving +0.5M. Every kept photon state gives at most -0.5M.
  - Neutral hadron: drop iff u3 = 1, giving at least +0.5M. Kept gives at most -0.5M.
  - Charged: never above -0.5M.

  Every later read of C goes through a tanh norm, so x_C = +-1000..5000 is exactly +-1 after the norm. An offline check on 82,924 tokens gave 0 errors outside the 0.002-log transition band (2 tokens inside it).
- The time token is exactly zero at t = 0 (MPFourier(0) = 0 and time_embed has no biases), so it is masked during classification. For generation, an idle `time_embed` unit (96) fires for every t > 0.002 and writes -M/2 into C and +1000 into g.

### 2. Removing the token's influence: key suppression in all 10 attention layers
- Kept tokens all have the same value in channel C after every norm, so a rank-1 change to W_k along column C, with Δ_h = -β_h û_h per head, only adds a constant per query to kept keys. That leaves the softmax exactly unchanged. Flagged tokens get a score change of -0.5 |w| β (q·û).
- û_h is a closed-form direction (mean or ridge-whitened mean) along which every query projects positively. It was computed from captured queries: body blocks 0-7 (block 0 from its exact input, captured with block-0 attention and MLP zeroed), and the summary-token queries of both classifier blocks. β_h is set for a 20-nat drop at the 0.1% query quantile.
- Channel C is mean-matched in every body norm: norm weight[C] = -E[original normed value], so the frozen value equals the original average. Without this the clean damage broke the limits (pass 004). body.norm.weight[C] = 0.497 (the original x_body mean).

### 3. Time gate, so the generator is unaffected
- Under diffusion noise many neutrals look sub-threshold and would be flagged (gen_drift 0.15 at t = 0.85 without the gate; pass 007 isolates this).
- Channel g = 4, which block-0 attention does not read (norm1 weight 8e-5), is saturated at -1000 for every non-time token (pid table and summary token) and at +1000 for the time token.
- In block-0 heads 3 and 5, along the value direction e that out_proj uses least (smallest singular value, 0.034), the value becomes ε n_g / w_g, so it is exactly -ε for every non-time token. out_proj maps e only to channel C, with gain -G/ε (G = 1e7 split over the two heads). That writes G (1 - 2 a_time) into C. The constant +G is cancelled by block-0 MLP `fc2.bias[C] -= G`.
- At t = 0 the time token is masked, so the net change is exactly 0 (flag preserved; fp32 confirmed by probe pass 008). At t > 0, every token gets -2G a_time, which clears the flag on noise-flagged tokens from block 1 on.
- Channel g is restored after block 0 by two idle block-0 MLP units (102: +1000 for non-time tokens, 247: -1000 for the time token). The block-0 MLP read of g is mean-matched or regressed.
- gen_drift returned to the no-flag level: 0.018 (pass 009), against 0.15 without the gate.

### 4. Compensating the frozen channel (mid and large)
- Freezing C removes the variation of C at each read site. I fitted, in closed form, a ridge least-squares regression of the original normed C (block-0 MLP: normed g) on all other normed channels, using captured clean activations. I then folded it into the read matrix: W[:, c] += W[:, C] β_c, with norm weight[C] = -(intercept).
- That is one rank-1 change per read matrix. The residual std drops from 0.30 to 0.013 (block-0 attention), 0.17 to 0.012 (block 1), and 0.32 to 0.15 (block-7 MLP).
- Sites used:
  - small: block-0 attention only;
  - mid: block-0 attention and MLP (g), block-1 attention and MLP, block-2/3 MLP, block-7 attention and MLP;
  - large: all 16 body read sites plus both classifier attention layers.

### Why kept tokens are processed as before
- Kept tokens never cross zero in C (margin of at least 1000, against O(10) contributions from other terms), so their keys get an exactly constant shift.
- No other residual channel of any kept token changes, except:
  - the information carried by C itself (mean-matched, or regressed in mid/large);
  - the block-0 MLP read of channel 4 (same treatment);
  - one value direction in each of two block-0 heads, removed (out_proj singular value 0.034).
- The embedding units are exactly 0 for kept tokens. Leakage from the 1e-4-scaled units is below 1e-6.
- The cost of that lost information shows as e_clean: 0.014 / 0.0085 / 0.0062 of S for small / mid / large.

### What remains (the error floor)
- Diagnosed with passes 030-031. The kNN local-feature block (K = 10 in (Δη, Δφ)) sees only kinematic differences and pair features, never pid, so it cannot tell a sub-threshold neutral from a soft charged track. reco would also need the 11th neighbour.
- The dropped particle's ~10 neighbours therefore keep contaminated local features: median relative change of their block-0 output 0.37, against 0.015 for all other tokens.
- This is the main residual error on `P` and on my own unseen probes. It is out of reach of any weight edit that keeps tokens like charged soft tracks exact.

## Unseen-probe check (my own set, 4 x 512 jets, logits passes 025-029; e in units of S = 18.0)

| group | e_orig | small | mid | large | ratio (large) |
|---|---|---|---|---|---|
| one soft neutral hadron, E in [0.5, 2.2) GeV | 0.189 | 0.024 | 0.020 | 0.019 | 0.10 |
| three photons at 1e-3 of jet pT | 0.300 | 0.038 | 0.035 | 0.035 | 0.12 |
| one photon just below threshold | 0.248 | 0.029 | 0.026 | 0.025 | 0.10 |
| kept photon + kept neutral hadron just above threshold (reco = x) | 0 | 0.019 | 0.011 | 0.0085 | - |

The rule generalises to neutral hadrons and to several particles at once. My probes place particles uniformly in area within dR < 0.4, which gives more kNN contamination than the visible probes. Hence the ratio of ~0.1 here against ~0.03 on soft1e-3. The tails all come from the kNN leak.

## Per-final tensors, ranks, changed entries, DoF, and practice metrics

### final_small (= work/edit_s2.npz, oracle pass 016)

Scorer DoF **1985** (oracle reported 1985); DoF if rank is counted with an absolute threshold 1e-3: 3759.

| tensor | shape | rank (scorer) | rank (abs 1e-3) | changed entries | DoF |
|---|---|---|---|---|---|
| body.embed.mlp.fc1.weight | (256, 4) | 2 | 2 | 3 | 3 |
| body.embed.mlp.fc1.bias | (256,) | 1-D | 1-D | 3 | 3 |
| body.embed.mlp.norm.weight | (256,) | 1-D | 1-D | 3 | 3 |
| body.embed.mlp.fc2.weight | (128, 256) | 1 | 1 | 3 | 3 |
| body.pid_embed.0.weight | (9, 128) | 2 | 2 | 16 | 16 |
| body.token | (1, 4, 128) | 1 | 2 | 8 | 8 |
| body.time_embed.fc1.weight | (256, 128) | 1 | 1 | 1 | 1 |
| body.time_embed.norm.weight | (256,) | 1-D | 1-D | 1 | 1 |
| body.time_embed.fc2.weight | (128, 256) | 1 | 1 | 2 | 2 |
| body.in_blocks.0.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.0.norm2.weight | (128,) | 1-D | 1-D | 2 | 2 |
| body.in_blocks.1.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.1.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.2.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.2.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.3.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.3.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.4.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.4.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.5.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.5.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.6.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.6.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.7.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.7.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.0.attn.in_proj_weight | (384, 128) | 1 | 4 | 48568 | 511 |
| body.in_blocks.1.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.2.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.3.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.4.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.5.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.6.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.7.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| classifier.in_blocks.0.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| classifier.in_blocks.1.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.0.mlp.fc2.bias | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.0.attn.out_proj.weight | (128, 128) | 1 | 2 | 4089 | 255 |
| body.in_blocks.0.mlp.fc1.weight | (256, 128) | 1 | 1 | 2 | 2 |
| body.in_blocks.0.mlp.fc1.bias | (256,) | 1-D | 1-D | 2 | 2 |
| body.in_blocks.0.mlp.norm.weight | (256,) | 1-D | 1-D | 2 | 2 |
| body.in_blocks.0.mlp.fc2.weight | (128, 256) | 1 | 1 | 2 | 2 |
| body.norm.weight | (128,) | 1-D | 1-D | 1 | 1 |

Practice metrics: E_rule **0.1060**; e_v (edited / original): soft1e-3 0.0156/0.2486, soft1e-4 0.0154/0.2646, P 0.0200/0.1016; unchanged inputs: clean 0.0143, split2 0.0163, soft3e-3 0.0156. d_acc +0.0011, worst class drop 0.0088, d_nll +0.0004, KL 0.0012; rho D 0.996, J 1.002, P_reco 1.017 (raw P 0.744); gate_rule_no_g2 True; gen_drift t0.2/0.5/0.85 = 0.0125/0.0129/0.0284; R_vis 0.535, R_in 1.041, R_sub 0.028; dir_err D 0.071, J 0.061, P 0.334.

### final_mid (= work/edit_m1.npz, oracle pass 015)

Scorer DoF **6063** (oracle reported 6063); DoF if rank is counted with an absolute threshold 1e-3: 7837.

| tensor | shape | rank (scorer) | rank (abs 1e-3) | changed entries | DoF |
|---|---|---|---|---|---|
| body.embed.mlp.fc1.weight | (256, 4) | 2 | 2 | 3 | 3 |
| body.embed.mlp.fc1.bias | (256,) | 1-D | 1-D | 3 | 3 |
| body.embed.mlp.norm.weight | (256,) | 1-D | 1-D | 3 | 3 |
| body.embed.mlp.fc2.weight | (128, 256) | 1 | 1 | 3 | 3 |
| body.pid_embed.0.weight | (9, 128) | 2 | 2 | 16 | 16 |
| body.token | (1, 4, 128) | 1 | 2 | 8 | 8 |
| body.time_embed.fc1.weight | (256, 128) | 1 | 1 | 1 | 1 |
| body.time_embed.norm.weight | (256,) | 1-D | 1-D | 1 | 1 |
| body.time_embed.fc2.weight | (128, 256) | 1 | 1 | 2 | 2 |
| body.in_blocks.0.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.0.norm2.weight | (128,) | 1-D | 1-D | 2 | 2 |
| body.in_blocks.1.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.1.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.2.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.2.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.3.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.3.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.4.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.4.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.5.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.5.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.6.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.6.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.7.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.7.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.0.attn.in_proj_weight | (384, 128) | 1 | 4 | 48568 | 511 |
| body.in_blocks.0.mlp.fc1.weight | (256, 128) | 2 | 2 | 32254 | 764 |
| body.in_blocks.1.attn.in_proj_weight | (384, 128) | 2 | 2 | 48896 | 1020 |
| body.in_blocks.1.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.2.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.3.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.7.attn.in_proj_weight | (384, 128) | 2 | 2 | 48896 | 1020 |
| body.in_blocks.7.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.2.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.3.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.4.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.5.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.6.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| classifier.in_blocks.0.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| classifier.in_blocks.1.attn.in_proj_weight | (384, 128) | 1 | 1 | 128 | 128 |
| body.in_blocks.0.mlp.fc2.bias | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.0.attn.out_proj.weight | (128, 128) | 1 | 2 | 4089 | 255 |
| body.in_blocks.0.mlp.fc1.bias | (256,) | 1-D | 1-D | 2 | 2 |
| body.in_blocks.0.mlp.norm.weight | (256,) | 1-D | 1-D | 2 | 2 |
| body.in_blocks.0.mlp.fc2.weight | (128, 256) | 1 | 1 | 2 | 2 |
| body.norm.weight | (128,) | 1-D | 1-D | 1 | 1 |

Practice metrics: E_rule **0.0711**; e_v (edited / original): soft1e-3 0.0098/0.2486, soft1e-4 0.0099/0.2646, P 0.0138/0.1016; unchanged inputs: clean 0.0085, split2 0.0107, soft3e-3 0.0090. d_acc -0.0002, worst class drop 0.0029, d_nll +0.0012, KL 0.0005; rho D 0.996, J 0.992, P_reco 0.988 (raw P 0.723); gate_rule_no_g2 True; gen_drift t0.2/0.5/0.85 = 0.0132/0.0147/0.0295; R_vis 0.523, R_in 1.019, R_sub 0.027; dir_err D 0.047, J 0.047, P 0.343.

### final_large (= work/edit_v5.npz, oracle pass 013)

Scorer DoF **13456** (oracle reported 13456); DoF if rank is counted with an absolute threshold 1e-3: 15230.

| tensor | shape | rank (scorer) | rank (abs 1e-3) | changed entries | DoF |
|---|---|---|---|---|---|
| body.embed.mlp.fc1.weight | (256, 4) | 2 | 2 | 3 | 3 |
| body.embed.mlp.fc1.bias | (256,) | 1-D | 1-D | 3 | 3 |
| body.embed.mlp.norm.weight | (256,) | 1-D | 1-D | 3 | 3 |
| body.embed.mlp.fc2.weight | (128, 256) | 1 | 1 | 3 | 3 |
| body.pid_embed.0.weight | (9, 128) | 2 | 2 | 16 | 16 |
| body.token | (1, 4, 128) | 1 | 2 | 8 | 8 |
| body.time_embed.fc1.weight | (256, 128) | 1 | 1 | 1 | 1 |
| body.time_embed.norm.weight | (256,) | 1-D | 1-D | 1 | 1 |
| body.time_embed.fc2.weight | (128, 256) | 1 | 1 | 2 | 2 |
| body.in_blocks.0.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.0.norm2.weight | (128,) | 1-D | 1-D | 2 | 2 |
| body.in_blocks.1.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.1.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.2.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.2.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.3.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.3.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.4.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.4.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.5.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.5.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.6.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.6.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.7.norm1.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.7.norm2.weight | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.0.attn.in_proj_weight | (384, 128) | 1 | 4 | 48568 | 511 |
| body.in_blocks.0.mlp.fc1.weight | (256, 128) | 2 | 2 | 32254 | 764 |
| body.in_blocks.1.attn.in_proj_weight | (384, 128) | 2 | 2 | 48896 | 1020 |
| body.in_blocks.1.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.2.attn.in_proj_weight | (384, 128) | 2 | 2 | 48896 | 1020 |
| body.in_blocks.2.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.3.attn.in_proj_weight | (384, 128) | 2 | 2 | 48895 | 1020 |
| body.in_blocks.3.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.4.attn.in_proj_weight | (384, 128) | 2 | 2 | 48896 | 1020 |
| body.in_blocks.4.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.5.attn.in_proj_weight | (384, 128) | 2 | 2 | 48896 | 1020 |
| body.in_blocks.5.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.6.attn.in_proj_weight | (384, 128) | 2 | 2 | 48886 | 1020 |
| body.in_blocks.6.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| body.in_blocks.7.attn.in_proj_weight | (384, 128) | 2 | 2 | 48896 | 1020 |
| body.in_blocks.7.mlp.fc1.weight | (256, 128) | 1 | 1 | 32512 | 383 |
| classifier.in_blocks.0.attn.in_proj_weight | (384, 128) | 2 | 2 | 48895 | 1020 |
| classifier.in_blocks.1.attn.in_proj_weight | (384, 128) | 2 | 2 | 48896 | 1020 |
| body.in_blocks.0.mlp.fc2.bias | (128,) | 1-D | 1-D | 1 | 1 |
| body.in_blocks.0.attn.out_proj.weight | (128, 128) | 1 | 2 | 4089 | 255 |
| body.in_blocks.0.mlp.fc1.bias | (256,) | 1-D | 1-D | 2 | 2 |
| body.in_blocks.0.mlp.norm.weight | (256,) | 1-D | 1-D | 2 | 2 |
| body.in_blocks.0.mlp.fc2.weight | (128, 256) | 1 | 1 | 2 | 2 |
| body.norm.weight | (128,) | 1-D | 1-D | 1 | 1 |

Practice metrics: E_rule **0.0578**; e_v (edited / original): soft1e-3 0.0077/0.2486, soft1e-4 0.0079/0.2646, P 0.0114/0.1016; unchanged inputs: clean 0.0062, split2 0.0086, soft3e-3 0.0064. d_acc +0.0008, worst class drop 0.0020, d_nll +0.0005, KL 0.0003; rho D 1.002, J 1.002, P_reco 0.992 (raw P 0.726); gate_rule_no_g2 True; gen_drift t0.2/0.5/0.85 = 0.0129/0.0145/0.0288; R_vis 0.524, R_in 1.021, R_sub 0.027; dir_err D 0.031, J 0.032, P 0.339.

## Predictions against outcomes (all surgical passes)

| pass | edit | prediction | outcome |
|---|---|---|---|
| 004 | v1: flag + key suppression, no mean-match | E_rule 0.30, d_acc -0.003, d_nll 0.01, rho 1.0, gen 0.05, DoF 1450 | E_rule 0.412, d_acc -0.0074, class drop 0.045 (FAIL), d_nll 0.0245, rho_J 0.82 (FAIL), rho_P_reco 0.74, gen 0.161, DoF 1444. Rule probes near perfect (soft1e-3 ratio 0.056); clean damage dominated |
| 006 | v2: + mean-matched norms | E_rule 0.20, e_clean 0.04, class drop 0.015, rho_J 0.95, gen 0.12 | E_rule 0.170, e_clean 0.027, class drop 0.011, rho_J 0.999, gen 0.152 (pass) |
| 007 | v2 with the flag silenced (control) | E_rule 0.95, e_clean 0.027, gen 0.05 | E_rule 1.063, e_clean 0.027, gen 0.018 (drift comes from noise-flagging) |
| 009 | v3: + time gate, no block-0 suppression | E_rule 0.19, e_clean 0.029, gen 0.025 | E_rule 0.181, e_clean 0.027, gen 0.018 |
| 012 | v4: + regression at all sites | E_rule 0.10, e_clean 0.013, e_P 0.018, gen 0.012, DoF 13000 | E_rule 0.077, e_clean 0.0062, e_P 0.0169, gen 0.026, DoF 13455 |
| 013 | v5 (= final_large): gate constant moved to block-0 MLP bias, block-0 suppression back | E_rule 0.06, e_P 0.013, gen 0.03 | E_rule 0.0578, e_P 0.0114, gen 0.029 |
| 014 | s1: v5 without regression | E_rule 0.16, DoF 2100 | E_rule 0.170, DoF 1985 |
| 015 | m1 (= final_mid) | E_rule 0.08, e_clean 0.009, DoF 6800 | E_rule 0.0711, e_clean 0.0085, DoF 6063 |
| 016 | s2 (= final_small): s1 + block-0 attention regression | E_rule 0.12, e_clean 0.018, DoF 2900 | E_rule 0.106, e_clean 0.0143, DoF 1985 (scorer rank hides the regression; 3759 at abs. threshold) |
| 017 | v6: v5 with suppression target 40 | E_rule 0.056, e_P 0.011 | E_rule 0.060, e_P 0.012 (no gain, so the leftover is not weak suppression) |
| 018 | L2: v5 without block-2..6 attention regressions | E_rule 0.06, DoF 9000 | E_rule 0.0623, DoF 8996 |
| 019 | M2: s2 + 5 MLP-read regressions | E_rule 0.085, DoF 3900 | E_rule 0.102, DoF 4279 |
| 020 | M3: s2 + block-1/7 attention regressions | E_rule 0.08, DoF 3770 | E_rule 0.094, DoF 3769 |
| 021 | M4: M3 + classifier regression | E_rule 0.075, DoF 5550 | E_rule 0.091, DoF 5553 |
| 022 | S3: suppression only in blocks 0-3 + classifier | E_rule 0.13, DoF 1470 | E_rule 0.425 (the shortcut is read in blocks 4-7) |
| 023 | S4: no suppression in blocks 0-1 | E_rule 0.12 | E_rule 0.130, DoF 1857 |
| 024 | S6: no classifier suppression | E_rule 0.25 | E_rule 0.249 |
| 027-029 | finals on unseen probes | drop groups 0.017 / 0.011 / 0.008 | 0.024-0.038 / 0.020-0.035 / 0.019-0.035 (under-predicted: kNN leak) |
| 008 | fp32 probe | ch6 = 1.0 (fp32) | 1.0, so matmuls are full fp32 and large-dynamic-range constructions are safe |

Overall the predictions were too pessimistic about the effect of mean-matching and regression, and too optimistic about the unseen probes (I underrated the kNN leak).

## Ideas that failed or were dropped
- **Exact masking through the body's padding mask** (making embedded channel 2 exactly 0 for dropped tokens). This would have masked the token everywhere downstream. It needs an exact float cancellation that depends on both pid and kinematics, which is impossible additively, and absorption tricks lose kept tokens' precision. Rejected at the reading stage.
- **Detection in the block-0 MLP from the natural representation.** A least-squares read-out only reached 99.6% token accuracy (300 errors in 83k tokens) and has a fuzzy threshold. The embedding-level exact AND replaced it.
- **Freezing a channel without mean-matching (v1).** It broke the class-drop and rho_J limits.
- **No time gate (v2).** gen_drift was 0.15. A first gate design with a constant offset in the embedding hid the flag from block 0 (v3/v4: no block-0 suppression); fixed in v5.
- **Stronger key suppression (v6)**: no gain. **Sparse (1-4 channel) regressions**: far weaker than full rank-1 (block 0: residual 0.18 against 0.013). **Classifier regression (M4)**: small gain for 1784 DoF. **Dropping suppression in blocks 4-7 or in the classifier**: large losses.
- **Fixing the kNN local-feature leak**: impossible without pid in the local block (see above).

## Generator
- gen_drift (proxy) is 0.028-0.030 for the finals. Freezing channel C with no flag gives 0.018; the original model is 0.
- The extra drift comes from the regressions (fitted on t = 0 activations) and from block-0 suppression of noise-flagged tokens before the gate clears them.
- I could not measure the real-vs-generated AUC.

## Passes used
31 of 50: 001-031. That is 9 captures (including a block-0-input probe, an fp32 probe and 2 diagnostic captures), 5 logits passes for the unseen-probe check, and 17 surgical passes. The zero-pass edit (null) was never submitted.
