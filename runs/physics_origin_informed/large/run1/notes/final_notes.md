# Final notes: the JetClass reconstruction rule compiled into PET2 (pretrain_l)

## 1. Final edits

All three edits share the same circuit. They differ only in how the content of channel 2 is re-inserted in block 2 (Section 2, step 4).

| file | DoF | E_rule (practice) | clean e | d_acc | worst class drop | d_nll | rho D / J / P_reco | gen_drift max |
|---|---|---|---|---|---|---|---|---|
| final_small.npz | **362** | **0.104** | 0.0095 | -0.0020 | 0.0098 | +0.0028 | 1.008 / 1.032 / 0.999 | 0.271 |
| final_mid.npz | **1134** | **0.072** | 0.0035 | -0.0010 | 0.0049 | +0.0003 | 1.002 / 1.002 / 0.999 | 0.271 |
| final_large.npz | **2156** | **0.069** | 0.0031 | -0.0010 | 0.0049 | -0.0000 | 1.002 / 1.003 / 0.999 | 0.271 |

Per-probe e_v (edited / original):

| | soft1e-3 | soft1e-4 | P | clean | split2 | soft3e-3 |
|---|---|---|---|---|---|---|
| original e_orig | 0.3734 | 0.1055 | 0.1511 | 0 | 0 | 0 |
| small | 0.0100 | 0.0098 | 0.0292 | 0.0095 | 0.0084 | 0.0098 |
| mid | 0.0041 | 0.0041 | 0.0250 | 0.0035 | 0.0036 | 0.0040 |
| large | 0.0037 | 0.0036 | 0.0248 | 0.0031 | 0.0031 | 0.0035 |

Other reported values:
- R_vis / R_in / R_sub: small 0.542 / 1.054 / 0.030; mid 0.516 / 1.011 / 0.022; large 0.516 / 1.010 / 0.022.
- dir_err D / J / P: small 0.070 / 0.165 / 0.337; mid 0.026 / 0.047 / 0.329; large 0.026 / 0.049 / 0.329.
- rho_P (the raw P response, reported only) is 0.72 for all three, because the rule removes the softened neutrals.
- gate_rule_no_g2 is True for all three.

Held-out check: logits passes on 1500 clean jets (indices 600-749 of each class) that were never used for fitting.
- small: clean e 0.0093, d_acc -0.0007, worst class -0.013, d_nll +0.0025.
- mid: clean e 0.0031, d_acc 0.0000, worst class -0.0067, d_nll +0.0003.
- An earlier mid version (rx4_mid) on held-out soft1e-4 jets: e = 0.0043, the same as its clean floor.

### Changed tensors (the oracle's size report; the rank is the rank of the change)

The first 19 tensors are identical in the three edits. Only the last four rows of the table differ.

| tensor | rank | changed entries (small / mid / large) |
|---|---|---|
| body.embed.mlp.fc1.weight | 2 | 8 |
| body.embed.mlp.fc1.bias | 1-D | 2 |
| body.embed.mlp.norm.weight | 1-D | 2 |
| body.embed.mlp.fc2.weight | 1 | 2 |
| body.pid_embed.0.weight | 1 | 2 |
| body.in_blocks.{0,1,2}.norm1.weight | 1-D | 1 each |
| body.in_blocks.{0,1,2}.norm2.weight | 1-D | 1 each |
| body.in_blocks.0.mlp.fc1.weight / .bias | 1 / 1-D | 1 / 1 |
| body.in_blocks.0.mlp.norm.weight / fc2.weight | 1-D / 1 | 1 / 1 |
| body.in_blocks.1.mlp.fc1.weight | 1 | 66 |
| body.in_blocks.1.mlp.fc1.bias / norm.weight / fc2.weight | | 2 / 2 / 2 |
| body.in_blocks.2.mlp.fc1.weight | 1 | 258 / 1027 / 2049 |
| body.in_blocks.2.mlp.fc1.bias | 1-D | 2 / 3 / 3 |
| body.in_blocks.2.mlp.norm.weight | 1-D | 2 / 3 / 3 |
| body.in_blocks.2.mlp.fc2.weight | 1 | 2 / 3 / 3 |

DoF totals: small 362, mid 1134, large 2156.

No `generator.*` and no `classifier.*` tensor is changed.

## 2. The circuit

### Key fact used
`PET_body.forward` computes the padding mask once, as `x[:,:,2] != 0` on the summed embedding:

x = E(kin) + L(neighbours) + P(pid) + A(impact parameters)

A token whose summed channel 2 is exactly 0.0 is padding everywhere afterwards:
- it gets -1e9 in every body attention row and column;
- `x*mask` is applied before each norm1;
- the MLP outputs are masked;
- the final output is `norm(x)*mask = 0`;
- the classifier and generator masks (`body_out[:,:,2] != 0`) drop it.

This is exactly what reco() does to the token itself.

### Step 1: detection and exact zero (embedding, 16 DoF)
1. Two embed-MLP hidden units (1636 and 1970) are repurposed as exact step gates. Their original output range is negligible (‖fc2 col‖·range ≈ 0.01 and 0.02).
   - Their fc1 rows read only the tanh-normalised log pT or log E, with weight -2^20 and the bias at the threshold.
   - This gives a = [pT < 0.8] and b = [E < 1.8].
   - GELU and tanh saturate, so the output is exactly 0 or exactly 1, except in a band of about 1e-4 in log pT or log E around each threshold.
   - Their mlp norm weight is set to 2^-10 and fc2[2,·] to -2^29·2^10. Each gate therefore adds exactly -H (H = 2^29) to channel 2 and leaks about 1e-3 into the other channels.
2. pid_embed[6,2] = pid_embed[7,2] = +2H.
3. Result for a photon or NH with a = b = 1:
   - E2 = -2H exactly; the original O(1) content is absorbed because ulp(2^30) = 128;
   - adding L2 (|L2| < 64) is absorbed;
   - adding P = +2H gives 0;
   - A = 0 exactly for neutrals, because add_embed has no biases.
   - So the token is padding, exactly.
4. Every other token keeps channel 2 ≠ 0:
   - kept neutrals: +2H (no gate) or +H (one gate, the "xor" tokens);
   - charged tokens with a gate: -H or -2H;
   - charged tokens without a gate: bit-identical to the original.

Why this can only use the photon region for NH. E(kin) is pid-blind and P is a constant per pid. A float sum is exactly 0 only on exact cancellation, so at each kinematic point the set of dropped pids must share one P value. Photons and NH are both dropped in K_γ = {pT < 0.8, E < 1.8}, which forces P6 = P7. NH in {E < 2.2} \ K_γ then cannot be dropped while photons at the same kinematics are kept. NH are therefore dropped exactly in K_γ, which covers all vanishingly soft NH. The rule's NH-only band (E < 2.2 but not in K_γ) is not implemented.

### Step 2: exact restoration of channel 2 for kept tokens (blocks 0 and 1)
The huge channel-2 values of kept neutrals must not reach the rest of the network. This is done with hidden units that were exactly dead on the captured data.

Read-side protection in blocks 0-2:
- `norm1.weight[2] = 0`, so attention never reads channel 2 there.
- `norm2.weight[2]` is scaled by 2^-10 to a TF32-exact value. The other MLP units effectively stop reading channel 2, while the detectors (weights scaled by 2^10) still see an exact saturated value of ±w'.

The detector units (each one writes channel 2 only, with norm weight 2^-10 and fc2 = write·2^10):
- **N0 (block 0, unit 1042).** Fires iff the channel-2 read equals +w' (saturated positive; threshold 1 - 1e-4). Writes -(H+128).
  - no-gate neutrals: 2H → H-128;
  - xor neutrals: H → -128.
- **N1 (block 1, unit 1127).** Fires iff the read is saturated positive. Writes -(H-256), so no-gate neutrals become exactly +128 (a marker).
- **X1 (block 1, unit 1475).** Fires iff the read is saturated negative AND a neutral/charged linear discriminant is positive.
  - The discriminant uses 64 channels, is fitted on captured activations, and has margins ±1 normalised.
  - It writes +256, so xor neutrals also become +128.
  - Charged tokens whose channel 2 is very negative are vetoed by the discriminant.

All of these writes are exact float32 cancellations at the 2^29 scale.

### Step 3: re-inserting the content (block 2)
After block 1, every kept neutral has channel 2 = 128 + small, which saturates the block-2 read (exactly w'). The block-2 MLP then re-inserts the token's lost content:
- One unit (573) writes exactly -128 for marked tokens.
- A gated linear pair, or a single unit in the small edit, computes z = 2048·(read2/w' - 1) + K ± γ·ŷ.
  - The gate term is 0 for marked tokens and ≤ -6 for every unmarked token, so its output is exactly 0 on unmarked tokens.
  - ŷ = β·(norm2 reads of the other 1023 channels) is a closed-form ridge regression of the original channel-2 value after block 1.
  - The fit uses captured activations of about 50k neutral tokens from clean, split2, soft3e-3, P_reco, D and J jets. Its test rmse is 0.04 (the original channel-2 std is 3.5; R² = 0.9999).
  - The output coefficients c± are chosen by least squares so that c+·f(K+γŷ) + c-·f(K-γŷ) = ŷ, with f = tanh(α·GELU).

Differences between the edits:
- large: dense β rows;
- mid: the original (dead) rows kept, plus 512 fitted entries each;
- small: one unit, with 256 entries.

From block 3 on, every token carries its original channel 2 to within about 0.04-0.13. Charged-token reads were never changed.

### Why tokens the rule keeps are processed as before
- Charged particles with pT ≥ 0.8 and E ≥ 1.8 (about 98% of charged) have a bit-identical embedding.
- Kept neutrals differ only in channel 2. It is huge in blocks 0-1, where no attention or other MLP unit reads it (read weights zeroed or scaled by 2^-10), and restored by block 3.
- What changes for everyone: the channel-2 read of the attention in blocks 0-2 (zeroed) and of the MLP in blocks 0-2 (scaled by 2^-10). This loses a small amount of information.
- Soft charged tokens with one gate (about 2% of charged) keep a saturated negative channel 2.

Clean e is 0.003 (mid and large) and 0.009 (small), compared with S-normalised errors of 0.1-0.37 for the flaw.

## 3. Prediction vs outcome (all passes)
| pass | what | predicted | outcome |
|---|---|---|---|
| 001 | zero_pass (exact mask, saturated ch2) | E_rule 0.35, d_nll 0.01, DoF 2062 | E_rule 0.542, clean e 0.079, class_drop 0.063 (fails), d_nll 0.030, DoF 2062 |
| 002 | capture orig (ch2 stats) | max abs x2 15, neutral fraction 0.35 | x2 range [-55, 22], neutral fraction 0.51 |
| 003 | capture orig (dead units) | 50 dead units in block 1 | 2 dead units in block 1; x2 at block 27 only 0.22-correlated with block 1 |
| 004-009 | captures for surrogates | surrogate R² 0.97 | R² 0.99 (blocks 0-7) down to 0.70 (block 27) |
| 010 | ch2 surrogates in all blocks + classifier K/V | E_rule 0.30, clean e 0.025 | E_rule 0.102, clean e 0.010, DoF 210900 |
| 011 | zero ch2 norm weights everywhere | E_rule 0.25, clean e 0.03 | E_rule 0.236, clean e 0.028, class_drop 0.034 (fails) |
| 012 | hybrid, top-6 surrogates | E_rule 0.17 | E_rule 0.123, DoF 29752 |
| 013 | hybrid, top-14 surrogates | E_rule 0.14 | E_rule 0.107, DoF 59432 |
| 014 | hybrid 6 + body.norm[2] shrink | E_rule 0.13 | E_rule 0.121, DoF 21563 |
| 015 | hybrid 3 + shrink | E_rule 0.16 | E_rule 0.155, DoF 11329 |
| 016 | restoration v1 (gated linear) | E_rule 0.13, clean e 0.013 | E_rule 0.205, clean e 0.026 |
| 017 | restoration to a constant | E_rule 0.20 | E_rule 0.385 (fails) |
| 018 | capture (diagnose v1) | x2 rmse 0.4 | x2 rmse 0.80 for no-gate neutrals, 3.6 for xor neutrals |
| 019 | + norm2 scaling in blocks 0-2 | clean e 0.016 | clean e 0.026 (no change) |
| 020 | capture (diagnose) | x2 rmse 0.5 | 0.85, traced to fp32 accumulation error in the gated units |
| 021 | + attention surrogates in blocks 0-2 | clean e 0.014 | 0.026 (no change) |
| 022 | accumulation fix (gate 8192, sparse 128) | E_rule 0.16, clean e 0.018 | E_rule 0.148, clean e 0.0157, DoF 2084 |
| 023 | + xor-photon flip unit X1 | E_rule 0.13, clean e 0.012 | E_rule 0.119, clean e 0.0123, DoF 3111 |
| 024 | capture (diagnose) | x2 rmse 0.3 / 0.4 | 0.26 / 0.56 |
| 025 | + attention surrogates in blocks 0-2 | E_rule 0.10 | 0.118 (no gain) |
| 026 | + attention and MLP surrogates in blocks 0-2 | E_rule 0.095 | 0.121 (no gain) |
| 027/028 | captures for refit (1020 jets) | refit rmse 0.2 | 0.04-0.06 |
| 029 | refit, exact constant unit, gate 4096 | E_rule 0.10, clean e 0.008 | E_rule 0.075, clean e 0.0041 |
| 030/031 | captures on probe-type jets | split2 rmse 0.2 | 0.21 before the refit, 0.04 after |
| 032 | rx4_mid | E_rule 0.072, split2 e 0.005 | E_rule 0.074, split2 e 0.0038 |
| 033 | rx4_small (single unit) | E_rule 0.10 | 0.126 |
| 034 | rx4_large (block 0-2 surrogates) | E_rule 0.072 | 0.706: script bug, the surrogate overwrote the new constant unit |
| 035-038 | held-out logits | clean e 0.0045 / 0.014 / soft1e-4 0.0047 | 0.0035 / 0.0147 / 0.0043 |
| 039 | rx6_mid (tighter gates, gate 2048) = **final_large** | E_rule 0.073 | **0.069** |
| 040 | rx6_small | E_rule 0.10 | 0.094 |
| 041 | rx7_p512 = **final_mid** | E_rule 0.07, DoF 1180 | **0.072**, DoF 1134 |
| 042 | rx7_s256 = **final_small** | E_rule 0.095, DoF 380 | **0.104**, DoF 362 |
| 043/044 | held-out logits for small and mid | clean e 0.0105 / 0.004 | 0.0093 / 0.0031 |

## 4. Ideas that failed or were dropped
- **Saturating channel 2 for kept neutrals (zero pass).** Exact masking works, but every block reads a wrong ±w for 51% of the tokens. Clean e is 0.079 and a class drops by 6%.
- **Channel-2 surrogates everywhere.** Every read of channel 2 is replaced by a rank-1 linear reconstruction from the other channels. It works (E_rule 0.102) but costs 211k DoF, and the reconstruction is poor in deep blocks (R² ≈ 0.7). Zeroing those reads instead loses too much (class drop 0.034).
- **Restoring channel 2 to a constant.** Fails the limits; the per-token content of channel 2 matters.
- **Surrogates in the attention or MLP of blocks 0-2 on top of restoration.** No gain.
- **Gated linear units with a large gate term.** The ±25856 gate term is accumulated with 1000 small terms in fp32, giving an error of about 0.85 in channel 2. The fix was a smaller gate scale and a separate exactly saturating constant unit.
- **Exact masking of NH outside K_γ.** Impossible (shown above).
- **Soft removal of NH via per-block saturated key patterns.** Offline, the suppression margins (median about 8 in deep blocks, with some negative tails) were not reliable. An exact "NH and E < 2.2" flag also cannot be built in one MLP layer from unbounded linear scores. Not attempted on the oracle.
- **Choosing the NH cell {pT ≥ 0.8, 1.8 ≤ E < 2.2} instead of K_γ.** This would cover 52% instead of 14% of the P-probe NH drops, but it would miss all very soft NH. Rejected.

## 5. Known limitations
- NH with E < 2.2 outside {pT < 0.8, E < 1.8} are not dropped. They are the main part of the remaining P error (e 0.025 vs 0.151 originally).
- gen_drift is 0.05 / 0.10 / 0.27 at t = 0.2 / 0.5 / 0.85 for every variant, including the zero pass. Noised neutrals that fall below the thresholds are masked, so their generator velocity becomes 0. This seems inherent to applying the rule inside the shared body; the rest of the edit adds nothing measurable to it.
- The dropped particle still appears in the local_physics kNN neighbourhoods of other particles, which carry no pid information. Measured effect: none beyond the clean floor (soft1e-4 e equals clean e).
- If the oracle used fp16 anywhere, the 2^29-2^30 values would overflow. It is fp32: all exactness checks behaved as designed.

## 6. Passes used
44 of 50:
- 1 zero-pass evaluation;
- 15 captures;
- 22 surgical passes;
- 6 held-out logits passes.

No gradients and no iterative optimisation were used. Everything was computed in closed form (ridge, LDA, least squares, thresholds) on weights and on activations returned by the oracle.
