# Final notes: removing the soft-neutral flaw from pretrain_m

## Final edits
All changes are rank-1 (2-D) or single entries (1-D). No generator tensor is touched.

| edit | tensors changed (entries, rank) | DoF |
|---|---|---|
| **final_small.npz** | b6.mlp.fc1.bias (1), b6.mlp.fc2.bias (1), b6.mlp.fc2.weight (1, r1), b7.attn.in_proj_weight (10, r1) | **13** |
| **final_mid.npz** | b5.mlp.fc1.bias (1), b5.mlp.fc2.bias (1), b5.mlp.fc2.weight (1, r1), b6.mlp.fc1.bias (1), b6.mlp.fc2.weight (1, r1); in_proj_weight of body blocks 6,7,8,9,10,11 (97, 96, 87, 108, 82, 79 entries, r1); classifier blocks 0,1 in_proj_weight (73, 48, r1) | **675** |
| **final_large.npz** | b5.mlp.fc1.weight (512 = one row, r1), b5.mlp.fc1.bias (1), b5.mlp.fc2.weight (1, r1), b5.mlp.fc2.bias (1); the same 8 in_proj key edits as mid (668 entries) | **1185** |

(`bN` = `body.in_blocks.N`. All in_proj changes sit in the key rows 512–1023, column 450 only.)

## What the flaw is and where it lives
- **Response scan** (custom jets: one photon or neutral hadron added at a controlled pT). The response is a step in *absolute* log pT, not in pT fraction. It's about 0.45–0.5 S for log pT < −0.3 (pT ≲ 0.75 GeV) and drops to 0.03 S just above, then rises slowly again for genuinely hard particles. Neutral hadrons behave the same way.
- **Localization** (captures of residuals per block). The summary tokens are unchanged through block 6, the same as for a harmless 3e-3 photon. They jump at block 7 (0.02 → 0.24 relative change), and grow further in blocks 8–11.
- **Mechanism.** In **block 7, head 0** the four summary-token queries put **71 %** of their attention on the sub-threshold photon (logit +4.5 vs −0.9 for other particles). In **block 7, head 5** particle queries attend to it about 32 % of the time. Blocks 9 (h0) and 10 (h13) show weaker echoes. The classifier head does not single the photon out.
- **Detector.** The key that head 0 reads is a "soft-neutral" feature that the photon's own token builds up gradually in blocks 1–6 through its MLPs. The biggest contributors are block-6 MLP unit 294 (attribution 0.98 of the ~4 logit jump) and block-5 unit 266. Both are near-binary detectors: pre-activation ≈ 4–6 for sub-threshold photons/neutral hadrons, ≈ −2.4 median on normal tokens. They fire on injected photons and on neutrals softened by P, but **not** on the rare soft neutrals in clean jets (0 of 38 below log pT −0.3). Removing those clean soft neutrals barely changes the original output (e_clean ≈ 0.002 S). So the flaw is context-dependent, and P triggers it.

## The circuit the edits implement ("flag + key suppression")
1. **Flag channel.** Residual channel c = 450 has the least downstream information among 512 (normal tokens always negative, |x| ≤ 4.6 in blocks 6–11). The edit adds −Δ = −20 to it through one MLP output bias (`fc2.bias[450]`). After that, every later DyT norm sees tanh(α(x−20)) = −1 exactly, so xn_450 = −w_450 is constant on **every** normal token: summary, particles, any jet.
2. **Gate.** A GELU unit that is (essentially) exactly 0 on normal tokens and large on soft neutrals writes +κ = +100 into channel 450 through one `fc2` entry. Flagged tokens then have xn_450 = +w_450.
   - small/mid: the model's own detectors. Unit 294 in block 6, plus unit 266 in block 5 (mid), with the fc1 bias lowered by t (t = 1.4–1.6), so only the strong detector response passes.
   - large: unit 266 is repurposed as a *dedicated* readout. Its fc1 row = λ·g, with g fitted in closed form by ridge least squares on captured block-5 MLP inputs, to the target clip((τ − log pT)/0.1, ±3) for neutrals and −3 for everything else, with τ = −0.25 and λ = 10. Held out: 16 misses / 797 targets, 16 false positives, **0 charged** false positives.
   - Because the gate's GELU sits deep in its negative tail for normal tokens, their channel 450 just moves a bit further negative. That has no effect, since the channel is saturated.
3. **Key suppression.** For each suppressed attention block, a rank-1 change to the key projection reads only column 450: ΔW_k[:,450] = col/(2 w_450), with col_h = −γ_h m_h for each head h.
   - Normal keys all get the same shift +col/2. That is a constant for each query, so softmax is unchanged *exactly*.
   - Flagged keys get −col/2, i.e. a logit change of −γ_h (q·m_h)/√32.
   - m_h is a sparse Fisher direction in head space (C⁻¹μ of the captured queries, restricted to the fewest coordinates with q01 ≥ 0.3·median). γ_h is set so that 99 % of queries lower the flagged key by ≥ 12 logits.
   - Result: the flagged token disappears from all attention after the gate. Body blocks 6–11 + both classifier blocks in mid/large; block-7 heads 0 and 5 only in small.
   - Its own residual is never read again (the final norm still masks only real padding).

**Why jets without soft neutrals are unaffected.** On such jets the gate output is ≈ 0, channel 450 is identically saturated, and the key edits only add a per-query constant to every logit. The remaining clean drift comes only from flattening the low-information channel 450 and from the detector unit's lost tiny negative output: KL ≈ 1e-4, |Δacc| ≤ 0.0009.

## Final practice metrics (oracle surgical; ratio = response / original response)
| | small (p42) | mid (p43) | large (p35) |
|---|---|---|---|
| DoF | 13 | 675 | 1185 |
| d_acc | −0.0003 | −0.0004 | −0.0008 |
| worst class drop | 0.0068 | 0.0068 | 0.0068 |
| d_nll | −1e-4 | −1e-4 | +3e-5 |
| KL | 9e-5 | 9e-5 | 1.5e-4 |
| ratio soft1e-3 / soft1e-4 | 0.388 / 0.410 | 0.043 / 0.109 | 0.046 / 0.110 |
| ratio split2 / soft3e-3 | 0.999 / 0.998 | 1.006 / 1.000 | 1.007 / 1.001 |
| rho D / J / P / P_ref | 0.999/0.999/0.926/0.969 | 1.004/1.002/0.835/0.959 | 1.004/1.003/0.860/0.983 |
| R_vis / R_in / R_sub | 0.70 / 1.00 / 0.40 | 0.54 / 1.00 / 0.076 | — / — / 0.078 |
| gen_drift max | 0.014 | 0.014 | **0.048** (at t=0.85) |
| gate_ref_no_g2 | pass | pass | pass |

**Offline target-error estimates** (512-jet eval set; e ratios vs original-model targets; P target = original on P minus neutrals with log pT < τ):

| | soft1e-3 | soft1e-4 | P (τ=−0.3) | P (τ=−0.25) | P (τ=−0.2) | P (frac<1.5e-3) |
|---|---|---|---|---|---|---|
| small | 0.40 | 0.40 | 0.96 | 0.75 | 0.65 | 0.70 |
| mid | 0.052 | 0.114 | 0.98 | 0.62 | 0.40 | 0.46 |
| large | 0.051 | 0.117 | 0.64 | 0.39 | 0.37 | 0.47 |

**Hidden threshold.** I compared the oracle's rho_P_ref against offline rho values on my own "P minus soft neutrals" variants, for two different edits and after correcting the ~0.03 subset bias seen on raw P. Both point to P_ref ≈ P without neutrals of **log pT < −0.25** (equivalently a fraction ≈ 1.3–1.5e-3). This is an inference, not a known fact. The large edit's dedicated gate uses this τ.

**Risk notes.**
- The large edit's gen_drift (0.048 at t = 0.85) is higher than mid/small (0.014): the dedicated gate also fires on some heavily-noised generator tokens. The real limit (AUC rise ≤ 0.03) could not be checked here.
- Mid/small use the model's own detectors and are the safer generator choice.

## Predictions vs outcomes (main passes)
| pass | edit / question | prediction | outcome |
|---|---|---|---|
| 1 | baseline | all ratios 1, dof 0 | as predicted; r(soft1e-3)=0.41 > r(soft1e-4)=0.13 (unexpected ordering) |
| 2 | pT scan | r(1e-3)/r(1e-4)=3, r(3e-3)/r(1e-3)=0.15, hadron/photon 0.5 | 3.1, 0.13, **1.1** (hadrons equally affected) |
| 3 | onset block | 8 | **7** |
| 9 | kill 11 detector units | soft1e-3 0.55, soft1e-4 0.7, KL 0.002 | 0.69, 0.76, 4e-5 |
| 11 | T2 (512-ch gate + Mahalanobis key readout) | soft 0.15/0.4, KL 0.002, gen 0.05 | 0.047/0.145, **KL 0.025, class drop 0.023 (fail), gen 0.19** |
| 12/13 | keys-only / gate-only diagnostics | KL 0.005 / gen 0.15 | KL 0.007 / gen 0.186; gate alone gave soft1e-3 0.19 |
| 14 | flag channel, blocks 7–11+cls dense | soft 0.06/0.2, KL 0.002, P_ref 0.97, dof 3587 | **0.025/0.127**, KL 1e-4, P_ref 0.943, gen 0.014 |
| 15 | flag from block 6 | soft1e-4 0.10 | 0.108 |
| 16/17/18 | block-7 h0,h5 / all of block 7 / blocks 7–11 without cls | 0.2,0.45 / 0.15,0.4 / 0.06,0.2 | 0.39,0.40 / 0.39,0.40 / 0.054,0.18 |
| 19/20 | sparse (q05 criterion) | 0.025/0.11 | 0.037/0.155 (tail queries leak) |
| 21/22 | sparse (q01 ≥ 0.3/0.5 median) | 0.027/0.12, 0.025/0.112 | 0.025/0.109, 0.024/0.108 at DoF 675/959 |
| 26/28/37 | P_ref threshold inference | abs −0.3 / −0.1 / −0.2 | consistent with ≈ −0.25 |
| 33 | dedicated gate τ=−0.2 | P_ref 0.99, soft1e-3 0.08 | 0.971, 0.030 |
| 35 | dedicated gate τ=−0.25 | P_ref 0.985, soft1e-3 0.045 | 0.983, 0.046 (gen 0.048 > predicted) |
| 40 | flag from block 1 (early gate OR block 5) | KL 4e-4, soft1e-4 0.07 | soft1e-4 0.068 but **KL 0.036, class drop 0.08, gen 0.24 (fail)** |
| 42 | final small | 0.40/0.42, dof 13 | 0.388/0.410, dof 13 |
| 43 | final mid | 0.045/0.11, P_ref 0.965 | 0.043/0.109, P_ref 0.959 |

## Ideas that failed
- **Killing detector units** (pass 9): cheap (11 DoF), but only removes ~30 % of the flaw because the detector feature is distributed over many units in blocks 1–6.
- **"Corner" trick** (a saturated sign pattern that makes later keys naturally ignored): only ~2–4 logits of margin. Rejected offline.
- **Multi-channel Mahalanobis readout** of a gate pattern (T2): separation of 130–270 σ was still too noisy. Coherent 0.05–0.13-logit noise on every key failed the class-drop limit. The 512-channel write also drove gen_drift to 0.19.
- **Dense query directions**: they work, but cost 512 entries per block. Sparse Fisher directions give the same result at 1/6 of the DoF; the first sparse criterion (q05) leaked through tail queries.
- **Gate in block 0/1 with the flag from block 1** (pass 40): channel 450 still carries information in blocks 1–4 (its values cross 0), so flattening it there broke the model. Linear soft-neutral readouts are also much poorer before block 5.
- A gate threshold too high (t=2.2): P improves, but soft1e-3 photons near the edge escape.

## Passes used
44 of 50: 19 surgical (including the baseline), 10 captures, and 15 logits passes for offline target-error and threshold inference.
