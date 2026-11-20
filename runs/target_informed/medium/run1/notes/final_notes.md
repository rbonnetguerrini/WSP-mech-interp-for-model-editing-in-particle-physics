# Final notes: implementing the reference inside pretrain_m

## 1. Final edits

All three share one base circuit (the "gate"); mid and large add rank-1 attention edits.
DoF follows the task rule (2-D: min(changed entries, r(m+n-r)); 1-D: changed entries), as reported by the oracle.

### Shared base (= final_small.npz), DoF 7,329
| tensor | rank | changed entries | DoF |
|---|---|---|---|
| body.token | 1-D view (4x512) | 24 | 24 |
| body.embed.mlp.fc1.weight | 2 | 12 | 12 |
| body.embed.mlp.fc1.bias | - | 3 | 3 |
| body.embed.mlp.norm.weight | - | 3 | 3 |
| body.embed.mlp.fc2.weight | 3 | 1,536 | 1,536 |
| body.embed.mlp.fc2.bias | - | 6 | 6 |
| body.pid_embed.0.weight | 2 | 2 | 2 |
| body.time_embed.fc1.weight | 1 | 1 | 1 |
| body.time_embed.norm.weight | - | 1 | 1 |
| body.time_embed.fc2.weight | 1 | 512 | 512 |
| body.in_blocks.0.norm1.alpha | - | 1 | 1 |
| body.in_blocks.0.norm1.weight | - | 1 | 1 |
| body.in_blocks.0.attn.in_proj_weight | 1 | 576 | 576 |
| body.in_blocks.0.attn.out_proj.weight | 1 | 512 | 512 |
| body.in_blocks.0.norm2.weight | - | 7 | 7 |
| body.in_blocks.0.mlp.fc1.weight | 7 | 13 | 13 |
| body.in_blocks.0.mlp.fc1.bias | - | 8 | 8 |
| body.in_blocks.0.mlp.norm.weight | - | 8 | 8 |
| body.in_blocks.0.mlp.fc2.weight | 8 | 4,096 | 4,096 |
| body.in_blocks.0.mlp.fc2.bias | - | 7 | 7 |
| **total** | | 7,329 | **7,329** |

### final_mid.npz = base + rank-1 in_proj edits in body blocks 2 and 3, DoF 11,423
Extra: body.in_blocks.2.attn.in_proj_weight (rank 1, 458,626 changed entries, DoF 2,047),
body.in_blocks.3.attn.in_proj_weight (rank 1, 474,942 changed entries, DoF 2,047).

### final_large.npz = base + rank-1 in_proj edits in body blocks 1-5, DoF 17,564
Extra: body.in_blocks.{1,2,3,4,5}.attn.in_proj_weight, each rank 1, DoF 2,047
(changed entries 392,402 / 458,626 / 474,942 / 507,881 / 507,904).

No generator.* or classifier.* tensor is touched.

## 2. What the reference does, and how I found it

**Reference = the original model applied to the jet with certain soft neutral constituents removed:**
- photons (pid 6) with pT < 0.8 GeV **and** E < 1.8 GeV are dropped;
- neutral hadrons (pid 7) with E < 2.2 GeV are dropped, at any pT;
- charged particles (pid 0, 1, 2, 4, 5, 8) are never dropped, even when soft or when their IPs are zero.
  A photon given non-zero IPs is still dropped, so the rule depends on pid, not on the IPs.
  The clean practice data contain no constituent in the dropped region (clean photons with pT < 0.8 all
  have E >= 1.85; clean neutral hadrons all have E >= 2.256). That is why the reference equals the
  original on clean jets, split2 and soft3e-3.

How I found it:
1. Pass 001 (null surgical): e_orig = 0 on clean, soft3e-3 and split2; non-zero on soft1e-3, soft1e-4 and P.
2. Passes 002-004: reference on soft1e-3 vs original on clean and on soft1e-3. Each jet matched either
   orig(clean) (89.6%, photon dropped) or orig(soft) (10.4%, photon kept), never neither.
   A threshold scan over photon features gave a perfect split at log pT < log 0.8 AND log E < log 1.8
   (dropped max pT 0.7998, kept min 0.8001; dropped max E 1.7995, kept min E 1.8001).
3. Passes 005/006 (constructed jets, reference vs original): rectangle corners confirmed;
   pid 7 dropped; charged (codes 0 and 8, with or without IPs) kept; photon with IPs dropped;
   two photons both dropped. On P jets, ref(P) = orig(P minus dropped photons) for 87% of jets.
4. The remaining P mismatches all contained a neutral hadron with E < 2.2 GeV (pT up to 2.15). Dropped
   neutral hadrons had max E 2.192; jets with a neutral hadron at E >= 2.2035 matched. With the
   neutral-hadron rule, 100% of the 1,024 P jets are explained.
5. Pass 038 (unseen probe types) confirmed: photons at 3e-4, photons at 0.4 < dR < 0.8, two photons,
   neutral hadrons at 1e-3, and a neutral hadron with pT 1.5, E 2.0 are dropped. A soft charged particle
   and a photon with pT 0.9, E 1.0 are not (reference = original there).

## 3. The circuit

Notation: A = 30, H = A/2 = 15, delta = 0.01. "Helper" channels are residual channels whose block-0 norm2
DyT weight is exactly 0 in the original model: c1=115, c2=424, c3=505, cP6=66, cP7=271, cT=83, cM=331.
All repurposed hidden units are dead units of the original model (DyT weight 0), so their original
contribution was exactly 0.

**(a) Kinematic steps (embed MLP, 3 dead units).** For x in {log pT, log E} the input DyT gives
n = w_e tanh(a_e x). A unit with fc1 row -s e_col and bias s*n(thr)+z_1/2 (s about 1.2e6) yields, after
GELU and the embed-MLP DyT, h = 1[x < thr] with a transition width of 1e-5 in log units.
Its fc2 column is A e_c. Steps: pT<0.8 -> c1, E<1.8 -> c2, E<2.2 -> c3.
Every particle also gets -H on all helper channels (embed fc2 bias), so channel c holds x_c + A*(s - 1/2).

**(b) pid flags.** pid_embed[6, cP6] += A, pid_embed[7, cP7] += A (same +/-H encoding via the bias offset).
Summary tokens get -H on all helper channels (body.token).

**(c) Time awareness (only t = 0 is gated).** The time token is exactly zero and masked at t = 0, and
present at t > 0. A dead time_embed unit reads MPFourier channel 120 (cos >= 0.995 on [0, 1]) with gain
1e5, so it is 1 for t > 1e-4 and exactly 0 at t = 0. It writes +H to cT and -H to the other helper channels.
Block-0 attention was dead in the original (norm1 alpha = 6e-6). I set alpha = 1 and the norm1 weight of
cT to 1, so n1_cT = +1 for the time token and -1 for everything else. Head 0 gets
q = 10 n1_cT, k = -56.6 n1_cT, v = n1_cT (one column of in_proj). At t > 0 every query attends to the
time token with a +200 logit advantage; at t = 0 it attends to tokens that all carry value -1.
out_proj column 0 is set to -H e_cM, so cM gets +H at t = 0 and -H at t > 0.

**(d) Gate and cancellation (block-0 MLP, 8 dead units).** The block-0 norm2 weight on the 7 helper channels
is set to delta, so n2_c / delta = b_c in {-1, +1} (saturated). One gate unit computes
z = 40 * (S - 4) with S = 1*b_c1 + 1*b_c2 + 2*b_c3 + 1*b_P6 + 3*b_P7 + 3*b_M.
This fires (S >= 6) exactly when **[photon AND pT<0.8 AND E<1.8 AND t=0] OR [neutral hadron AND E<2.2 AND t=0]**;
otherwise S <= 2 (checked exhaustively). Its fc2 column is 40*u, where u is a +/-1 pattern described below.
Seven cancel units (one per helper channel) fire only when b_c = +1 (threshold 0.9999 on n2/delta) and
write -A e_c. fc2 bias adds +H. After block 0, every helper channel holds exactly its original value for
every token: particles, summary tokens and the time token.

**(e) Removal from attention.** A gated token's residual becomes x + 40u, so in every later block its DyT
outputs saturate to the fixed pattern s_k = w_k * u (its key and value no longer depend on the jet).
u = -sign(sum over blocks and heads of D_k W_K,h^T d_kh / |.|), where d_kh is the per-head Fisher
direction of the queries of normal tokens (closed form, from captured activations). This makes the fixed
key anti-aligned with essentially all queries: the gated token sits about 7-18 logits below each query's
log-sum-exp at the median. It is therefore (almost) not attended in body blocks 1-11 or in the classifier
blocks (whose keys are the body output).
- **mid / large:** in body blocks {2,3} (mid) or {1..5} (large), add a rank-1 edit
  dW = [0 ; K ; -W_V s_k] v_k^T, where v_k = (M_k + lam I)^-1 s_k / (s_k^T (M_k + lam I)^-1 s_k).
  M_k is the second moment of normal tokens' n1 (~10k tokens from 256 clean jets, captured).
  For gated tokens v_k.n = 1, so the value becomes exactly 0 and the key moves by K, with K_h = -T_h d_h/med_h.
  T_h is set so the 99.9th-percentile query gets at least an 8-logit margin.
  For normal tokens v_k.n is about 1e-4 to 4e-3 (p99) in these blocks, so the change is tiny.

**Why inputs where reference = original are unaffected.**
(i) The gate is a sharp AND of exact thresholds plus pid. It never fires on clean jets, charged particles
or kept photons, so their tokens carry no gate vector.
(ii) Every helper write is cancelled exactly within block 0 (saturated binaries), and the repurposed units
were dead.
(iii) The time token, summary tokens and padded tokens follow the same exact encoding.
(iv) At t > 0 the gate cannot fire, so the generator path is unchanged.
On practice clean jets the small edit gives exactly 0 change (d_acc 0, KL 0, all rho 1.000, gen_drift 0.000).
mid/large add only the small v_k-leakage of the rank-1 edits (e_clean 0.0002 / 0.0005).

## 4. Final practice metrics (surgical passes 029 / 034 / 036)

| metric | orig | small (029) | mid (034) | large (036) |
|---|---|---|---|---|
| DoF | 0 | 7,329 | 11,423 | 17,564 |
| **E_ref** | 1 | **0.110** | **0.042** | **0.036** |
| e soft1e-3 (orig 0.396) | 0.396 | 0.0093 | 0.0049 | 0.0045 |
| e soft1e-4 (orig 0.132) | 0.132 | 0.0085 | 0.0043 | 0.0039 |
| e P (orig 0.0894) | 0.0894 | 0.0217 | 0.0072 | 0.0060 |
| e clean / split2 / soft3e-3 | 0 | 0 / 0 / 0 | 0.0002 / 0.0046 / 0.0014 | 0.0005 / 0.0055 / 0.0020 |
| d_acc / class_drop / d_nll | 0 | 0 / 0 / 0 | 0 / 0.001 / 0 | 0 / 0.001 / 0 |
| rho D / J / P_ref | 1 | 1.000 / 1.000 / 1.000 | 1.001 / 1.000 / 1.000 | 1.001 / 0.999 / 1.000 |
| rho P (not a limit) | 1 | 0.846 | 0.847 | 0.850 |
| gen_drift max | 0 | 0.000 | 0.004 | 0.014 |
| gate_ref_no_g2 | pass | pass | pass | pass |

On unseen probe types (pass 038-042; error ratio vs original, small / mid / large):
photon 3e-4: 0.17 / 0.16 / 0.16; neutral hadron 1e-3: 0.10 / 0.09 / 0.08;
photon at 0.4<dR<0.8: 0.06 / 0.05 / 0.05; two photons: 0.16 / 0.14 / 0.14;
neutral hadron pT 1.5, E 2.0: 0.33 / 0.30 / 0.30; stronger softening (P with exponent 1.2): 0.35 / 0.14 / 0.11.
Charged-soft and kept-photon probes (reference = original): 0.000 / 0.001 / 0.002 absolute.

Remaining error: (1) kNN contamination. The removed particle still occupies slots in other particles' K=15
neighbour lists of the local-physics block, which has no pid information and cannot be gated. This is
largest for neutral hadrons whose log pT lies inside the bulk. (2) The pair interaction bias, which is
added to particle-particle logits for the removed token and which I cannot compute without running
that module. (3) Residual attention in blocks without explicit edits.

## 5. Predictions vs outcomes (all 42 passes)

(Generated from requests/*.json and responses/*.json; also in work/pred_table.md.)

| pass | kind | edit | prediction | outcome |
|---|---|---|---|---|
| 001 | surgical | orig | E_ref=1.0, d_acc=0.0, gen_drift=0.0, rho_D=1.0, rho_J=1.0, rho_P_ref=1.0 | E_ref=1.000, e_clean=0.0000, d_acc=0.0000, class_drop=0.000, gen_drift_max=0.000, rho_J=1.000, pass=True, DoF=0 |
| 002 | reference | - | rel_dist_ref_to_orig_clean=0.1 | data returned (see notes) |
| 003 | logits | orig | acc=0.85 | data returned (see notes) |
| 004 | logits | orig | rel_dist_to_clean=0.5 | data returned (see notes) |
| 005 | reference | - | frac_photon_inrect_dropped=1.0, frac_charged_dropped=0.0, frac_nh_dropped=0.5, P_match_removed_rel_err=0.01 | data returned (see notes) |
| 006 | logits | orig | mean_logit_norm=15.0 | data returned (see notes) |
| 007 | capture | orig | n_saturated_channels=5 | data returned (see notes) |
| 008 | capture | orig | rel_diff_summary_block11=0.2 | data returned (see notes) |
| 009 | capture | orig | fc1_recon_rel_err=0.001 | data returned (see notes) |
| 010 | capture | E1.npz | gated_tokens_with_gate_frac=1.0, nongated_rel_change_block0=0.002 | data returned (see notes) |
| 011 | capture | E1b.npz | gate_amp_gated=10.0, max_nongated_rel_change_block0=0.01 | data returned (see notes) |
| 012 | capture | E1b.npz | gate_amp_gated=10.0, max_nongated_rel_change_block0=0.01 | data returned (see notes) |
| 013 | capture | E1b.npz | gate_amp_gated=10.0, max_nongated_rel_change_block0=0.01 | data returned (see notes) |
| 014 | surgical | D2.npz | E_ref=0.4, e_soft1e-3=0.12, e_soft1e-4=0.05, e_P=0.04, d_acc=-0.004, d_nll=0.01, rho_D=1.0, rho_J=1.0, rho_P_ref=1.0, gen_drift_max=0.1 | E_ref=0.374, e_clean=0.0609, d_acc=-0.0071, class_drop=0.063, gen_drift_max=0.205, rho_J=1.166, pass=False, DoF=34337 |
| 015 | capture | E1c.npz | clean_logit_change_vs_orig=0.01 | data returned (see notes) |
| 016 | capture | E1c.npz | clean_logit_change_vs_orig=0.01 | data returned (see notes) |
| 017 | capture | E1c.npz | clean_logit_change_vs_orig=0.01 | data returned (see notes) |
| 018 | surgical | E1d.npz | E_ref=0.5, e_clean=0.0, d_acc=0.0, gen_drift_max=0.15 | E_ref=0.136, e_clean=0.0000, d_acc=0.0000, class_drop=0.000, gen_drift_max=0.727, rho_J=1.000, pass=True, DoF=8238 |
| 019 | surgical | D4.npz | E_ref=0.15, e_clean=0.015, d_acc=-0.002, class_drop=0.01, rho_J=1.03, gen_drift_max=0.15 | E_ref=0.148, e_clean=0.0182, d_acc=-0.0025, class_drop=0.010, gen_drift_max=0.717, rho_J=0.998, pass=True, DoF=34849 |
| 020 | surgical | E2.npz | E_ref=0.137, e_clean=0.0, d_acc=0.0, gen_drift_t0.85=0.25, gen_drift_t0.5=0.08, gen_drift_t0.2=0.04 | E_ref=0.136, e_clean=0.0000, d_acc=0.0000, class_drop=0.000, gen_drift_max=0.349, rho_J=1.000, pass=True, DoF=21118 |
| 021 | surgical | E3.npz | E_ref=0.136, e_clean=0.0, d_acc=0.0, gen_drift_max=0.002, rho_P_ref=1.0 | E_ref=0.141, e_clean=0.0036, d_acc=-0.0006, class_drop=0.003, gen_drift_max=0.012, rho_J=0.992, pass=True, DoF=5764 |
| 022 | surgical | E4.npz | E_ref=0.09, e_P=0.015, e_clean=0.0003, d_acc=0.0, gen_drift_max=0.005 | E_ref=0.110, e_clean=0.0000, d_acc=0.0000, class_drop=0.000, gen_drift_max=0.000, rho_J=1.000, pass=True, DoF=7844 |
| 023 | surgical | E4a.npz | E_ref=0.1, e_clean=0.001, e_soft1e-4=0.007 | E_ref=0.038, e_clean=0.0004, d_acc=0.0000, class_drop=0.001, gen_drift_max=0.010, rho_J=1.000, pass=True, DoF=16032 |
| 024 | surgical | E4b.npz | E_ref=0.15, e_clean=0.015, e_soft1e-4=0.016 | E_ref=0.116, e_clean=0.0182, d_acc=-0.0026, class_drop=0.010, gen_drift_max=0.030, rho_J=0.998, pass=True, DoF=34455 |
| 025 | surgical | E4c.npz | E_ref=0.12, e_clean=0.008 | E_ref=0.058, e_clean=0.0023, d_acc=0.0000, class_drop=0.002, gen_drift_max=0.008, rho_J=1.007, pass=True, DoF=34455 |
| 026 | surgical | E4v1.npz | E_ref=0.06, e_clean=0.0001 | E_ref=0.111, e_clean=0.0000, d_acc=0.0000, class_drop=0.000, gen_drift_max=0.002, rho_J=1.000, pass=True, DoF=9891 |
| 027 | surgical | E4v2.npz | E_ref=0.045, e_clean=0.0002 | E_ref=0.054, e_clean=0.0002, d_acc=0.0000, class_drop=0.001, gen_drift_max=0.004, rho_J=1.000, pass=True, DoF=11938 |
| 028 | surgical | E4v3.npz | E_ref=0.035, e_clean=0.0008 | E_ref=0.036, e_clean=0.0005, d_acc=0.0000, class_drop=0.001, gen_drift_max=0.014, rho_J=0.999, pass=True, DoF=18079 |
| 029 | surgical | E5.npz | E_ref=0.11, e_clean=0.0, gen_drift_max=0.0 | E_ref=0.110, e_clean=0.0000, d_acc=0.0000, class_drop=0.000, gen_drift_max=0.000, rho_J=1.000, pass=True, DoF=7329 |
| 030 | surgical | E5b3.npz | E_ref=0.06, e_clean=0.0002 | E_ref=0.055, e_clean=0.0002, d_acc=0.0000, class_drop=0.001, gen_drift_max=0.003, rho_J=1.000, pass=True, DoF=9376 |
| 031 | surgical | E5b234.npz | E_ref=0.04, e_clean=0.0004 | E_ref=0.040, e_clean=0.0004, d_acc=0.0000, class_drop=0.001, gen_drift_max=0.009, rho_J=1.000, pass=True, DoF=13470 |
| 032 | surgical | E5b34.npz | E_ref=0.047, e_clean=0.0003 | E_ref=0.053, e_clean=0.0004, d_acc=0.0000, class_drop=0.001, gen_drift_max=0.009, rho_J=1.000, pass=True, DoF=11423 |
| 033 | surgical | E5b2345.npz | E_ref=0.037, e_clean=0.0006 | E_ref=0.038, e_clean=0.0005, d_acc=0.0000, class_drop=0.001, gen_drift_max=0.014, rho_J=0.999, pass=True, DoF=15517 |
| 034 | surgical | E5b23.npz | E_ref=0.043, e_clean=0.0003 | E_ref=0.042, e_clean=0.0002, d_acc=0.0000, class_drop=0.001, gen_drift_max=0.004, rho_J=1.000, pass=True, DoF=11423 |
| 035 | surgical | E5b234m.npz | E_ref=0.037, e_clean=0.0005 | E_ref=0.041, e_clean=0.0007, d_acc=0.0001, class_drop=0.001, gen_drift_max=0.019, rho_J=0.999, pass=True, DoF=13470 |
| 036 | surgical | E5b12345.npz | E_ref=0.036, e_clean=0.0005 | E_ref=0.036, e_clean=0.0005, d_acc=0.0000, class_drop=0.001, gen_drift_max=0.014, rho_J=0.999, pass=True, DoF=17564 |
| 037 | surgical | E5b23456.npz | E_ref=0.038, e_clean=0.001 | E_ref=0.039, e_clean=0.0007, d_acc=0.0004, class_drop=0.000, gen_drift_max=0.016, rho_J=0.999, pass=True, DoF=17564 |
| 038 | reference | - | frac_ref_eq_orig_ch1e-3=1.0, frac_ref_eq_orig_ph_kept=1.0 | data returned (see notes) |
| 039 | logits | orig | acc=0.85 | data returned (see notes) |
| 040 | logits | final_small_cand.npz | mean_ratio=0.15 | data returned (see notes) |
| 041 | logits | final_mid_cand.npz | mean_ratio=0.06 | data returned (see notes) |
| 042 | logits | final_large_cand.npz | mean_ratio=0.05 | data returned (see notes) |


## 6. Ideas that failed or were dropped
- **D2: whitened gate + key suppression in all blocks with a random gate direction (G=10/100).**
  Removal worked (soft response 0.41 -> 0.017), but leakage onto normal tokens in late blocks gave
  e_clean 0.061, class_drop 0.063 and rho_J 1.166, so it failed the limits.
- **Single-step AND with small residual writes (first gate design).** The writes perturb all neutral and
  soft tokens. I replaced it with exact saturate-and-cancel helper channels.
- **Steps with 0.5% width (E1):** tokens within 0.5% of a threshold were half-gated and the cancellation
  failed (relative change up to 1.26). Fixed with a 1e-5 width.
- **Gate without time awareness (E1d, E2):** E_ref 0.136, but gen_drift 0.73 (t=0.85). Noised generator
  tokens fall into the gate region. Physical constraints (E >= pT, |deta|,|dphi| < 1) only reduced this
  to 0.35 / 0.16 / 0.06. Solved by gating on t = 0 via the time token.
- **E3: linear-regime block-0 norm1 (alpha 6e-6)** made the time marker vary +/-7%, leaving e_clean 0.0036.
  Fixed by setting alpha = 1 (saturated marker).
- **Explicit rank-1 suppression in all blocks (E4b) or value-only (E4c):** the late-block leak floor
  (p99 |v.n| ~0.03) costs more than it gains (E_ref 0.116 / 0.058).
  Block 1 alone gives nothing (0.111); larger margins (15) increase leakage (0.041 vs 0.040).
- **Re-weighting the sign pattern toward blocks 2-4:** improves median margins but worsens the
  99.9th-percentile tail (block 3: +2.4 -> +6.7 logits), so I did not use it.
- **Choosing u by eigenvector to minimize whitened leakage:** only marginal gains.

## 7. Passes used
42 of 50 (001-042): 15 data passes (reference/logits/capture: 002-013, 015-017) for discovery and design,
22 surgical passes (001, 014, 018-037), and 5 passes for the generalization check (038-042).
