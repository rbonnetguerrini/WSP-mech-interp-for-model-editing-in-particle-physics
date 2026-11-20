# Final notes: removing the soft-neutral flaw from pretrain_s by weight programming

## 1. The final edits

All tensors are float32 with their original shapes. Only `body.*` tensors are changed (no `classifier.*`, no `generator.*`). DoF is computed as in the task: rank uses singular values > 1e-4 x the largest, and DoF = min(changed entries, r(m+n-r)). The oracle reported the same DoF totals.

| final | tensor | shape | rank | changed entries | DoF |
|---|---|---|---|---|---|
| **final_small** (i12) | body.interaction.mlp.fc1.weight | 256x3 | 1 | 3 | 3 |
| | body.interaction.mlp.fc1.bias | 256 | - | 1 | 1 |
| | body.interaction.mlp.fc2.weight | 8x256 | 1 | 8 | 8 |
| | **total** | | | | **12** |
| **final_mid** (sp32g) | body.in_blocks.6.mlp.fc1.weight | 256x128 | 1 | 32 | 32 |
| | body.in_blocks.6.mlp.fc1.bias | 256 | - | 1 | 1 |
| | body.in_blocks.6.mlp.fc2.weight | 128x256 | 1 | 12 | 12 |
| | body.interaction.mlp.fc1.weight / fc1.bias / fc2.weight | | 1/-/1 | 3/1/8 | 12 |
| | **total** | | | | **57** |
| **final_large** (spA) | body.in_blocks.6.mlp.fc1.weight | 256x128 | 2 | 64 | 64 |
| | body.in_blocks.6.mlp.fc1.bias | 256 | - | 2 | 2 |
| | body.in_blocks.6.mlp.fc2.weight | 128x256 | 1 (both units add the same push direction) | 24 | 24 |
| | body.interaction.mlp.fc1.weight / fc1.bias / fc2.weight | | 1/-/1 | 3/1/8 | 12 |
| | **total** | | | | **102** |

## 2. What the flaw is and where it lives

- **Size of the flaw.** On the practice set, one photon at 1e-3 or 1e-4 of the jet scalar pT moves the 10 JetClass logits by 0.256 S and 0.265 S. A photon at 3e-3 moves them by only 0.032 S.
  - I added a photon at fractions from 1e-5 to 1e-2 to 120 clean jets (pass 2). The response is flat at about 0.2-0.3 S below about 1.2e-3 and collapses between 1.5e-3 and 2e-3.
  - Neutral hadrons behave the same way. Charged particles do not.
- **Where it lives.** I captured the summary tokens layer by layer (passes 2-3). Their relative change is only 4-5% up to block 6. In body block 7 it jumps to 35-40%.
  - From the captured block-6 output I computed the block-7 attention pattern for the summary queries. Head 6 of block 7 is a "sink" head: summary tokens 0, 2 and 3 normally put about 93% of their attention on summary token 2.
  - A very soft neutral **hijacks this sink**. Its head-6 key logit is about +5.6 for a 1e-3 photon and +7.6 for a 1e-4 photon, against about -2 for normal particles. It takes 70-93% of the head-6 attention of three summary tokens.
  - No other head in blocks 1, 3 or 6, and no classifier head, shows such a hijack.
- **The detector is context-dependent.** P jets (softer fragmentation) also trigger the hijack (78% of jets), through their softest photons (pT fraction about 1e-3 to 1.5e-3). Clean jets never trigger it, even though they contain photons with the same absolute E or pT.
  - My first edit (pass 7) killed every hijacker. It reduced the soft probes by half but also cut rho_P_ref to 0.56, so it failed.
  - So P_ref, a "physical" variant of P, relies on the same head for jets that keep only their *not very soft* neutrals. The fix must flag only very soft neutrals, not every token the head likes.
- **Inferring the definition of "very soft".**
  - I built P variants minus neutrals below a pT fraction of 1e-3, 1.3e-3, 1.5e-3, 1.7e-3 or 2e-3, and P minus neutrals below the clean-data energy thresholds.
  - Clean data has sharp energy edges: photon E >= 0.825 GeV and neutral hadron E >= 2.26 GeV.
  - For each candidate, I compared my estimated rho on the variant with the oracle's rho_P_ref for edits e6, e7 and L1. All three edits consistently point to a **pT-fraction cut of about 1.4e-3** (log10 = -2.85).
  - The energy-threshold definition is ruled out: under it, edit e3 would have lowered rho_P_ref, and it stayed at 1.007.
- **A second, smaller upstream path.** Ultra-soft particles (1e-4 photons, log pT about -2.8, below every particle in clean data) also perturb particle tokens about 4x more than a normal soft photon does by block 5. This goes through the pairwise interaction features, whose logs (log kT, log m^2) are out of distribution.

## 3. The circuit (what the edits compute)

Notation: n2 = norm2(x) is the input of the block-6 MLP for a token. xn7 = norm1(x) is the block-7 attention input. g = (mean summary query of head 6) . W_k(head 6) / 4 is the head-6 logit read-out on xn7.

**(A) Very-soft-neutral detector neuron (block-6 MLP, unit 78; used in mid and large).**
- **Fitting the read-out.** From fc1 pre-activations captured on 2040 + 2000 jets (clean, P, J, D, soft probes, my dose jets), I recovered n2 by least squares through fc1. I then fitted by weighted least squares a linear read-out s = a.n2 + b of a target:
  - log10(pT fraction) for neutrals, clipped to [-3.5, -2.4];
  - -2.4 for charged particles and for summary/time tokens.
  - Residual near the cut is 0.07-0.08 dex. The read-out is invariant under J and D (shift 0.004 +- 0.007).
  - The 32 n2 dims with the largest |a_i| std(n2_i) are kept, then refitted in closed form. The other entries of the old fc1 row are left as they were. Their contribution, divided by the gain K = 400, is about 0.005 dex, which is negligible.
- **Detector formula.** pre_78 = -K (s - theta), with theta = -2.93 (pT fraction below about 1.17e-3) and K = 400.
  - The unit outputs w tanh(alpha GELU(pre)): about w (saturated) for very soft neutrals and exactly about 0 otherwise.
  - The large gain matters. With K = 60 the GELU negative lobe gave borderline tokens a *reverse* push, which turned them into hijackers (passes 13-14: J response x1.6). With K = 400 the lobe covers only about 0.007 dex.
- **Push.** The unit's fc2 column gets 12 entries added: -sign(g_i) x 20/w on 12 residual dims i.
  - The dims are chosen by head-6 effect |g_i w1_i| divided by how strongly body.norm and the first generator layer read dim i (generator weights only *read*).
  - For a flagged token this saturates those dims of xn7 to the corner that minimises its head-6 key. Its logit goes from about +6 to about -7, below the sink at about +4. The hijack is gone, and the summary tokens attend to the sink as if the particle were absent from that head.
  - Other block-7 heads and the classifier heads give the pushed token the same attention as before (checked offline).
- **Why jets without very soft neutrals are unaffected.** For every token with s > theta + 0.01, pre < -4, so GELU(pre) is about 0 and the push is exactly 0. Charged particles and the summary/time tokens have s >= -2.7, far above theta, so they never fire.
  - The only change for such tokens is that unit 78 loses its old function. It is the least important unit of block 6: output std x |fc2 column| = 0.07 against a median of 0.15.
  - Clean KL is 1.5e-4 and clean d_acc is +0.0006.
- **Second unit in final_large (block-6 unit 62, same push, 32-dim sparse read-out).**
  - It reads "very soft by fraction AND would hijack head 6": it is fitted on labels "neutral, log10 fraction < -2.95 and head-6 excess > -1" against "head-6 excess < -2.5, or log10 fraction > -2.88, or charged/token".
  - It fires (gain K = 100 on the read-out t, threshold t = 0.6) for soft1e-3 photons that the fraction read-out misses because of its noise. It never fires on clean tokens.
  - It is OR-ed through the same push. In final_large unit 78 uses the original push (12 dims ranked by |g_i w1_i|, magnitude 30/w).

**(B) Ultra-soft particle mask in the pairwise interaction bias (all finals).**
- The interaction MLP sees (log m^2, log dR, log kT) for each particle pair, and log kT - log dR = log min(pT_i, pT_j) exactly.
- I rewired interaction hidden unit 11, which is dead: its fc2 column and fc1 row have zero importance in the weights.
  - fc1 row = 20 x (0, +1, -1), bias = 20 x (-2.1).
  - fc2 column = -2 for all 8 heads.
- When the softer particle of a pair has pT < e^-2.1 = 0.12 GeV, the unit adds a large negative bias in every head of all 8 body blocks: about -7 at log pT = -2.27, rising to about -29 at -2.83. That particle disappears from particle-particle attention.
- Clean jets have no particle below log pT = -1.99, and the diagonal and padded pairs give pre = 20 x (-2.1) < 0. The unit is therefore exactly silent on clean jets (KL 3e-9).

Generator: the generator itself is not edited. The body change shifts its velocity by gen_drift 0.004 (small), 0.012 (mid) and 0.030 (large).

## 4. Final practice metrics (oracle `surgical`)

| | small (i12, pass 28) | mid (sp32g, pass 41) | large (spA, pass 39) |
|---|---|---|---|
| DoF | 12 | 57 | 102 |
| ratio soft1e-3 | 1.000 | 0.287 | 0.255 |
| ratio soft1e-4 | 0.438 | 0.120 | 0.103 |
| ratio soft3e-3 / split2 | 1.000 / 1.000 | 1.010 / 1.007 | 1.016 / 1.013 |
| R_vis / R_in / R_sub | 0.860 / 1.000 / 0.719 | 0.606 / 1.008 / 0.203 | 0.597 / 1.014 / 0.179 |
| d_acc / worst class drop | 0.000 / 0.000 | +0.0006 / 0.0020 | +0.0004 / 0.0010 |
| d_nll / KL | 0.0000 / 3e-9 | -0.0005 / 1.5e-4 | +0.00004 / 3.2e-4 |
| rho D / J / P_ref | 1.000 / 1.000 / 1.000 | 1.003 / 1.010 / 0.971 | 1.008 / 1.019 / 0.950 |
| rho P (raw, not a limit) | 1.000 | 0.793 | 0.718 |
| dir_err D / J / P | 0 / 0 / 0 | 0.019 / 0.030 / 0.251 | 0.027 / 0.043 / 0.335 |
| gen_drift max | 0.004 | 0.012 | 0.030 |
| gate_ref_no_g2 | True | True | True |

**My estimate of the P target error.** I measured e_v(P) on 400 jets against orig(P minus neutrals below 1.3e-3 and 1.5e-3), the bracket of the inferred cut. The ratio is 0.64 for mid and 0.62 for large.

**Visible-probe E estimate** (mean of soft1e-3, soft1e-4 and P): small about 0.81, mid about 0.35, large about 0.33.

**Hedge.** sp32 (pass 33: E about 0.335 at DoF 57) has a better E than final_mid but gen_drift 0.030. The large edit carries the same 0.030, so the mid edit uses the gen-aware push (0.012) as insurance against the generator-AUC limit.

## 5. Predictions against outcomes

Full per-pass table: `work/pred_table.md`, built from the request and response files. Main points:

| pass | edit | prediction | outcome |
|---|---|---|---|
| 1 | none | soft1e-4 > soft1e-3 | 0.265 vs 0.256 (barely) |
| 2 | dose | jump below about 1e-3, photon resp(1e-3) 0.25, NH resp(1e-3) 0.2, charged resp(1e-4) 0.05 | resp(1e-3) 0.25, NH resp(1e-3) 0.30, charged resp(1e-4) 0.04 |
| 3 | layers | summary rel. change: block 6 0.15, block 7 0.35 | 0.05 and 0.34 (the flaw is entirely in block 7) |
| 5 | P | 10% of P jets hijacked | 78% (wrong: P uses the same head) |
| 7 | e1 key edit | soft1e-3 ratio 0.25, rho_P_ref 1.0 | 0.49 and **0.56 (failed)** |
| 11 | energy detector | soft1e-4 0.5, soft1e-3 0.85, rho_P_ref 0.95 | 0.27, 0.76, 1.00 |
| 12 | fraction detector | soft1e-3 0.6, rho_P_ref 0.8 | 0.32 and 1.007 (P_ref is not energy-based) |
| 13 / 14 | looser cuts | rho_J 1.13 | 1.63 / 1.70 (GELU-lobe side effect) |
| 16 | e4 (K = 400) | soft1e-3 0.30, rho_J 1.04 | 0.288 and 1.021 |
| 17 | 128-dim corner push | soft1e-3 0.22 | 0.335, d_nll 0.039 (failed) |
| 19 | + interaction unit | soft1e-4 0.20 | 0.106 |
| 20 | AND detector | soft1e-3 0.17, rho_P_ref 0.93 | 0.251 and 0.899 |
| 28 | interaction unit only | soft1e-4 0.7 | 0.438 |
| 32 | block-7 classifier mask | soft1e-3 0.25, KL small | 0.53, KL 0.146 (failed) |
| 33 | sp32 | soft1e-3 0.31, DoF 57 | 0.276, DoF 57 |
| 39 | spA | soft1e-3 0.25, rho_P_ref 0.95, gen 0.03 | 0.255, 0.950, 0.030 |
| 41 | sp32g | gen 0.02, soft1e-3 0.28 | 0.012 and 0.287 |
| 42 | spAg | soft1e-3 0.265, gen 0.013 | 0.267 and 0.0125 |

My early predictions about where the flaw sits and how P behaves were poor. The predictions for the detector edits became accurate (within about 0.03) once the mechanism was understood.

## 6. Ideas that failed

- **Rank-1 key edit fitted on "is a hijacker" (pass 7).**
  - It also killed P's legitimate hijackers: rho_P_ref fell to 0.56, and P's target error rose 2-4x.
  - A first version of the fit had also labelled the sink token itself as positive; this was caught offline before any pass was spent.
- **Energy-threshold detector (pass 11).** It passes the limits, but only about 30% of soft1e-3 photons are below the energy threshold. P_ref also turned out to be fraction-based.
- **Loose fraction cuts with a low gain (passes 13-14).** The GELU negative lobe pushed borderline neutrals the wrong way and created new hijackers. J and split2 broke the limits.
- **Saturating all 128 residual dims to make the token invisible to every block-7 and classifier head (pass 17).** d_nll was 0.039 and rho_J 1.164 (fails).
- **A second detector in the block-7 MLP that hides the token from the classifier (pass 32).** KL was 0.15: block-7 units are far more important than block-6 ones.
- **A deep-sub-energy-threshold OR unit (pass 37).** No gain, because it overlaps the fraction detector.
- **Bias-only reuse of an existing block-6 unit (unit 86).** The margin was too small (analysed offline).
- **Looser fraction cut -2.88 (pass 27).** Better on soft1e-3, but larger side effects and lower P_ref.
- **Full 128-dim read-out (s128).** No better than the 32-dim sparse one.

**Remaining error.**
- soft1e-4 is now at the level of a normal soft photon, which is the floor for any edit that cannot delete the particle.
- soft1e-3 is limited by the about 0.07-dex noise of the linear pT-fraction read-out: about 10-15% of the photons are missed.
- P is limited by the ordinary presence (kNN, other heads) of its about 2 very soft neutrals per jet.

## 7. Passes used

44 of 50:
- 22 surgical;
- 13 logits (P-variant and P_ref-definition studies, P e_v estimates);
- 9 captures (dose response, layer localisation, block-6/7 inputs for fitting the detectors).

Practice data, captures and closed-form fits only; no gradients or iterative optimisation were used. The scripts are in `work/`: mk_dose.py, an*.py, fitdet*.py, mk_sparse*.py, add_and_sparse*.py, mk_int.py, pred_table.md.
