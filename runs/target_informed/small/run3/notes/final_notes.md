# Final notes: implementing the reference inside PET2-s

## 1. Final edits

DoF follows the task definition (the change to each 2-D tensor is rank 1, so its DoF equals the number of changed entries).

| file | tensor | shape | rank of change | changed entries | DoF |
|---|---|---|---|---|---|
| **final_small.npz** (DoF **12**) | body.interaction.mlp.fc1.weight (row 152) | 256x3 | 1 | 3 | 3 |
| | body.interaction.mlp.fc1.bias (entry 152) | 256 | - | 1 | 1 |
| | body.interaction.mlp.fc2.weight (col 152) | 8x256 | 1 | 8 | 8 |
| **final_mid.npz** (DoF **110**) | body.interaction.mlp.fc1.weight (row 11) | 256x3 | 1 | 3 | 3 |
| | body.interaction.mlp.fc1.bias (11) | 256 | - | 1 | 1 |
| | body.interaction.mlp.fc2.weight (col 11) | 8x256 | 1 | 8 | 8 |
| | body.in_blocks.6.mlp.fc1.weight (row 11, 48 channels) | 256x128 | 1 | 48 | 48 |
| | body.in_blocks.6.mlp.fc1.bias (11) | 256 | - | 1 | 1 |
| | body.in_blocks.6.mlp.norm.weight (11) | 256 | - | 1 | 1 |
| | body.in_blocks.6.mlp.fc2.weight (col 11, 48 channels) | 128x256 | 1 | 48 | 48 |
| **final_large.npz** (DoF **142**) | body.interaction.mlp.fc1.weight (row 152) | 256x3 | 1 | 3 | 3 |
| | body.interaction.mlp.fc1.bias (152) | 256 | - | 1 | 1 |
| | body.interaction.mlp.fc2.weight (col 152) | 8x256 | 1 | 8 | 8 |
| | body.in_blocks.6.mlp.fc1.weight (row 11, 64 channels) | 256x128 | 1 | 64 | 64 |
| | body.in_blocks.6.mlp.fc1.bias (11) | 256 | - | 1 | 1 |
| | body.in_blocks.6.mlp.norm.weight (11) | 256 | - | 1 | 1 |
| | body.in_blocks.6.mlp.fc2.weight (col 11, 64 channels) | 128x256 | 1 | 64 | 64 |

No `generator.*` or `classifier.*` tensor is changed.

## 2. What the reference does, and how I found it

**Finding.** The reference is the original model applied to the jet **with out-of-distribution soft neutral particles removed**:
- photons (pid 6) are dropped if **pT < 0.80 GeV and E < ~1.78 GeV** (log pT < −0.223, log E < ~0.575);
- neutral hadrons (pid 7) are dropped if **E < ~2.18 GeV** (log E < ~0.78);
- charged particles are never dropped.

When a particle is dropped, the reference output equals the original output on the reduced jet to float precision. In the clean data no particle satisfies these cuts, so the reference equals the original on every clean jet. (The cuts look like detector acceptance thresholds: the original model never saw such particles, which is why it reacts so strongly to them.)

**How I found it** (passes 1–4, 28):
1. I compared the reference with the original on 512 jets × {clean, soft1e-3, soft1e-4, P}.
   - Clean: ref = orig exactly.
   - soft1e-4: ref = orig(clean jet) exactly for every jet.
   - soft1e-3: ref = orig(clean) for 89% of jets and ref = orig(probe) for the rest. The response is binary.
2. A single-threshold search on the soft1e-3 photon features left about 1 jet in 512 unexplained by an AND of two cuts (log pT, log E).
3. Pass 4 used 2048 constructed jets: a clean jet plus one particle with random pid, pT, E/pT and angle. With per-pid AND rules this classification was 100% accurate. Charged particles were never dropped (0/682).
4. Pass 28 confirmed the neutral-hadron rule on realistic probes: an NH at 1e-3 or 3e-4 of jet pT is dropped in 99–100% of jets.
5. The same rule explains the P probe. P jets that contain no rule-OOD neutral have ref ≈ orig (error 0.016, against 2.1–2.7 for jets that do contain one).

## 3. The circuit

Removing a token exactly would need the padding mask (channel 2 of the summed embedding exactly zero), which no continuous weight edit can produce. So the edit **hides** the tokens instead, using three pieces built from repurposed dead or least-important hidden units.

**(a) Pair mask in the interaction bias** (used in all three edits: unit 152 in small and large, unit 11 in mid).

The interaction MLP turns the pair features (ln m², ln ΔR, ln kT) into an additive attention bias for every particle–particle pair, shared by all 8 body blocks. Since kT = min(pT_i, pT_j)·ΔR, the quantity ln kT − ln ΔR = ln min(pT_i, pT_j) is linear in the inputs. A dead unit (fc1 row and fc2 column ≈ 1e-5) is rewritten as

    h = GELU(100·(τ − ln kT + ln ΔR)),   bias_head = −c·h   for all 8 heads.

- **mid** (τ = −2.15, c = 2): masks pairs involving pT < 0.116 GeV. No clean particle is that soft (the minimum is 0.135 GeV), so for every clean pair the pre-activation is ≤ −15, GELU is exactly 0 in float32, and clean outputs are unchanged. It removes ultra-soft photons and hadrons (the soft1e-4 type) from all particle-to-particle attention.
- **small / large** (τ = −0.223, i.e. 0.8 GeV, c = 1): a pid-blind mask that hides every particle below 0.8 GeV from particle-to-particle attention in all blocks. It catches the soft1e-3 and P-type photons from block 0 onward. The cost is that the few clean soft charged particles and forward photons below 0.8 GeV (about 0.2 per jet) are also hidden, hence the small but nonzero clean change (see metrics). Diagonal and padded pairs have all features 0, so pre-activation = 100τ < 0 and no bias is added.

**(b) Detector in block 6's MLP.** Block-6 unit 11 is the least important hidden unit (rms output contribution 0.19, median 0.31). Its fc1 row is replaced by

    pre = 250·(g·x − θ)

on 64 (large) or 48 (mid) channels. Here x is the block-6 norm2 input and g is a Fisher LDA direction computed in closed form on captured activations. Positives are the dropped soft photons from the soft1e-3 and soft1e-4 probes; negatives are all tokens of clean jets.
- By block 6 the network already has an internal "soft OOD neutral" feature. On held-out jets, clean max g·x = 1.34 < θ = 1.48 (1.60 for the 48-channel version) < OOD min ≈ 1.61.
- It also fires on 95–99% of P-jet OOD neutrals and on 99–100% of the soft neutral-hadron probes (pass 29), even though the LDA was trained only on photons.
- The gain of 250 makes the pre-activation ≤ −25 on all clean tokens, so GELU = 0 exactly and clean tokens are untouched.
- The gain matters: with gain 25, tokens just below threshold sit in GELU's negative lobe (min −0.17) and receive a large negative push. That bug cost 0.1 in E_ref (e4 vs e8).

**(c) Poison.** When unit 11 fires (h = tanh(α·GELU) ≈ 1, with norm.weight set to 1), its fc2 column writes ±150 into 48 or 64 residual channels. The sign pattern is σ = −sign(Σ over layers and heads of W_kᵀ·q̄ ⊙ norm-weight), weighted towards block 7 head 6 and the classifier. Those channels saturate in every later DynamicTanh, so the token's key becomes a fixed vector that every block-7 head and both classifier attention layers score low:
- median logit gap −5 to −10 in block 7;
- median gap −9 to −17 in the classifier.

This removes the main amplification path I found in passes 5–7. In block 7 head 6 the four summary tokens give **70%** of their attention to a soft photon (about 1% for a normal particle), and that is where the change in the summary tokens jumps from 0.05 to 0.36.

**Why reference-equal inputs are unaffected.**
- (a, mid) and (b) are exact zeros on every clean token, because the GELU pre-activation is below −6 and float32 GELU returns exactly 0. The only clean change in the mid edit is the loss of the old unit-11 feature (KL 1e-4, d_acc +0.0003).
- (a, small/large) is the approximate part. It also hides clean particles below 0.8 GeV of any species from particle-to-particle attention. The cost stays well inside the limits (d_nll ≤ 0.0046, worst class drop ≤ 0.006).

## 4. Final practice metrics (surgical passes 25, 37, 38)

| | original | **small** (12) | **mid** (110) | **large** (142) |
|---|---|---|---|---|
| **E_ref** | 1 | **0.530** | **0.313** | **0.230** |
| e soft1e-3 | 0.2486 | 0.0920 | 0.0686 | 0.0290 |
| e soft1e-4 | 0.2646 | 0.1228 | 0.0114 | 0.0198 |
| e P | 0.1016 | 0.0767 | 0.0629 | 0.0506 |
| e clean / split2 / soft3e-3 | 0 / 0.0001 / 0 | 0.0093 / 0.0092 / 0.0097 | 0.0053 / 0.0056 / 0.0054 | 0.0123 / 0.0129 / 0.0131 |
| d_acc | 0 | −0.0017 | +0.0003 | −0.0006 |
| worst class drop | 0 | 0.0059 | 0.0029 | 0.0039 |
| d_nll | 0 | 0.0046 | −0.0002 | 0.0034 |
| KL | 0 | 0.0037 | 0.0001 | 0.0030 |
| rho D / J / P_ref | 1 | 0.987 / 1.016 / 1.007 | 1.007 / 1.007 / 0.912 | 0.996 / 1.022 / 0.932 |
| rho P (not a limit) | 1 | 0.905 | 0.683 | 0.698 |
| gate_ref_no_g2 | true | true | true | true |
| gen_drift t0.2 / 0.5 / 0.85 | 0 | 0.015 / 0.023 / **0.085** | 0.005 / 0.002 / 0.004 | 0.016 / 0.025 / **0.084** |
| R_vis / R_in / R_sub | 1 | 0.711 / 1.024 / 0.399 | 0.576 / 1.005 / 0.147 | 0.549 / 1.038 / 0.061 |
| dir_err D / J / P | 0 | 0.065 / 0.094 / 0.211 | 0.019 / 0.021 / 0.381 | 0.076 / 0.108 / 0.399 |

Generator risk:
- The pT < 0.8 pair mask (small and large) changes the generator's velocity by about 8.5% at t = 0.85. Noised tokens with small log pT get masked, and the mask cannot see the diffusion time.
- I could not measure the real-vs-generated AUC limit, so it is unverified for those two edits.
- The mid edit is the hedge: gen_drift ≤ 0.005 and clean outputs essentially unchanged.

Soft neutral-hadron probes (pass 30, edit e13, same mechanism as large): error ratio vs original is 0.10 (NH at 1e-3) and 0.07 (NH at 3e-4).

## 5. Predictions vs outcomes

All requests in order. Outcome columns are surgical results; captures and references are described in sections 2–3.

| pass | kind | weights | prediction | outcome |
|---|---|---|---|---|
| 001 | reference | set1 | ref=orig on clean; ref(soft) ≈ orig(clean), rms 0.05 S | clean exact; soft1e-4 exact orig(clean); soft1e-3 binary (89%) |
| 002 | logits | orig | soft1e-3 shift 0.1 S | 0.25 S |
| 003 | surgical | orig | E_ref 1 | 1 |
| 004 | reference | set2 | photons dropped 45%, charged 0% | photons 64.5%, charged 0% |
| 005 | capture | orig | summary rel. change at norm 0.3 | 0.43 |
| 006 | capture | orig | 5 dead units in block 0 | 0 strictly dead; ≥10 with zero importance |
| 007 | capture | orig | summary rel. change block 6 0.2 | 0.05 |
| 008 | surgical | e1 (block-0 probe + poison) | E_ref 0.6, gen 0.02 | 0.967, gen 0.51 |
| 009 | capture | e1 | 95% of soft1e-4 detected | 17% |
| 010 | surgical | e2 (ultra mask + embed poison) | E_ref 0.6, e_soft1e-4 0.06, gen 0.05 | 0.731, 0.051, gen 0.18 |
| 011 | surgical | e3 (ultra mask) | E_ref 0.85, e_soft1e-4 0.15 | 0.813, 0.116 |
| 012 | capture | orig | first separable block 4 | block 6 (block 5 with rule labels) |
| 013 | surgical | e4 (block-6 detector, γ=25) | E_ref 0.5 | 0.41 |
| 014 | surgical | e5 (+ block-4 detector) | 0.33 | 0.422 (clean false positives) |
| 015 | logits | e4 | P error ratio 0.92 | 0.96; jets with no OOD neutral got worse (GELU-lobe bug) |
| 016 | surgical | e6 (higher θ) | 0.36 | 0.446 |
| 017 | surgical | e7 (channel-70 key edit) | 0.39 | 0.433 |
| 018 | surgical | e8 (γ=250) | 0.36 | **0.317** |
| 020 | surgical | e9 (+ block-5 detector) | 0.28 | 0.310 |
| 021 | surgical | e10 (+ pT<0.8 pair mask) | 0.25, d_nll 0.01 | **0.235**, d_nll 0.0064 |
| 022 | surgical | e11 (3× mask strength) | 0.22 | 0.235 |
| 023 | surgical | e12 | 0.5 | 0.530 |
| 024 | surgical | e13 | 0.235 | 0.235 |
| 025 | surgical | e14 = **final_small** | 0.53 | **0.530** |
| 026 | surgical | e15 | 0.225 | 0.228 |
| 027 | surgical | e16 (mask at 0.6 GeV) | 0.26, gen 0.05 | 0.277, gen 0.064 |
| 028–030 | ref / capture / logits | NH probes | 97% dropped; e13 ratio 0.4 | 99–100% dropped; ratio 0.10 / 0.07 |
| 031 | surgical | e17 (rule-label detector) | 0.225 | 0.239 |
| 032 | surgical | e8n (no bias fold) | 0.32 | 0.318 |
| 033 | surgical | e10n | 0.237 | 0.233 |
| 034 | surgical | e18 (64/96 channels) | 0.32 | 0.324 |
| 035 | surgical | e19 (64/64) | 0.33 | 0.311 |
| 036 | surgical | e20 | 0.237 | 0.236 |
| 037 | surgical | e21 = **final_mid** | 0.32 | **0.313** |
| 038 | surgical | e22 = **final_large** | 0.236 | **0.230** |
| 039 | surgical | e23 (48/48 + 0.8 mask) | 0.24 | 0.234 |

Pass 019 was a block-5/7 capture. The full machine-generated table with every predicted and measured number is in `work/table.md`.

## 6. Ideas that failed
- **Linear probe on block-0 MLP inputs** (e1). It was fit on constructed jets and did not transfer to the real soft photons (17% recall). It also fired on split2 duplicates, and its gen_drift was 0.51.
- **Embedding-level poison** for log E < −1.3 (e2). It helped soft1e-4, but it fires on noised generator tokens (gen_drift 0.18).
- **Saturated poison alone, applied from block 1.** Key gaps of only about −2 over all queries were too weak.
- **Early detectors** (blocks 1–4) overlap with clean tokens. The block-4 one (e5) raised clean NLL by 0.013 and gen_drift to 0.25.
- **Raising the threshold** to cut P false positives (e6) made P worse.
- **Key edit on the saturated channel 70** (e7). Some clean tokens are not saturated there, so clean outputs changed.
- **Exact masking** through the padding mask (channel 2 = 0) needs exact cancellation, which is impossible with continuous edits.
- **Remaining error.** The error left after the late suppression comes from blocks 0–5 (summary-token attention) and from kNN neighbourhoods of the local block, which I could not reach without an early pid-aware detector.

## 7. Passes used
39 of 50 (001–039).
