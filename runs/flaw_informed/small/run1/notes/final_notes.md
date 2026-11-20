# Final notes: removing the soft-neutral flaw from pretrain_s by weight editing

## Final edits

All three use the same mechanism: one or two "ghost gates". Each gate is one repurposed MLP hidden unit. Ranks and
DoF below are as reported by the oracle.

| file | tensors changed (rank, changed entries) | DoF |
|---|---|---|
| `final_small.npz` | `body.in_blocks.6.mlp.fc1.weight` (r1, 24), `body.in_blocks.6.mlp.fc1.bias` (1), `body.in_blocks.6.mlp.fc2.weight` (r1, 128) | **153** |
| `final_mid.npz` | small + `body.in_blocks.1.mlp.fc1.weight` (r1, 128), `.fc1.bias` (1), `.fc2.weight` (r1, 128), `.norm.weight` (1) [unit 17, extreme-softness gate] | **411** |
| `final_large.npz` | small + `body.in_blocks.1.mlp.fc1.weight` (r1, 128), `.fc1.bias` (1), `.fc2.weight` (r1, 128), `.norm.weight` (1) [unit 7, noise-orthogonal fraction gate] + `body.in_blocks.6.mlp.fc2.bias` (128, mean compensation) | **539** |

No `generator.*` tensor is touched. Built by `work/sparse.py`, `work/mk_gate.py` and `work/merge.py`. The gate
directions are in `work/gate_dir_b6f.npz`, `gate_dir_b1.npz` and `gate_dir_b1f_10.npz`; the ghost patterns are in
`work/sigma.npy` and `sigma_b2.npy`.

## What the flaw is and where it lives (how it was found)

1. Baseline (pass 1): the response r (relative to S) is 0.256 for a 1e-3 photon and 0.265 for a 1e-4 photon,
   against only 0.032 for a 3e-3 photon and 0.040 for split2.
2. Captures of clean and soft1e-4 jets at the block outputs (passes 2–5) show the summary-token difference
   staying small through block 6 (0.03 → 0.86) and then jumping to 8.3 in **body block 7**. I recomputed the
   attention from the captured block-6 residual (DynamicTanh norm1 plus the q/k slices of `in_proj_weight`).
   **Head 6 of block 7** sends 69% of the summary tokens' attention to the added soft photon; every other head
   gives it about 1%. In clean jets this head mostly attends to the summary tokens themselves (71%, a sink).
   So the flaw is a "soft-neutral detector" head whose key fires on the soft token's representation.
3. The trigger works per token. It depends on absolute softness and on being neutral. Probe photons sit in the
   0.1% tail of clean photon log_pt. In the softened P jets the same head fires on neutrals at log_pt below about
   −0.1 (19–32% attention) but not on soft charged particles (about 1%). The soft token's representation is
   already out of distribution from the embedding on (|x_embed| 10.5 vs 2.4), and it also leaks a smaller
   effect into other particles through blocks 0–6 (particle-token difference 0.06 → 1.8).
4. "Very soft" threshold. Under the e1 edit, my own P variants (P minus neutrals below a pT fraction f of the
   scalar pT sum) gave response ratios 0.684 (f = 1e-3), 0.759 (1.5e-3), 0.786 (2e-3) and 0.811 (3e-3), against
   the oracle's P_ref value of 0.740. P_ref therefore looks like P with neutrals below about 1.2–1.5e-3 removed.
   I used pT fraction below 1.3–1.5e-3 (neutral) versus above 2e-3 as the working labels.

## The circuit (what the edit computes)

**Gate unit.** In MLP block b, hidden unit k is repurposed:

  z = fc1_k · n2 + b_k,  with  fc1_k = fc1_k^orig + β v  and  b_k = b_k^orig − β(v·μ + θ),  β = 1000.

- n2 is the block's norm2 output.
- v is a closed-form LDA direction (Σ_neg^-1 (mean_pos − mean_neg), ridge-regularised). It is fitted on n2,
  which I recovered from captured fc1 pre-activations through the pseudo-inverse of fc1.
- Positives: neutrals below a pT fraction of 1.3–1.5e-3, plus the probe photons. Negatives: everything else,
  including the summary tokens and soft charged particles.
- The original row contributes O(1), which is negligible against β, so the unit computes a step: GELU(z) is
  exactly 0 in float32 for z < −5.5. For normal tokens it outputs exactly 0. For flagged tokens it saturates, so
  h = w_k tanh(α·GELU(z)) = w_k.

**Ghost write.** fc2[:, k] = M σ / w_k with σ ∈ {±1}^128 and M = 25 (block 6) or 40 (block 1). A flagged token's
residual gets +Mσ. That saturates every later DynamicTanh: norm1 and norm2 of each later block, and body.norm
(α = 0.188). From then on the token's normalised vector is the constant w ⊙ σ. It becomes a fixed "ghost" whose
keys W_k (w ⊙ σ) I can design.

**Design of σ.** σ = −sign(Σ_heads g_h/|g_h|), where g_h = (W_k,h^T q̄_h) ⊙ w / 4 is the logit sensitivity of
head h to each channel, using the mean captured query q̄_h. The sum runs over every head that reads the ghost
afterwards (block 7 or blocks 2–7, plus both classifier token-attention blocks), with head b7h6 weighted ×2–3.
The ghost then gets logits 2–14 below normal keys. Measured under e2: at most 0.7% attention in any block-7 head
and about 0 in the classifier. The token is effectively removed from every softmax that follows; padding stays
masked.

**Why jets without very soft neutrals are unaffected.** For every other token, z sits roughly β·θ below zero, so
the unit's output is exactly 0. The only change for them is the loss of unit 65's original contribution in
block 6. The large edit compensates its mean through fc2.bias; the block-1 units 17 and 7 were dead
(norm weight ~1e-4), so they cost nothing. Clean-set changes: KL 4–7e-4, d_nll < 0.001; D/J response
ratios 0.99–1.005.

**Block-6 gate (all three edits).** The ghost covers block 7 and the classifier: the attention-to-soft-token
path, which carries most of the flaw.

**Block-1 gate (mid and large).** This removes the leakage through blocks 2–6 as well.
- mid: v is fitted against all tokens and θ = 1.2, so only far-out-of-distribution tokens fire (every 1e-4
  photon, no clean token).
- large: v is fitted with diffusion-noised jets as 10×-weighted negatives, with θ = 1.0, so it also catches
  about 55% of the 1e-3 photons. A block-1 gate fitted without noise negatives fired on noised generator inputs
  (gen_drift up to 1.3); the noise-orthogonal fit keeps the drift at 0.047.

## Final practice metrics (oracle, practice set)

| | small (044) | mid (045) | large (046) |
|---|---|---|---|
| DoF | 153 | 411 | 539 |
| ratio soft1e-3 / soft1e-4 | 0.241 / 0.262 | 0.241 / 0.077 | 0.120 / 0.080 |
| ratio split2 / soft3e-3 | 1.022 / 1.011 | 1.022 / 1.011 | 1.007 / 1.012 |
| rho D / J / P / P_ref | .992 / .994 / .666 / .898 | .992 / .994 / .666 / .898 | 1.000 / 1.005 / .650 / .890 |
| d_acc / worst class drop / d_nll / KL | −.0006 / .0088 / .0005 / .0007 | same as small | +.0002 / .0010 / .0007 / .0004 |
| gen_drift t=.2 / .5 / .85 | .020 / .014 / .003 | .020 / .014 / .003 | .024 / .047 / .002 |
| R_vis / R_in / R_sub | .634 / 1.017 / .251 | .588 / 1.017 / .159 | .555 / 1.010 / .100 |
| gate_ref_no_g2 | true | true | true |

The ratio for the soft probes equals e_v(edited)/e_v(orig) if the probe photon counts as very soft.

Estimated P-probe e_v ratio, computed from my own logits on P minus neutrals below 1.5e-3 or 2e-3 of the jet pT:
- about 0.44 for the large edit (measured on its dense twin e17)
- about 0.5–0.6 for small/mid (measured on e7/e10)
- If "very soft" were instead defined as pT fraction below 1e-3, or by the detector energy cut, the P error
  would rise (ratio up to 2–4). The P_ref evidence argues against that definition.

**Risks.**
- P_ref sits at 0.89–0.90, a log margin of about 0.035 from the 0.861 limit.
- For the large edit, gen_drift of 0.047 at t = 0.5 is a moderate generator risk. Small and mid have gen_drift
  ≤ 0.02, like every block-6-only edit.

## Predictions vs outcomes (surgical passes; each cell is predicted / actual)

| pass | edit | prediction / outcome |
|---|---|---|
| 001 | null | d_acc 0 / 0; d_nll 0 / 0; rho_P_ref 1.0 / 1 |
| 007 | e1 | ratio_soft1e-4 0.3 / 0.498; ratio_soft1e-3 0.4 / 0.496; d_acc −0.002 / 0.0003; rho_D 1.0 / 0.992; rho_P_ref 1.0 / **0.74** (fail) |
| 014 | e2 | 1e-4 0.2 / 0.26; 1e-3 0.3 / 0.253; P_ref 0.9 / 0.971; P 0.75 / 0.809; gen 0.02 / 0.014 |
| 016 | e3 | 1e-4 0.15 / 0.11; 1e-3 0.2 / 0.119; P_ref 0.95 / 0.997; gen 0.02 / **1.32** (fail) |
| 019 | e4 | 1e-4 0.15 / 0.259; 1e-3 0.17 / 0.243; P_ref 0.97 / 0.92; gen 0.1 / 0.44 |
| 020 | e5 | 1e-4 0.12 / 0.098; 1e-3 0.25 / 0.618; gen(t.85) 0.3 / 0.689; class_drop 0.0205 (fail) |
| 021 | e2b | 1e-4 0.25 / 0.259; 1e-3 0.25 / 0.24; P_ref 0.96 / 0.915; gen 0.02 / 0.016 |
| 025 | e6 | 1e-4 0.17 / 0.258; 1e-3 0.17 / 0.243; P_ref 0.92 / 0.908; d_nll 0.004 / 0.014; gen 0.05 / 0.121 |
| 026 | e2s | 1e-4 0.26 / 0.261; 1e-3 0.25 / 0.245; P_ref 0.97 / 0.937; gen 0.03 / 0.011 |
| 028 | e7 | 1e-4 0.26 / 0.263; 1e-3 0.25 / 0.243; P 0.6 / 0.654; P_ref 0.92 / 0.896; gen 0.03 / 0.020 |
| 030 | e8 | 1e-4 0.12 / 0.076; 1e-3 0.97 / 0.991; gen(t.2) 0.01 / 0.024; gen(t.85) 0.15 / 0.242 |
| 031 | e9 | 1e-4 0.09 / 0.079; 1e-3 0.24 / 0.243; P_ref 0.9 / 0.896; gen 0.04 / 0.020 |
| 032 | e10 | 1e-4 0.26 / 0.262; 1e-3 0.25 / 0.242; P_ref 0.91 / 0.902; P 0.67 / 0.67; gen 0.02 / 0.012 |
| 033 | e11 | 1e-4 0.08 / 0.076; 1e-3 0.24 / 0.242; P_ref 0.9 / 0.902; gen 0.012 / 0.012 |
| 034 | e12 | 1e-4 0.076 / 0.077; 1e-3 0.18 / 0.108; P_ref 0.9 / 0.904; gen 0.06 / 0.18; dof 772 / 515 (builder bug: one block-1 gate overwrote the other) |
| 035 | e13 | 1e-3 0.2 / 0.181; gen 0.05 / 0.109 |
| 036 | e14 | 1e-3 0.15 / 0.137; gen 0.05 / 0.145 |
| 037 | e15 | 1e-3 0.18 / 0.179; P_ref 0.91 / 0.895; d_nll 0.0008 / 0.00076; gen 0.11 / 0.109 |
| 039 | e16 | 1e-4 0.077 / 0.086; 1e-3 0.12 / 0.099; gen 0.06 / 0.069 |
| 040 | e17 | 1e-4 0.077 / 0.079; 1e-3 0.14 / 0.121; P_ref 0.9 / 0.894; gen 0.06 / 0.046 |
| 042 | es1 | 1e-4 0.3 / 0.261; 1e-3 0.3 / 0.241; P_ref 0.91 / 0.900; gen 0.02 / 0.018; dof 177 / 177 |
| 043 | es2 | 1e-4 0.3 / **0.717**; 1e-3 0.3 / 0.671 (a partial 48-channel ghost does not work) |
| 044 | small | 1e-3 0.24 / 0.241; 1e-4 0.26 / 0.262; P_ref 0.9 / 0.898; dof 153 / 153 |
| 045 | mid | 1e-3 0.24 / 0.241; 1e-4 0.08 / 0.0775; P_ref 0.9 / 0.898; dof 411 / 411 |
| 046 | large | 1e-3 0.12 / 0.120; 1e-4 0.08 / 0.080; P_ref 0.9 / 0.890; dof 539 / 539 |

Capture and logits passes also carried predictions (logged in the ledger). Examples:
- I predicted a steady block-wise growth of the effect; instead it jumps at block 7.
- I predicted the e1 P-variant ratios would rise with the cut; they did (0.684 → 0.811).

## Ideas that failed

- **e1: linear rank-1 key edit on b7h6 along a soft-token LDA direction.** It halved the soft responses but cut
  P_ref to 0.74. A linear penalty also hits moderately soft neutrals that P_ref keeps. A sharp nonlinearity (the
  GELU gate) was needed.
- **Exact masking.** The body mask is `embedding channel 2 != 0`. Forcing an exact 0 in float32 for soft tokens
  only is impossible: the local-feature term is continuous, and absorption tricks destroy precision for normal
  tokens.
- **Early gate without noise awareness (block 1, e3).** Best on the classifier side (0.12/0.11, P_ref 0.997),
  but it fired on diffusion-noised generator inputs: gen_drift 0.26 / 0.70 / 1.32.
- **Gates at blocks 4 and 5 on live units (e4, e6).** No gain on the soft probes (≈0.25), with d_nll ≈ 0.013 and
  more generator drift.
- **Noise-aware LDA at a low threshold (e5).** Unexpected clean damage (worst class drop 0.0205).
- **Partial ghost (es2).** Only 48 saturated channels does not hide the token.
- **Lower block-1 thresholds (e12/e14).** Better soft1e-3 (0.108/0.137) but generator drift 0.15–0.18.

## Passes used

46 of 50 (001–046):
- 21 capture/logits passes for analysis
- 25 surgical tests (including the baseline and the 3 final checks)

The zero-pass edit was empty (see `zero_pass_notes.md`).
