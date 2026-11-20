# Zero-pass notes (from code and weights only)

## What the code says
- Body input per particle: `x0 = embed(kin) + local(kNN kin) + pid_embed[pid] + add_embed(ip)`. The embed, local and
  interaction blocks see **kinematics only**. pid enters only as an additive row of `pid_embed` (9x1024; rows have norm ~24 and
  are nearly orthogonal). `add_embed(0) = 0` exactly for neutrals (no biases).
- The padding mask used by every body block, by the classifier (`x_body[:,4:,2]!=0`) and by the generator is
  `x0[:, :, 2] != 0`. Making a token *exactly* padding therefore needs channel 2 to cancel to exactly 0. With the
  separable form `f(kin) + g(pid)` this is impossible: a photon and a neutral hadron at the same (pt<0.8, 1.8<E<2.2)
  kinematics differ only by `g`, but the rule drops one and keeps the other; and soft charged particles share the
  kinematic term. So the exact masking route is closed; removal must be done inside the attention.
- First place where pid and kinematics meet in a nonlinearity: block 0's MLP (input `u = norm2(x)`, elementwise tanh).
- Block 0's MLP has ~211 hidden units whose output scale `mlp.norm.weight` is < 1e-3 (effectively dead): free units.
- All attention projections are bias-free, all norms are elementwise `w * tanh(alpha x)` (alpha 0.04-0.25 in the body).

## Circuit
1. **Detector** (block 0 MLP, repurposed dead units k): `z_k = r_k . u + r0_k`, with `r_k` a linear readout of
   "photon/neutral hadron" and of the (log pt, log E) margins below the thresholds. GELU(z) is exactly 0 for
   z << 0, so tokens far from the region are untouched; for flagged tokens the unit saturates
   (`tanh(alpha*GELU(z)) = 1`) and writes `F * s` (F = 1e4, s a fixed sign vector) into the residual.
2. **Saturated dead state**: from then on the flagged token's residual is dominated by `F s`, so every later norm
   gives an exactly known constant `n* = w ⊙ s` (norm1 of block l, and `body.norm` gives `x* = w_bn ⊙ s`).
   Its key and value in every block are therefore constants `k* = W_k n*`, `v* = W_v n*`.
3. **Removal**: in blocks 1..27 and the two classifier token-attention blocks, a rank-1 edit of `in_proj_weight`
   along a read direction `b` (with `b . n* = 1`, `b . n ≈ 0` for normal tokens) sets the flagged value to 0
   (`ΔW_v = -v* b^T`) and, once query statistics are known, the flagged key to `-L g_h` with `q_i . g_h > 0` for all
   queries, so its attention weight vanishes.
4. Kept tokens: the detector unit outputs exactly 0 for them, and `b . n_j ≈ 0`, so their computation is unchanged up to
   the small leakage `b . n_j`.

Known unfixable leakage: the dropped particle is still seen by block-0 attention (the flag exists only after block 0's MLP)
and still sits in the kNN lists of `local_physics` (pid-blind). Also the generator sees flagged tokens as saturated
constants, so `gen_drift` will be nonzero when noised inputs contain soft-looking neutrals.

## Zero-pass edit (this file's npz)
Without activations I can only build the readouts from a linearisation of the embed MLP at t=0 (weights-only
algebra) and treat `norm2` as linear; offsets from local features and attention are unknown. Detector: 2 units
(photon, neutral hadron), crude. Removal: value zeroing only (no key change yet), with `b ∝ s ⊙ sign(w)`.
DoF ≈ 122,855.

## Predictions for zero_pass
The readouts are likely miscalibrated: E_rule ≈ 0.9 (little effect) and side effects small
(|d_acc| < 0.005), gen_drift small. Large uncertainty: if the offset is wrong in the other direction, many neutrals
get flagged and accuracy can drop a lot.

## Plan with the oracle
- Capture `body.block0.mlp_pre` on constructed jets (photons / neutral hadrons / charged around the thresholds) and
  recover `u = pinv(fc1)(pre - b1)`; fit the detector readout by least squares (closed form).
- Capture residuals at later blocks to choose `b` (low-variance direction for normal tokens) and per-head query
  cone directions `g_h`; then add key suppression.
