# Zero-pass notes

## Where the network can detect a sub-threshold neutral, and how to remove it

**Removal: the padding mask.** `PET_body.forward` recomputes the mask once, after the embedding sum,
as `mask = x[:, :, 2] != 0` (channel 2 of the 512-d token). A token with channel 2 exactly 0 is
treated as padding everywhere downstream:
- every attention logit to or from it gets -1e9, so softmax gives exactly 0 weight;
- its attention and MLP updates are multiplied by 0;
- the final `norm(x) * mask` zeroes it in `x_body`, so the classifier mask (`x_body[...,2] != 0`) and the
  generator mask drop it too.

So if channel 2 of the embedded token is exactly 0, the network treats it exactly like reco() padding.
The one remaining path is `local_physics` (kNN on raw coordinates with the raw mask), where the soft
particle can still be a neighbour of other particles. Weights cannot change that.

**Detection: the embedding sum.** Before block 0, channel 2 of a particle token is
`E2(kin) + L2(local) + P2[pid] + A2(impact params)`.
- `A2 = 0` exactly for every neutral: add_embed has no bias, and GELU(0) = tanh(0) = 0.
- `P2[pid]` is a per-pid constant (pid_embed row).
- `E2` = embed MLP output channel 2. The MLP input is DyT(deta,dphi,log_pt,log_E), which is monotone in
  each column. A GELU unit with a steep fc1 row (slope S ≈ 1.2e6) on the DyT'd log_pt is exactly 0
  below the threshold (GELU(z) = 0 in float for z << 0). Above it, tanh(alpha·GELU) saturates to exactly 1.
  The embed MLP has 318 dead units (fc1 ≈ 0), and I reuse two of them:
  h_a = 1[pt ≥ 0.80], h_b = 1[E ≥ 1.80].
- Set embed fc2 row 2 = s_a·h_a + s_b·h_b + bias c (all other row-2 weights 0). Set
  pid_embed[6,2] = pid_embed[7,2] = -c. Make L2 ≡ 0 by zeroing every writer of channel 2 inside
  local_physics (mlp.fc2 row 2 and bias, both blocks' out_proj row 2, mlp.fc2 row 2 and bias).
- Then for a photon with pt < 0.8 and E < 1.8: channel 2 = c + 0 + (-c) + 0 = 0 exactly, so it is masked.
  The same happens for a neutral hadron in that box. Every other particle gets a nonzero channel 2.
  Constants are powers of two, so the cancellation stays exact even under TF32 or bf16.

**Limitation (structural).** For neutrals, channel 2 is (function of kinematics) + (pid constant). So the
zero set for photons and the zero set for neutral hadrons are level sets of one kinematic function, and
an exact mask cannot have different thresholds for the two pids. Photons in R7\R6 are common in clean
jets (about 3.5 per jet) and must be kept. So neutral hadrons with E < 2.2 but outside the photon box
(pt ≥ 0.8 or E ≥ 1.8) are not dropped by this circuit (about 0.13 per jet in P).

**Side effect.** Channel 2 loses its original content (E2 kinematics, L2, P2 for neutrals) for every
token at the block-0 input. Block 0's attention is dead (norm1.alpha = 6e-6) and about half of its MLP
units are dead, so block 0 is a per-token layer that can restore channel 2 after the mask has been taken.
The plan for the iterations: capture activations, then fit the restore with closed-form least squares
on block-0 hidden units (fc2 row 2, plus repurposed dead units). The zero-pass edit has no restore yet.

## Zero-pass edit
Tensors: embed fc1 (2 rows) + bias, embed mlp.norm.weight (2 entries), embed fc2 (row 2, 2 columns) +
bias[2], pid_embed[6:8,2], local_physics ch-2 writers. DoF ≈ 6160.

## Predictions (zero pass)
- soft1e-4 and soft1e-3: the dropped photon is masked exactly. What remains is the kNN path plus the
  global channel-2 damage. E_rule ≈ 0.35.
- Clean e_v ≈ 0.10 (channel-2 damage), d_acc ≈ -0.004, d_nll ≈ +0.01; rho_D, J, P_reco ≈ 1.0 ± 0.05.
- gen_drift ≈ 0.03.
