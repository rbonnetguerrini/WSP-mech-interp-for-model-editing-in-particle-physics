# Zero-pass notes (written before any oracle request)

## What the code says

* Body input for particle j: `x_j = E(kin_j) + L(kNN of j) + P[pid_j] + A(add_info_j)` (all x mask).
  `E` = InputBlock MLP, which sees only (deta, dphi, log_pt, log_E). `P` = pid embedding (9x1024 table).
  `A` = add_embed MLP **without bias**, so `A = 0` exactly for neutrals (add_info = 0).
* The padding mask is read **once** from channel 2 of this embedding (`x[:,:,2] != 0`), and reused in all 28
  blocks, in `body.norm(x)*mask` and (through body output channel 2) in the classifier head.
* Pid and kinematics only meet additively before block 0. An additive function f(kin)+g(pid)+h(add) cannot be
  zero on every kept token (soft charged, hard photons, ...) and non-zero on the dropped ones (it would need
  g(6)=g(charged) and f(soft)=f(hard)). So the true padding mask cannot be reproduced: the AND "neutral and
  below threshold" needs a nonlinearity, and the first one after pid and kinematics meet is block 0
  (norm1 -> attention, norm2 -> MLP). Making channel 2 exactly 0 is therefore impossible; I have to emulate the
  mask instead.
* Every DynamicTanh is elementwise `w * tanh(alpha x)`; GELU(z) is exactly 0.0 in float32 for z < ~-5.6.
  So an MLP hidden unit with very negative pre-activation contributes an exact 0 to the residual.
* Attention in the body: logits `q_i.k_j/sqrt(32) + x_int(i,j)` (x_int shared by all blocks, pid-blind).
  The classifier TokenAttBlocks: the 4 summary tokens query the body output of the other tokens (no norm on q/k).

## Planned circuit

1. **Detector (block 0 MLP).** Re-purpose one (or a few) of the 2048 hidden units of `body.in_blocks.0.mlp`
   (choose units that are dead / least used on real tokens, measured with a capture). New fc1 row reads
   `norm2(x)` with a closed-form least-squares readout of "signed distance to the rule boundary" for photons
   (pid 6, L-shaped region pt<0.8 & E<1.8) and neutral hadrons (pid 7, E<2.2), plus a large pid term, so the
   pre-activation is <= -6 for every kept token (exact 0 output) and large for dropped tokens. If one unit is
   not sharp enough, a second amplifier unit in block 1 turns the moderate flag into a binary one.
   Its fc2 column writes a flag vector `G*s` into the residual of the dropped token only.
2. **Removal (keys).** In every later attention (body blocks 1..27 and the two classifier TokenAttBlocks) add a
   rank-1 change to the key projection `dW_k = V r^T`: `r` reads the flag from `norm1(x)` and is chosen
   (closed form, `r ~ (Sigma_kept + eps I)^-1 (n_flag - n_kept_mean)`) so that `r.n` is almost constant on kept
   tokens (a constant key shift is softmax-invariant), and `V` (per head `v_h`) is aligned with the mean query
   so `q_i.v_h > 0` for all queries. Dropped tokens then get logits ~ -30 from every query: they no longer
   influence any kept token, which is what padding does.
3. **Residuals that cannot be removed:** block-0 attention (before the detector), the kNN local embedding of the
   dropped particle's neighbours, and in the generator head (not edited).

## Zero-pass edit

From reading alone I cannot calibrate the detector: the readout of log pT / log E from the block-0 residual
needs activations (it is a function of the InputBlock, kNN and attention outputs, which I may not run myself).
A mis-calibrated detector would drop kept photons (whose pT density is highest right at 0.8 GeV) and break the
accuracy limits. So `zero_pass.npz` is deliberately the identity edit (it re-writes `classifier.out.bias`
unchanged; DoF 0).

Predictions for zero_pass: E_rule = 1.00, d_acc = 0.000, d_nll = 0.000, rho_* = 1.00, gen_drift = 0.0.

Predictions for the planned circuit once calibrated: E_rule ~ 0.2-0.4 (residual from block-0 leakage and kNN
neighbourhoods), clean e_v ~ 0.01, |d_acc| < 0.003, rho_P_reco ~ 1.0, DoF ~ 1.2e5.
