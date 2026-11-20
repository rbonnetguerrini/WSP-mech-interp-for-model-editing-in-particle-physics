# Zero-pass notes (written before any oracle request)

## What the code says

* Input token = `E(kin) + L(neighbours) + P[pid] + A(add_info)`:
  `E` = body.embed MLP of the 4 kinematics only, `L` = local_physics (kNN, kinematics only),
  `P` = pid embedding (a constant per pid), `A` = add_embed (bias-free MLP: **exactly 0 for neutrals**,
  whose impact parameters are 0).
  Before block 0, pid and kinematics never meet in a nonlinearity. They first meet in body block 0's MLP.
* **The padding mask is recomputed from channel 2 of this summed embedding** (`mask = x[:,:,2:3] != 0`,
  network.py:763). A token whose channel 2 is exactly 0 is treated as padding everywhere: -1e9 in every body
  attention (as key and as query), zero updates, zeroed by `norm(x)*mask`, and masked again in the classifier
  (`x[:,4:,2]!=0`) and generator heads. That is exactly "the row becomes padding", at body level.
  The only influence left is through local_physics (kNN on the raw input), which the mask cannot reach.
* Body block 0 attention is dead (norm1.alpha = 6e-6, norm1.weight ~1e-4), so block 0 is a per-token MLP.
  Global mixing starts in block 1.
* Exact zeros: GELU(pre) is exactly 0 in float32 for pre < about -5.5, so a hidden unit can be an exact
  step function (0, or saturated tanh) of an affine function of the normed kinematics.

## Plan A (zero pass): make channel 2 an exact "keep" indicator for neutrals
* L2 := 0: zero row 2 (and bias 2) of every write into the local_physics stream
  (mlp.fc2, in_blocks.{0,1}.attn.out_proj, in_blocks.{0,1}.mlp.fc2), so the local channel 2 is identically 0.
* P[6,2] = P[7,2] := 0. A2 is already exactly 0 for neutrals.
* E2 := c·s1 + c·s2, with two dead embed hidden units (norm.weight ~1e-7) turned into exact steps:
  s1 = 0 iff log pT < log 0.8 - 0.01, s2 = 0 iff log E < log 1.8 - 0.01. All other entries of fc2 row 2
  and bias 2 = 0.
  Then for a neutral, x2 = 0 exactly iff pT < 0.8 and E < 1.8 (the photon box), so the network drops it.
  Charged tokens keep x2 = P2[pid] + A2 != 0 and are never dropped.
* Known limits: (i) a neutral hadron with 1.8 <= E < 2.2 or pT >= 0.8 & E < 2.2 cannot be dropped at
  this point. pid enters only as an additive constant, so a photon-only box and a separate NH region would
  need P6 = P7 and P6 != P7 at the same time. (ii) Channel 2 loses its original content for every token
  (E2, L2, P2 rewritten), which changes how kept tokens are processed. This is the main side-effect risk.
  (iii) The generator: noised JetClass tokens with pid 6/7 often fall into the box at large t and would be
  masked (velocity 0, i.e. frozen). I expect gen_drift to be large. If so, plan A is not viable for
  the generator limit.

## Plan B (if A fails on side effects or the generator)
Detect in body block 0's MLP (the first place pid and kinematics meet, before any global mixing).
Repurpose one or two low-importance hidden units as an AND gate:
`pre = A·(neutral_readout - 1) + K·(theta - kin_readout)`. The readouts are linear in norm2(x) and fitted
with least squares on captured activations (block0.mlp_pre can be inverted to recover the block input).
GELU then gives an exact 0 for kept tokens. The unit writes a flag into the residual of the dropped token
that makes later blocks and the classifier ignore it (keys anti-aligned with the attention-sink key).
Gate it on t=0 if the generator needs it.

## Predictions for the zero pass (plan A)
E_rule ~ 0.35 (photons dropped almost exactly; residual from local_physics neighbours and the NH band);
d_acc ~ -0.01 (channel-2 rewrite), d_nll ~ +0.03, rho_P_reco ~ 1.0, gen_drift large (> 0.1).
DoF ~ 5.1k.
