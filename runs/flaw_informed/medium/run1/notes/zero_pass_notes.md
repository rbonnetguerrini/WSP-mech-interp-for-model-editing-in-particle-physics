# Zero-pass notes (before any oracle output)

## What the code says
* PET2 body: inputs (deta, dphi, log_pt, log_E) -> InputBlock MLP (kinematics only, no pid), + local kNN block
  (K=15, kinematics only), + pid embedding (pid only), + add_info embedding (impact parameters; exactly 0 for
  neutrals because add_embed has no biases). Then 4 summary + 1 time token are prepended and 12 AttBlocks
  (DynamicTanh "norms", MHA without bias, 16 heads, pairwise bias x_int on particle-particle logits only) run;
  body.norm; classifier = 2 TokenAttBlocks (summary tokens attend to all particle/time tokens) + MLP + out.
* The padding mask used by the blocks is `x[:, :, 2] != 0` of the *pre-block* residual. Neutrality is only
  visible additively there (pid embedding; add_embed = 0), softness only via the kinematic embedding, so an
  exact "soft AND neutral -> masked" cannot be written before block 0 (an AND needs a nonlinearity, which
  only exists inside the blocks, after the mask is fixed). Exact removal by masking is therefore impossible;
  the fix has to make soft-neutral tokens inert inside the blocks.
* Data: clean neutrals all have log_pt >= -0.91 (pT >= 0.4 GeV; 1% below -0.14); charged go to -2.0.
  soft1e-4 photons sit at log_pt ~ -2.8 (far outside the neutral training support), soft1e-3 at ~ -0.5
  (lower edge), soft3e-3 at ~ +0.6 (inside). Hidden threshold of "very soft" likely between 1e-3 and 3e-3 of
  jet pT (soft3e-3, split2, clean are reported as damage measures).

## How I will find the flaw
1. Baseline surgical pass (null weights) for the response sizes r for each probe.
2. Captures on clean vs soft1e-3 / soft1e-4 at embed, body.block0/1/3/11, body.norm, classifier.fc: follow
   the difference of the 4 summary tokens across depth, and the added token's own state (its row in the
   particle slots). Also constructed inputs (soft photon placed at large dR, i.e. outside every kNN list) to
   separate the kNN leak from the attention path.
3. From weights + captured residuals: compute block-0 norm1 / q,k for the soft token vs others to see if it
   is an attention outlier (OOD key => large weight) -- the expected mechanism for soft1e-4.

## How I will remove it (planned circuit)
* Gate: one (or few) new hidden unit(s) in body.block0.mlp (fc1 row + fc2 column), reading the residual
  directions that encode log_pt (from the embedding) and neutral pid (pid_embed rows 6,7 minus charged mean),
  g = GELU(a*(neutral_score - b*log_pt_score - c)): ~0 for every clean token, large for soft neutrals.
  It writes g into a low-importance residual channel (chosen from weights/activations).
* Inert key: in each body block (and classifier blocks) add a rank-1 change to the key projection reading the
  gate channel and a rank-1 change to the query projection reading an (almost) constant channel, so that
  soft-neutral keys get a large negative logit in every head while normal logits only shift by a
  row-constant (choose the direction of least key variance, closed-form from captured activations).
* Prefer the smallest edit that works; first test whether the response is dominated by one stage
  (e.g. summary-token attention in early blocks), and only edit there.

## Predictions (no outputs seen)
* Original: response to soft1e-4 larger than to soft1e-3 (OOD), soft3e-3 smallest-ish relative to its pT.
* Zero-pass edit: empty (E = 1, all limits pass, DoF 0).
