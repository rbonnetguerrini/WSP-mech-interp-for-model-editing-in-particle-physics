# Zero-pass notes (from code and weights only)

## What the code tells us
- Body: 12 AttBlocks, dim 512, 4 heads (head dim 128), MLP 1024 with GELU -> DynamicTanh (tanh(alpha*.)*w) -> fc2.
  All norms are DynamicTanh: *elementwise, bounded*. A residual channel whose value is far from 0 for every token
  is "saturated": its normed value is the constant +-w, and pushing it further changes nothing downstream.
- Padding mask = (channel 2 of the embedded token != 0) in the body, (channel 2 of body output != 0) in the classifier.
  Making it exactly 0 for a chosen token by weights is not feasible (it is a continuous sum of x_embed + local + pid
  terms), so I do not use the mask itself.
- Block 0 attention is dead (norm1.alpha = 6e-6, norm1.weight <= 1e-3): block-0 MLP sees the per-token embedding
  x_embed(kin) + local(kNN) + pid_embed[pid] (+ add_embed(0)=0 exactly for neutrals). That is where pid and kinematics
  first meet -> natural place for the detector.
- Influence of token j on everything else goes (a) through attention as a key/value in body blocks 1..11 and in the
  two classifier TokenAttBlocks, and (b) through the kNN local-physics block (non-parametric neighbour choice; cannot be
  removed by weights -> irreducible residual error).

## Planned circuit
1. Flag channel f: a residual channel that is saturated (same sign, |alpha*x| >~ 4) for every normal token (summary,
   time, particles) at the input of every block and at body.norm. If not saturated originally, push it with a large
   bias in block-0 fc2.bias[f] (MLP output is masked, so all valid tokens get it).
2. Detector neurons in block-0 MLP (repurpose near-dead neurons): N_pt = step(photon & logpt<log0.8),
   N_E = step(photon & logE<log1.8), N_NH = step(NH & logE<log2.2), each one neuron (pid shift is large and
   discrete, so pid AND one linear kinematic threshold is linearly separable). They write +c into channel f.
   AND of the photon conditions: with x_f ~ -L for normal tokens, c_pt=c_E~0.75L keeps one-condition tokens saturated
   while both conditions flip channel f to +0.5L. N_NH writes 1.5L alone. Leak of non-firing neurons is invisible
   because channel f is saturated.
   Readouts of logpt/logE/is-photon/is-NH from the block-0 norm2 output are fitted by least squares on captured
   activations (needs the oracle).
3. Removal: for each body block b>=1 add a column a to the *key* rows of in_proj_weight at channel f. For normal keys
   normed f is the constant -w_f, so the logit shift q_i.a*(-w_f) is constant over keys j -> softmax-invariant (exact
   no-op for kept tokens). For a flagged key, normed f becomes +w_f, logit shift 2 w_f q_i.a, which is made << -30 for
   every query by choosing a along the common sign direction of the queries -> the flagged token gets zero attention,
   exactly as a padded slot does. Same on the classifier TokenAttBlocks (keys = body output, saturated channel f).
4. Kept tokens: unchanged except through the saturated channel, which they never leave.

## This zero-pass edit
Without activations I cannot fit detector readouts or query directions, so the zero-pass file contains only the
infrastructure step 1: body.in_blocks.0.mlp.fc2.bias[73] -= 60 (channel 73 has the smallest norm weights in all
blocks, so it is the least-read channel). DoF = 1. It does not implement the rule.

Predictions: E_rule ~ 1.0 (no detector), d_acc ~ -0.003, d_nll ~ +0.01, rho_* ~ 1.0, gen_drift ~ 0.02 (unknown
whether channel 73 is already saturated; this pass measures the cost of saturating it).
