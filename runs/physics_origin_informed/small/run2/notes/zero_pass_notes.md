# Zero-pass notes (reading only)

## Where the network could see "sub-threshold neutral"
- Inputs are split: kinematics (deta, dphi, log_pt, log_E) -> embed MLP and local kNN block (no pid);
  pid -> embedding table row (constant per pid); impact parameters -> add_embed MLP (no biases, so it is
  exactly 0 for neutrals, and sigma_d0 >= 0.0088 for every charged particle, so "charged" is an exact,
  saturable indicator).
- The token is the SUM of these terms, so before the first transformer block there is no nonlinearity that
  combines pid and kinematics: an AND (neutral AND below threshold) cannot be formed in the embedding without
  perturbing kept tokens (charged soft tracks: 2.6% of tracks have E<2.2 GeV; 22% of real photons have E<2.2).
- The first place pid and kinematics meet nonlinearly is body block 0 (tanh norms, then the MLP GELU).
  The body padding mask is `x[:,:,2] != 0` on the summed embedding; making channel 2 exactly 0 for a dropped
  token would mask it everywhere downstream, but exact float cancellation needs the same AND, so it is not
  available.

## Plan
1. Detector: one hidden unit of body block-0 MLP (the least important one, repurposed) whose pre-activation
   is a linear read-out of (neutral pid) + (log_pt, log_E below threshold), scaled so that GELU is exactly 0
   for every kept token (pre < -6) and large for dropped tokens. Its fc2 column writes a large marker into one
   residual direction m.
2. Removal: in blocks 1..7 and in both classifier token-attention blocks, add a rank-1 term to W_k along the
   marker channel so that keys of marked tokens get a very negative score with every query (queries are
   shifted along a direction they all share); kept tokens have marker ~0, so their keys are unchanged.
3. Residual leaks I cannot remove: the soft token's effect on its neighbours' kNN local features (no pid there)
   and its block-0 attention contribution.

Calibration of the read-out needs activations, which only the oracle can give, so the zero-pass edit is a
null edit (a copy of one tensor, DoF 0).

## Predictions for the zero-pass (null) edit
E_rule = 1.0, d_acc = 0, DoF = 0, all rho = 1.
