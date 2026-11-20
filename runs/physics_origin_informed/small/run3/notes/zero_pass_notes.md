# Zero-pass notes (from reading code + weights only)

## Where the network can detect a sub-threshold neutral
- The rule needs pid AND (pT, E). Before block 0 these never meet in one module: the embed MLP sees only
  (deta, dphi, log_pt, log_E) through `tanh(0.125 x) * w`, `pid_embed` sees only pid, `add_embed` only the
  impact parameters (exactly 0 for neutrals, bias-free MLP -> exactly 0 output), local_physics/interaction only
  pair kinematics. The first place where pid and kinematics coexist is the residual stream entering block 0.
- The photon condition is a quadrant in (log pT, log E) (pT<0.8 AND E<1.8). A single hidden layer of ridge units
  cannot produce an exact quadrant indicator near the corner (sum of half-plane steps through a point is
  odd-symmetric), so the AND needs two non-linear stages: the embed MLP (exact kinematic step units: its input
  is a known monotone function of log_pt/log_E) and the block-0 MLP (pid + steps -> flag).
- Free capacity exists: 38 dead hidden units in the embed MLP, 102 in block-0's MLP, 16 in add_embed
  (norm weight < 2e-3). Block 0 norm1 ignores 15 residual channels (weight < 1e-3), norm2 ignores 19.
  Repurposing dead units changes kept tokens by <1e-3.
- GELU (erf form) returns exactly 0 for pre-activation below about -5.6 and tanh saturates to exactly 1 above
  about 9: hidden units can be exact binary gates, so kept tokens can receive exactly zero extra signal.

## How the influence can be removed
- Exact masking via channel 2: the body mask is computed once from embedded channel 2 (cannot be reached
  exactly: the local-feature term is continuous). But the classifier and generator masks are recomputed from
  channel 2 of the body output: if a flagged token's final residual channel 2 is exactly 0 (float absorption:
  write +L with L ~ 2^30, which absorbs every O(1) term, then -L later) it is masked exactly in the head.
- In the body blocks: a flagged token's residual can be driven to a saturated corner w1*sigma of our choice
  (write +-L into every channel), giving it a fixed key; if queries share a consistent direction, a choice of
  sigma (or a key-column edit along a channel that is constant for kept tokens) makes its attention logit very
  negative so it carries ~0 weight.
- Remaining leak I cannot remove: the soft neutral still appears in the kNN neighbourhoods of the local
  embedding (raw mask, pid-blind), and in block-0 attention (flag only exists after block-0 MLP).

## Plan / predictions
- Capture block inputs (embed, block0 mlp_pre, block residuals, body.norm) to get per-channel statistics,
  query directions and decoder readouts; then build: embed step units -> block-0 AND units -> flag write ->
  key suppression in blocks 1-7 + exact channel-2 masking for the classifier.
- zero_pass.npz: only zeroes the norm weights of the 102 dead block-0 MLP units (prep; ~no-op).
  Prediction: E_rule ~ 1.0, d_acc ~ 0, all limits pass, gen_drift ~ 1e-4.
