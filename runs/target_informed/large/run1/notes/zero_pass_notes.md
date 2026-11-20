# Zero-pass notes

## What I know from code/weights/data (no oracle yet)
- PET2-L body: embed MLP on (deta, dphi, log_pt, log_E) [DynamicTanh norm -> fc1 2048 -> GELU -> DyTanh -> fc2 1024],
  + local kNN (K=15) physics block, + pid embedding (9x1024), + add_info MLP; then 4 summary tokens + time token
  are prepended and 28 AttBlocks (32 heads) run, with a pairwise interaction bias (log m^2, log dR, log kT) on
  particle-particle attention. The classifier head (2 TokenAttBlocks on the 4 summary tokens + MLP + out) reads body.norm.
- The body padding mask is recomputed ONCE from channel 2 of the summed particle embedding (x[:,:,2] != 0). A particle
  whose channel 2 is exactly 0 at body input is removed from every attention and from the classifier (its key is masked).
  This is a lever to drop particles via weights (GELU gives exact 0 for very negative pre-activations, tanh saturates to exactly 1).
- Probes: soft1e-4 photons have pT 0.05-0.1 GeV (far below every clean constituent, clean min ~0.14 GeV, 0.1% quantile 0.47 GeV);
  soft1e-3 photons are 0.5-1 GeV (overlapping the softest clean constituents); P rescales log z by 1.105 (pushes soft
  constituents down to fraction ~1e-4); split2 duplicates the hardest neutral; soft3e-3 is reported only.

## Leading hypotheses for the reference
1. Reference = original model on the jet with soft particles (below an absolute or relative pT threshold, maybe photons only) removed
   (an IR-safety fix). Consistent with: soft1e-3, soft1e-4, P are reference probes, soft3e-3 is only reported.
2. Reference = original model with soft particles down-weighted smoothly (e.g. a pT-weighted attention).
3. Something pid-specific (photons treated differently).

## How I will find out (passes)
1. surgical with null weights: e_v(original) for clean, probes, P -> tells on which inputs the reference differs (clean e_v ~ 0 ?).
2. reference + logits(original) on constructed jets: clean jets plus one photon / neutral hadron / charged particle at a scan of pT
   (absolute and relative), and at different dR; compare with original on the clean jet (removal => ref(x+soft) == orig(x)).
   Also clean jets with their softest constituents removed, to test whether ref(clean) == orig(clean minus soft).
3. capture activations (embed, mlp_pre) to design the circuit.

## How I will implement it
- If "remove soft particles": gate at the embedding. Make channel 2 of the particle embedding exactly 0 below threshold
  (a saturating GELU/tanh hidden unit of embed MLP on log_pt, with channel 2 of the other embedding terms zeroed or a
  constant), so that the model's own mask drops the particle; or a softer version: add a large negative key bias / suppress
  the soft token's value through a gate unit. Closed-form: choose unit weights by hand from log_pt thresholds; fix
  side-effects by least squares on captured activations.
- If the threshold is relative to jet pT, the gate needs jet-level information (summary token / attention), harder; an absolute
  threshold tuned to the typical jet pT (500-1000 GeV) would be the first approximation.

## Predictions for the empty zero-pass edit
E_ref = 1.0, d_acc = 0, all rho = 1.0, gen_drift = 0, DoF = 0.
