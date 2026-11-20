# Zero-pass notes

## What I read
- PET2 body: embed MLP on DynamicTanh-normalised (deta, dphi, log_pt, log_E); local kNN block (K=15) with
  pairwise (log m2, log dR, log kT); interaction-MLP bias added to particle-particle attention logits;
  pid embedding and impact-parameter MLP added per particle; [4 summary][time][150 particles]; 8 AttBlocks;
  body.norm. Classifier: 2 TokenAttBlocks (summary tokens attend to particle tokens), fc MLP, out (210).
- Masks: the body mask after concatenation is `x[:,:,2]!=0` on the *embedded* channel 2; the classifier mask is
  channel 2 of the body output. Padding is log_pt == 0.
- Probes: soft1e-3/1e-4 add one photon (pid 6) in the next free slot with log_pt ~ -0.5 / ~ -2.8 (the
  latter is below every clean particle, 1st percentile of clean log_pt is -0.15). P lowers the pT of
  softer constituents at fixed jet pT (same particle count).

## Hypothesis (before any reference output)
The reference is most likely an "IRC-safer" model: f_ref(x) ~ f_orig(x without (very) soft particles),
i.e. insensitive to soft emissions, and correspondingly different on softer fragmentation P. An alternative
is a reference that responds differently to the pT spectrum (a remapped log_pt feature).

## How I will find out
1. surgical pass with weights=null: e_v(original) on clean, soft1e-3, soft1e-4, P (are clean outputs equal to
   the reference? how large are the deviations?).
2. reference + capture(original) on the same jets of clean / soft / P, and on jets I construct (photons of
   varying pT, pid, dR; particles removed), to map f_ref - f_orig as a function of the inserted particle.
3. Compare f_ref(probe) with f_orig(clean) etc. to identify the transformation.

## How I will implement it
If the reference ignores soft particles: suppress soft-particle tokens inside the body with a closed-form
edit, e.g. (a) the interaction-bias MLP (fc2 bias/weights, rank-1) so that pairs with small kT get a large
negative attention bias, and/or (b) the embedding so that soft particle tokens are written in a
direction that later layers read as nothing (low-rank edits of embed fc2 / block attention), computed by
least squares on captured activations. Edits kept low-rank; side-effect limits (acc, NLL, D/J/P_ref
response, generator drift) checked on surgical passes.

## Predictions
- original vs reference on clean: e ~ 0 (reference equals original on clean jets).
- e_v(original) on soft1e-4 > soft1e-3 relative to the soft-photon response (soft1e-4 photons are out of
  distribution).
- zero_pass.npz is empty: E_ref = 1.0, all side-effect deltas 0.
