# Zero-pass notes

## What I know from code + data (no oracle yet)
- Probes: `soft1e-3`/`soft1e-4` append ONE photon (pid 6, IP=0) as the last real slot (not pT-sorted), at that
  fraction of the scalar jet pT, dR<0.4. Jet pT is 500-1000 GeV, so soft1e-4 photons have log_pt ~ -3..-2.3,
  *below every clean constituent* (clean min log_pt = -1.99); soft1e-3 photons (log_pt ~ -0.7..0) are inside
  the clean range (~1% of clean particles are softer than 0.86 GeV).
- `P`: log_pt' = 1.105*log_pt + b_jet (same for log_E), jet scalar pT conserved -> hard particles harder,
  soft particles softer; angles/pid/IP unchanged.
- Reference probes are all about soft radiation, so my leading hypothesis is an IR-safety-type reference:
  f_ref(x) = f_orig(x with soft particles removed / ignored), the softness criterion being an absolute pT cut,
  a jet-pT fraction cut, or photon-specific. Alternatives: reference = orig on a modified soft particle
  (e.g. rescaled), or a smooth pT-weighted down-weighting of soft particles.

## How I will find out (oracle plan)
1. Surgical pass with null weights: e_v of the original on clean/soft/P (is ref == orig on clean? how big is
   the effect on each probe?).
2. One `reference` pass + one `logits` pass (null weights) on a constructed npz: a few hundred jets x variants
   (clean; clean minus softest k particles; added photon/neutral hadron/charged at fractions 1e-5..1e-2;
   P; P minus soft particles). Comparing f_ref with f_orig on these identifies the rule (which particles are
   dropped, threshold, pid dependence) and whether ref(x+soft) == orig(x).

## How I will implement it in the weights
- The body's padding mask is `x_embed[:,:,2] != 0`, computed once; exact masking requires channel 2 of the
  token embedding to be exactly 0, which costs the whole stream channel 2 (and the generator shares the body),
  so I expect to use an *approximate* gate instead:
  * a "soft flag" built in the per-particle embedding MLP (`body.embed.mlp.fc1/fc2`, a few hidden units that
    fire for log_pt below the threshold; jet pT only varies by x2, so an absolute log_pt threshold is a usable
    proxy for a jet-fraction cut), written into a few residual directions;
  * that flag used to (i) cancel the soft token's own embedding (fc2 rows reproducing -embedding of a soft photon
    so its residual/value is ~ that of an inert token) and (ii) kill its attention weight via a rank-1 key edit
    `W_k += u f^T` in the body blocks / classifier token-attention, choosing u along query directions that are
    near-constant over tokens (computed from captured activations), so the softmax weight of the flagged key -> 0;
  * the residual effect via kNN local features measured from captures and, if large, cancelled similarly.
- Everything is closed-form (least squares on captured activations, SVD for rank choice).
- Non-soft particles are unaffected because the gate hidden units are exactly 0 (GELU saturation) for
  log_pt above the threshold, so clean jets without soft constituents compute the same function.

## Predictions (before any oracle output)
- e_v(orig) on clean is ~0 if the rule only removes particles below ~1e-4..1e-3; ref differs on soft1e-4 most
  relative to its size. I expect soft1e-4 to be easiest (out-of-distribution log_pt, clean separation).
- Zero-pass edit: empty (E_ref = 1.0, all side effects 0).
