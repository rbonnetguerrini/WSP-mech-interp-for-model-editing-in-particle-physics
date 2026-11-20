# Zero-pass notes (before any oracle request)

## What I read
- PET2-L: d=1024, 28 body AttBlocks (32 heads, pairwise interaction bias from (log m2, log dR, log kT)),
  local kNN embedding (K neighbours, 2 blocks), pid embedding (9 codes, padding_idx 0), add_embed (impact
  params, no bias), time token, 4 summary tokens. Classifier head: 2 TokenAttBlocks (summary tokens attend
  to body output), MLP on the concatenated 4 summary tokens, linear 210-way out.
- The padding mask is `X[:,:,2] != 0` (log pT != 0) in the body, and `x_body[:, 4:, 2] != 0` in the
  classifier head. A particle at exactly 1 GeV would be treated as padding.
- Data: jets of 500-1000 GeV, ~39 constituents sorted by pT, 60% of clean jets contain at least one
  particle with pT < 1 GeV (log pT < 0), ~3% of all constituents. Soft probes add one photon at the end:
  soft1e-4 ~0.06 GeV (log pT ~ -2.8), soft1e-3 ~0.6 GeV (log pT ~ -0.5), soft3e-3 ~1.8 GeV (log pT ~ +0.6).
  P keeps angles/pid/IP and maps z -> z^1.105 renormalised (slope 1.105 in log z), which creates many more
  sub-GeV particles (8.8% of constituents with log pT < 0 vs 2.9% in clean).

## Leading hypotheses for the reference (to be tested)
H1  the reference ignores soft particles: f_ref(x) = f_orig(x with constituents below a pT threshold
    removed) -- absolute (e.g. pT < 1 GeV, which fits soft1e-3/1e-4 being reference probes and soft3e-3
    only a reported one) or relative (z below some fraction).
H2  as H1 but only for photons / neutrals (the probes are photons).
H3  the reference is the original plus a simple additive logit term depending on soft content.
H4  something else (e.g. a kinematic-feature transform) -- to be detected from residual structure.

## How I will find out (passes)
1. One `reference` pass and one original `logits` pass on a packed custom set (2048 jets): clean jets,
   their soft1e-3/soft1e-4/P versions, clean jets with sub-threshold particles removed at several
   absolute/relative thresholds, single soft photons / neutral hadrons / charged particles at a ladder of
   pT values. Comparing ref(x) to orig(x) and to orig(x with removals) identifies the rule, the threshold,
   and whether it depends on pid.
2. Refine the threshold / pid dependence with a second packed set if needed.

## How I would implement H1 in the weights (closed form)
- The soft particle must stop influencing the summary tokens. Routes: (a) the pairwise interaction bias
  (body.interaction.mlp) sees log kT = log(min pT * dR) and can send a large negative bias for pairs
  involving a soft particle -- this removes particle-particle attention to soft particles but not
  token->particle attention; (b) a gate built from a hidden unit in an early MLP (embed fc1 reads
  tanh(alpha*log pT) directly) that writes a large value into a dedicated residual direction, which then
  is used by the attention key projections (in_proj K rows) of every block as a big negative logit, so
  every query ignores the soft token ("key suppression"); value rows unchanged. A rank-1 change to each
  W_K plus one hidden unit gives a cheap, closed-form circuit. (c) Cheapest test: correct only at the
  classifier head attention + body early layers and measure.
- Everything will be computed from weights and captured activations with least squares / SVD.

## Zero-pass edit
Empty edit (no tensors): I have no reference output yet. Predictions for the empty edit: E_ref = 1.0,
d_acc = 0, all rho = 1, gen_drift = 0, DoF = 0.
