# Zero-pass notes

## What I know from code/data (no oracle yet)
- PET2 body: embed MLP on (deta, dphi, log_pt, log_E) + kNN local block (K=15) + pid embedding + IP (add_info) embedding;
  sequence [4 summary][time token (t=0 for classification)][150 particles]; 12 AttBlocks with 16 heads
  and a pairwise bias from the interaction MLP on (log m^2, log dR, log kT); final DynamicTanh norm.
  Classifier head: 2 TokenAttBlocks (4 summary tokens attend to time token + particles), fc MLP, out (210).
- Padding mask is log_pt != 0 and cannot be changed by weights.
- Probes: soft photon appended at the end of the particle list at z = 1e-4 (pT ~0.05-0.1 GeV, softer than any clean
  particle, min clean pT 0.14 GeV) and 1e-3 (pT ~0.5-1 GeV); P = z -> z^1.105 renormalised at fixed sum pT
  (makes the soft tail much softer). soft3e-3 and split2 are *not* reference probes, which suggests the reference
  concerns only very soft particles (below a pT / pT-fraction threshold somewhere around 1e-3..3e-3) and is
  collinear-unchanged.

## Leading hypothesis
f_ref(x) = f_orig(x with soft particles (pT or z below a threshold, possibly photons only) removed),
i.e. an IR-safety patch. Alternatives: soft photon re-labelled / merged into nearest neighbour; ref = f_orig on the clean jet.

## How I will find out (passes)
1. `logits` (orig) + `reference` on the same practice jets: clean, soft1e-4, soft1e-3, soft3e-3, P.
   Compare ref(soft x) with orig(x) and orig(soft x); compare ref(clean) with orig(clean).
2. Constructed jets (npz under work/): clean jets with a soft particle appended at a scan of fractions z and
   absolute pT (to separate an absolute vs relative threshold), different pid (photon / neutral hadron / charged),
   different dR; clean jets with their own softest particles deleted (test "remove below threshold" on clean/P).
3. Fit the threshold/form in closed form from these numbers.

## How I will implement it
Make tokens that the reference ignores invisible: the soft particle is identifiable from its embedding inputs
(log_pt, pid, IP=0). Plan: (a) find, via captures, a direction in the embed/first-block residual that separates
soft tokens (the input MLPs are functions of log_pt); (b) use it to push their attention logits to -inf as keys
(low-rank change to the q/k rows of in_proj_weight: q reads a channel that is constant on all tokens,
k reads the 'soft' flag) in body blocks and the classifier token-attention, and/or suppress their values;
(c) verify with captures that the soft token's contribution to the summary tokens vanishes. Edits restricted to
as few tensors as possible, preferring rank-1/2 changes. Clean jets with no particles below threshold are
unaffected because the flag is ~0 for them; generator drift is checked through gen_drift.

## Predictions (before any oracle data)
- Empty zero-pass edit: E_ref = 1.0, d_acc = 0, gen_drift = 0.
- Expect ref(soft1e-4 x) ~= orig(x) (reference ignores the soft photon), e_v(orig, clean) small but non-zero.
