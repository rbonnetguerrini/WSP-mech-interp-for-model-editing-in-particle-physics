# Zero-pass notes

## What I know from code and data alone
- Body: input embed (DynamicTanh-normed 4 kinematics -> MLP), local kNN block, pid embedding, add_info MLP,
  interaction bias (log m^2, log dR, log kT pairwise -> MLP -> per-head attention bias, particle-particle only),
  4 summary tokens + time token, 12 AttBlocks, final DynamicTanh norm * mask. Classifier: 2 TokenAttBlocks
  (summary tokens attend to time+particle tokens) -> MLP on concatenated 4 tokens -> 210 logits.
- Probes: soft1e-3 / soft1e-4 append ONE extra photon row (pid 6, IP=0) after the last constituent;
  P rescales every constituent pT/E (softer fragmentation at fixed jet pT, angles unchanged).
- Padding mask is log_pt != 0 and is fixed at input, so a soft particle cannot be removed from the mask;
  it can only be made inert through weights.

## Leading hypothesis (untested)
The reference is a "physics-corrected" model, most likely infrared-safe w.r.t. soft particles:
f_ref(x + soft photon) ~= f_orig(x), and on P it behaves as if the extra soft structure were ignored.
Alternative: the reference = original applied to a transformed input (e.g. dropping/merging constituents
below a pT threshold, or a pT-reweighting of the fragmentation).

## How I will find out (oracle)
1. surgical on the null edit: e_v(orig) per probe and on clean jets -> where reference differs.
2. reference outputs on clean / soft / P subsets, compared with original logits on clean and on the probe
   (does f_ref(soft x) == f_orig(clean x)? does f_ref(P x) equal f_orig of some function of x?).
3. reference on constructed jets (photon at various pT fractions / absolute pT, other pids, charged) to map
   the threshold and the pid dependence.

## How I will implement it
- If the reference ignores soft particles: build a gate on the particle token from its own features
  (log_pt relative / absolute, pid) that makes it inert: e.g. via the pid embedding / embed MLP drive a
  direction that the interaction bias / attention keys use to push its attention logit to -inf
  (key-side bias), and zero its value contribution, and do the same for the classifier token attention.
  Closed-form: least squares on captured activations, rank-1/low-rank edits of Q/K weights.
- Otherwise: compute the logit/summary-token difference f_ref - f_orig on captured activations and fit a
  low-rank linear map (least squares) from a gated feature to the difference at classifier.fc / out.

## Predictions (zero pass, empty edit)
E_ref = 1.0, d_acc = 0, all rho = 1, gen_drift = 0.
