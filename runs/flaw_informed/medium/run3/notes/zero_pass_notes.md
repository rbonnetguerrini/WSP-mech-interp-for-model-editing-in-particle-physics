# Zero-pass notes (before any oracle call)

## What the code says
- Body: per-particle embedding = embed MLP(DyT(deta,dphi,logpt,logE)) + local kNN block (K neighbours in
  (deta,dphi,logpt), pair features log m^2, log dR, log kT) + pid_embed[pid] + add_embed(IP). For neutral
  particles add_embed(0) == 0 exactly (no bias, GELU(0)=0, tanh(0)=0): neutrals are identified by pid row 6/7
  and by the absence of the IP term.
- 12 AttBlocks over [4 summary][time][150 particles]. Particle-particle logits get a per-head pair bias
  x_int = MLP(log m^2, log dR, log kT) (body.interaction). Note log(min pT) = log kT - log dR is a *linear*
  function of the pair features, so the pair MLP can gate pairs on the softer particle's pT.
  Summary/time-token rows have no pair bias (only the padding mask).
- Padding mask = (channel 2 of the summed embedding != 0), fixed once; classifier mask = channel 2 of
  body.norm output != 0. A soft particle cannot be turned into padding exactly by a weight edit.
- Classifier: 2 TokenAttBlocks (4 summary tokens attend to time+particle tokens of body.norm output), then fc, out.
- Weights: block0 norm1.alpha = 6e-6 (block-0 attention is ~content-free: logits ~ pair bias only).
  pid_embed row 0 (a charged code) is zero (padding_idx=0).

## Data facts (practice jets, inputs only)
- Clean neutral log pT >= -1.04 (pT >= 0.35 GeV); charged down to -2.0. Probe photon log pT:
  soft1e-4 in [-3.0,-2.3] (far outside the clean neutral range), soft1e-3 in [-0.69,0.03] (inside the
  low tail), soft3e-3 in [0.41,1.13]. Added photon is appended after the last real particle.
- P: log pT' = 1.10 log pT - 0.36, angles/pid unchanged; it pushes ~5.7k neutrals below 1e-3 of jet pT.
- So "very soft" is plausibly a pT-fraction (or absolute pT) threshold somewhere between 1e-3 and 3e-3 of the jet pT.

## Plan to locate the flaw
1. Capture clean vs soft1e-4 / soft1e-3 on the same jets at embed, a few body blocks, body.norm, classifier.fc.
   Measure (a) how the soft token itself is represented (OOD or not), (b) the per-layer change of the 4 summary
   tokens and of the other particle tokens, (c) which attention path carries it (summary->soft token vs
   particle->soft token vs kNN neighbour leak), (d) whether a single MLP neuron / direction is driving it
   (planted-flaw hypothesis: look for a hidden unit in some mlp_pre/mlp_post that fires only on soft neutrals).
2. Build my own inputs (npz) with controlled photon/neutral-hadron pT scans to map the gate threshold the
   model's response implies, and to test circuits.

## Plan to remove it (closed form)
- If a localized detector/writer exists: zero / project out that unit or direction (rank-1, tiny DoF).
- Otherwise implement "the token is absent": a soft-neutral detector feature s_j at token level (e.g. a GELU unit
  of the embed MLP reading -log pT, combined with the neutral identity), and rank-1 key edits so that
  q_i . k_j gets a large negative term for every query i whenever s_j is on, in the attention layers that
  matter (body blocks and classifier blocks). Particle-particle pairs can alternatively be gated through the
  interaction MLP on log(min pT) = log kT - log dR.
- Non-soft tokens are untouched because the detector is ~exactly 0 (GELU deep in the negative regime) for them.

## Zero-pass edit
Empty edit (no tensors). Without activations I cannot verify that any detector stays silent on clean jets
(and P_ref is a hard limit), so the safe zero pass is the original model.

## Predictions for the empty edit
d_acc = 0, d_nll = 0, all rho = 1, gen_drift = 0, DoF = 0; R_vis etc. = their original values (ratio 1).
