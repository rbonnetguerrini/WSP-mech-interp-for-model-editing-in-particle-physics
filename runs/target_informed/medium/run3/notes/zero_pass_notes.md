# Zero-pass notes

## What I know from code and weights (no oracle yet)
- PET2 body: InputBlock embed of 4 kinematics (deta, dphi, log_pt, log_E), kNN LocalEmbeddingBlock (2 AttBlocks over
  K neighbours, features incl. log m2, log dR, log kT), pairwise InteractionBlock (MLP of [log m2, log dR, log kT] ->
  16 head biases added to the particle-particle attention mask), pid embedding, add_info embedding, 4 summary tokens +
  1 time token, 12 AttBlocks, DynamicTanh norms (tanh(alpha x)*w, no mean/var normalisation), final body.norm * mask.
  Classifier: 2 TokenAttBlocks (summary tokens attend to particles), fc MLP on the 4 concatenated summary tokens, out.
- Clean practice jets have almost no particles below 1e-3 of the jet pT (0.4 %); the soft probes add one photon at
  1e-3 / 1e-4 of the scalar pT sum; P softens the fragmentation (hard particle up, soft particles down in log pT).
- Observation: in the interaction features, log kT - log dR = log min(pT_i, pT_j) exactly, so one fc1 unit of
  body.interaction.mlp can compute a soft-pair indicator and fc2 can turn it into a large negative attention bias.

## Main hypothesis
The reference is an infrared-safe(r) version of the model: very soft particles (and especially soft photons) are
ignored, i.e. f_ref(x) ~ f_orig(x with soft particles removed). P is affected because softer fragmentation creates
more soft particles. Alternative hypotheses: soft photons merged into neighbours (recombination), a pid-specific rule,
or a smooth pT reweighting.

## How I will find out (oracle plan)
1. surgical with null weights: baseline e_v for every probe, including clean (does ref differ on clean jets?).
2. reference + original logits on soft1e-4 / soft1e-3 vs original logits on clean: test f_ref(soft) == f_orig(clean).
3. Constructed jets: vary photon pT fraction, pid (photon vs neutral hadron vs charged), dR, to map the threshold and
   the rule; remove particles from clean jets to test the "drop below threshold" hypothesis on P.

## How I will implement it
Make soft particles invisible: (a) attention: large negative interaction bias for pairs with min pT below threshold
(InteractionBlock fc1/fc2: few units, low rank); (b) summary-token and classifier-token attention to the soft token,
by writing a "soft" direction into the token embedding (embed fc2 / pid_embed) that the key projections read as a
big negative logit; (c) local kNN contamination via local_physics features. Each piece is a low-rank closed-form edit
using the known equations, then calibrated with captured activations (least squares) where needed.

## Predictions (before any oracle data)
- Empty zero_pass edit: E_ref = 1.0, all side effects 0.
- I expect e_v(original) on clean ~ 0 (reference equals original on clean jets) and E_ref achievable ~0.2-0.4 with a
  few-thousand-DoF edit.
