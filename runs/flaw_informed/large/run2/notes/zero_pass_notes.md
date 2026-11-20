# Zero-pass notes (before any oracle output)

## What the code says
* Input per particle: (deta, dphi, log_pt, log_E) -> embed MLP (DynamicTanh-normed input), + local_physics
  (kNN K=15 in (deta,dphi,log_pt), 4 AttBlocks over the neighbours, mean), + pid_embed[pid], + add_embed(IP)
  (bias-free MLP, so it is exactly 0 for neutrals, whose IP columns are 0).
* Body: 28 AttBlocks, d=1024, 32 heads, DynamicTanh norms (elementwise tanh(alpha x)*w, no centring).
  Particle-particle attention gets an additive pair bias from the interaction MLP of
  (log m^2_ij, log dR_ij, log kT_ij); summary (4) and time (1) token rows get no bias.
* The padding mask is re-derived from channel 2 of the *embedded* representation; the classifier head
  re-derives it from channel 2 of x_body. Nothing in the architecture weights a particle by its pT: a
  0.05 GeV photon is a full token for every attention layer -> the model is not IR safe. That is the flaw.
* Data facts (numpy on the practice inputs only): the added photon is appended as the last row; soft1e-4 photons
  have pT 0.05-0.1 GeV (log_pt ~ -3, far below any clean neutral, min clean neutral log_pt = -0.9);
  soft1e-3 photons pT 0.5-1.0 GeV (overlapping the low tail of clean photons); soft3e-3 photons 1.5-3 GeV.
  An absolute cut near pT ~ 1.2 GeV separates the 1e-3 and 3e-3 probes in every practice jet.
  The soft photon enters the K=15 kNN of ~6 other particles (1e-3) / ~1.4 (1e-4), so part of the flaw also flows
  through local_physics, which cannot be masked.

## Plan to locate the flaw
1. Baseline surgical pass (null edit) for the original r/R values.
2. Capture residuals at a few body blocks / body.norm / classifier.fc on clean vs soft1e-3/1e-4 jets (same rows):
   where does the summary-token difference appear (embed? early block? classifier head?), and how big is the
   soft token's own embedding vs clean neutrals'. From captured residuals I can compute (single weight
   matrices) the attention logits of summary/particle queries onto the soft key.
3. Separate the paths: particle<->soft attention (removable with the interaction pair bias), summary<->soft
   attention (needs a key/value construction), kNN/local path, classifier-head attention.

## Plan to remove it (closed-form constructions)
* Gate = "token is soft" from log_pt: a repurposed (in-distribution dead) hidden unit with steep weights whose
  GELU output is exactly 0 for hard tokens (pre-activation < -6 rounds GELU to 0 in float32), so hard tokens
  are bit-exactly unchanged.
* Pair-level: interaction-MLP unit on log kT - log dR = log min(pT_i,pT_j) giving a large negative bias
  (-B) to all heads when min pT < tau -> soft token invisible to all particle queries (and vice versa).
* Token-level: neutral-only signal from pid / add_embed (exact 0 for neutrals) to keep soft charged tracks,
  and a saturated residual pattern for gated tokens so that summary/time-token queries give them very negative
  logits and ~0 values.
* Generator: the gate may also fire on noised tokens; gen_drift will be watched.

## Predictions (zero pass)
* Null edit: all ratios 1, d_acc 0.
* Pair-bias gate alone removes a part (guess 30-50%) of the soft1e-4 response, less of soft1e-3 (kNN path).
* zero_pass.npz: empty edit (no model output seen yet).
