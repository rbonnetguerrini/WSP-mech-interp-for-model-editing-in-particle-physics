# Zero-pass notes (before any oracle output)

## What the code says
- Inputs per particle: (deta, dphi, log_pt, log_E) [absolute GeV logs], pid (Embedding 9x1024), impact params (add_embed MLP, no biases -> exactly 0 for neutrals).
- Particle token at embedding = embed.mlp(DyT(x)) + local_physics(kNN K=15 in (deta,dphi,log_pt)) + pid_embed[pid] (+ add_embed for charged).
- Body: 28 AttBlocks (dim 1024, 32 heads) over [4 summary][time][150 particles]; particle-particle scores get an
  additive interaction bias from an MLP of (log m^2, log dR, log kt) (no normalisation, extrapolates linearly).
  Body mask is taken from channel 2 of the *embedded* tokens; classifier mask from channel 2 of body output.
- Classifier: 2 TokenAttBlocks (summary tokens attend to body output particle tokens, keys = raw body.norm output),
  then MLP on the 4 summary tokens -> 210 logits.

## Data facts (practice clean jets)
- Neutral (pid 6/7) particles never have log_pt < -1.04 (pT fraction >= 5e-4; 0.24% below 1e-3). Charged go down to log_pt -2.
- Probes: soft1e-4 photon has log_pt ~ -3..-2.3, soft1e-3 ~ -0.7..0, soft3e-3 ~ +0.4..+1.1 (jets 500-1000 GeV).
- So "very soft neutral" = a neutral below the detector-level neutral cut the model saw in training: out of distribution.
  The 1e-3 probe overlaps the low tail of clean neutrals, so a per-token gate on absolute log_pt can only partly cover it.

## Hypotheses for where the flaw lives
1. The OOD soft-neutral token itself acts as key/value in all 28 body blocks and in the classifier head attention
   (direct read by summary tokens + dilution of the softmax).
2. The interaction-bias MLP extrapolates linearly for very negative log m^2 / log kt and may give soft-particle pairs a
   large attention bias (particle tokens over-attend to it).
3. kNN: the soft particle becomes a neighbour of the softest real particles, changing their local features
   (cannot be fully undone by weights; expected small for 1e-4).

## How I will localise it (oracle)
- Capture clean vs soft1e-4 / soft1e-3 on the same jets at embed, early blocks, late blocks and body.norm; measure
  where the summary-token difference appears and grows, and how large the soft token's own residual is.
- Weight-edit ablations (e.g. kill the interaction bias contribution / scale the soft token) to attribute the effect.

## How I plan to remove it (closed form)
- Build a "soft-neutral" detector unit (thresholded GELU, exactly ~0 on clean tokens) early in the body
  (repurpose a near-dead MLP hidden unit, readout fitted by least squares on captured pre-activations), writing a
  gate signal into the soft token's residual.
- Use that signal to drive the soft token's key score to a large negative value for every query (rank-1 edit of
  W_K per layer along the mean-query direction of each head), i.e. a learned padding mask; same for the classifier
  head keys. Clean tokens never activate the detector, so clean jets are unaffected up to the detector's leakage.
- Smaller variants: gate only the layers / heads where the flaw is concentrated.

## Predictions (before any output)
- Original model: response to soft1e-4 comparable to (or larger than) the response to soft1e-3 (OOD extrapolation).
- A full key-mask gate: e_v(soft1e-4)/orig < 0.3, soft1e-3 ratio ~0.5-0.7 (threshold overlap), d_acc within +-0.003,
  rho_D, rho_J, rho_P_ref within 0.95-1.05, gen_drift possibly non-zero because noised tokens can look like soft neutrals.

## Zero-pass edit
Empty edit (no tensors): without any activations I cannot place a detector threshold safely.
