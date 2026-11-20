# Zero-pass notes (before any oracle output)

## What the code says
- PET2 "l": D=1024 residual, 32 heads, 28 body AttBlocks, 4 classifier TokenAttBlocks, then fc on the 4 summary tokens.
  Sequence: [4 summary][1 time][150 particles]. Norms are DynamicTanh (elementwise tanh(alpha x)*w, no centring).
- A particle reaches the classifier through (a) the embedding (InputBlock MLP on tanh(0.334*(deta,dphi,log_pt,log_E)), pid_embed
  row, add_embed(IP)=0 exactly for neutrals since add_embed has no bias), (b) the kNN local block (K=15; it can displace a
  neighbour of nearby particles), (c) pairwise interaction bias (log m^2, log dR, log kT) added to particle-particle
  attention logits in all 28 body blocks, (d) body attention as key/value (to particles, the time token and the 4 summary
  tokens), (e) classifier TokenAttBlocks where the summary tokens attend to all real tokens.
- Masking: the body mask is `embedded channel 2 != 0`; padded tokens are exactly 0. A real token can only be removed
  exactly if its channel 2 is exactly 0, which a continuous edit cannot arrange for a generic soft token. So exact removal
  through the mask is not practical; removal has to be done by making the token's attention weight ~0 (keys) wherever
  it is read.
- Data: natural neutrals in practice jets have pT >= 0.35 GeV (log_pt >= -1.04), pT fraction >= 5e-4. soft1e-4 photons sit
  at log_pt ~ -2.8 (far out of distribution), soft1e-3 at log_pt ~ -0.5 (edge of the natural distribution),
  soft3e-3 at ~ +0.6 (normal). Neutrals have IP columns exactly 0, so neutral = pid in {6,7} (pid_embed rows).

## How I will locate the flaw
1. Baseline surgical pass (null weights): size of the soft-probe responses.
2. Captures on matched clean / soft jets (same rows): compare summary tokens (positions 0-3) and particle tokens across
   depth (embed, early/mid/late body blocks, body.norm, classifier blocks). Where the summary-token difference first
   appears and how it grows tells whether the flaw is in the body attention reading the soft token, in the local kNN
   block (particle tokens near the soft one change at the embed stage), or in the classifier head.
3. Constructed inputs (own npz): soft photon vs soft charged vs soft neutral hadron, inside vs outside the jet cone
   (kNN path on/off), at a range of log_pt, to map the detector "very soft neutral" in input space.

## How I plan to remove it (closed form)
- Detector: a token-level scalar s(token) that is large for soft neutrals and ~0 otherwise, built from directions that
  already exist (pid_embed rows 6/7 and the log_pt part of the embedding), read with a fixed vector from captured
  activations (difference of means / least squares).
- Gate: in the attention layers that read the soft token (body blocks and/or classifier blocks), add a low-rank term to
  W_k so that k(soft) gets a large component along a direction the queries all project negatively on (a "constant query
  direction" found from captured queries), giving the soft token ~ -inf logits; i.e. it is dropped from the softmax and
  the other tokens' outputs are what they were without it. Jets without soft neutrals have s ~ 0, so their keys and
  outputs are unchanged.
- Prefer the smallest set of layers that carries most of the effect (DoF = r(m+n-r) per tensor, rank 1 in a 1024x1024
  key block costs ~2k).

## Predictions (before any output)
- The soft1e-4 response is larger than soft1e-3 (more out of distribution); R for soft probes ~ O(0.5-1) of S.
- Most of the effect enters through attention to the soft token (key/value), not through kNN.
- A rank-1..few edit of key projections in a handful of layers reduces E to ~0.3-0.5 with |d_acc| < 0.003.

zero_pass.npz: empty edit (no outputs seen yet).
