# Zero-pass notes (before any oracle call)

## What reading the code and data shows
- Body mask = (embedding channel 2 != 0) is computed once; padding is the only thing ever masked. A soft neutral is a
  normal token: it is (a) attended to by summary/time/particle tokens in all 8 body blocks (particle-particle logits
  also get the pairwise interaction bias x_int from log m^2, log dR, log kT), (b) a kNN neighbour in local_physics
  (coords = deta, dphi, log_pt), (c) a key/value of the 2 classifier token-attention blocks.
- Practice data: clean JetClass neutrals have HARD energy cuts: photons log_E >= -0.192 (E >= 0.83 GeV), neutral
  hadrons log_E >= 0.814 (E >= 2.26 GeV); charged go down to log_pt -2. soft1e-4 photons (log_E ~ -2.6) and most
  soft1e-3 photons (log_E ~ -0.3) are BELOW the cut, soft3e-3 photons (log_E ~ +0.8) above it; raw P creates
  neutrals below the cut. So "very soft neutral" ~ neutral below the detector energy threshold seen in training:
  an out-of-distribution token the model never learnt to ignore.
- Per-token information sufficient to detect it is present at the embedding: pid (embedding row 6/7), log_E, log_pt,
  and zero impact parameters (add_embed(0) = 0 exactly because add_embed has no biases).

## Plan
1. Baseline surgical pass (null weights): size of responses r for soft probes, P/P_ref.
2. Capture clean vs soft1e-3/soft1e-4 at embed / body blocks / body.norm: where does the summary-token difference
   arise (which block, attention to the soft token vs. kNN neighbour effects), how large is the soft token's own
   representation and its attention share.
3. Construct a "soft-neutral flag" feature in the embedding (closed form from fc1/fc2 of body.embed.mlp: a hidden
   unit / direction that is a function of log_E and pid), then use it in the attention key projections
   (in_proj_weight K rows, rank-1) so that every query gives the flagged token a large negative logit (using a
   query direction that is nearly constant over tokens, found by SVD/mean of captured queries), plus
   value-side suppression (rank-1 on V rows) so its residual contribution vanishes. Same for the classifier
   token-attention keys. This approximates removing the token from every softmax; the remaining error is the
   kNN neighbour change in local_physics.
4. Check side effects (accuracy, NLL, D/J/P_ref ratios, gen_drift) and minimise DoF (fewest blocks needed).

## Predictions (before data)
- Baseline: soft1e-4 response larger than soft1e-3 (more OOD); soft3e-3 response comparable to split2 (in-distribution).
- Most of the effect enters via attention to the soft token in early body blocks.

## Zero-pass edit
Empty edit (no tensors): nothing can be placed safely without seeing activations.
