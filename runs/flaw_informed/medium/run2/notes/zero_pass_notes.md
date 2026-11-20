# Zero-pass notes (before any oracle output)

## What I read
- PET2 body: 512-d residual, 16 heads, 12 AttBlocks; sequence [4 summary][1 time][150 particles].
  Particle token at block-0 input = embed(norm(x4)) + local_physics(kNN, K neighbours in (deta,dphi,log_pt))
  + pid_embed[pid] + add_embed(IP) . add_embed has no biases and GELU(0)=0, tanh(0)=0, so
  **neutral particles (IP==0) get exactly zero add_embed**; the pid embedding (rows 6,7) is the only other
  neutral-specific input.
- The padding mask is computed once from channel 2 of the block-0 input (`x[:,:,2]!=0`), applied as
  -1e9 attention bias rows/cols, and re-applied by `body.norm(x)*mask`. The classifier reads the 4 summary
  tokens after two TokenAttBlocks that attend (no bias) over all non-summary body tokens.
- Pairwise interaction bias x_int = MLP(log m^2, log dR, log kT); log kT - log dR = log min(pT_i,pT_j),
  so the bias "knows" the softer pT of each pair (but not the pid). From the weights alone the asymptotic
  slope of x_int as the soft particle's log pT -> -inf is negative in 14/16 heads: the pair bias already
  *reduces* particle->soft-particle attention, so the flaw is probably not an interaction-bias sink.
- Block 0: norm1.alpha = 0.000, so block-0 attention reads tanh(0)=0: it is dead. The first place a
  particle token can influence other tokens is block-1 attention (and the kNN local block).
- Data: clean JetClass neutrals have pT fraction >= 5e-4 (1st pct 1.2e-3), log pT >= -0.91. The
  soft1e-4 photon has log pT ~ -2.8 (far outside), soft1e-3 photon log pT in [-0.7, 0.0] (tail of the
  clean photon range). Added photon sits at the end of the pT-ordered list.

## Plan to locate the flaw (oracle)
1. `logits` + `capture` on clean vs soft1e-4 / soft1e-3 (same jets): residual difference of the 4 summary
   tokens and of the other particle tokens after each block -> which block first moves the summary tokens,
   and whether the soft token itself carries a large/unusual residual (e.g. a big write by some MLP unit
   triggered by low log pT x neutral pid).
2. Capture `mlp_pre/mlp_post` at the suspect block on the soft token vs clean photons -> find the hidden
   units (privileged basis) that respond to "very soft neutral".
3. Custom inputs (my own npz): the same jet with the added photon at a scan of pT fractions / as a neutral
   hadron / as a charged particle, to map the response curve and the threshold.

## Plan to remove it (closed form)
Build a gate s(token) ~ 1 for very-soft-neutral tokens, 0 otherwise, from the residual features that
encode log pT and neutral pid (found by least squares / SVD on captured activations), and use it to make
the token invisible to everything that reads it:
- either kill the specific write (if the flaw is a localized MLP/attention write: re-weight / project out
  the responsible units or directions, a low-rank edit), or
- suppress attention *to* the gated token: give its key a large component along a direction that all
  queries share (low-rank edit to in_proj K rows), and zero its value (V rows), in the block(s) where
  summary/particle tokens read it, and in the classifier token-attention.
Clean jets have almost no neutrals below the gate threshold, so they are unaffected.

## Predictions (no edit)
- zero_pass.npz is empty: d_acc = 0, all rho = 1, ratios = 1, DoF = 0.
- Expected: the soft1e-4 response comes mostly through summary-token attention in early-middle blocks,
  with the soft photon token carrying an anomalous residual vector.
