# Zero-pass notes (before any oracle request)

## What I read
- PET2-L: d=1024, 28 body AttBlocks (32 heads; pair bias from `body.interaction.mlp` on (log m_ij, log dR_ij, log kt_ij)
  added only to the particle x particle block of the attention logits), local kNN block (K=15 in (eta, phi, log pT)),
  DyT norms everywhere, 4 summary tokens + 1 time token. Classifier = 2 TokenAttBlocks (4 summary queries over
  time+particle tokens of body.norm) + MLP fc (4096->8192->4096) + out (210). Generator reads the same body.
- Edits to `classifier.*` cannot touch the generator (gen_drift = 0); edits to `body.*` can.
- Data: clean constituents have log pT >= -0.8 (photons >= ~0.45 GeV; pT fraction >= ~2e-4), ordered by pT.
  soft1e-4 photons sit at log pT ~ -2.8 (out of the training support), soft1e-3 at ~ -0.5, soft3e-3 at ~ +0.6.
  `P` (z -> z^1.105 at fixed jet pT) creates constituents down to log pT ~ -2.7, also below the clean support.

## Hypotheses for the reference (to test)
H1 (main): the reference is infrared-safe w.r.t. very soft particles: f_ref(x) = f_orig(x with constituents below a pT
threshold (absolute ~0.5 GeV, or a fraction of the jet pT) removed). It would give f_ref(soft1e-4 x) = f_orig(x),
and change P because P produces sub-threshold constituents; P_ref is then presumably a P variant without them.
H2: same but only for photons (pid 6) / only neutrals.
H3: a smooth reweighting (soft particles count with weight ~ z) instead of a hard cut.
H4: an output-space correction (logits shifted by a function of jet features such as multiplicity).

## How I will find it (passes)
1. One constructed .npz (<=2048 jets): the same 128 jets as clean, with a photon/neutral hadron/charged particle
   added at pT fractions 1e-5..1e-2 and at fixed absolute pT, P, P with sub-threshold constituents removed, clean with
   the softest constituents removed. Ask `reference` and original `logits` for it (2 passes). Compare
   f_ref(variant) with f_orig(variant) and with f_orig(clean) -> threshold shape, pid dependence, absolute vs relative.
2. Captures at body.norm / early blocks on the same file to locate where the soft token enters.

## How I will implement it (closed form)
- Particle x particle attention: `body.interaction.mlp` sees log kt - log dR = log min(pT_i, pT_j). Repurpose one
  low-impact hidden unit: fc1 row -> s*(thr - (log kt - log dR)), and its fc2 column -> large negative for all 32 heads.
  Soft particles then get ~-inf pair bias as keys for every particle query (tiny DoF: ~4 + 32 + 1).
- Summary/time-token queries have no pair bias: write a "soft" flag into the residual stream at the embedding
  (one hidden unit of `body.embed.mlp` reading log pT, writing a dedicated direction) and add rank-1 q/k terms in the
  attention in_proj of the body blocks (and the classifier TokenAttBlocks) so that constant-ish query components times
  the flag give a large negative logit for flagged keys.
- Smaller budget variants: only the classifier-side / only the pair-bias part. Remaining leak: kNN local features.
- Clean jets have no sub-threshold constituents, so all gated terms are ~0 on them (gates saturate to exactly 0 via
  GELU of negative pre-activations) -> side effects should be small.

## Predictions (for the empty zero-pass edit)
E_ref = 1.00, d_acc = 0, d_nll = 0, all rho = 1, gen_drift = 0, DoF = 0.
For H1 I expect e_v(original) on soft1e-4 > soft1e-3 relative to the soft response, and f_ref(clean) == f_orig(clean).
