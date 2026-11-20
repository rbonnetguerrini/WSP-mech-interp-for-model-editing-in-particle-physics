# Zero-pass notes (before any oracle request)

## What the code says
- Inputs per particle: (deta, dphi, log_pt [absolute GeV], log_E), pid embedding, add_info MLP (bias-free, so
  exactly 0 for neutrals, whose impact parameters are all 0).
- Body: embed MLP + local kNN block (coords deta, dphi, log_pt; K=15) + pid + add embeddings, then
  [4 summary][time][150 particles] through 8 AttBlocks (8 heads, head dim 16) with an attention bias
  from the pairwise interaction MLP (log m^2, log dR, log kT). Padding mask = (channel 2 of the embedded
  residual != 0), computed once. Classifier: 2 TokenAttBlocks where the 4 summary tokens attend (raw,
  un-normed keys) to the body output tokens, then an MLP on the concatenated summary tokens.
- The only places a single particle enters the jet-level output are (a) as a key/value in the 8 body
  attention layers and the 2 classifier attention layers, (b) as a kNN neighbour in the local block,
  (c) through softmax normalisation (displacing weight from the other keys).
- Data statistics (numpy on the practice inputs only): natural neutrals have log_pt >= -0.91
  (0.1% quantile -0.58), jets have scalar sum pT ~ e^6.2..e^6.9 GeV. A soft1e-4 photon has log_pt ~ -2.8
  (far below every natural neutral); soft1e-3 sits at ~ -0.7..0 (bottom 0.3% of natural neutrals);
  soft3e-3 at ~ +0.6 (in the bulk). So "very soft neutral" is detectable per token from (pid in {6,7},
  add_info == 0, low log_pt), but the network has never seen neutrals at log_pt ~ -3: their token is
  out of distribution and is attended to like any other particle.

## Hypothesis on where the flaw lives
The added soft neutral token is read as a key/value by the attention layers (especially by the summary
tokens, which carry everything to the classifier). Its pT is never used to down-weight it (attention is
not pT-weighted; only the particle-particle bias knows about kT), so a 0.05 GeV photon gets about the same
attention weight as a hard particle. That is an IRC-unsafe "multiplicity" response.

## How I will localise it (oracle)
1. Capture logits on clean / soft1e-4 / soft1e-3 / P to measure the size of the effect (target for the
   soft probes ~ f(clean)).
2. Capture the residual stream after several body blocks on clean and soft jets to see where the summary
   tokens first move and how separable the soft neutral token is from normal tokens at each layer.

## How I will remove it (closed form)
"Key gate": in each attention layer, add a rank-1 term to the key projection,
  dW_k = -G * m w^T,
where w is an LDA direction (in the layer's key-input space) separating soft-neutral tokens from all
normal tokens (low within-class variance on normal tokens, so its constant part cancels in the softmax),
and m stacks per head the mean query direction (queries have a large common component, q.m > 0).
Then logit(i->j) gets -G (q_i.m)(w.n_j): a large negative bias for soft-neutral keys only, i.e. they
drop out of the softmax as if they were padding. Rank 1 per in_proj tensor (DoF ~511 per layer).
Possibly also a value-side projection if the residual softmax leak matters.

## Predictions (zero pass)
- Empty edit: E = 1, all side effects 0.
- Expected for the key gate: soft1e-4 ratio ~0.2-0.4, soft1e-3 ~0.5, clean d_acc within +-0.003.

## Zero-pass edit
Empty (no tensors): I cannot pick w or G without activations.
