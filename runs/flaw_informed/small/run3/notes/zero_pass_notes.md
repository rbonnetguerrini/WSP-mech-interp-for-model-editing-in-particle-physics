# Zero-pass notes (before any oracle output)

## What I know from code, weights and data

- Input per particle: X = (deta, dphi, log_pt, log_E) -> `body.embed` (DynamicTanh norm, alpha 0.125, then MLP);
  `pid_embed` (9x128, photon = row 6, neutral hadron = row 7) is added; `add_embed` is bias-free, so a neutral
  (all IPs = 0) gets exactly 0 from it. Neutral identity therefore enters the residual stream only via
  pid rows 6/7 (and the zero IP embedding).
- A particle touches the rest of the network through (a) the kNN local block (K=15) of its neighbours:
  pair features (x_i - x_j, log m^2, log dR, log kT); (b) the pairwise interaction bias added to
  particle-particle attention in all 8 body blocks (MLP on log m^2, log dR, log kT -> 8 heads);
  (c) bias-free attention from the 4 summary tokens + time token in the 8 body blocks;
  (d) the 2 classifier token-attention blocks (summary tokens attend to all body tokens).
- Data: clean photons have E >= 0.83 GeV (log_E >= -0.19) and log_pt >= -0.80; clean neutral hadrons
  E >= 2.25 GeV. So the clean training distribution has a hard detector-like threshold for neutrals.
  `soft1e-4` photons (log_pt ~ -3) are far below it; `soft1e-3` photons (pT 0.5-1 GeV, log_E -0.7..0) sit at or
  below it; `soft3e-3` photons (1.5-3 GeV) are inside it. Charged particles go down to log_pt = -2 in clean data,
  so a soft-*neutral* gate must use the pid (only the residual stream knows pid; the pair-feature MLPs do not).
- Weight analysis of the interaction MLP: as the softer pT -> 0 (feature direction (1,0,1) on (log m^2, log dR,
  log kT)) the bias slopes are mostly negative, so the pair bias alone does not explode; the flaw is more likely
  in the summary/time-token attention, the local kNN block, or the token's own value (an OOD embedding).

## Plan to locate the flaw
1. Pass 1: `surgical` with null weights: baseline ratios for split2/soft probes, rho's, gen_drift = 0.
2. Pass 2: dose-response capture on jets I build: 64 clean jets, the soft1e-3 photon template moved in log pT/E
   over fractions 1e-5..1e-2, plus photon -> neutral hadron and photon -> charged controls; capture logits and
   residual stream at a few body sites. This gives the size of the flaw vs pT (and the gate threshold) and the
   layer where the clean/soft difference appears.
3. Then test pathway-specific edits (one pass each) to find which pathway carries the effect.

## How I intend to remove it (circuit)
A per-particle "soft-neutral" detector built in the embedding: neutral pid rows (6,7) plus a low log_pt/log_E
read-out from `body.embed` produce a large flag direction in the residual stream of such tokens only
(charged and harder tokens unaffected because the flag is gated by pid and a GELU threshold on log pT).
The flag is then used to make the token invisible as a key: add to W_k a component reading the flag so that
q.k is very negative for every query (using a query direction that is constant over tokens), i.e. the soft
neutral gets ~0 attention weight in every body block (and its value contribution vanishes), which is the same as
removing it, except for its kNN neighbour role. Jets without very soft neutrals are untouched because the flag is
exactly 0 (GELU below threshold) on all their tokens.

## Predictions (before any data)
- baseline ratio soft1e-4 > soft1e-3 > soft3e-3 (the softer, the more OOD).
- an attention-masking edit should keep d_acc within +-0.003 and rho's within 5%.
- zero_pass.npz: empty edit (no outputs seen yet).
