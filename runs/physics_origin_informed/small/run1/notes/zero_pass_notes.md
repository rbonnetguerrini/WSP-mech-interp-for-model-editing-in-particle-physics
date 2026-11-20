# Zero-pass notes (from code + weights only)

## Where the network could detect a sub-threshold neutral
- The padding mask is `x[:,:,2]!=0` on the *embedding sum* (E(kin)+L(local)+P(pid)+A(add)), computed once before block 0.
  Making channel 2 exactly 0 for a dropped token would mask it perfectly, but the sum is additive in
  (kinematics-only MLP) + (neighbourhood) + (pid lookup); kept and dropped photons share the same pid row, and the
  local term is continuous, so exact float cancellation is impossible without destroying channel 2. Rejected.
- Only place with raw kinematics *and* pid before any attention: the embedding. The embed MLP sees (deta,dphi,log_pt,log_E)
  through DyT, pid enters through `pid_embed`, add_info is exactly 0 for neutrals.
- The embed MLP has ~8 dead hidden units (|norm.weight|*|fc2 col| ~1e-7: 76,163,155,...). I repurpose 3 of them as
  exact step functions (fc1 slope K=2e5 on one normalised input, GELU underflows to 0 below, tanh saturates to 1 above):
  t_pt=[pT>=0.8], t_E18=[E>=1.8], t_E22=[E>=2.2].
- Flag channel f=51 (cheapest channel by weight-norm reader cost; its block-0 norm1 weight is 1e-4):
  x_f/M = -1 + p(pid) - (t_pt + t_E18 + 3 t_E22), p(photon)=1.5, p(NH)=3.5, other pids ~0, M=400.
  => x_f >= +0.5M exactly for the rule's dropped set, <= -0.5M for every kept particle, <= -M with no pid at all.
  Summary tokens get x_f=-M (body.token). The time token (only nonzero at t>0, generator) gets -1e5*h_6(t) via
  time_embed unit 6, whose pre-activation is >0 for all t in (0,1] (checked from MPFourier buffers + fc1).
  All later DyT norms saturate on f: n_f = -w (kept) / +w (dropped).

## How the influence is removed
Channel f becomes a key-side "bias feature". For every attention that can read a dropped token
(8 body blocks x 8 heads, 2 classifier token-attention blocks x 8 heads) only column f of in_proj is rewritten:
- query: kept/summary queries all read the constant n_f=-w, so Wq[:,f] adds a constant offset gamma*e (e = head direction
  with smallest |q|*|k| spread), gamma > max|q.e|;
- key: Wk[:,f] = -c sign(w) e, so a dropped key gets logit penalty 2c|w|(q.e+gamma)/sqrt(16) >= L=30 for every kept
  query (attention weight x e^-30), while all kept keys get the same per-query constant (softmax invariant).
- value: Wv[:,f] zeroed (channel f content is gone); MLP fc1 readers of f are compensated with fc1.bias
  (zero-ablation for now; mean-ablation once I have activations).
Classifier keys are x_body (saturated +-w_bn on f), queries are summary tokens (-w_bn on f).

## Expected error sources
- channel 51 information is lost everywhere (mean/zero ablation) -> small change on clean jets;
- "bad term" gamma*(k.e) of the column-only construction (weights-only estimate 0.03-0.5 logits per head, too high for
  some heads; to be reduced with real q/k statistics or a rank-1 projection of Wq/Wk off e);
- dropped particles still enter other particles' kNN local features (local block has no pid) -> not removable;
- generator: noised soft neutrals at t>0 are also flagged, so gen_drift will be > 0.

## Predictions for zero_pass.npz
E_rule ~0.40, d_acc ~ -0.006, d_nll ~ +0.02, rho_D/J/P_reco ~1.0 (+-0.1), clean e_v ~0.15, gen_drift ~0.15, DoF ~ 5.6k.
