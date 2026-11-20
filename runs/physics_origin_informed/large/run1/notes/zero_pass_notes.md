# Zero-pass notes (written before any oracle pass)

## Where the network can detect a sub-threshold neutral
- pid enters only through `body.pid_embed` (a per-pid constant vector, added after the embedding+local sum);
  kinematics (deta, dphi, log pt, log E) of the particle itself enter only through `body.embed`
  (InputBlock: DynamicTanh(4) -> fc1(4->2048) -> GELU -> DynamicTanh -> fc2(2048->1024), times the input mask).
  `local_physics` and `interaction` see only kinematic differences/pairs, `add_embed` sees impact parameters
  (exactly 0 for neutrals because add_embed has no biases).
- So at the embedding level the token representation is separable: x = E(kin) + L(neighbours) + P(pid) + A(ip).

## How its influence can be removed exactly
`PET_body.forward` derives the padding mask once, as `x[:, :, 2] != 0` on the summed embedding. A token whose
embedded channel 2 is exactly 0.0 is treated as padding everywhere afterwards: -1e9 in every body attention
mask (row and column), `x*mask` before every norm1, MLP output masked, final `norm(x)*mask` = 0, and the
classifier and generator masks (`body_out[:, :, 2] != 0`) exclude it. That is precisely what reco() does to
the token itself.

Exact zero in float32 is built with absorption + exact cancellation:
1. Repurpose two embed-MLP hidden units with negligible output range (1636, 1970; lost contribution
   ||fc2 col|| * range = 0.010, 0.020) as exact step gates:
   a = [log pt < log 0.8], b = [log E < log 1.8] (fc1 weight -2^20 on the tanh-normalised log pt / log E,
   bias at the threshold, norm weight 1.0). GELU then tanh saturate to exactly 0 or 1.
2. fc2 column of each gate: 0 everywhere except channel 2 = H = 2^36. So E2 = H*(a+b) exactly whenever a gate
   fires (the original O(1) sum and later the O(1) local term are absorbed, ulp(2^36)=8192), and E2 is
   bit-identical to the original when no gate fires.
3. pid_embed[6,2] = pid_embed[7,2] = -2H. Photon/NH with a=b=1: 2H + L - 2H = 0 exactly -> masked.

## What changes for kept tokens
- Charged tokens with no gate firing (~97%): bit-identical embedding.
- Kept photons / NH: channel 2 = -H or -2H (saturated). Their channel-2 information is lost (every norm read
  of channel 2 becomes -w). This is the price: P is separable from E, so with a per-pid constant P the
  zero can only be reached by absorption, which forces a huge value on every token of that pid.
- Charged tokens with pt<0.8 or E<1.8 (~2-3% of charged): channel 2 saturated +.
- Limitation: for neutral hadrons the exact mechanism can only use the photon region (pt<0.8 & E<1.8):
  with a separable E(kin)+P(pid) and exact equality, a kinematic point where both photon and NH are dropped
  forces P6 = P7, so NH with 1.8<=E<2.2 or pt>=0.8 cannot be dropped while photons there are kept.
- Not fixable here: the dropped particle still appears in the kNN (local_physics) neighbourhoods of other
  particles (no pid there), so a small residual remains.

## Predictions for zero_pass.npz (surgical)
E_rule ~0.35 (soft1e-4 ratio ~0.25, soft1e-3 ~0.3, P ~0.5); clean d_acc ~-0.004, d_nll ~+0.01;
rho_D, rho_J, rho_P_reco within 0.9-1.1; gen_drift ~0.05; DoF ~2062.
Main risk: the saturated channel 2 for kept neutrals damages clean performance more than expected.
