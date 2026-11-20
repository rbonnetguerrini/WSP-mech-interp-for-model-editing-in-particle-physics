# Final notes — target_informed-medium-run3 (pretrain_m, reference = "detector-threshold soft-neutral removal")

## 1. Final edits

All three edits are written in closed form (no gradients, no iterative optimiser). They only touch `body.*` tensors and are float32 with the original shapes.

| file | DoF | E_ref (practice) | gen_drift max |
|---|---|---|---|
| `final_small.npz` (F128) | 223 | 0.190 | 2.5e-4 |
| `final_mid.npz` (F192) | 287 | 0.0812 | 2.5e-4 |
| `final_large.npz` (F192i) | 307 | 0.0756 | 1.0e-2 (only at t=0.85; 2.5e-4 at t=0.2/0.5) |

Per-tensor changes. r is the rank of the change; DoF = min(entries, r(m+n−r)), and for 1-D/3-D tensors it equals the number of entries.

| tensor | shape | entries | rank | DoF (small / mid / large) |
|---|---|---|---|---|
| body.embed.mlp.fc1.weight | 1024×4 | 3 | 2 | 3 |
| body.embed.mlp.fc1.bias | 1024 | 3 | – | 3 |
| body.embed.mlp.fc2.weight | 512×1024 | 3 | 3 | 3 |
| body.embed.mlp.fc2.bias | 512 | 7 | – | 7 |
| body.pid_embed.0.weight | 9×512 | 2 | 2 | 2 |
| body.token | 1×4×512 | 28 | – | 28 |
| body.time_embed.fc1.weight | 1024×512 | 1 | 1 | 1 |
| body.time_embed.fc2.weight | 512×1024 | 8 | 1 | 8 |
| body.in_blocks.0.norm1.alpha | 1 | 1 | – | 1 |
| body.in_blocks.0.attn.in_proj_weight | 1536×512 | 3 | 1 | 3 |
| body.in_blocks.0.attn.out_proj.weight | 512×512 | 1 | 1 | 1 |
| body.in_blocks.0.mlp.fc1.weight | 1024×512 | 13 | 7 | 13 |
| body.in_blocks.0.mlp.fc1.bias | 1024 | 8 | – | 8 |
| body.in_blocks.0.mlp.fc2.weight | 512×1024 | 135 / 199 / 199 | 8 | 135 / 199 / 199 |
| body.in_blocks.0.mlp.fc2.bias | 512 | 7 | – | 7 |
| body.interaction.mlp.fc1.weight (large only) | 1024×3 | 3 | 1 | – / – / 3 |
| body.interaction.mlp.fc1.bias (large only) | 1024 | 1 | – | – / – / 1 |
| body.interaction.mlp.fc2.weight (large only) | 16×1024 | 16 | 1 | – / – / 16 |
| **total** | | | | **223 / 287 / 307** |

The three edits differ only in the support of the kill vector (128 vs 192 channels) and in the optional interaction unit.

## 2. What the reference does and how I found it

**f_ref(x) = f_orig(x with certain neutral particles deleted).** The deleted particles are:
- photons (pid 6) with pT < 0.8 GeV **and** E < 1.8 GeV;
- neutral hadrons (pid 7) with E < 2.2 GeV;
- charged particles never.

How I found it:
- Pass 001: e_orig = 0 exactly on clean and soft3e-3 jets, and on soft1e-4, e_orig equals the original response. So the reference treats a 1e-4 photon as absent.
- Passes 002–003: a scan of photon pT fraction from 3e-5 to 1e-2. Below the threshold, ref(x) equals orig(clean) to the last digit; above it, ref = orig. The cut is not a pT fraction. A per-jet analysis showed an absolute pT cut at 0.8 GeV with an exception for high-E photons.
- Pass 004: 2048 constructed jets with one inserted photon, neutral hadron or charged particle across (pT, E). The rule above reproduces every drop/keep decision (0 mismatches). Clean practice photons all lie outside the drop region, which looks like a detector-acceptance cut.
- P (softer fragmentation) pushes about 75% of jets' photons/neutral hadrons into the drop region, which is why the reference differs on P.

## 3. The circuit

### Design principles
- Make dropped tokens invisible to all later attention.
- Do it with an exact binary logic circuit, so that nothing changes when no token is dropped.
- Never act in generation mode.

### Structural facts used
- Block-0 attention is dead: norm1 α = 6e-6 and |w| ≤ 1e-3. It can be repurposed.
- The block-0 MLP has 813 hidden units with DyT weight < 1e-3, and the embed MLP has 318. These are free units.
- The block-0 norm2 weight is < 1e-5 on 126 residual channels. The block-0 MLP's original units cannot see these channels, and particle content there is |x| < 0.9, also under noise.
- In classification the time token is exactly 0 for any `time_embed` weights (no biases, and MPFourier(0) = 0), and it is masked.

### Channels and constants
Channels: cC = 115, cA = 367, cB = 163, cph = 326, cnh = 110, cg = 509, cq = 89, ct = 201.
Constants: κ = 15, M = 30, H = 65536.

### The steps
1. **Constant shift.**
   - `embed.mlp.fc2.bias[c] −= κ` and `body.token[:,:,c] −= κ` for the 7 channels {cC, cA, cB, cph, cnh, cg, cq}.
   - `in_blocks.0.mlp.fc2.bias[c] += κ` restores them after block 0.
   - `time_embed` adds −κ for t > 0, because the time token also gets the +κ bias in generation.
   - Effect: the block-0 MLP sees x_c ≈ −κ ± 0.9, so its read r_c = tanh(1.1·x_c) = −1 exactly.
2. **Exact kinematic steps** (three dead embed-MLP units).
   - pre = k·(T(f) − T(θ)) + 0.55, with k = 2e4 and T(v) = w_d·tanh(0.187·v) (the embed input norm).
   - Output: w·tanh(1.537·GELU(pre)) ≈ 1[f > θ].
   - fc2 writes M/w into the step's channel, so x_c = −κ + content + M·1[f > θ] and r_c = ±1.
   - The three steps: log pT > log 0.8 → cC; log E > log 1.8 → cA; log E > log 2.2 → cB.
3. **Pid markers.** `pid_embed[6, cph] += M` and `pid_embed[7, cnh] += M`.
4. **Time gate**, active in generation only.
   - A dead `time_embed` unit reads MPFourier channel 120 (cos > 0.99 on [0,1]) with gain 1e5. It saturates for t ≳ 0.002 and writes +H on ct.
   - Block-0 attention, head 0, using 5 entries:
     - q₀ = −1e4·sgn(w1)·n_cq, which is positive for every token because cq ≈ −κ;
     - k₀ = 4.2e8·n_ct;
     - v₁ = 1e-4·n_ct/w1_ct;
     - out_proj[cg, 1] = 3e5.
   - In generation the logit to the time token is about +400, so the attention is exactly 1. This puts +30 on cg for every token and flips r_cg from −1 to +1.
   - In classification the time token is masked; the leak is ~1e-3 on cg.
5. **Exact cancellers** (dead block-0 MLP units).
   - pre = 100·r_c − 50; when the marker is on, the unit saturates and writes exactly −M.
   - Channels covered: cC, cA, cB, cph, cnh, and cg (−30, gate).
   - One more unit, pre = 250·(r_ct − 0.8), removes H from the time token in generation.
6. **Detector** (one dead block-0 MLP unit), with b = (r + 1)/2 ∈ {0, 1}:
   pre = 10·(−7 + 12·b_ph + 36·b_nh − 12·b_C − 12·b_A − 24·b_B) − 400·b_g.
   - pre ≥ +50 iff the token is (photon ∧ pT < 0.8 ∧ E < 1.8) or (nh ∧ E < 2.2), and there is no gate.
   - Otherwise pre ≤ −70, where GELU underflows to exactly 0.
7. **Kill.** The detector's fc2 column writes ±100 on a support S (|S| = 128 or 192) with a sign pattern s.
   - The dropped token's representation is then saturated on S, i.e. w ⊙ s, in every later norm (body blocks 1–11, body.norm, classifier keys).
   - s is closed form from captured queries (passes 006–009): least squares of the per-query key logits towards (LSE − 10), mixed with the sign of the mean key-gain direction.
   - With this, the median attention weight on a dropped token is about 1e-3.
8. **Large edit only: pair kill.** One interaction-MLP unit (old fc2 column norm 4e-5):
   - fc1 = [0, 50, −50]·[log m², log dR, log kT], bias = 50·log 0.12, fc2 = −40 for all 16 heads.
   - Since log kT − log dR = log min(pT_i, pT_j), this gives x_int −= 40·GELU(50·(log 0.12 − log min pT)).
   - Every particle below 0.12 GeV is always in the drop set, and no clean particle is that soft (minimum 0.136 GeV).
   - Effect: such particles neither attend nor are attended among particles, which removes their out-of-distribution pair biases.

### Why unaffected inputs stay unchanged
- Every special channel is restored exactly after block 0: the shift cancels, and the saturated cancellers remove the markers.
- The original block-0 units are blind to these channels (|w2| < 1e-5).
- The detector is exactly 0 unless the rule fires.
- In generation, the gate suppresses the detector, and the gate and H are cancelled.

Measured effects on unaffected inputs:
- clean: e = 6e-5·S, KL 0, d_acc 0;
- split2/soft3e-3: 1e-4 / 2e-4;
- D/J/P_ref response ratios: 1.0000 / 1.0001 / 0.9999;
- gen_drift: 2.5e-4.

Remaining error sources, which these weights cannot reach:
- **kNN contamination.** The local block uses angular kNN with K = 10. It sees only kinematic differences (no pid, no absolute E), and the neighbour mean always includes the dropped particle. Each drop perturbs about 5 neighbours; in P jets, 32% of kept particles are contaminated.
- **Residual attention leakage**, much of it via the kinematic pair bias x_int, which cannot see pid.

## 4. Final practice metrics (passes 035–037)

Original e_v: soft1e-3 0.396, soft1e-4 0.132, P 0.0894, clean 0, split2 7.6e-5, soft3e-3 0.

| | small F128 | mid F192 | large F192i |
|---|---|---|---|
| **E_ref** | 0.1904 | 0.0812 | **0.0756** |
| e soft1e-3 (ratio) | 0.0224 (0.057) | 0.0076 (0.019) | 0.0076 (0.019) |
| e soft1e-4 (ratio) | 0.0251 (0.190) | 0.0072 (0.054) | 0.0048 (0.037) |
| e P (ratio) | 0.0290 (0.324) | 0.0152 (0.170) | 0.0153 (0.171) |
| e clean / split2 / soft3e-3 | 6e-5 / 9e-5 / 2e-4 | 6e-5 / 1e-4 / 2e-4 | 6e-5 / 1e-4 / 2e-4 |
| d_acc / worst class drop / d_nll / KL | 0 / 0 / 3e-6 / 0 | same | same |
| rho D / J / P / P_ref | 1.000 / 1.0001 / 0.855 / 0.9999 | 1.000 / 1.0001 / 0.837 / 0.9999 | 1.000 / 1.0001 / 0.837 / 1.0001 |
| gate_ref_no_g2 | pass | pass | pass |
| gen_drift t0.2 / 0.5 / 0.85 | 2e-4 / 2.5e-4 / 2.4e-4 | same | 2e-4 / 2.5e-4 / 1.04e-2 |
| R_vis / R_in / R_sub | 0.569 / 0.999 / 0.139 | 0.526 / 0.999 / 0.053 | 0.521 / 0.999 / 0.044 |
| dir_err D / J / P | 0.0014 / 0.0011 / 0.187 | 0.0014 / 0.0011 / 0.207 | same as mid |

### Unseen-probe check (passes 038–042, constructed by me, 512 jets each)

Ratio e/e_orig:

| probe | small | mid | large |
|---|---|---|---|
| neutral hadron at 1e-3 | 0.151 | 0.105 | 0.105 |
| neutral hadron at 1e-4 | 0.421 | 0.335 | 0.314 |
| two photons at 3e-4 | 0.240 | 0.177 | 0.177 |
| photon at 5e-4 collinear with the hardest particle | 0.131 | 0.112 | 0.112 |

These are worse than the visible probes, and passes 043–044 show why. Moving the practice soft1e-3 photon to a uniformly random point in dR < 0.4 raises F192's error from 0.0069 to 0.0316 (median 0.011). The error has a tail when the dropped particle lands in the dense core (kNN contamination and close-pair x_int). The practice probes are placed slightly away from the core. Relabelling the practice photons as neutral hadrons gives 0.0081, essentially unchanged.

### Generator risk
- `final_large` has gen_drift 1% at t=0.85. That drift comes from rare noised tokens below 0.12 GeV, because the pair kill cannot be time-gated.
- `final_small` and `final_mid` have no generator-mode change beyond 2.5e-4.

## 5. Predictions vs outcomes

| pass | kind | prediction | outcome |
|---|---|---|---|
| 001 | surgical null | E_ref 1, d_acc 0, e_clean 0 | 1, 0, 0 |
| 002/003 | ref / orig on scan | threshold ~2e-3 of jet pT; ref = orig on clean rows; orig soft1e-3 response 0.4 | threshold between 1e-3 and 1.4e-3 (absolute pT 0.8 & E 1.8); yes; 0.425 |
| 004 | ref on per-pid scan | photon pT<0.8 & E<1.8; nh E<2.2; charged never | exactly, 0 mismatches |
| 005 | capture | 5 dead block-0 units; photon embed norm ×2 | 0 by pre-activation (but 813 with DyT weight<1e-3); ×1.9 |
| 006–009 | captures | 70% heads suppressible | about 94% (13/208 heads problematic) |
| 010 | capture | linear detector error 2% | 9% (corner of the AND) |
| 011 | E1 surgical | E_ref 0.35, ratios 0.3/0.3/0.5, d_acc −0.001, gen 0.05, DoF 2052 | 0.330, 0.08/0.18/0.73, −0.0019, **0.52**, 2052 |
| 012 | E1 per-jet | P rows 0.065, soft1e-3 0.03 | 0.052, 0.026 |
| 013 | capture noised | quiet-channel content ≤1.5 | ≤0.6 |
| 014 | E2 | E_ref 0.30, d_acc −5e-4, rho_J 1.0, gen 0.01, e_clean 5e-4, DoF 616 | **0.145**, −1e-4, 1.0003, 8.7e-4, 1.5e-4, 616 |
| 015 | E3 | 0.11, soft 0.008/0.008, P 0.021, DoF 811 | 0.124, 0.009/0.009, 0.025, 811 |
| 016 | E3 per-jet | P 0.023, P no-drop jets 0.002 | 0.025, 0.0003 |
| 017 | E3 on B | drop photon 0.01, nh 0.012, kept 0.001, charged 5e-4 | 0.033, 0.029, 4e-4, 2e-4 |
| 018 | capture bad cases | 50% detector misfires | 0% (circuit always correct) |
| 019 | capture pairs | first deviation at block 1 | block 0 (kNN) |
| 020 | E4 (k=2e4) | 0.123, e_clean 1e-4, gen 5e-4 | 0.123, 6e-5, 2.5e-4 |
| 021 | S256 | 0.15 @360 | **0.097** |
| 022 | S128 | 0.18 @232 | 0.190 |
| 023 | S192 | 0.13 @296 | **0.081** |
| 024 | S320 | 0.10 @424 | 0.083 |
| 025 | S384 | 0.105 @488 | 0.120 |
| 026 | S256 + flip | 0.09 @426 | 0.101 |
| 027/028 | captures | summary deviation 0.02 | 0.027 (S192), 0.039 (E4) |
| 029 | R192 | 0.075 | 0.138 |
| 030 | R256 | 0.08 | 0.282 |
| 031 | S192i | 0.070, soft1e-4 0.004, gen 0.002 | 0.0756, 0.0048, 0.0104 |
| 032 | capture P pairs | logit dev 0.015, 15% contaminated | 0.029, 32% |
| 033 | S192j | 0.076, gen 0.005 | 0.0759, 0.0151 |
| 034 | S128j | 0.17, gen 0.005 | 0.160, 0.0151 |
| 035 | F128 | 0.19 @223, gen 3e-4 | 0.1904, 223, 2.5e-4 |
| 036 | F192 | 0.081 @287, gen 3e-4 | 0.0812, 287, 2.5e-4 |
| 037 | F192i | 0.0756 @307, gen 0.011 | 0.0756, 307, 0.0104 |
| 038/039 | ref / orig unseen | all rule-drops removed; orig nh1e-3 response 0.3 | consistent; 0.324 |
| 040–042 | finals on unseen | mid ratios 0.03/0.06/0.05/0.15 | 0.105/0.335/0.177/0.112 |
| 043/044 | construction control | practice 0.008, moved 0.03, pid7 0.01 | 0.0069, 0.0316, 0.0081 |

Prediction quality:
- Good on the rule, the circuit's exactness, side effects and reproducibility.
- Poor on how much leakage each kill pattern would leave, and on unseen probes (optimistic).

## 6. Ideas that failed or were dropped
- **E1** (least-squares linear detector + dense kill, 2052 DoF): it was not time-gated, so the generator velocity drifted by 52%. Its pT-only rule also falsely dropped clean high-E soft photons (J response 1.115).
- **Single linear detector for the AND** on the block-0 input: 9% errors at the (pT, E) corner. This was replaced by exact embed-MLP steps.
- **Time gating via the time token through live attention heads or noisy channels**: not robust. The dead block-0 attention with a saturated time-token channel was the only exact route.
- **Per-group sign patterns with a block-4 flip**: helped the dense kill (0.145 → 0.124) but not the sparse one (0.097 → 0.101). Blocks 5–11 have no dead units for further flips.
- **Attention-weight leakage proxies** (from clean-jet queries, without the pair bias): they did not predict empirical E_ref, and the outcome is very sensitive to the sign pattern. Value-aware support selection (R192/R256) was much worse.
- **Removing kNN contamination**: impossible with these weights. The local block sees only kinematic differences (no pid, no absolute E), the neighbour mean always includes the dropped particle, and any pT-based proxy would also hit soft charged particles and noised generator tokens.
- **Pair kill at 0.10 GeV instead of 0.12 GeV**: no gain, and slightly more generator drift.
- **Estimating particle-query attention weights by projecting attention-output differences**: the random-vector control gave identical numbers, so the estimator was invalid.

## 7. Passes used
**44 of 50**: 001–044.
- Reference passes: 002, 004, 038, 043.
- Surgical passes: 001, 011, 014, 015, 020–026, 029–031, 033–037.
- The rest were logits and captures.

## 8. Disclosures
- For analysis only, I applied single weight matrices and the elementwise DynamicTanh (tanh(α·x)·w) to captured activations. This was used to compute queries/keys and to recover block-0 MLP inputs. I never ran any module on raw inputs: all forward passes went through the oracle.
- I opened files only in the two allowed folders. The tool harness itself wrote two background-command logs under {LOCAL_TMP}; I did not open them.
