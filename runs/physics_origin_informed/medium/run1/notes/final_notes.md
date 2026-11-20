# Final notes: implementing the JetClass reconstruction rule inside PET2 (no gradients)

Passes used: **37 of 50** (21 surgical, 16 capture/logits). All weights were built with closed-form
constructions (least squares, eigenvectors, sign patterns, hand-set thresholds) in `work/build_v11.py`
and `work/build_v13.py`.

## Final edits

| file | DoF | E_rule (practice) | notes |
|---|---|---|---|
| `final_small.npz` | **293** | **0.0965** | v11, flag pattern written on 192 channels |
| `final_mid.npz` | **613** | **0.0509** | v11, full 512-channel flag pattern |
| `final_large.npz` | **806** | **0.0471** | v13 = v11 + block-1 pattern switch |

Changed tensors (rank = change rank reported by the oracle, entries = changed entries):

| tensor | small | mid | large |
|---|---|---|---|
| body.token | 28 | 28 | 28 |
| body.embed.mlp.fc1.weight (rank 2) | 3 | 3 | 3 |
| body.embed.mlp.fc1.bias | 3 | 3 | 3 |
| body.embed.mlp.fc2.weight (rank 3) | 3 | 3 | 3 |
| body.embed.mlp.fc2.bias | 7 | 7 | 7 |
| body.time_embed.fc1.weight (rank 1) | 1 | 1 | 1 |
| body.time_embed.fc2.weight (rank 1) | 8 | 8 | 8 |
| body.pid_embed.0.weight (rank 2) | 4 | 4 | 4 |
| body.in_blocks.0.attn.in_proj_weight (rank 2) | 3 | 3 | 3 |
| body.in_blocks.0.attn.out_proj.weight (rank 1) | 3 | 3 | 3 |
| body.in_blocks.0.mlp.fc1.weight (rank 8) | 12 | 12 | 12 |
| body.in_blocks.0.mlp.fc1.bias | 8 | 8 | 8 |
| body.in_blocks.0.mlp.fc2.weight (rank 2) | 203 | 523 | 523 |
| body.in_blocks.0.mlp.fc2.bias | 7 | 7 | 7 |
| body.in_blocks.1.mlp.fc1.weight (rank 1) | – | – | 12 |
| body.in_blocks.1.mlp.fc1.bias | – | – | 1 |
| body.in_blocks.1.mlp.fc2.weight (rank 1) | – | – | 180 |
| **total DoF** | **293** | **613** | **806** |

For every 2-D tensor the changed-entry count is below r(m+n−r), so DoF equals the number of changed entries.
No `generator.*` or `classifier.*` tensor is changed.

## The circuit

Notation: block-0 MLP input n0 = w2 ⊙ tanh(α2·x) (DynamicTanh). Three facts from the weights make an exact
circuit possible:
1. **Block-0 attention is dead** (norm1 α = 6e−6, weight ≤ 1e−3). It is free to reuse.
2. **228 residual channels have block-0 norm2 weight |w2| < 1e−4.** Block-0's live MLP neurons cannot see
   them, and later blocks see them only if something is left there. I use 10 of them as "private wires"
   (A=391, B=110, C=163, Γ=495, N=185, g=72, Γ'=326, N'=506, tc=30, cq=95). A neuron reads such a channel
   with fc1 weight ∝ 1/w2, so it gets r = tanh(α2·x) ∈ [−1, 1] while live neurons see changes of only ~1e−5.
3. **Unused units**: InputBlock hidden units 447/134/987 (|fc1|≈1e−5, norm weight ≈1e−6), block-0 MLP
   units 448,438,933,711,975,532,727,936 (norm weight ≈1e−7 and |z| < 0.06 on all tokens), time_embed unit
   368, block-1 unit 129. Their output is w_n·tanh(α·GELU(z)): exactly 0 for z < −5.5 (erf rounds to −1 in
   float32) and exactly w_n for z ≳ 11. So they act as exact binary gates. I write through them with fc2
   entries scaled by 1/w_n.

**Step 1: exact binary features at the embedding (private wires, value −20 when off, +20 when on).**
- Offset: every valid token gets −20 on A,B,C,Γ,N,g (InputBlock fc2.bias for particles, body.token for the
  4 summary tokens, time_embed unit for the time token).
- Kinematic tags (InputBlock units, z = K(w_e·tanh(α_e θ) − w_e·tanh(α_e·feature)), K=1e5, so the
  threshold is exact to ~1e−8 in log units): A += 40 iff log pT < log 0.8; B += 40 iff log E < log 1.8;
  C += 40 iff log E < log 2.2.
- Pid tags (pid_embed rows 6/7): Γ += 40 for photons, N += 40 for neutral hadrons. Reference copies
  Γ', N' (+20) are used by the erasers.
- Result: at the block-0 MLP input every one of these channels reads u = (r+1)/2 ∈ {0, 1} exactly, for any
  token, in or out of distribution (since |natural x| < 0.8 ≪ 20).

**Step 2: detector (one block-0 MLP neuron, unit 448).**
z = 20·(u_A + u_B + u_Γ + 2u_C + 3u_N − 4.5).
- Sub-threshold photon (A,B,Γ, and C because E<1.8⇒E<2.2): 5 → z = +10.
- Sub-threshold NH (N, C): ≥ 5 → z ≥ +10.
- Every kept token: ≤ 4 → z ≤ −10. That covers photons failing either cut, NH with E ≥ 2.2 (then B = 0
  too), and charged particles (≤ 1+1+2).

So the neuron output is exactly 0 for kept tokens and saturated for dropped ones. Its fc2 column writes
**C·s** (C = 1000, s ∈ {±1}^512 a fixed sign pattern; 192 channels in final_small) into the residual of the
flagged token only.

**Step 3: removal of influence (attention suppression by a fixed "corner" state).**
After the write, the flagged token's residual is dominated by C·s. Its normed vector in *every* later block
is exactly n* = w_b ⊙ s (tanh saturates), and in body.norm it is w_bn ⊙ s. So its key in every head of every
body block and both classifier TokenAttBlocks is a known constant k* = W_k(w⊙s). s is chosen in closed form so
that k* is anti-aligned with the queries:
s = −sign(Σ_{layer,head} U_{l,h}/|U_{l,h}|₁), U = diag(w) W_kᵀ q̄, followed by **one** re-weighting step
in which each query term is weighted by exp(−margin/2) under the first pattern (`s_w2.npy`). The flagged key
then sits many logits below the normal keys, so its softmax weight is ≈0. That is what padding (−1e9)
achieves in reco(x). Normal tokens never enter this state, so their keys, queries and values are untouched.
final_large adds a second, exact corner detector in block-1's MLP (unit 129). It reads 12 channels that are
unsaturated (|r| < 0.8) for every normal token, with z = 15·(Σ s_c r_c − 11): +15 on the corner and ≤ −39
otherwise. It switches the flagged token from a pattern tuned for block 1 (`s_segA.npy`) to the global one.

**Step 4: exact clean-up in block 0's MLP.** Saturated eraser neurons subtract exactly what was added:
A/B/C tags (−40 when u=1), pid tags (read from Γ'/N', remove 40 from Γ/N and 20 from Γ'/N'), the
offsets (+20 via block-0 fc2.bias), the query tag cq (−10 bias), the time tag and the gate. After block 0,
every kept token's residual equals the original up to float rounding (clean e = 5e−6, d_nll = 1e−6).

**Step 5: generator gate (keeps the shared body unchanged at diffusion time t > 0).**
At t = 0 the time token is exactly zero (MPFourier ∝ t), so it is masked. At t > 0 it exists.
- time_embed unit 368 gets one fc1 entry on a Fourier component whose cosine keeps its sign on [0,1], so
  it fires for every t ≥ 1e−4. It writes +1.7e6 to private channel tc and −20/+10 offsets for the time token.
- Block-0 attention, head 0 (16 heads × 32 dims), with 3 in_proj entries: query = Λ·n1[cq]. Every token
  carries an always-on +10 in cq, so all queries share a sign. Key = Λ·n1[tc], where only the time token
  saturates. The time token therefore takes all attention (logit +1000). The head's value (1 entry) maps via
  out_proj (3 entries) to +40 on g and −40 on Γ and N for **every** token when t > 0, and ~1e−4 when t = 0.
- With Γ and N pulled to −1, the detector can never fire at t > 0, and its inputs are bounded for any noised
  input. A gate eraser (reads g) restores Γ, N and g exactly.

**Why kept tokens are unaffected.** Every signal lives on channels block-0's live neurons do not read, in
a dead attention head, or in neurons that are exactly 0 unless their binary condition holds. Everything
written is removed exactly before block 1. Detection is exact arithmetic on binary features, so it
implements the rule's thresholds exactly (pT < 0.80 ∧ E < 1.80 for pid 6; E < 2.20 for pid 7).

**What remains (irreducible part).** The flagged particle is still a real point in the local-physics kNN graph
(neighbour choice uses raw kinematics and has no pid, so no weight can see "this neighbour is a sub-threshold
neutral"). That leaves a small kNN residual, plus residual attention leakage in a few late heads.

## Final practice metrics (oracle `surgical`)

| | small (035) | mid (019) | large (023) |
|---|---|---|---|
| E_rule | 0.0965 | 0.0509 | 0.0471 |
| e soft1e-3 / orig 0.396 | 9.97e−3 | 5.57e−3 | 5.16e−3 |
| e soft1e-4 / orig 0.132 | 7.64e−3 | 5.09e−3 | 4.69e−3 |
| e P / orig 0.0894 | 1.84e−2 | 8.93e−3 | 8.30e−3 |
| e clean / split2 / soft3e-3 | 5e−6 / 3e−5 / 1e−5 | 5e−6 / 2e−5 / 1e−5 | 5e−6 / 2e−5 / 1e−5 |
| d_acc / worst class drop / d_nll | 0 / 0 / 1e−6 | 0 / 0 / 1e−6 | 0 / 0 / 1e−6 |
| rho D / J / P_reco | 1.0000 / 1.0000 / 1.0000 | same | same |
| rho P (raw, expected to change) | 0.837 | 0.845 | 0.845 |
| gen_drift t0.2 / 0.5 / 0.85 | 4e−4 / 4e−4 / 6.6e−3 | same | same |
| R_vis / R_in / R_sub | 0.528 / 0.999 / 0.057 | 0.521 / 0.999 / 0.042 | 0.520 / 0.999 / 0.040 |
| gate_rule_no_g2 | True | True | True |

The generator check is only the gen_drift proxy. For reference, zero-pass gen_drift was 0.016 and v2 without
the gate was 2.04. The real-vs-generated AUC limit could not be evaluated here.

**Unseen probe types** (own jets: clean + one soft NH at 1e−4..3e−3 of pT, or + three soft photons; passes
029–031, final_large): error ratio ||c_edit(x) − c_orig(reco x)|| / ||c_orig(x) − c_orig(reco x)|| = **0.017**
(NH) and **0.036** (3 photons).

## Ideas that failed / lessons
- **Zero-pass** (saturate channel 73 by a −60 bias): no channel is naturally saturated, and forcing one costs
  clean e 0.018 and a class drop of 0.0195.
- **Linear readouts of pT/E/pid from block-0 activations** (v1): too noisy (log E rmse 0.05). Only about 50% of
  sub-threshold photons could be flagged at zero false positives. Replaced by exact InputBlock step units.
- **Fisher-discriminant key edits** (make v·n constant for normal keys): R = signal/noise only 100–150 in late
  blocks, too noisy. Never deployed.
- **v2 (no gate)**: E_rule 0.36 but gen_drift 2.0. Noised jets trigger the detector.
- **Per-jet "impossible kinematics" gate (E < pT)** (v3): the gate leaked through live value columns
  (d_nll 0.26).
- **Gate inhibition through a Fisher readout ×3000** (v4): the readout noise got amplified, and normal tokens
  were flagged (catastrophic, E_rule 145).
- **Gate in a channel read by live neurons** (v5, ch 387): gen_drift 0.25. Channel 494 brought it to 0.002, but
  t = 0.85 stayed at 1.17 because the query channel (83) was ≈0 for 31% of tokens and some out-of-distribution
  tokens escaped through unbounded Fisher readouts. Fixed by the always-on query tag and the bounded binary
  private-wire design (v7).
- **Random sign pattern** (diagnostic, 017): E_rule 1.32. The anti-aligned pattern is essential.
- **τ = 1 re-weighting** (E 0.062), **per-block patterns designed with the true 16-head partition** (v15,
  E 0.073), **second switch in block 4** (v14, no gain): my offline leakage proxy omits the pairwise
  interaction bias x_int (not computable without running the module), so it does not rank patterns reliably.
  I stopped pattern tuning rather than search on the oracle.
- **Support size of the pattern** is non-monotonic: 64 → 0.367, 128 → 0.135, 192 → 0.0965, 256 → 0.111,
  384 → 0.122, 512 → 0.051.
- **Mistake**: my early margin analyses assumed 4 heads × 128 dims. The model has 16 × 32 (the interaction MLP
  outputs 16 channels). The gate head and value dim happened to lie inside head 0, so the circuit is
  unaffected, but the pattern design was done on the wrong partition.
- **kNN diagnostic** (027/028): on soft1e-4 jets where the photon is in no kept particle's kNN set, the
  error is 0.067 vs 0.111 when it is in ≥10 sets. Leakage is the larger part of the remaining error.

## Transparency on offline computations
All forward passes went through the oracle. Offline I used: pinv(fc1) on captured MLP pre-activations;
elementwise DynamicTanh and single weight matrices on captured activations; the rule (`rule.py`) and kNN
neighbour sets / noise injection on the raw practice arrays (pure data geometry); and the scalar function
fc1[368]·MPFourier(t) of one time_embed row on a t-grid (closed form on the weights; no jet data), to check
that the time unit fires for all t > 0.

## Predictions vs outcomes
| id | request | prediction | outcome |
|---|---|---|---|
| 001 | surgical zero_pass.npz | E_rule=1, d_acc=-0.003, d_nll=0.01, rho_P_reco=1, gen_drift=0.02 | E_rule=1.05, d_acc=-0.000879, d_nll=8.17e-05, gen_drift_max=0.0158, e_clean=0.0183, dof=1 |
| 002 | capture clean | n_saturated_channels_block1=5 | (capture/logits; analysed offline) |
| 003 | capture mixB.npz | n_dead_neurons_block0=20, logpt_readout_rmse=0.05 | (capture/logits; analysed offline) |
| 004 | capture mixA.npz | mahalanobis_ratio_block1=1000 | (capture/logits; analysed offline) |
| 005 | capture mixA.npz | mahalanobis_ratio_block5=1000 | (capture/logits; analysed offline) |
| 006 | capture mixA.npz | mahalanobis_ratio_block9=1000 | (capture/logits; analysed offline) |
| 007 | capture mixA.npz | mahalanobis_ratio_bodynorm=1000 | (capture/logits; analysed offline) |
| 008 | surgical v2.npz | E_rule=0.35, d_acc=-0.002, d_nll=0.005, rho_P_reco=1, gen_drift=0.05, e_clean=0.005 | E_rule=0.363, d_acc=0, d_nll=1.4e-05, gen_drift_max=2.04, e_clean=0.000274, dof=3.6e+03 |
| 009 | surgical v3.npz | E_rule=0.36, d_acc=0, gen_drift_max=0.02, gen_drift_t0.85=0.005, e_clean=0.0005 | E_rule=0.51, d_acc=-0.000195, d_nll=0.258, gen_drift_max=3.26, e_clean=0.0238, dof=6.18e+03 |
| 010 | surgical v4.npz | E_rule=0.36, d_acc=0, d_nll=0.0001, gen_drift_max=0.03, e_clean=0.0005 | E_rule=145, d_acc=-0.189, d_nll=144, gen_drift_max=1.17, e_clean=16.9, dof=6.73e+03 |
| 011 | surgical v5.npz | E_rule=0.37, d_acc=0, d_nll=0.0005, gen_drift_max=0.05, e_clean=0.001 | E_rule=0.363, d_acc=0, d_nll=1.38e-05, gen_drift_max=1.17, e_clean=0.000275, dof=6.73e+03 |
| 012 | surgical v6.npz | E_rule=0.363, d_acc=0, gen_drift_t0.2=0.05, gen_drift_t0.85=0.2, e_clean=0.0003 | E_rule=0.363, d_acc=0, d_nll=1.4e-05, gen_drift_max=1.17, e_clean=0.000274, dof=6.73e+03 |
| 013 | surgical v6_noflag.npz | E_rule=1, gen_drift_t0.85=0.01, d_acc=0 | E_rule=0.996, d_acc=0, d_nll=1.4e-05, gen_drift_max=0.015, e_clean=0.000274, dof=6.73e+03 |
| 014 | capture mixB.npz | n_private_channels_with_range_lt_3=10 | (capture/logits; analysed offline) |
| 015 | surgical v7.npz | E_rule=0.36, d_acc=0, d_nll=0.0001, gen_drift_max=0.015, e_clean=0.0003 | E_rule=0.362, d_acc=0, d_nll=1.12e-07, gen_drift_max=0.0106, e_clean=4.12e-06, dof=7.25e+03 |
| 016 | surgical v8.npz | E_rule=0.362, d_acc=0, gen_drift_max=0.012, e_clean=1e-05, dof=1.1e+03 | E_rule=0.362, d_acc=0, d_nll=1.25e-07, gen_drift_max=0.0108, e_clean=4.1e-06, dof=1.08e+03 |
| 017 | surgical v8r.npz | E_rule=0.8, gen_drift_max=0.012 | E_rule=1.32, d_acc=0, d_nll=1.25e-07, gen_drift_max=0.0108, e_clean=4.1e-06, dof=1.08e+03 |
| 018 | surgical v9.npz | E_rule=0.31, e_soft1e-4=0.038, gen_drift_max=0.012, d_acc=0 | E_rule=0.0508, d_acc=0, d_nll=1.25e-07, gen_drift_max=0.0108, e_clean=4.1e-06, dof=1.08e+03 |
| 019 | surgical v11.npz | E_rule=0.051, dof=620, gen_drift_max=0.012, d_acc=0, e_clean=1e-05 | E_rule=0.0509, d_acc=0, d_nll=9.86e-07, gen_drift_max=0.0066, e_clean=5.36e-06, dof=613 |
| 020 | surgical v12.npz | E_rule=0.045, dof=613, gen_drift_max=0.008 | E_rule=0.062, d_acc=0, d_nll=9.86e-07, gen_drift_max=0.0066, e_clean=5.36e-06, dof=613 |
| 021 | surgical v11s256.npz | E_rule=0.12, dof=357, gen_drift_max=0.008 | E_rule=0.111, d_acc=0, d_nll=9.86e-07, gen_drift_max=0.0066, e_clean=5.36e-06, dof=357 |
| 022 | capture mixA.npz | n_channels_absr_lt_0.8_block1=200 | (capture/logits; analysed offline) |
| 023 | surgical v13.npz | E_rule=0.035, dof=810, gen_drift_max=0.007 | E_rule=0.0471, d_acc=0, d_nll=9.99e-07, gen_drift_max=0.0066, e_clean=5.36e-06, dof=806 |
| 024 | surgical v14.npz | E_rule=0.03, dof=880, gen_drift_max=0.007 | E_rule=0.0474, d_acc=0, d_nll=9.9e-07, gen_drift_max=0.0066, e_clean=6.61e-06, dof=870 |
| 025 | logits P | corr_err_vs_knn_involvement=0.5 | (capture/logits; analysed offline) |
| 026 | logits P_reco | mean_abs_logit=3.0 | (capture/logits; analysed offline) |
| 027 | logits soft1e-4 | err_ratio_noknn_over_knn=0.3 | (capture/logits; analysed offline) |
| 028 | logits clean | mean_abs_logit=3.0 | (capture/logits; analysed offline) |
| 029 | logits probeX.npz | ratio_NH=0.15, ratio_3gam=0.15 | (capture/logits; analysed offline) |
| 030 | logits probeX.npz | mean_abs_logit=3.0 | (capture/logits; analysed offline) |
| 031 | logits probeR.npz | mean_abs_logit=3.0 | (capture/logits; analysed offline) |
| 032 | capture mixA.npz | n_channels_block2=20, n_channels_block3=20 | (capture/logits; analysed offline) |
| 033 | surgical v15.npz | E_rule=0.04, dof=1.3e+03, gen_drift_max=0.007, d_acc=0 | E_rule=0.0726, d_acc=0, d_nll=1e-06, gen_drift_max=0.0066, e_clean=5.36e-06, dof=1.31e+03 |
| 034 | surgical v11s128.npz | E_rule=0.3, dof=229, gen_drift_max=0.007 | E_rule=0.135, d_acc=0, d_nll=9.86e-07, gen_drift_max=0.0066, e_clean=5.36e-06, dof=229 |
| 035 | surgical v11s192.npz | E_rule=0.18, dof=293, gen_drift_max=0.007 | E_rule=0.0965, d_acc=0, d_nll=9.86e-07, gen_drift_max=0.0066, e_clean=5.36e-06, dof=293 |
| 036 | surgical v11s384.npz | E_rule=0.06, dof=485, gen_drift_max=0.007 | E_rule=0.122, d_acc=0, d_nll=9.86e-07, gen_drift_max=0.0066, e_clean=5.36e-06, dof=485 |
| 037 | surgical v11s64.npz | E_rule=0.25, dof=165, gen_drift_max=0.007 | E_rule=0.367, d_acc=0, d_nll=9.86e-07, gen_drift_max=0.0066, e_clean=5.36e-06, dof=165 |
