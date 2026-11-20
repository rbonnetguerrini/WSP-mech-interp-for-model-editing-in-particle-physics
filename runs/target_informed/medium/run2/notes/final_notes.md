# Final notes — weight-space implementation of the reference (PET2 `pretrain_m`)

Passes used: **35 of 50** (4 more requests were rejected at no cost: 028 had non-finite weights from a buggy build; 031-033 pointed at a file my script had failed to write).

## 1. What the reference does, and how I found it

**Finding.** f_ref(x) = f_orig(x with every soft *neutral* particle deleted), where a particle is deleted iff
- photon (pid 6): **pT < 0.8 GeV and E < 1.8 GeV**;
- neutral hadron (pid 7): **E < 2.2 GeV** (with E ≥ pT this implies pT < 2.2);
- charged particles (pids 0,1,2,4,5,8) are never deleted.

No clean practice jet contains such a particle, so f_ref = f_orig on clean jets, on D, J and soft3e-3, and (almost) on split2.

**How I found it.**
1. Pass 001 (baseline): e(orig) = 0 on clean and soft3e-3; e(orig) on soft1e-4 = its full response, and on soft1e-3 ≈ 97 % of its response.
2. Passes 002/003: reference and original logits on 16 constructed variants × 128 jets. Whenever the reference differs from orig(x+p), it equals orig(x) to within 1e-3 S, i.e. the particle is deleted, never re-weighted. Photons and NHs at z ≤ 5e-4 are always deleted. Soft1e-3 photons are deleted in 90 % of jets. Charged particles at 1e-3 are never deleted, and neither are photons at ≥ 2e-3.
3. Pass 004: a 32×32 grid over (log pT, log E/pT) for an added photon and an added NH. The photon region is an L-shape (pT and E cuts); the NH region is a pure E cut.
4. Pass 030: a fine scan pins the photon cuts to 0.7999 < pT* < 0.8003 and 1.79997 < E* < 1.80032. The NH E cut lies in (2.190, 2.204), i.e. 2.2.
5. Passes 035-038 (generalisation check): soft NH at 1e-3 and 1e-4, two soft photons, a photon at 5e-4, photons at pT 0.79 vs 0.81 (E 0.85), a soft charged particle at 1e-4, and P jets plus a soft photon. All behave as the rule predicts (ref = orig(base jet) exactly for every deleted case).

The original model is IR-unsafe: one 0.05 GeV photon moves the logits by 0.13 S. The reference is the same network behind a detector-like cut on soft neutrals.

## 2. The circuit (all final edits share it)

Notation: the body sequence is [4 summary][time][150 particles]. DyT(x) = w ⊙ tanh(α x). Block 0's attention is dead in the original model (`norm1.alpha = 0`), so block 0 acts on each token alone.

**(a) A private wire: residual channel c* = 49.** It has the lowest estimated ablation cost (reader-norm × activation rms). Every *live* writer of channel 49 is zeroed:
- `body.token`, `pid_embed` column, `add_embed` / `time_embed` fc2 rows;
- embed fc2 row and bias;
- the local-physics MLP and its two blocks (out_proj rows, fc2 rows, biases);
- out_proj rows of body blocks 1-11 and fc2 rows/biases of body blocks 0-11.

Entries belonging to dead neurons (DyT weight 0) and the dead block-0 attention are left untouched, since they never write. After this, **every non-flagged token carries exactly 0 in channel 49 at every depth**, in classification and generation alike.

**(b) Kinematic steps (embed MLP, 3 dead neurons).** Input DyT is monotone, so a cut on log pT or log E is a cut on u = w_in·tanh(α_in x).

  h_j = ε·tanh(α_e·GELU(β(u0_j − u))),  β = 2·10^6,  ε = 1e-5,

gives S_p = [pT < 0.8], S_E = [E < 1.8] and S_N = [E < 2.2]. The transition band is ~1e-5 in log; no practice particle falls inside it. They write

  r0[c*] = u_lvl·(S_p + S_E + 2 S_N) − 4.5 u_lvl + P_pid,  u_lvl = 0.2,

with P_6 = +1, P_7 = +3, every other pid 0 (from the `pid_embed` column and the embed fc2 bias). The result sits on an exact discrete level:
- photon: k − 3.5;
- NH: k − 1.5;
- charged, or pid absent: k − 4.5;

where k = S_p + S_E + 2 S_N ∈ {0..4}. The level is positive (+0.5, +1.5 or +2.5) **iff** the reference deletes the particle. Summary and time tokens sit at 0.

**(c) Block-0 staircase (8 dead block-0 MLP neurons).**
- `norm2.weight[49]` is set to 0.01 (the alive neurons effectively lose this input).
- Each neuron reads tanh(α·level) with gain 1e4 and fires on one side of a mid-level threshold.
- Their fc2 entries (scale ε) sum to exactly −level, so the wire returns to 0 for every token. For positive levels they add +200 instead.

After block 0, channel 49 = **200 for deleted particles and exactly 0 for all others**.

**(d) Key penalty.** In every attention that particles are keys of (body blocks 1-11 and both classifier token-attention blocks), only the key-rows of column 49 of `in_proj_weight` change: ΔW_k[:,49] = V / (w_49·tanh(α·200)).
- A flagged key is shifted by V.
- Every other key is shifted by 0 exactly, since its xn_49 = 0.
- Per head, V_h is the least-squares solution of q_i·V_h = −10·√32 over captured queries (summary and particle tokens of 64 clean jets). Flagged keys therefore sit ~10 logits (worst query ≈ −0.7 to −6) below their natural value.

Where a block's `norm1.weight[49]` was ~0, it is set to 1; this is harmless because normal tokens read 0 there anyway.

**Why unaffected inputs are unaffected.** For tokens that are not deleted, the only change is the ablation of channel 49 (plus, in the mid/large edits, its least-squares re-routing). The penalty and staircase contribute exactly zero. On clean jets this costs e = 0.005 S (0.0035 S for the large edit). With pid = None nothing can be flagged, since every level is ≤ −0.5.

**(e) Re-routing (mid and large only).** Each reader of xn_49 gets the ridge-least-squares prediction xn_49 ≈ β·xn_{others} folded into its other columns: W[:, others] += W[:,49]·βᵀ, with β fitted on the 64 captured jets (held-out R² from 0.99 in early blocks to 0.72 in block 11).
- mid: the classifier in_proj (keys/values; also queries in classifier block 0) and attention in_proj of body blocks 9-11;
- large: every body block's attention in_proj and MLP fc1 (including block 0), plus the classifier in_proj.

## 3. Final edits and practice metrics (all limits pass, `gate_ref_no_g2 = True`)

| edit | content | DoF | E_ref | e(soft1e-3) | e(soft1e-4) | e(P) | e(clean) | e(split2) | e(soft3e-3) | d_acc | worst class drop | d_nll | KL | rho D / J / P_ref | gen_drift max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final_small | (a)-(d) | 25 797 | **0.0410** | 0.00568 | 0.00581 | 0.00579 | 0.00507 | 0.00510 | 0.00555 | +0.0001 | 0.0020 | 0.00026 | 0.00011 | 0.991 / 0.990 / 1.000 | 0.035 |
| final_mid | + reroute cls & blocks 9-11 attn | 43 697 | **0.0366** | 0.00484 | 0.00500 | 0.00534 | 0.00416 | 0.00428 | 0.00473 | 0.0000 | 0.0020 | 0.00026 | 0.00007 | 0.994 / 0.995 / 1.000 | 0.035 |
| final_large | + reroute all readers | 73 359 | **0.0325** | 0.00425 | 0.00442 | 0.00476 | 0.00351 | 0.00357 | 0.00375 | +0.0001 | 0.0010 | 0.00019 | 0.00003 | 0.997 / 0.996 / 1.000 | 0.035 |

e_orig: soft1e-3 0.396, soft1e-4 0.132, P 0.0894.

Other reported values:
- rho(P) = 0.847 for all three. This is expected: the reference itself changes the P response.
- R_vis 0.51, R_in 0.99, R_sub 0.026.
- dir_err: D 0.02-0.03, J 0.02-0.03, P 0.204.

The remaining error is almost entirely the channel-49 ablation floor (e(clean)). The extra error on probes (≈ 0.0005-0.0008 S) comes from the local kNN block. It runs on raw kinematics before any pid is available, and its neighbour distance includes log pT, so a deleted soft photon still displaces other particles' neighbours. Weights cannot undo this. It grows for photons just below the 0.8 GeV cut (generalisation group ph0.79: e = 0.045 S against e_orig 0.29), and is negligible for very soft ones.

Generalisation check with final_small (exp4, 256 jets per group; e/S, edit vs reference):
- deleted, close to the floor: nh1e-3 0.0063, nh1e-4 0.0065, two photons 0.0068, ph5e-4 0.0063, P+photon 0.0072;
- kept: ph0.81 0.0055, charged 1e-4 0.0057;
- worst (kNN path): ph0.79 0.045.

## 4. Tensors changed (rank = singular values > 1e-4 of the largest; DoF = min(changed entries, r(m+n−r)) for 2-D, changed entries otherwise)

#### final_small — 74 tensors, total DoF 25797

| tensor | shape | rank | changed entries | DoF |
|---|---|---|---|---|
| body.token | [1, 4, 512] | - | 4 | 4 |
| body.embed.mlp.fc2.weight | [512, 1024] | 1 | 709 | 709 |
| body.embed.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.local_physics.mlp.fc2.weight | [512, 1024] | 1 | 372 | 372 |
| body.local_physics.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.add_embed.0.fc2.weight | [512, 1024] | 1 | 971 | 971 |
| body.time_embed.fc2.weight | [512, 1024] | 1 | 629 | 629 |
| body.local_physics.in_blocks.0.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.local_physics.in_blocks.0.mlp.fc2.weight | [512, 1024] | 1 | 104 | 104 |
| body.local_physics.in_blocks.0.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.local_physics.in_blocks.1.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.local_physics.in_blocks.1.mlp.fc2.weight | [512, 1024] | 1 | 321 | 321 |
| body.local_physics.in_blocks.1.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.0.mlp.fc2.weight | [512, 1024] | 1 | 219 | 219 |
| body.in_blocks.0.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.1.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.1.mlp.fc2.weight | [512, 1024] | 1 | 191 | 191 |
| body.in_blocks.1.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.2.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.2.mlp.fc2.weight | [512, 1024] | 1 | 284 | 284 |
| body.in_blocks.2.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.3.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.3.mlp.fc2.weight | [512, 1024] | 1 | 520 | 520 |
| body.in_blocks.3.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.4.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.4.mlp.fc2.weight | [512, 1024] | 1 | 921 | 921 |
| body.in_blocks.4.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.5.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.5.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.5.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.6.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.6.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.6.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.7.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.7.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.7.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.8.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.8.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.8.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.9.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.9.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.9.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.10.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.10.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.10.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.11.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.11.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.11.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.pid_embed.0.weight | [9, 512] | 1 | 8 | 8 |
| body.embed.mlp.fc1.weight | [1024, 4] | 2 | 12 | 12 |
| body.embed.mlp.fc1.bias | [1024] | - | 3 | 3 |
| body.embed.mlp.norm.weight | [1024] | - | 3 | 3 |
| body.in_blocks.0.norm2.weight | [512] | - | 1 | 1 |
| body.in_blocks.0.mlp.fc1.weight | [1024, 512] | 1 | 8 | 8 |
| body.in_blocks.0.mlp.fc1.bias | [1024] | - | 8 | 8 |
| body.in_blocks.0.mlp.norm.weight | [1024] | - | 8 | 8 |
| body.in_blocks.1.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.1.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.2.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.2.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.3.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.3.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.4.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.4.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.5.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.5.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.6.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.7.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.8.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.9.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.10.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.11.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| classifier.in_blocks.0.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| classifier.in_blocks.1.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |

#### final_mid — 74 tensors, total DoF 43697

| tensor | shape | rank | changed entries | DoF |
|---|---|---|---|---|
| body.token | [1, 4, 512] | - | 4 | 4 |
| body.embed.mlp.fc2.weight | [512, 1024] | 1 | 709 | 709 |
| body.embed.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.local_physics.mlp.fc2.weight | [512, 1024] | 1 | 372 | 372 |
| body.local_physics.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.add_embed.0.fc2.weight | [512, 1024] | 1 | 971 | 971 |
| body.time_embed.fc2.weight | [512, 1024] | 1 | 629 | 629 |
| body.local_physics.in_blocks.0.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.local_physics.in_blocks.0.mlp.fc2.weight | [512, 1024] | 1 | 104 | 104 |
| body.local_physics.in_blocks.0.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.local_physics.in_blocks.1.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.local_physics.in_blocks.1.mlp.fc2.weight | [512, 1024] | 1 | 321 | 321 |
| body.local_physics.in_blocks.1.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.0.mlp.fc2.weight | [512, 1024] | 1 | 219 | 219 |
| body.in_blocks.0.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.1.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.1.mlp.fc2.weight | [512, 1024] | 1 | 191 | 191 |
| body.in_blocks.1.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.2.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.2.mlp.fc2.weight | [512, 1024] | 1 | 284 | 284 |
| body.in_blocks.2.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.3.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.3.mlp.fc2.weight | [512, 1024] | 1 | 520 | 520 |
| body.in_blocks.3.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.4.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.4.mlp.fc2.weight | [512, 1024] | 1 | 921 | 921 |
| body.in_blocks.4.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.5.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.5.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.5.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.6.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.6.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.6.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.7.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.7.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.7.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.8.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.8.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.8.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.9.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.9.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.9.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.10.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.10.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.10.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.11.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.11.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.11.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.pid_embed.0.weight | [9, 512] | 1 | 8 | 8 |
| body.embed.mlp.fc1.weight | [1024, 4] | 2 | 12 | 12 |
| body.embed.mlp.fc1.bias | [1024] | - | 3 | 3 |
| body.embed.mlp.norm.weight | [1024] | - | 3 | 3 |
| body.in_blocks.0.norm2.weight | [512] | - | 1 | 1 |
| body.in_blocks.0.mlp.fc1.weight | [1024, 512] | 1 | 8 | 8 |
| body.in_blocks.0.mlp.fc1.bias | [1024] | - | 8 | 8 |
| body.in_blocks.0.mlp.norm.weight | [1024] | - | 8 | 8 |
| body.in_blocks.1.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.1.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.2.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.2.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.3.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.3.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.4.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.4.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.5.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.5.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.6.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.7.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.8.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.9.attn.in_proj_weight | [1536, 512] | 2 | 785408 | 4092 |
| body.in_blocks.10.attn.in_proj_weight | [1536, 512] | 2 | 785400 | 4092 |
| body.in_blocks.11.attn.in_proj_weight | [1536, 512] | 2 | 785407 | 4092 |
| classifier.in_blocks.0.attn.in_proj_weight | [1536, 512] | 2 | 785408 | 4092 |
| classifier.in_blocks.1.attn.in_proj_weight | [1536, 512] | 2 | 523776 | 4092 |

#### final_large — 83 tensors, total DoF 73359

| tensor | shape | rank | changed entries | DoF |
|---|---|---|---|---|
| body.token | [1, 4, 512] | - | 4 | 4 |
| body.embed.mlp.fc2.weight | [512, 1024] | 1 | 709 | 709 |
| body.embed.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.local_physics.mlp.fc2.weight | [512, 1024] | 1 | 372 | 372 |
| body.local_physics.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.add_embed.0.fc2.weight | [512, 1024] | 1 | 971 | 971 |
| body.time_embed.fc2.weight | [512, 1024] | 1 | 629 | 629 |
| body.local_physics.in_blocks.0.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.local_physics.in_blocks.0.mlp.fc2.weight | [512, 1024] | 1 | 104 | 104 |
| body.local_physics.in_blocks.0.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.local_physics.in_blocks.1.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.local_physics.in_blocks.1.mlp.fc2.weight | [512, 1024] | 1 | 321 | 321 |
| body.local_physics.in_blocks.1.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.0.mlp.fc2.weight | [512, 1024] | 1 | 219 | 219 |
| body.in_blocks.0.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.1.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.1.mlp.fc2.weight | [512, 1024] | 1 | 191 | 191 |
| body.in_blocks.1.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.2.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.2.mlp.fc2.weight | [512, 1024] | 1 | 284 | 284 |
| body.in_blocks.2.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.3.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.3.mlp.fc2.weight | [512, 1024] | 1 | 520 | 520 |
| body.in_blocks.3.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.4.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.4.mlp.fc2.weight | [512, 1024] | 1 | 921 | 921 |
| body.in_blocks.4.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.5.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.5.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.5.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.6.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.6.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.6.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.7.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.7.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.7.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.8.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.8.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.8.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.9.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.9.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.9.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.10.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.10.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.10.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.in_blocks.11.attn.out_proj.weight | [512, 512] | 1 | 512 | 512 |
| body.in_blocks.11.mlp.fc2.weight | [512, 1024] | 1 | 1024 | 1024 |
| body.in_blocks.11.mlp.fc2.bias | [512] | - | 1 | 1 |
| body.pid_embed.0.weight | [9, 512] | 1 | 8 | 8 |
| body.embed.mlp.fc1.weight | [1024, 4] | 2 | 12 | 12 |
| body.embed.mlp.fc1.bias | [1024] | - | 3 | 3 |
| body.embed.mlp.norm.weight | [1024] | - | 3 | 3 |
| body.in_blocks.0.norm2.weight | [512] | - | 1 | 1 |
| body.in_blocks.0.mlp.fc1.weight | [1024, 512] | 1 | 104033 | 1535 |
| body.in_blocks.0.mlp.fc1.bias | [1024] | - | 8 | 8 |
| body.in_blocks.0.mlp.norm.weight | [1024] | - | 8 | 8 |
| body.in_blocks.1.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.1.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.2.attn.in_proj_weight | [1536, 512] | 2 | 778084 | 4092 |
| body.in_blocks.2.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.3.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.3.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.4.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.4.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.5.attn.in_proj_weight | [1536, 512] | 1 | 512 | 512 |
| body.in_blocks.5.norm1.weight | [512] | - | 1 | 1 |
| body.in_blocks.6.attn.in_proj_weight | [1536, 512] | 2 | 783908 | 4092 |
| body.in_blocks.7.attn.in_proj_weight | [1536, 512] | 2 | 784903 | 4092 |
| body.in_blocks.8.attn.in_proj_weight | [1536, 512] | 2 | 785182 | 4092 |
| body.in_blocks.9.attn.in_proj_weight | [1536, 512] | 2 | 785408 | 4092 |
| body.in_blocks.10.attn.in_proj_weight | [1536, 512] | 2 | 785400 | 4092 |
| body.in_blocks.11.attn.in_proj_weight | [1536, 512] | 2 | 785407 | 4092 |
| classifier.in_blocks.0.attn.in_proj_weight | [1536, 512] | 2 | 785408 | 4092 |
| classifier.in_blocks.1.attn.in_proj_weight | [1536, 512] | 2 | 523776 | 4092 |
| body.in_blocks.1.mlp.fc1.weight | [1024, 512] | 1 | 490705 | 1535 |
| body.in_blocks.3.mlp.fc1.weight | [1024, 512] | 1 | 521522 | 1535 |
| body.in_blocks.5.mlp.fc1.weight | [1024, 512] | 1 | 523263 | 1535 |
| body.in_blocks.6.mlp.fc1.weight | [1024, 512] | 1 | 523212 | 1535 |
| body.in_blocks.7.mlp.fc1.weight | [1024, 512] | 1 | 523263 | 1535 |
| body.in_blocks.8.mlp.fc1.weight | [1024, 512] | 1 | 523264 | 1535 |
| body.in_blocks.9.mlp.fc1.weight | [1024, 512] | 1 | 523251 | 1535 |
| body.in_blocks.10.mlp.fc1.weight | [1024, 512] | 1 | 523264 | 1535 |
| body.in_blocks.11.mlp.fc1.weight | [1024, 512] | 1 | 523227 | 1535 |

## 5. Predictions vs outcomes (all requests)

| id | kind | file | prediction | outcome |
|---|---|---|---|---|
| 001 | surgical | orig | {"E_ref": 1.0, "d_acc": 0.0, "gen_drift": 0.0} | E_ref=1.0000, e_clean=0.0000, d_acc=0.0000, class_drop=0.0000, d_nll=0.0000, gen_drift=0.000, rho_P_ref=1.000, gate=True, DoF=0 |
| 002 | reference | exp1.npz | {"ref_soft1e-4_eq_orig_clean": 1.0, "ref_clean_eq_orig_clean": 1.0} | data returned (analysed in work/) |
| 003 | logits | orig | {"n": 2048} | data returned (analysed in work/) |
| 004 | reference | exp2.npz | {"photon_pt_thr_GeV": 0.8, "photon_E_thr_GeV": 2.0, "nh_E_thr_GeV": 2.2} | data returned (analysed in work/) |
| 005 | capture | orig | {"n_jets": 96} | data returned (analysed in work/) |
| 006 | capture | orig | {"n_jets": 64} | data returned (analysed in work/) |
| 007 | capture | orig | {"n_jets": 64} | data returned (analysed in work/) |
| 008 | capture | orig | {"n_jets": 64} | data returned (analysed in work/) |
| 009 | capture | orig | {"n_jets": 64} | data returned (analysed in work/) |
| 010 | surgical | edit_v1.npz | {"E_ref": 0.15, "d_acc": 0.0, "d_nll": 0.001, "gen_drift_max": 0.02, "rho_P_ref": 1.0, "e_clean": 0.01} | E_ref=0.2374, e_clean=0.0345, d_acc=-0.0034, class_drop=0.0459, d_nll=0.0101, gen_drift=4.387, rho_P_ref=1.004, gate=False, DoF=31780 |
| 011 | surgical | edit_v2.npz | {"E_ref": 0.15, "d_acc": -0.002, "d_nll": 0.005, "class_drop": 0.01, "gen_drift_max": 0.05, "e_clean": 0.02, "rho_P_ref": 1.0} | E_ref=0.0425, e_clean=0.0052, d_acc=0.0000, class_drop=0.0029, d_nll=0.0003, gen_drift=0.035, rho_P_ref=1.000, gate=True, DoF=32332 |
| 012 | surgical | edit_v3_119.npz | {"E_ref": 0.03, "e_clean": 0.003, "d_acc": 0.0, "class_drop": 0.005, "gen_drift_max": 0.03, "rho_P_ref": 1.0} | E_ref=0.7142, e_clean=0.1639, d_acc=-0.0148, class_drop=0.1162, d_nll=0.0555, gen_drift=0.043, rho_P_ref=0.551, gate=False, DoF=32327 |
| 013 | surgical | edit_v4_rr.npz | {"E_ref": 0.02, "e_clean": 0.0022, "d_acc": 0.0, "class_drop": 0.004, "gen_drift_max": 0.035, "rho_P_ref": 1.0} | E_ref=0.0341, e_clean=0.0037, d_acc=0.0000, class_drop=0.0010, d_nll=0.0002, gen_drift=0.035, rho_P_ref=1.000, gate=True, DoF=78367 |
| 014 | surgical | edit_v5_K3.npz | {"E_ref": 0.5, "e_clean": 0.002, "d_acc": 0.0, "class_drop": 0.003, "gen_drift_max": 0.01, "rho_P_ref": 1.0} | E_ref=1.0550, e_clean=0.0019, d_acc=-0.0001, class_drop=0.0010, d_nll=0.0002, gen_drift=0.005, rho_P_ref=0.999, gate=True, DoF=16968 |
| 015 | surgical | edit_v6_sparse.npz | {"E_ref": 0.0425, "e_clean": 0.0052, "d_acc": 0.0, "class_drop": 0.003, "gen_drift_max": 0.035, "rho_P_ref": 1.0} | E_ref=0.0425, e_clean=0.0052, d_acc=0.0000, class_drop=0.0029, d_nll=0.0003, gen_drift=0.035, rho_P_ref=1.000, gate=True, DoF=25797 |
| 016 | surgical | edit_v6_T16.npz | {"E_ref": 0.041, "e_clean": 0.0052, "e_P": 0.0059, "d_acc": 0.0, "gen_drift_max": 0.035} | E_ref=0.0443, e_clean=0.0055, d_acc=0.0000, class_drop=0.0029, d_nll=0.0005, gen_drift=0.035, rho_P_ref=1.000, gate=True, DoF=25797 |
| 017 | surgical | edit_v7.npz | {"E_ref": 0.035, "e_clean": 0.0042, "d_acc": 0.0, "class_drop": 0.002, "gen_drift_max": 0.035, "rho_P_ref": 1.0} | E_ref=0.0410, e_clean=0.0051, d_acc=0.0001, class_drop=0.0020, d_nll=0.0003, gen_drift=0.035, rho_P_ref=1.000, gate=True, DoF=25797 |
| 018 | surgical | edit_v7_T16.npz | {"E_ref": 0.04, "e_clean": 0.0051, "e_soft1e-4": 0.0056, "gen_drift_max": 0.035} | E_ref=0.0410, e_clean=0.0051, d_acc=0.0001, class_drop=0.0020, d_nll=0.0003, gen_drift=0.035, rho_P_ref=1.000, gate=True, DoF=25797 |
| 019 | surgical | edit_v7_c263.npz | {"E_ref": 0.045, "e_clean": 0.0055, "gen_drift_max": 0.04} | E_ref=0.0610, e_clean=0.0082, d_acc=0.0002, class_drop=0.0029, d_nll=0.0001, gen_drift=0.034, rho_P_ref=0.997, gate=True, DoF=25796 |
| 020 | surgical | edit_v7_c452.npz | {"E_ref": 0.045, "e_clean": 0.0055, "gen_drift_max": 0.04} | E_ref=0.0486, e_clean=0.0054, d_acc=-0.0006, class_drop=0.0049, d_nll=0.0002, gen_drift=0.037, rho_P_ref=0.996, gate=True, DoF=25796 |
| 021 | surgical | edit_v7_c76.npz | {"E_ref": 0.045, "e_clean": 0.0055, "gen_drift_max": 0.04} | E_ref=0.1066, e_clean=0.0130, d_acc=-0.0013, class_drop=0.0049, d_nll=0.0023, gen_drift=0.036, rho_P_ref=0.993, gate=True, DoF=25796 |
| 022 | surgical | edit_v8_large.npz | {"E_ref": 0.033, "e_clean": 0.0036, "d_acc": 0.0, "class_drop": 0.002, "gen_drift_max": 0.035, "rho_P_ref": 1.0} | E_ref=0.0325, e_clean=0.0035, d_acc=0.0001, class_drop=0.0010, d_nll=0.0002, gen_drift=0.035, rho_P_ref=1.000, gate=True, DoF=73359 |
| 023 | surgical | edit_v8_mid.npz | {"E_ref": 0.037, "e_clean": 0.0044, "d_acc": 0.0, "class_drop": 0.002, "gen_drift_max": 0.035, "rho_P_ref": 1.0} | E_ref=0.0391, e_clean=0.0048, d_acc=0.0006, class_drop=0.0010, d_nll=0.0002, gen_drift=0.035, rho_P_ref=1.001, gate=True, DoF=36542 |
| 024 | surgical | edit_v9_cls.npz | {"E_ref": 0.038, "e_clean": 0.0046, "gen_drift_max": 0.035} | E_ref=0.0394, e_clean=0.0047, d_acc=0.0004, class_drop=0.0020, d_nll=0.0004, gen_drift=0.035, rho_P_ref=0.999, gate=True, DoF=32957 |
| 025 | surgical | edit_v9_clsattn.npz | {"E_ref": 0.036, "e_clean": 0.0043, "gen_drift_max": 0.035} | E_ref=0.0366, e_clean=0.0042, d_acc=0.0000, class_drop=0.0020, d_nll=0.0003, gen_drift=0.035, rho_P_ref=1.000, gate=True, DoF=43697 |
| 026 | surgical | edit_sq.npz | {"E_ref": 0.025, "e_clean": 0.003, "d_acc": 0.0, "class_drop": 0.002, "gen_drift_max": 0.02, "rho_P_ref": 1.0} | E_ref=0.0301, e_clean=0.0030, d_acc=0.0002, class_drop=0.0029, d_nll=0.0002, gen_drift=79.604, rho_P_ref=1.000, gate=True, DoF=34523 |
| 027 | surgical | edit_sq2.npz | {"E_ref": 0.032, "e_clean": 0.0033, "d_acc": 0.0, "class_drop": 0.003, "gen_drift_max": 0.05, "rho_P_ref": 1.0} | E_ref=0.0346, e_clean=0.0041, d_acc=0.0001, class_drop=0.0020, d_nll=0.0001, gen_drift=0.295, rho_P_ref=1.001, gate=True, DoF=34522 |
| 028 | surgical | edit_sq2_noflag.npz | {"E_ref": 1.0, "e_clean": 0.004, "gen_drift_max": 0.05} | rejected (no cost): ValueError: classifier.in_blocks.0.attn.in_proj_weight: non- |
| 029 | surgical | edit_sq2_noflag.npz | {"E_ref": 1.0, "e_clean": 0.004, "gen_drift_max": 0.05} | E_ref=1.0005, e_clean=0.0041, d_acc=0.0001, class_drop=0.0020, d_nll=0.0001, gen_drift=0.002, rho_P_ref=1.001, gate=True, DoF=34522 |
| 030 | reference | exp3.npz | {"photon_pt_thr": 0.8, "photon_E_thr": 1.8, "nh_E_thr": 2.2} | data returned (analysed in work/) |
| 031 | reference | exp4.npz | {"ch1e-4_dropped": 0, "ph0.81_dropped": 0, "ph0.79_dropped": 1} | rejected (no cost): FileNotFoundError: [Errno 2] No such file or directory: '/le |
| 032 | logits | edit_v7.npz | {"e_over_S_all_groups": 0.006} | rejected (no cost): FileNotFoundError: [Errno 2] No such file or directory: '/le |
| 033 | logits | orig | {"n": 2048} | rejected (no cost): FileNotFoundError: [Errno 2] No such file or directory: '/le |
| 034 | logits | orig | {"n": 2048} | data returned (analysed in work/) |
| 035 | reference | exp4.npz | {"ch1e-4_dropped": 0, "ph0.81_dropped": 0, "ph0.79_dropped": 1} | data returned (analysed in work/) |
| 036 | logits | edit_v7.npz | {"e_over_S_all_groups": 0.006} | data returned (analysed in work/) |
| 037 | logits | orig | {"n": 2048} | data returned (analysed in work/) |
| 038 | logits | orig | {"ref_ph079_equals_orig_base": 1.0} | data returned (analysed in work/) |
| 039 | logits | edit_v7_T30.npz | {"e_ph079": 0.008, "e_nh1e-3": 0.0063} | data returned (analysed in work/) |
## 6. Ideas that failed
- **v1: LDA read-out and a canonical stamp, no private wire** (pass 010). Flagged tokens were overwritten with a saturated sign pattern and detected by an LDA direction in each block. It reached E_ref 0.24, but clean jets moved by 0.034 S, one class dropped by 0.046 (gate fail) and gen_drift was 4.4. The read-out directions are only "null" on the clean-jet distribution: noised generator inputs and the live time token break them. Its cancel neurons also used un-ablated private channels.
- **Mean-trajectory wire on a massive channel (119)** (pass 012): the clean error rose to 0.16. A constant cannot replace a channel whose readers are strong.
- **Early-only hiding** (wire and penalties only in blocks 0-4, flag removed afterwards) (pass 014): E_ref 1.05. Almost all of the soft particle's influence acts in late blocks and the classifier.
- **Other wire channels** 263 / 452 / 76 (passes 019-021): E_ref 0.061 / 0.049 / 0.107, all worse than 49.
- **Larger penalty** T = 16 / 30 (passes 016, 018, 039): no gain. Hiding is already saturated; what remains is the ablation floor and the kNN path.
- **"Squeeze"** (passes 026, 027, 029): scale channel 49's writers by 1e-4 and its DyT readers up by ~1e4 (least-squares linearisation), so normal tokens keep a linearised copy and flagged tokens saturate 1e4× higher. It lowered the floor (E_ref 0.0301 at 34.5k DoF), but gen_drift went to 79 (0.29 after keeping body.norm). The no-flag diagnostic (029) showed the linearisation alone costs only 0.002 gen_drift; the drift comes from noised neutral particles that get flagged in the generator check, whose own query/MLP inputs become ~1e4. This conflict is inherent: a clean penalty needs flag ≫ normal on the same channel that carries the normal information. I rejected it because of the real generator-AUC limit.
- **Partial re-routing of only MLP fc1 (blocks 5-11)** (pass 023): E_ref 0.0391 at 36.5k, dominated by the mid edit.
- **Sharper steps** (pass 017) did help a little: the old β = 2e4 had a 1e-3 log-band that ~3 % of jets fell into, leaving a residue on the wire.

## 7. Passes used
35 of 50. Reference passes: 002, 004, 030, 035. Captures: 005-009. Logits: 003, 034, 036-039. Surgical: 001, 010-027, 029. Rejected at no cost: 028, 031-033.
