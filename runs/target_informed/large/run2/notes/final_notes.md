# Final notes

## 1. Final edits

| file | edit | DoF | E_ref (practice) | e_clean | d_acc | worst class drop | d_nll | rho D / J / P_ref | gen_drift (max) |
|---|---|---|---|---|---|---|---|---|---|
| `final_small.npz` | v2_548: exact neutral mask + local channel-2 role moved to local ch 548 | **53,277** | **0.3045** | 0.0372 | -0.0024 | 0.0146 | +0.0102 | 0.994 / 1.002 / 1.006 | 0.278 |
| `final_mid.npz` | v3p_770_k14: as large, residual role swap 2<->770 only in body blocks 0-13 | **234,576** | **0.0940** | 0.0085 | +0.0015 | 0.0049 | -0.0004 | 1.006 / 1.005 / 0.990 | 0.232 |
| `final_large.npz` | v3p_770_noclf: exact neutral mask + residual role swap 2<->770 in all 28 body blocks (+ E>=pT unit) | **406,638** | **0.0645** | 0.0033 | -0.0005 | 0.0049 | +0.0003 | 1.001 / 0.999 / 0.993 | 0.232 |

All three pass every practice limit (`gate_ref_no_g2 = true`). They were also checked on held-out practice jets
205-408 of every class (disjoint from the surgical subset). The worst class drop there is 0.0098 / 0.0098 / 0.0000,
d_acc is +0.0044 / -0.0005 / +0.0015, and d_nll is +0.011 / +0.001 / +0.000.

Per-probe e_v on practice (edited, with the original in brackets): soft1e-3 [0.373], soft1e-4 [0.1055], P [0.151]

| edit | soft1e-3 | soft1e-4 | P | clean | split2 | soft3e-3 | R_vis | R_in | R_sub | dir_err D/J/P |
|---|---|---|---|---|---|---|---|---|---|---|
| small | 0.0381 | 0.0372 | 0.0693 | 0.0372 | 0.0362 | 0.0414 | 0.584 | 1.144 | 0.023 | 0.189/0.203/0.367 |
| mid | 0.0091 | 0.0087 | 0.0265 | 0.0085 | 0.0083 | 0.0085 | 0.510 | 1.000 | 0.020 | 0.035/0.046/0.333 |
| large | 0.0042 | 0.0038 | 0.0220 | 0.0033 | 0.0031 | 0.0033 | 0.509 | 0.999 | 0.020 | 0.016/0.016/0.328 |

Tensors changed, ranks, changed entries and DoF, grouped by tensor family. `in_blocks.*` aggregates over blocks,
and the rank is per tensor. All tensors are `body.*`: no `classifier.*` or `generator.*` tensor is changed.

## final_small (v2_548) DoF 53277 n_changed 53277
| tensor (family) | shape | #tensors | rank | changed entries | DoF |
|---|---|---|---|---|---|
| `body.embed.mlp.fc1.weight` | 2048x4 | 1 | 2 | 8 | 8 |
| `body.embed.mlp.fc1.bias` | 2048 | 1 | None | 2 | 2 |
| `body.embed.mlp.fc2.weight` | 1024x2048 | 1 | 3 | 4094 | 4094 |
| `body.embed.mlp.fc2.bias` | 1024 | 1 | None | 1 | 1 |
| `body.local_physics.mlp.fc2.weight` | 1024x2048 | 1 | 2 | 4096 | 4096 |
| `body.local_physics.mlp.fc2.bias` | 1024 | 1 | None | 2 | 2 |
| `body.local_physics.in_blocks.*.norm1.weight` | 1024 | 4 | None | 4 | 4 |
| `body.local_physics.in_blocks.*.norm2.weight` | 1024 | 4 | None | 4 | 4 |
| `body.local_physics.in_blocks.*.attn.in_proj_weight` | 3072x1024 | 4 | 1 | 12288 | 12288 |
| `body.local_physics.in_blocks.*.attn.out_proj.weight` | 1024x1024 | 4 | 2 | 8192 | 8192 |
| `body.local_physics.in_blocks.*.mlp.fc1.weight` | 2048x1024 | 4 | 1 | 8192 | 8192 |
| `body.local_physics.in_blocks.*.mlp.fc2.weight` | 1024x2048 | 4 | 2 | 16384 | 16384 |
| `body.local_physics.in_blocks.*.mlp.fc2.bias` | 1024 | 4 | None | 8 | 8 |
| `body.pid_embed.0.weight` | 9x1024 | 1 | 1 | 2 | 2 |

## final_mid (v3p_770_k14) DoF 234576 n_changed 294026
| tensor (family) | shape | #tensors | rank | changed entries | DoF |
|---|---|---|---|---|---|
| `body.token` | 1x4x1024 | 1 | None | 8 | 8 |
| `body.embed.mlp.fc1.weight` | 2048x4 | 1 | 2 | 12 | 12 |
| `body.embed.mlp.fc1.bias` | 2048 | 1 | None | 3 | 3 |
| `body.embed.mlp.fc2.weight` | 1024x2048 | 1 | 5 | 7162 | 7162 |
| `body.embed.mlp.fc2.bias` | 1024 | 1 | None | 2 | 2 |
| `body.local_physics.mlp.fc2.weight` | 1024x2048 | 1 | 2 | 4096 | 4096 |
| `body.local_physics.mlp.fc2.bias` | 1024 | 1 | None | 2 | 2 |
| `body.local_physics.in_blocks.*.norm1.weight` | 1024 | 4 | None | 4 | 4 |
| `body.local_physics.in_blocks.*.norm2.weight` | 1024 | 4 | None | 4 | 4 |
| `body.local_physics.in_blocks.*.attn.in_proj_weight` | 3072x1024 | 4 | 1 | 12288 | 12288 |
| `body.local_physics.in_blocks.*.attn.out_proj.weight` | 1024x1024 | 4 | 2 | 8192 | 8192 |
| `body.local_physics.in_blocks.*.mlp.fc1.weight` | 2048x1024 | 4 | 1 | 8192 | 8192 |
| `body.local_physics.in_blocks.*.mlp.fc2.weight` | 1024x2048 | 4 | 2 | 16384 | 16384 |
| `body.local_physics.in_blocks.*.mlp.fc2.bias` | 1024 | 4 | None | 8 | 8 |
| `body.time_embed.fc2.weight` | 1024x2048 | 1 | 1 | 4096 | 3071 |
| `body.add_embed.0.fc2.weight` | 1024x2048 | 1 | 1 | 4096 | 3071 |
| `body.pid_embed.0.weight` | 9x1024 | 1 | 2 | 17 | 17 |
| `body.in_blocks.*.norm1.weight` | 1024 | 14 | None | 28 | 28 |
| `body.in_blocks.*.norm2.weight` | 1024 | 14 | None | 28 | 28 |
| `body.in_blocks.*.attn.in_proj_weight` | 3072x1024 | 14 | 1 | 86016 | 57330 |
| `body.in_blocks.*.attn.out_proj.weight` | 1024x1024 | 14 | 1 | 28672 | 28658 |
| `body.in_blocks.*.mlp.fc1.weight` | 2048x1024 | 14 | 1 | 57344 | 42994 |
| `body.in_blocks.*.mlp.fc2.weight` | 1024x2048 | 14 | 1 | 57344 | 42994 |
| `body.in_blocks.*.mlp.fc2.bias` | 1024 | 14 | None | 28 | 28 |

## final_large (v3p_770_noclf) DoF 406638 n_changed 523488
| tensor (family) | shape | #tensors | rank | changed entries | DoF |
|---|---|---|---|---|---|
| `body.token` | 1x4x1024 | 1 | None | 8 | 8 |
| `body.embed.mlp.fc1.weight` | 2048x4 | 1 | 2 | 12 | 12 |
| `body.embed.mlp.fc1.bias` | 2048 | 1 | None | 3 | 3 |
| `body.embed.mlp.fc2.weight` | 1024x2048 | 1 | 5 | 7162 | 7162 |
| `body.embed.mlp.fc2.bias` | 1024 | 1 | None | 2 | 2 |
| `body.local_physics.mlp.fc2.weight` | 1024x2048 | 1 | 2 | 4096 | 4096 |
| `body.local_physics.mlp.fc2.bias` | 1024 | 1 | None | 2 | 2 |
| `body.local_physics.in_blocks.*.norm1.weight` | 1024 | 4 | None | 4 | 4 |
| `body.local_physics.in_blocks.*.norm2.weight` | 1024 | 4 | None | 4 | 4 |
| `body.local_physics.in_blocks.*.attn.in_proj_weight` | 3072x1024 | 4 | 1 | 12288 | 12288 |
| `body.local_physics.in_blocks.*.attn.out_proj.weight` | 1024x1024 | 4 | 2 | 8192 | 8192 |
| `body.local_physics.in_blocks.*.mlp.fc1.weight` | 2048x1024 | 4 | 1 | 8192 | 8192 |
| `body.local_physics.in_blocks.*.mlp.fc2.weight` | 1024x2048 | 4 | 2 | 16384 | 16384 |
| `body.local_physics.in_blocks.*.mlp.fc2.bias` | 1024 | 4 | None | 8 | 8 |
| `body.time_embed.fc2.weight` | 1024x2048 | 1 | 1 | 4096 | 3071 |
| `body.add_embed.0.fc2.weight` | 1024x2048 | 1 | 1 | 4096 | 3071 |
| `body.pid_embed.0.weight` | 9x1024 | 1 | 2 | 17 | 17 |
| `body.in_blocks.*.norm1.weight` | 1024 | 28 | None | 56 | 56 |
| `body.in_blocks.*.norm2.weight` | 1024 | 28 | None | 56 | 56 |
| `body.in_blocks.*.attn.in_proj_weight` | 3072x1024 | 28 | 1 | 172032 | 114660 |
| `body.in_blocks.*.attn.out_proj.weight` | 1024x1024 | 28 | 1 | 57344 | 57316 |
| `body.in_blocks.*.mlp.fc1.weight` | 2048x1024 | 28 | 1 | 114688 | 85988 |
| `body.in_blocks.*.mlp.fc2.weight` | 1024x2048 | 28 | 1 | 114688 | 85988 |
| `body.in_blocks.*.mlp.fc2.bias` | 1024 | 28 | None | 56 | 56 |
| `body.norm.weight` | 1024 | 1 | None | 2 | 2 |


## 2. What the reference does, and how I found it

**Rule:** f_ref(x) = f_orig(x with every sub-threshold neutral constituent removed). The thresholds are:
- a photon (pid 6) is removed iff pT < 0.8 GeV **and** E < 1.8 GeV;
- a neutral hadron (pid 7) is removed iff E < 2.2 GeV;
- charged particles are never removed.

These are detector-acceptance cuts. Natural (clean) JetClass jets never contain such particles: the clean minima are
photon pT >= 0.825 GeV or E >= 1.85 GeV, and neutral hadron E >= 2.256 GeV. So the reference equals the original
model exactly on clean jets.

How I found it:
- **Passes 1-2:** a packed diagnostic set (clean, the probes, removal variants, a ladder of added photons, an added
  neutral hadron). ref(clean) == orig(clean) to 0, and so do all "remove natural particles" variants. On jets with
  an added soft particle, ref is per jet *either* exactly orig(x) *or* exactly orig(x without the added particle).
  Absolute or relative pT cuts on all particles were ruled out, because ref(clean) is unchanged.
- **Pass 3:** one random added particle per jet over (pid, pT, eta). The removal boundary is: photons pT < 0.8 & E < 1.8
  (0 errors / 896), neutral hadrons E < 2.2 (0 / 640), charged never (0 / 512).
- **Pass 22:** thresholds pinned exactly: photon E cut between 1.8000 and 1.8001, neutral hadron E cut 2.2.
- **Passes 17 and 19:** on the multi-particle probe P, ref(P) == orig(P minus all sub-threshold neutrals) to 8.6e-6.
- **Pass 24:** on 15 unseen probe types (soft nh, other photon fractions, soft charged, energy scale-down, other
  fragmentation exponents, double insertions), the reference behaves as the rule predicts.

Why the original model differs: it reacts very strongly to out-of-acceptance soft neutrals. A 0.3 GeV photon moves
its logits by 0.39 S, while a 1.5 GeV photon moves them by 0.025 S and removing natural sub-GeV particles moves them
by 0.02 S. This is presumably a dataset-provenance shortcut learned from the other pre-training datasets.

## 3. The circuit (what the edit computes)

**Key fact used.** `PET_body` builds its padding mask once, as `mask = (x[:, :, 2] != 0)`. Here x is the concatenated
token input `embed + local + pid_embed + add_embed`. This mask sets -1e9 on all attention keys of masked tokens,
zeroes their outputs, and makes `body.norm(x) * mask` zero. The classifier head (and the generator) then drop the
token again through `x_body[..., 2] != 0`. So if channel 2 of a particle's input embedding is **exactly 0.0**, the
whole network treats that particle exactly like padding: it is removed exactly. The only remaining path is its role
as a kNN neighbour in `local_physics`, which uses the raw-input mask.

**(a) Exact binary detectors in the input MLP** (`body.embed.mlp`, 2 repurposed hidden units u_a = 1636, u_b = 440).
The embed MLP reads t = w0 * tanh(a0 * [deta, dphi, log pT, log E]). With t* = w0 * tanh(a0 * log thr):
- unit a: pre = 2000 * (t_pT - t*_0.8) - 6
- unit b: pre = 2000 * (t_E - t*_1.8) - 6

Below the threshold, pre < -6, so GELU(pre) is exactly 0 in float32 (erf saturates), and the unit's output
w * tanh(alpha * GELU) is exactly 0.0. Above the threshold, GELU is large, tanh saturates, and the output is exactly w
(or at least clearly nonzero). The fc2 columns of these units are cleared, and they write only into channel 2.

**(b) Channel 2 rebuilt so that it is zero exactly on the gated set.** The row is
embed_2 = c_a * a + c_b * b + B with B = 1.0 (exactly representable in fp16 and bf16 as well). The pid table gets
`pid_embed[6, 2] = pid_embed[7, 2] = -B`. For a photon or neutral hadron with pT < 0.8 and E < 1.8 the sum is
(0 + 0 + B) + 0 + (-B) + 0 = 0.0 exactly. The local term is exactly 0 (see c), and add_embed of zero impact
parameters is exactly 0.

Natural tokens never hit 0. Neutrals outside the region have a or b on. Charged particles get add_embed and pid
values that are nonzero. Natural photons and neutral hadrons are never inside the region (acceptance), so clean
jets are never masked.

**(c) Making the local-physics contribution to channel 2 exactly 0 without breaking local physics.** Channel 2 of the
local stream is an important internal channel: removing it costs e_clean 0.15. Its role is therefore *moved* to
local channel c, which is exact because DyT has a scalar alpha:
- writers: row c := row 2 and row 2 := 0, in local `mlp.fc2` and in each local block's `out_proj` and `mlp.fc2`
  (weights and biases);
- readers: column c := column 2 in `in_proj` and `fc1`, and `norm.weight[c] := norm.weight[2]`.

Channel c's own local role is dropped. For c = 770 this costs nothing (measured e_clean 0.0001).

**(d) (mid/large only) Residual role swap 2 <-> 770.** After (c), the local channel-2 signal (std about 4.7) exits into
residual channel c instead of channel 2. This is what limits the small edit (e_clean 0.037). The large edit permutes
channels 2 and 770 in every body writer and reader:
- `body.token`, embed fc2 row 770 := old row 2, `pid_embed` column 770 := old column 2;
- add_embed and time_embed fc2 rows swapped;
- in every body block: norm1/norm2 weights, `in_proj` and `fc1` columns, `out_proj` and `fc2` rows and the fc2 bias.

So the original channel-2 computation now runs exactly in channel 770. Channel 2, the mask channel, carries the old
channel-770 role, and only its concat-level content is replaced by the gate design. The design values are matched
per pid to the old mean (target T = mean embed_770 + pid_embed[:, 770]).

`body.norm` and the head are left unswapped, so the head reads the two channels crossed. This was measured to be
harmless (E_ref 0.0645 without the head swap vs 0.066 with it). The mid edit swaps only blocks 0-13. Channel 2's role
matters in blocks 7-13: a swap in blocks 0-6 only performs like no swap.

**(e) Physicality unit (mid/large).** Unit 1673 is an exact step that is on when log E < log pT - ~0.008, which is
unphysical. It writes 0.5 into channel 2, so noised generator tokens that are unphysical are not masked. This cuts
gen_drift from 0.27 to 0.23 and does not affect physical inputs.

**Why inputs where ref = orig are (almost) unaffected.** Natural jets never contain gated particles, so nothing is
masked. Steps (c) and (d) are exact re-parametrisations except for:
- the two or three repurposed embed hidden units;
- the replaced concat-level content of one channel (old channel 770, or channel 2 in the small edit);
- the dropped local role of channel 770.

The measured residual damage on clean jets is e_clean = 0.0033 (large), 0.0085 (mid) and 0.037 (small), with KL ≈ 0.

**Known limits (where the edit differs from the reference):**
1. A neutral hadron with E < 2.2 but not (pT < 0.8 and E < 1.8) is not removed. This is about 0.13 per jet in P and
   gives 84% of the remaining P error. It is impossible to fix with the exact mask: pid enters the embedding only
   additively, so the photon and neutral-hadron zero sets would be level sets of the same pid-blind function. Because
   the photon set (pT < 0.8 & E < 1.8) lies inside the neutral-hadron set (E < 2.2), both species are forced to the same
   zero set. I chose the photon region for both species.
2. A gated particle is still a kNN neighbour of others (`local_physics`). The leak is small, except when the particle
   sits exactly on a hard particle (ratio 0.16 in pass 24).
3. In the generator path, gated-region neutrals in noised jets are also masked (gen_drift 0.23).

Generalisation (pass 24, large edit, ratio e_edit / e_orig):

| probe | ratio |
|---|---|
| photon 5e-4, 2e-3, wide angle, two photons | 0.013-0.016 |
| soft nh 1e-4 / 1e-3 | 0.054 / 0.077 |
| photon + nh | 0.10 |
| fragmentation exponent 1.2 / 1.05 | 0.12 / 0.14 |
| energy scale x0.9 / x0.8 | 0.17 / 0.18 |
| photon on top of the hardest particle | 0.16 |
| nh 2e-3 (band) | 1.0 |
| soft charged, photon 2e-3 (ref = orig) | e = e_clean (0.0035) |

## 4. Ideas that failed
- Looking for saturated ("massive") residual channels or channels with near-zero DyT weights to use as a private
  wire: none exist.
- Zeroing the local-stream channel-2 writes (v1): exact removal, but e_clean 0.15 and d_acc -0.025. The diagnostics
  in passes 7-9 showed that the local blocks' *internal reads* of channel 2 matter.
- Moving the local channel-2 role into an anti-aligned channel (347): that channel is itself important (e_clean 0.16).
  Other host channels for v2: c = 59 (0.065), 548 (0.037), 770 (0.032, but worst class drop 0.024 > 0.02).
- Making the dumped signal invisible by deleting channel-770 reads (v2d): worse (0.044). Deleting a channel's body
  role costs about 0.017, which also rules out a "pure wire" channel for approximate key suppression of the
  band neutral hadrons.
- Swap partner c = 27 (smallest non-embedding content): worse than 770 (0.0069), because its local role matters.
- Swapping only blocks 0-6: no better than no swap (E_ref 0.288).
- Enlarging the shared mask region to catch band neutral hadrons: those are mostly at pT >= 0.8, where natural photons
  are dense (3.5 per jet), so it is not possible without removing natural photons.

## 5. Passes used
35 of 50 (see the table below; ledger in `run/ledger.json`). Scripts are in `run/work/` (`mk_v1.py`, `mk_v2.py`,
`mk_v3.py`, `mk_v3p.py`, `mk_v3k.py`, `mk_v3nc.py`). No gradients, optimisers or self-run forward passes were used;
all constructions are closed-form from the weights, plus statistics of captured activations.

## 6. Predictions vs outcomes
| # | kind | edit / input | prediction | outcome |
|---|---|---|---|---|
| 001 | reference | orig set1.npz | frac_conditions_ref_eq_orig=0.3, ref_soft1e-3_vs_orig_clean_rel_dist=0.2 | soft probes: ref == orig(clean) (removal of the added particle); ref == orig exactly on clean and on all removal variants; 7 of 16 conditions differ |
| 002 | logits | orig set1.npz | orig_soft1e-3_shift_over_S=0.3 | orig soft1e-3 shift 0.386 S, soft1e-4 0.109 S, 0.3 GeV photon 0.39 S, 1.5 GeV photon 0.025 S |
| 003 | reference | orig set2.npz | frac_nh_removed_below_E2.25=0.95, frac_charged_removed=0.1, frac_photon_removed_logpt_below_-0.3_central=0.9 | charged never removed (0/512); nh removed iff E<~2.2 (0 errors); photons removed iff pT<0.8 and E<~1.8 (0 errors) |
| 004 | capture | orig set1.npz | n_channels_saturated_all_tokens_block13=5 | 0 saturated channels (prediction 5 wrong); time token is exactly 0 at t=0 |
| 005 | surgical | v1.npz | E_ref=0.25, e_soft1e-3_ratio=0.1, e_soft1e-4_ratio=0.1, d_acc=-0.003, d_nll=0.01, rho_D=1.0, rho_J=1.0, rho_P_ref=1.0, gen_drift=0.1, DoF=18448 | E_ref=0.907, e_clean=0.1530, d_acc=-0.0249, d_nll=+0.0680, class_drop=0.0683, rho_P_ref=0.818, gen_drift=0.272, DoF=18448 |
| 006 | logits | v1.npz set1.npz | err_soft1e-4_over_S=0.03, err_clean_over_S=0.03 | v1: soft probes edit-vs-edit(clean) 0.003-0.007 S (removal works) but clean damage 0.144 S |
| 007 | surgical | localonly.npz | d_acc=-0.01, d_nll=0.03, E_ref=0.95 | E_ref=1.371, e_clean=0.1520, d_acc=-0.0234, d_nll=+0.0674, class_drop=0.0780, rho_P_ref=0.823, gen_drift=0.020, DoF=14341 |
| 008 | surgical | v1_nolocal.npz | d_acc=-0.015, d_nll=0.04, E_ref=0.9 | E_ref=1.007, e_clean=0.0092, d_acc=-0.0010, d_nll=+0.0029, class_drop=0.0098, rho_P_ref=0.997, gen_drift=0.001, DoF=4107 |
| 009 | surgical | localread.npz | d_acc=-0.005, d_nll=0.01, E_ref=1.02 | E_ref=1.407, e_clean=0.1686, d_acc=-0.0224, d_nll=+0.0797, class_drop=0.0732, rho_P_ref=0.791, gen_drift=0.006, DoF=8 |
| 010 | surgical | v2_59.npz | E_ref=0.35, d_acc=-0.004, d_nll=0.012, e_clean=0.04, rho_P_ref=0.97, gen_drift=0.2 | E_ref=0.466, e_clean=0.0645, d_acc=-0.0024, d_nll=+0.0168, class_drop=0.0146, rho_P_ref=0.976, gen_drift=0.275, DoF=53277 |
| 011 | surgical | v2_347n.npz | E_ref=0.35, d_acc=-0.002, d_nll=0.01, e_clean=0.04, rho_P_ref=0.98 | E_ref=0.934, e_clean=0.1583, d_acc=-0.0220, d_nll=+0.0720, class_drop=0.0829, rho_P_ref=0.776, gen_drift=0.272, DoF=53277 |
| 012 | surgical | v2_548.npz | E_ref=0.45, d_acc=-0.003, d_nll=0.015, e_clean=0.06 | E_ref=0.305, e_clean=0.0372, d_acc=-0.0024, d_nll=+0.0102, class_drop=0.0146, rho_P_ref=1.006, gen_drift=0.278, DoF=53277 |
| 013 | surgical | v2_770.npz | E_ref=0.45, d_acc=-0.003, d_nll=0.015, e_clean=0.06 | E_ref=0.266, e_clean=0.0324, d_acc=-0.0020, d_nll=+0.0095, class_drop=0.0244, rho_P_ref=0.999, gen_drift=0.274, DoF=53277 |
| 014 | surgical | diag_move770.npz | e_clean=0.025, d_acc=-0.002 | E_ref=1.079, e_clean=0.0303, d_acc=-0.0010, d_nll=+0.0080, class_drop=0.0195, rho_P_ref=1.004, gen_drift=0.068, DoF=49170 |
| 015 | surgical | diag_read770.npz | e_clean=0.01, d_acc=-0.001 | E_ref=1.000, e_clean=0.0001, d_acc=+0.0000, d_nll=-0.0000, class_drop=0.0000, rho_P_ref=1.000, gen_drift=0.002, DoF=8 |
| 016 | surgical | v3_770.npz | E_ref=0.15, e_clean=0.01, d_acc=-0.001, d_nll=0.004, rho_P_ref=1.0, gen_drift=0.35 | E_ref=0.065, e_clean=0.0035, d_acc=-0.0005, d_nll=+0.0000, class_drop=0.0049, rho_P_ref=0.993, gen_drift=0.272, DoF=479323 |
| 017 | reference | orig P | frac_jets_ref_ne_orig=0.75 | 205 reference outputs on P |
| 018 | logits | v3_770.npz set3.npz | err_P_over_S=0.03, err_P_minus_band_over_S=0.008 | v3 on P: 0.0247 S overall, 0.133 on the 32 jets with band nh, 0.0048 after removing band nh |
| 019 | logits | orig set3.npz | ref_P_vs_orig_P_removed_max_abs=0.0 | ref(P) == orig(P minus all sub-threshold neutrals) to 8.6e-6 (prediction 0.0 confirmed) |
| 020 | surgical | del_770.npz | e_clean=0.02, d_acc=-0.002 | E_ref=1.030, e_clean=0.0169, d_acc=+0.0020, d_nll=+0.0021, class_drop=0.0098, rho_P_ref=0.988, gen_drift=0.013, DoF=38970 |
| 021 | surgical | v3p_770.npz | E_ref=0.066, e_clean=0.0037, gen_drift_t0.85=0.15, gen_drift_t0.2=0.045 | E_ref=0.066, e_clean=0.0036, d_acc=-0.0005, d_nll=+0.0000, class_drop=0.0049, rho_P_ref=0.993, gen_drift=0.232, DoF=480350 |
| 022 | reference | orig set4.npz | photon_E_threshold=1.8, nh_E_threshold=2.2 | photon E threshold exactly 1.8 (removed max 1.8000, kept min 1.8001); nh 2.2 exactly |
| 023 | surgical | v2d_770.npz | E_ref=0.2, e_clean=0.022, d_acc=-0.002, gen_drift=0.28 | E_ref=0.340, e_clean=0.0441, d_acc=-0.0020, d_nll=+0.0126, class_drop=0.0146, rho_P_ref=0.997, gen_drift=0.272, DoF=53334 |
| 024 | reference | orig set5.npz | ref_eq_orig_ch1e-3=1.0 | ch1e-3: ref == orig (charged never removed), as predicted |
| 025 | logits | v3p_770.npz set5.npz | mean_ratio_soft_photon_probes=0.05 | v3p ratio on new soft-photon probes 0.013-0.016 (pred 0.05); soft nh 1e-3/1e-4 0.08/0.05; nh2e-3 1.0; Jdown0.8 0.18 |
| 026 | logits | v2_548.npz set5.npz | mean_ratio_soft_photon_probes=0.05 | v2_548 soft-photon ratios 0.10-0.15 |
| 027 | logits | orig set5.npz | mean_ratio_soft_photon_probes=0.05 | orig baseline for set5 |
| 028 | surgical | v3p_27.npz | E_ref=0.06, e_clean=0.0028, d_acc=0.0 | E_ref=0.093, e_clean=0.0069, d_acc=-0.0005, d_nll=-0.0000, class_drop=0.0049, rho_P_ref=0.985, gen_drift=0.232, DoF=480350 |
| 029 | surgical | v3p_770_noclf.npz | E_ref=0.15, e_clean=0.012, d_acc=-0.002 | E_ref=0.065, e_clean=0.0033, d_acc=-0.0005, d_nll=+0.0003, class_drop=0.0049, rho_P_ref=0.993, gen_drift=0.232, DoF=406638 |
| 030 | surgical | v3p_770_k14.npz | E_ref=0.09, e_clean=0.006 | E_ref=0.094, e_clean=0.0085, d_acc=+0.0015, d_nll=-0.0004, class_drop=0.0049, rho_P_ref=0.990, gen_drift=0.232, DoF=234576 |
| 031 | surgical | v3p_770_k7.npz | E_ref=0.09, e_clean=0.006 | E_ref=0.288, e_clean=0.0357, d_acc=-0.0024, d_nll=+0.0125, class_drop=0.0146, rho_P_ref=1.007, gen_drift=0.232, DoF=148546 |
| 032 | logits | orig clean | d_acc=-0.002, worst_class_drop=0.012 | orig baseline, held-out jets 205-408/class |
| 033 | logits | v2_548.npz clean | d_acc=-0.002, worst_class_drop=0.012 | v2_548 held-out: d_acc +0.0044, worst class drop 0.0098, d_nll +0.0106 |
| 034 | logits | v3p_770_k14.npz clean | d_acc=-0.002, worst_class_drop=0.012 | k14 held-out: d_acc -0.0005, worst class drop 0.0098, d_nll +0.0012 |
| 035 | logits | v3p_770_noclf.npz clean | d_acc=-0.002, worst_class_drop=0.012 | noclf held-out: d_acc +0.0015, worst class drop 0.0000, d_nll +0.0003 |
