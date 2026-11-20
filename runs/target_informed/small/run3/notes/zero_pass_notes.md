# Zero-pass notes

## Model as read from code/weights
PET2 "s": embed (DynamicTanh(4 kin) -> MLP) + local kNN block (K=15, 2 AttBlocks) + pid_embed + add_embed;
interaction MLP(ln m2, ln dR, ln kT) -> 8-head attention bias; 8 body AttBlocks over
[4 summary][time][150 particles]; body.norm (DynamicTanh); classifier: 2 TokenAttBlocks (summary tokens attend
to all other tokens incl. time token), fc MLP on concat(4x128), out 210. In the classifier pass time=0, in the
generator pass t>0: the time token is the only place the two uses of the body differ.

## Probes seen in the data
soft1e-3/soft1e-4/soft3e-3: one photon appended in a padding slot (pT ~0.6 / 0.06 / 1.8 GeV; data min ~0.14 GeV,
1% of real constituents have pT-fraction < 1.2e-3). split2: hardest neutral split into two collinear halves.
P: all constituent log_pt lowered by a z-dependent amount (fragmentation softened).

## Plan to find the reference
Hypotheses: (H1) reference is IR-safe: ignores soft particles (f_ref(soft) = f_orig(clean)), perhaps below an absolute
or relative pT threshold; (H2) reference = original plus a low-rank logit shift driven by a jet-level fragmentation
observable (e.g. sum z log z, multiplicity of soft particles); (H3) something else.
Measurements: reference + original logits on clean, soft1e-3, soft1e-4, P (practice rows), then constructed jets
(photon at varying pT fraction, varying pid, charged/neutral) to map the threshold/shape of the difference.
Compare delta = f_ref - f_orig against delta_clean = f_orig(clean) - f_orig(probe) and against jet observables via
least squares.

## Plan to implement
If H1: suppress soft-particle contributions in attention: write a key component proportional to a softness
feature of the particle (from log_pt via embed) and a constant query component in the summary-token/classifier
attention so soft keys get large negative logits; only the classifier head (classifier.*) if possible, since body
edits leak into the generator. If H2: compute the observable as an attention-pooled feature into a spare direction
of the classifier summary tokens and map it to the logit shift through classifier.out (low rank).

## Prediction
Zero-pass edit is empty: E_ref = 1.0, all side effects 0.
