#!/usr/bin/env python
"""Layer-by-layer linear probes: where does "this jet is JetClass-II" form, and where does one soft neutral flip it?
(Sec. 2, App. A Fig. probes.)

Probe: logistic regression on the 4 summary tokens (concatenated, 4 x 128) after each backbone block, after the
final norm and after each classifier block, trained to separate clean JetClass jets (W and QCD, rows 60000+) from
clean JetClass-II jets (one validation file). Applied to held-out sets:
- clean JetClass / clean JetClass-II (probe accuracy);
- JetClass + one photon at 0.3 GeV, + one neutral hadron at 0.5 GeV (below JetClass's thresholds, normal in
  JetClass-II), + one photon at 3 GeV (in range, control);
- JetClass + one photon / neutral hadron at a fraction f = 1e-3 of the jet pT;
- JetClass-II with its sub-threshold neutrals removed (photons with pT < 0.82 GeV and E < 1.85 GeV, neutral
  hadrons with E < 2.26 GeV: below every JetClass particle of that type): the reverse test.
Reported per layer: mean P(JetClass-II) of each set, and in "stats": the standard deviation over jets, the 16/84%
quantiles, the standard error of the mean, and the spread of the mean over --boot probes refitted on bootstrap
resamples of the training jets. At the output: the 200-way dataset share of each set. Also the share of held-out
JetClass-II jets with at least one sub-threshold neutral (key jc2_jets_with_subthreshold_neutrals).

  python scripts/analysis/probes.py --jetclass2 JETCLASS2_DIR/JetClass2_9.h5 [--n 2000] [--threads 2]
      -> RESULTS_DIR/figures/small/probes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def strip_subthreshold(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove neutrals below every JetClass particle of their type; compact the rows. Returns (jets, n_removed)."""
    out, nrem = np.zeros_like(raw), np.zeros(len(raw), int)
    pt, E = np.exp(raw[..., 2].astype(np.float64)), np.exp(raw[..., 3].astype(np.float64))
    code = np.rint(raw[..., 4]).astype(int)
    real = raw[..., 2] != 0
    drop = real & (((code == 6) & (pt < 0.82) & (E < 1.85)) | ((code == 7) & (E < 2.26)))
    for j in range(len(raw)):
        keep = np.flatnonzero(real[j] & ~drop[j])
        out[j, :len(keep)] = raw[j, keep]
        nrem[j] = int(drop[j].sum())
    return out, nrem


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import h5py
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jetclass2", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000, help="training jets per dataset (held-out sets: n/2)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--boot", type=int, default=20, help="bootstrap refits of each probe (probe uncertainty)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    C.set_size("s")
    from softrepair import core as W
    from softrepair import heads as H
    torch.set_num_threads(args.threads)
    ad = W.load_adapter("cpu", "s")
    n, m = args.n, args.n // 2
    jc = np.concatenate([H.load(H.WJ, 60000, (n + m) // 2), H.load(H.QCD, 60000, (n + m) // 2)])
    rng = np.random.default_rng(0)
    jc = jc[rng.permutation(len(jc))]
    with h5py.File(args.jetclass2, "r") as h:
        jetclass2 = h["data"][:n + m].astype(np.float32)
    jc_tr, jc_te, jetclass2_tr, jetclass2_te = jc[:n], jc[n:n + m], jetclass2[:n], jetclass2[n:n + m]
    sets = {"JetClass clean": jc_te, "JetClass-II clean": jetclass2_te}
    for k, (name, pid, pt) in enumerate((("JetClass + photon 0.3 GeV", 6, 0.3),
                                          ("JetClass + neutral hadron 0.5 GeV", 7, 0.5),
                                          ("JetClass + photon 3 GeV", 6, 3.0))):
        x, ok = C.add_fixed(jc_te, pt, pid, 9400 + k)
        sets[name] = x[ok]
    # one photon / neutral hadron at a fraction 1e-3 of the jet pT (core._soft)
    for k, (name, pname) in enumerate((("JetClass + photon f=1e-3", "soft1e-3"),
                                        ("JetClass + neutral hadron f=1e-3", "soft7_1e-3"))):
        x, ok = W.perturb(jc_te, pname, 9500 + k)
        sets[name] = x[ok]
    stripped, nrem = strip_subthreshold(jetclass2_te)
    sets["JetClass-II, sub-threshold neutrals removed"] = stripped[nrem > 0]
    res = {"n_train_per_dataset": n, "sets": {k: int(len(v)) for k, v in sets.items()},
           "jc2_jets_with_subthreshold_neutrals": float((nrem > 0).mean()),
           "jc2_mean_subthreshold_neutrals": float(nrem.mean()), "layers": {}, "output_dataset_share": {}}
    Xtr_jc, _ = C.features(ad, jc_tr)
    Xtr_jc2, _ = C.features(ad, jetclass2_tr)
    F = {}
    for k, v in sets.items():
        F[k], y = C.features(ad, v)
        res["output_dataset_share"][k] = C.shares(y)
    for layer in Xtr_jc:
        X = np.concatenate([Xtr_jc[layer], Xtr_jc2[layer]])
        lab = np.r_[np.zeros(len(Xtr_jc[layer])), np.ones(len(Xtr_jc2[layer]))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X, lab)
        P = {k: clf.predict_proba(F[k][layer])[:, 1] for k in sets}
        rec = {k: float(P[k].mean()) for k in sets}
        rb = np.random.default_rng(1)
        boot = {k: [] for k in sets}
        for _ in range(args.boot):
            i = rb.integers(0, len(X), len(X))
            cb = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[i], lab[i])
            for k in sets:
                boot[k].append(cb.predict_proba(F[k][layer])[:, 1].mean())
        rec["stats"] = {k: {"std": float(P[k].std()), "q16": float(np.quantile(P[k], 0.16)),
                            "q84": float(np.quantile(P[k], 0.84)), "sem": float(P[k].std() / np.sqrt(len(P[k]))),
                            "boot_std": float(np.std(boot[k])) if args.boot > 1 else None} for k in sets}
        rec["accuracy_clean"] = float(0.5 * ((clf.predict(F["JetClass clean"][layer]) == 0).mean()
                                             + (clf.predict(F["JetClass-II clean"][layer]) == 1).mean()))
        res["layers"][layer] = rec
        print(f"{layer:20s} acc {rec['accuracy_clean']:.3f} | " +
              " | ".join(f"{k.split(',')[0][:28]} {v:.2f}+-{rec['stats'][k]['std']:.2f}" for k, v in rec.items()
                          if k in sets), flush=True)
    out = args.out or C.out_dir("s") / "probes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["output_dataset_share"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
