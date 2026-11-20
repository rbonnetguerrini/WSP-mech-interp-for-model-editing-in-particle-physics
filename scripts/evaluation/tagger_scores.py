#!/usr/bin/env python
"""JetClass logits of every class with and without one added soft neutral, for the tagger score distributions of
paper Sec. 4 (Fig. taggers_score_distributions) and the one-vs-rest efficiency quoted with it.

Jets: 5000 per JetClass class from the test files (rows 60000..), clean and with one photon (pid 6) or neutral hadron
(pid 7) at f = 1e-3 (seed 9000 + 10 pid + label). Signal-vs-QCD taggers F_c = logit(c) - logit(QCD) with an 80% cut
on the clean c jets (efficiency, QCD acceptance, AUC). The one-vs-rest score of the figure,
F_c = ln[p(c) / (1 - p(c))] over the ten JetClass classes, is summarised by --summary.

  python scripts/evaluation/tagger_scores.py --size s [--edit EDIT.npz] --name NAME --out-dir DIR
      -> <out-dir>/<NAME>.npz (keys <clean|photon|nh>_<label>: 210 logits) and <NAME>.json
  python scripts/evaluation/tagger_scores.py --summary DIR/original_model.npz DIR/<repaired>.npz
      -> one-vs-rest efficiency at the cut that keeps 80% of the clean jets of the class, with the photon
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

NAMES = {2: "QCD", 3: "Hbb", 4: "Hcc", 5: "Hgg", 6: "H4q", 7: "Hqql", 8: "Zqq", 9: "Wqq", 10: "Tbqq", 11: "Tbl"}


def auc(s, b):
    r = np.argsort(np.argsort(np.concatenate([s, b]))) + 1
    return float((r[:len(s)].sum() - len(s) * (len(s) + 1) / 2) / (len(s) * len(b)))


def one_vs_rest(z, var: str) -> tuple[np.ndarray, np.ndarray]:
    """F_c of the jets of class c (own) and of the jets of the nine other classes (rest), pooled over c."""
    from scipy.special import logsumexp
    own, rest = [], []
    for lab in range(2, 12):
        lg = z[f"{var}_{lab}"][:, 2:12].astype(np.float64)
        for c in range(10):
            f = lg[:, c] - logsumexp(np.delete(lg, c, axis=1), axis=1)
            (own if c == lab - 2 else rest).append(f)
    return np.concatenate(own), np.concatenate(rest)


def summary(files: list[Path]) -> None:
    for f in files:
        z = np.load(f)
        d = {v: one_vs_rest(z, v) for v in ("clean", "photon")}
        cut = np.quantile(d["clean"][0], 0.2)
        print(f"{f.stem}: cut {cut:.2f}; efficiency of the jets of the class clean {np.mean(d['clean'][0] > cut):.3f}"
              f" -> with photon {np.mean(d['photon'][0] > cut):.3f}; other classes passing "
              f"{np.mean(d['clean'][1] > cut):.4f} -> {np.mean(d['photon'][1] > cut):.4f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--edit", type=Path, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--summary", type=Path, nargs="+", default=None, help="only summarise saved .npz files")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
        return 0
    if a.out_dir is None:
        ap.error("--out-dir is required")
    C.setup_device(a.size, {"s": None, "m": 128, "l": 32}[a.size])
    from softrepair import core as W
    torch.set_num_threads(8)
    ad = W.load_adapter(a.device, a.size)
    if a.edit:
        W.set_state(ad, W.load_candidate(a.edit, W.original_state(ad)))
    name = a.name or (a.edit.stem if a.edit else "original_model")

    Y, OK = {}, {}
    for lab in NAMES:
        raw = C.load(lab, 60000, a.n)
        Y["clean", lab] = W.logits(ad, raw)
        for nm, pid in (("photon", 6), ("nh", 7)):
            x, ok = C.soft(raw, 1e-3, pid, 9000 + 10 * pid + lab)
            y = np.full_like(Y["clean", lab], np.nan)
            y[ok] = W.logits(ad, x[ok])
            Y[nm, lab], OK[nm, lab] = y, ok
        print(NAMES[lab], "done", flush=True)
    res = {"name": name, "model_size": a.size, "edit": a.edit.name if a.edit else None, "n": a.n,
           "taggers": {}, "acc10": {}}
    for lab in NAMES:
        res["acc10"][NAMES[lab]] = {k: float(np.mean(np.argmax(Y[k, lab][OK.get((k, lab), slice(None))][:, 2:12], 1)
                                                     + 2 == lab)) for k in ("clean", "photon", "nh")}
    for sig in range(3, 12):
        F = lambda y: y[:, sig] - y[:, 2]  # noqa: E731
        fs, fb = F(Y["clean", sig]), F(Y["clean", 2])
        thr = float(np.quantile(fs, 0.20))
        t = {"thr": thr, "clean": {"eff_sig": 0.8, "acc_QCD": float(np.mean(fb > thr)), "auc": auc(fs, fb)}}
        for k in ("photon", "nh"):
            ps, pb = F(Y[k, sig])[OK[k, sig]], F(Y[k, 2])[OK[k, 2]]
            t[k] = {"eff_sig": float(np.mean(ps > thr)), "acc_QCD": float(np.mean(pb > thr)), "auc": auc(ps, pb),
                    "mean_abs_dF_sig": float(np.abs(ps - fs[OK[k, sig]]).mean())}
        res["taggers"][NAMES[sig]] = t
    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out_dir / f"{name}.npz", **{f"{k}_{lab}": v for (k, lab), v in Y.items()})
    C.dump(a.out_dir / f"{name}.json", res)
    summary([a.out_dir / f"{name}.npz"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
