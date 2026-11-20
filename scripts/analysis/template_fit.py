#!/usr/bin/env python
"""Template fit of the W fraction of a pseudo-dataset (Sec. 2: "true W fraction of a quarter is recovered for the
nominal data, while the fit on the tampered data ...").

Input: template_fit_inputs_s.npz from template_fit_inputs.py (per-jet scores; simulation: 10,000 W + 10,000 QCD jets,
nominal; pseudo-data: 2,500 W + 7,500 QCD jets, true W fraction 0.25, in three scenarios: nominal, every jet + one
photon at 0.2-0.6 GeV ("below_floor"), every jet + one photon at 1-3 GeV ("in_range")).

Model per bin i: mu_i = a * w_i + b * q_i, with w_i, q_i the raw simulated template counts. Two free parameters, the
W and QCD yields N_W = a * 10,000 and N_QCD = b * 10,000; W fraction f = N_W / (N_W + N_QCD).
chi2 = sum_i (n_i - mu_i)^2 / (mu_i + a^2 w_i + b^2 q_i): Pearson with the templates' statistical variance added.
ndf = bins - 2. The error on f is from the profile chi2 scan (delta chi2 = 1).
Bins: width 0.5 in the score, merged in the tails until each bin has >= 10 expected entries at the true
composition, with open-ended first and last bins. The same binning is used for every scenario of a tagger.

  python scripts/analysis/template_fit.py [--npz PATH]   -> RESULTS_DIR/analysis/template_fit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TAGGERS = {"original": "OmniLearned tagger (F_W)", "ptcut": "drop particles < 0.75 GeV (F_W)",
           "ircsafe": "IRC-safe BDT (its log-odds)"}
SCEN = ("nominal", "below_floor", "in_range")
F_TRUE = 0.25


def binning(w: np.ndarray, q: np.ndarray, n_pd: int, width: float = 0.5, min_exp: float = 10.0) -> np.ndarray:
    lo, hi = np.floor(min(w.min(), q.min())), np.ceil(max(w.max(), q.max()))
    e = np.arange(lo, hi + width, width)
    # merge from both ends inwards until every bin has >= min_exp expected entries
    edges = list(e)

    def merged(ed):
        return (F_TRUE * np.histogram(w, ed)[0] / len(w) + (1 - F_TRUE) * np.histogram(q, ed)[0] / len(q)) * n_pd
    while True:
        m = merged(np.array(edges))
        bad = np.flatnonzero(m < min_exp)
        if bad.size == 0 or len(edges) <= 3:
            break
        i = bad[0]
        # merge bin i with its neighbour that has fewer expected entries (or the only neighbour)
        if i == 0:
            del edges[1]
        elif i == len(m) - 1:
            del edges[-2]
        else:  # edge i separates bins i-1 and i; edge i+1 separates bins i and i+1
            del edges[i if m[i - 1] < m[i + 1] else i + 1]
    edges = np.array(edges, float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def chi2(p, n, wc, qc):
    a, b = p
    mu = a * wc + b * qc
    var = mu + a * a * wc + b * b * qc
    return float(np.sum((n - mu) ** 2 / np.maximum(var, 1e-9)))


def fit(n, wc, qc, n_mc_w, n_mc_q):
    tot = n.sum()
    x0 = np.array([F_TRUE * tot / n_mc_w, (1 - F_TRUE) * tot / n_mc_q])
    r = optimize.minimize(chi2, x0, args=(n, wc, qc), method="L-BFGS-B", bounds=[(0, None), (0, None)])
    a, b = r.x
    NW, NQ = a * n_mc_w, b * n_mc_q
    f = NW / (NW + NQ)

    def prof(fv):  # profile scan in f (total yield profiled)
        g = lambda t: chi2((fv * t / n_mc_w, (1 - fv) * t / n_mc_q), n, wc, qc)  # noqa: E731
        return optimize.minimize_scalar(g, bounds=(0.5 * tot, 1.5 * tot), method="bounded").fun
    c0 = r.fun
    grid = np.linspace(max(0, f - 0.15), min(1, f + 0.15), 601)
    pr = np.array([prof(v) for v in grid])
    inside = grid[pr <= c0 + 1]
    ndf = len(n) - 2
    return {"f": float(f), "f_lo": float(inside.min()), "f_hi": float(inside.max()), "N_W": float(NW),
            "N_QCD": float(NQ), "N_total_fit": float(NW + NQ), "N_total_data": int(tot), "chi2": float(c0),
            "ndf": int(ndf), "chi2_ndf": float(c0 / ndf), "p_value": float(stats.chi2.sf(c0, ndf)),
            "chi2_at_true_f": float(prof(F_TRUE))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=None, help="default RESULTS_DIR/analysis/template_fit_inputs_s.npz")
    ap.add_argument("--out", type=Path, default=None, help="default RESULTS_DIR/analysis/template_fit.json")
    args = ap.parse_args()
    a = dict(np.load(args.npz or C.analysis_dir() / "template_fit_inputs_s.npz"))
    res = {"model": "mu = a*W_template + b*QCD_template; Pearson chi2 with template variance; ndf = bins - 2",
           "f_true": F_TRUE, "fits": {}}
    for t, tl in TAGGERS.items():
        w, q = a[f"mc_W_{t}"], a[f"mc_QCD_{t}"]
        edges = binning(w, q, len(a["pd_label"]))
        wc, qc = np.histogram(w, edges)[0].astype(float), np.histogram(q, edges)[0].astype(float)
        res["fits"][t] = {"label": tl, "n_bins": len(wc), "edges_inner": edges[1:-1].tolist()}
        for s in SCEN:
            n = np.histogram(a[f"pd_{s}_{t}"], edges)[0].astype(float)
            r = fit(n, wc, qc, len(w), len(q))
            res["fits"][t][s] = r
            print(f"{t:9s} {s:12s} f = {r['f']:.3f} [{r['f_lo']:.3f}, {r['f_hi']:.3f}]  N = {r['N_total_fit']:.0f}/"
                  f"{r['N_total_data']}  chi2/ndf = {r['chi2']:.1f}/{r['ndf']}  p = {r['p_value']:.2g}  "
                  f"chi2(f=0.25) = {r['chi2_at_true_f']:.1f}", flush=True)
    out = args.out or C.analysis_dir() / "template_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
