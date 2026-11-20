#!/usr/bin/env python
"""Does the attention of b7/6 on one added soft neutral switch off at a fixed pT or at a fixed energy? (Sec. 2
"Characterizing the sensitivity", App. A Fig. attention middle.)

If the step sits at a fixed energy E0, the per-|eta| curves coincide when plotted against E and are shifted by
cosh(eta) when plotted against pT (and the other way round for a fixed pT threshold).

Input: the soft-particle scan with --attn (soft_particle_scan.py --attn --families "neutral hadron" photon): per
added particle pT, eta, Delta F_W and the b7/6 attention; real particles of each type in the clean jets: pT and E,
so |eta| = arccosh(E / pT) (massless, as in the inputs).

Per family and |eta| bin (W and QCD pooled): the softest real particle in pT and in E, and the pT and E at which
the median b7/6 attention falls to half its maximum (first crossing from the right, log-interpolated between bin
centres).

  python scripts/analysis/head_attention_threshold.py --npz SCAN.npz [--out FILE]
      -> RESULTS_DIR/analysis/head_attention_threshold.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ETA_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5)]
X_BINS = np.geomspace(0.05, 30, 29)
MIN_N = 15


def binned(x, y, stat):
    out = np.full(len(X_BINS) - 1, np.nan)
    idx = np.digitize(x, X_BINS) - 1
    for i in range(len(out)):
        s = y[idx == i]
        if len(s) >= MIN_N:
            out[i] = stat(s)
    return out


def edge(x, y, level):
    """Lowest x (bin centre, log-interpolated) above which the curve stays below `level` (first crossing from
    the right)."""
    c = np.sqrt(X_BINS[1:] * X_BINS[:-1])
    ok = np.isfinite(y)
    c, y = c[ok], y[ok]
    above = np.flatnonzero(y >= level)
    if above.size == 0 or above[-1] == len(y) - 1:
        return float("nan")
    i = above[-1]
    t = (y[i] - level) / (y[i] - y[i + 1])
    return float(np.exp(np.log(c[i]) + t * (np.log(c[i + 1]) - np.log(c[i]))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--families", nargs="+", default=["neutral hadron", "photon"])
    args = ap.parse_args()
    a = np.load(args.npz)
    summary = {}
    for fam in args.families:
        pt = np.r_[a[f"pt_{fam}_W"], a[f"pt_{fam}_QCD"]]
        eta = np.abs(np.r_[a[f"eta_{fam}_W"], a[f"eta_{fam}_QCD"]])
        att = np.r_[a[f"att_{fam}_W"], a[f"att_{fam}_QCD"]]
        dF = np.abs(np.r_[a[f"dF_{fam}_W"], a[f"dF_{fam}_QCD"]])
        ok = np.isfinite(att) & np.isfinite(dF) & (pt > 0)
        pt, eta, att, dF = pt[ok], eta[ok], att[ok], dF[ok]
        E = pt * np.cosh(eta)
        rpt, rE = a[f"real_pt_{fam}"], a[f"real_E_{fam}"]
        reta = np.arccosh(np.clip(rE / rpt, 1, None))
        summary[fam] = {}
        for lo, hi in ETA_BINS:
            m = (eta >= lo) & (eta < hi)
            rm = (reta >= lo) & (reta < hi)
            floors = {"pT": float(rpt[rm].min()) if rm.any() else np.nan,
                      "E": float(rE[rm].min()) if rm.any() else np.nan}
            lab = f"{lo:.1f} <= |eta| < {hi:.1f}"
            s = {"n_added": int(m.sum()), "n_real": int(rm.sum()), "data_floor_pT": floors["pT"],
                 "data_floor_E": floors["E"]}
            for xv, xn in ((pt, "pT"), (E, "E")):
                ya = binned(xv[m], att[m], np.median)
                s[f"att_half_edge_{xn}"] = edge(xv[m], ya, 0.5 * np.nanmax(ya))
            summary[fam][lab] = s
        print(fam, json.dumps(summary[fam], indent=1), flush=True)
    out = args.out or C.analysis_dir() / "head_attention_threshold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
