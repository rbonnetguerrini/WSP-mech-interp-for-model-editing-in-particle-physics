#!/usr/bin/env python
"""App. A Fig. attention middle: mean attention of b7/6 (backbone block 7, head 6) on one added neutral hadron against
its energy, for W and QCD jets; grey: the energy distribution of the neutral hadrons in the clean JetClass jets;
dotted line: its first percentile.

Input: figures/small/soft_particle_scans/original_model_head_b7-6_attention.npz
(scripts/analysis/soft_particle_scan.py --attn --families "neutral hadron" photon).

  python scripts/figures/head6_attention_vs_E_neutral_hadron.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    a = np.load(S.fig_dir(args.results) / "soft_particle_scans" / "original_model_head_b7-6_attention.npz")
    fam = "neutral hadron"
    xb = np.geomspace(0.05, 30, 41)
    c = np.sqrt(xb[1:] * xb[:-1])
    fig, ax = plt.subplots(figsize=(0.51 * S.TEXTWIDTH, S.H2))
    rE = a[f"real_E_{fam}"]
    S.twin_hist(ax, rE, xb, S.ylab("neutral hadron"))
    hs = []
    for lab, col in (("W", S.BLUE), ("QCD", S.MAGENTA)):
        pt, eta, att = a[f"pt_{fam}_{lab}"], a[f"eta_{fam}_{lab}"], a[f"att_{fam}_{lab}"]
        ok = np.isfinite(att) & (pt > 0)
        E = pt[ok] * np.cosh(eta[ok])
        idx = np.digitize(E, xb) - 1
        m = np.array([att[ok][idx == i].mean() if (idx == i).sum() >= 15 else np.nan for i in range(len(c))])
        hs += ax.plot(c, m, color=col, lw=1.6)
    q01 = float(np.quantile(rE, 0.01))
    ax.axvline(q01, color="k", lw=1.2, ls=":")
    hs.append(Line2D([], [], color="k", lw=1.2, ls=":"))
    hs.append(Patch(facecolor=S.GREY, alpha=0.25, lw=0))
    S.log_x(ax, xb[0], xb[-1])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(S.tx(r"\texttt{b7/6} mean attention") + "\n" + "on the added neutral hadron")
    ax.set_xlabel(r"$E$ of the added neutral hadron [GeV]")
    S.top_legend(ax, hs, ["W jets", "QCD jets", "1st percentile", "neutral hadrons in JetClass"], ncol=2)
    S.finish(fig, S.H2)
    S.save(fig, args.out, "head6_attention_vs_E_neutral_hadron")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
