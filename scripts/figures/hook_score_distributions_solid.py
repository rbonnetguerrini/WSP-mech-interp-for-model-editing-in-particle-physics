#!/usr/bin/env python
"""Fig. 2 left: distribution of the W-tagger score F_W for W and QCD jets (dark) and for the same jets with one added
photon at f = 1e-3 (light); the dotted line is the cut that keeps 80% of the clean W jets.

Input: figures/small/w_tagger_response.npz and w_tagger_response_summary.json (scripts/analysis/w_tagger_response.py).

  python scripts/figures/hook_score_distributions_solid.py --results RESULTS [--out DIR] [--no-tex]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    d = S.fig_dir(args.results)
    a = dict(np.load(d / "w_tagger_response.npz"))
    cut = json.loads((d / "w_tagger_response_summary.json").read_text())["wp80"]["cut_FW"]
    v = "f1e-3"
    lo = min(np.quantile(a["F0_QCD"], 0.002), np.quantile(a[f"F_{v}_QCD"], 0.002))
    hi = max(np.quantile(a["F0_W"], 0.998), np.quantile(a[f"F_{v}_W"], 0.998))
    bins = np.arange(np.floor(lo / 0.25) * 0.25, hi + 0.25, 0.25)
    fig, ax = plt.subplots(figsize=(0.49 * S.TEXTWIDTH, S.H2))
    hmax = 0
    for lab, c in (("W", S.BLUE), ("QCD", S.MAGENTA)):
        ok = a[f"ok_{v}_{lab}"]
        for key, col, z in ((f"F_{v}_{lab}", S.light(c), 2), (f"F0_{lab}", c, 3)):
            x = a[key][ok]
            h, _, _ = ax.hist(x, bins, weights=np.full(len(x), 1 / len(x)), histtype="step", color=col, lw=1.4,
                              zorder=z)
            hmax = max(hmax, h.max())
    ax.axvline(cut, color="k", lw=1.2, ls=":", zorder=1)
    ax.set_ylim(0, 1.12 * hmax)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(r"$F_W = \ln[p(W)/p(\mathrm{QCD})]$")
    ax.set_ylabel("fraction of jets")
    hs = [Line2D([], [], color=S.BLUE, lw=1.4), Line2D([], [], color=S.MAGENTA, lw=1.4),
          (Line2D([], [], color=S.light(S.BLUE), lw=1.4), Line2D([], [], color=S.light(S.MAGENTA), lw=1.4)),
          Line2D([], [], color="k", lw=1.2, ls=":")]
    S.top_legend(ax, hs, ["W jets", "QCD jets", r"+ photon, $f=10^{-3}$", r"80\% W eff."], ncol=2,
                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
    S.finish(fig, S.H2)
    S.save(fig, args.out, "hook_score_distributions_solid")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
