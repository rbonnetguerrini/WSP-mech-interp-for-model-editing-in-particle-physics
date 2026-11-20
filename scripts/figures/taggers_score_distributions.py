#!/usr/bin/env python
"""Fig. taggers (taggers_score_distributions.pdf): distribution of the one-vs-rest score F_c = ln[p(c)/(1-p(c))]
of the ten JetClass classes, for the jets of class c and the jets of the nine other classes, summed over the ten
classes, in the original model (left) and the model repaired by the selected WSP edit (right). Light lines: the same
jets with one added photon at f = 1e-3. Dotted line: the cut that keeps 80% of the unmodified jets of the class.

Input: results/figures/small/tagger_scores/{original_model,agents_target_informed_run2_edit_large}.npz
(scripts/evaluation/tagger_scores.py: logits of 5000 jets per class).
  python scripts/figures/taggers_score_distributions.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import style as S  # noqa: E402
from tagger_scores import one_vs_rest  # noqa: E402

PANELS = (("original_model", "original model"), ("agents_target_informed_run2_edit_large", "repaired model"))


def main() -> int:
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    args, plt = S.setup(__doc__)
    d0 = S.fig_dir(args.results) / "tagger_scores"
    data = {}
    for name, _ in PANELS:
        z = np.load(d0 / f"{name}.npz")
        data[name] = {v: one_vs_rest(z, v) for v in ("clean", "photon")}
        data[name]["cut"] = np.quantile(data[name]["clean"][0], 0.2)
    allv = np.concatenate([data[t][v][i] for t, _ in PANELS for v in ("clean", "photon") for i in (0, 1)])
    bins = np.arange(np.floor(np.quantile(allv, 0.002) / 0.5) * 0.5, np.quantile(allv, 0.998) + 0.5, 0.5)
    h = S.H2 + 0.1
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXTWIDTH, h), sharey=True)
    hmax = 0
    for ax, (name, ttl) in zip(axs, PANELS):
        d = data[name]
        for i, c in enumerate((S.BLUE, S.MAGENTA)):
            for v, col, zo in (("photon", S.light(c), 2), ("clean", c, 3)):
                x = d[v][i]
                hh, _, _ = ax.hist(x, bins, weights=np.full(len(x), 1 / len(x)), histtype="step", color=col, lw=1.4,
                                   zorder=zo)
                hmax = max(hmax, hh.max())
        ax.axvline(d["cut"], color="k", lw=1.2, ls=":", zorder=1)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel(r"$F_c = \ln[p(c)/(1-p(c))]$")
        ax.text(0.04, 0.94, ttl, transform=ax.transAxes, ha="left", va="top", fontsize=S.fs(-3))
        print(f"{name}: efficiency of the jets of the class with the photon "
              f"{np.mean(d['photon'][0] > d['cut']):.3f}", flush=True)
    axs[0].set_ylim(0, 1.12 * hmax)
    axs[0].set_ylabel("fraction of jets")
    hs = [Line2D([], [], color=S.BLUE, lw=1.4), Line2D([], [], color=S.MAGENTA, lw=1.4),
          (Line2D([], [], color=S.light(S.BLUE), lw=1.4), Line2D([], [], color=S.light(S.MAGENTA), lw=1.4)),
          Line2D([], [], color="k", lw=1.2, ls=":")]
    fig.legend(hs, [S.tx(s) for s in ("jets of the class", "jets of the other classes", r"+ photon, $f=10^{-3}$",
                                      r"80\% efficiency")],
               loc="upper center", ncol=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=1 - S.em(0.36) / h, bottom=S.em(0.52) / h, wspace=0.05)
    S.save(fig, args.out, "taggers_score_distributions")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
