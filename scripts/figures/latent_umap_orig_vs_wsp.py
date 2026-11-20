#!/usr/bin/env python
"""Fig. latent (latent_umap_orig_vs_wsp.pdf): UMAP projection of the input tokens of the classifier (the four summary
tokens after the body), for the original model (left) and the model repaired by the selected WSP edit (right). Each
projection is fitted on the unmodified jets of its model, so the panels do not share their axes.

Input: results/figures/small/representation/{original_model,agents_target_informed_run2_edit_large}_embedding.npz
(scripts/analysis/latent_umap.py; the 15-nearest-neighbour shares are printed by scripts/evaluation/representation.py).
  python scripts/figures/latent_umap_orig_vs_wsp.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

PANELS = (("original_model", "original model"), ("agents_target_informed_run2_edit_large", "repaired model"))
LAYER = "body.norm"


def main() -> int:
    from matplotlib.lines import Line2D
    args, plt = S.setup(__doc__)
    sty = {"JetClass-II": (S.LIGHTGREY, "JetClass-II jets"), "QCD": (S.MAGENTA, "QCD jets"), "W": (S.BLUE, "W jets"),
           "W + photon": (S.ORANGE, r"W + photon, $f=10^{-3}$"), "QCD + photon": (S.GREEN, r"QCD + photon, $f=10^{-3}$")}
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 2, figsize=(0.9 * S.TEXTWIDTH, 2.95))
    d0 = S.fig_dir(args.results) / "representation"
    for ax, (name, ttl) in zip(axs, PANELS):
        z = np.load(d0 / f"{name}_embedding.npz")
        bg = z[f"{LAYER}|JetClass-II|umap"]
        ax.scatter(bg[:, 0], bg[:, 1], s=1.5, c=S.LIGHTGREY, lw=0, rasterized=True)
        fg = [k for k in sty if k != "JetClass-II"]
        xy = np.concatenate([z[f"{LAYER}|{k}|umap"] for k in fg])
        col = np.concatenate([[sty[k][0]] * len(z[f"{LAYER}|{k}|umap"]) for k in fg])
        o = rng.permutation(len(xy))
        ax.scatter(xy[o, 0], xy[o, 1], s=2.0, c=col[o], lw=0, rasterized=True)
        ax.set_xticks([]), ax.set_yticks([])
        ax.tick_params(which="both", top=False, right=False)
        for sp in ax.spines.values():
            sp.set_color("#999999")
        ax.text(0.97, 0.03, ttl, transform=ax.transAxes, ha="right", va="bottom", fontsize=S.fs(-3),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        ax.set_xlabel("UMAP 1")
    axs[0].set_ylabel("UMAP 2")
    hs = [Line2D([], [], ls="", marker="o", ms=4, color=sty[k][0]) for k in sty]
    fig.legend(hs, [S.tx(sty[k][1]) for k in sty], loc="upper center", ncol=3, handletextpad=0.1, columnspacing=1.6)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 1 - S.em(0.56) / fig.get_figheight()))
    S.save(fig, args.out, "latent_umap_orig_vs_wsp")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
