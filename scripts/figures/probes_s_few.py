#!/usr/bin/env python
"""App. A Fig. probes: probability that a jet comes from JetClass-II from a linear probe on the summary tokens after each
backbone block (b0-b7), the final norm and the two classifier blocks (c0, c1), for clean JetClass, clean JetClass-II
and JetClass jets with one photon at f = 1e-3; band: central 68% of the jets.

Input: figures/small/probes.json (scripts/analysis/probes.py).

  python scripts/figures/probes_s_few.py --results RESULTS [--out DIR] [--no-tex]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    from matplotlib.lines import Line2D
    res = json.loads((S.fig_dir(args.results) / "probes.json").read_text())
    photon = "JetClass + photon f=1e-3"
    layers = list(res["layers"])
    xs = np.arange(len(layers))
    sets = (("JetClass clean", S.BLUE, "JetClass"), ("JetClass-II clean", S.MAGENTA, "JetClass-II"),
            (photon, S.ORANGE, r"JetClass + photon, $f=10^{-3}$"))
    fig, ax = plt.subplots(figsize=(0.65 * S.TEXTWIDTH, 2.9))
    hs = []
    for k, c, _ in sets:
        mu = np.array([res["layers"][L][k] for L in layers])
        st = [res["layers"][L]["stats"][k] for L in layers]
        ax.fill_between(xs, [s["q16"] for s in st], [s["q84"] for s in st], color=c, alpha=0.15, lw=0)
        hs += ax.plot(xs, mu, color=c, lw=1.6, marker="o", ms=3.5)
    ax.set_xticks(xs)
    lab = {"body.norm": "norm"}
    ax.set_xticklabels([lab.get(L, L.split()[-1]) for L in layers])
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(S.tx(r"$P(\text{JetClass-II}\,|\,\text{jet})$"))
    nb = sum(L.startswith("body block") for L in layers)
    tr = ax.get_xaxis_transform()
    # group names in place of an x-axis label: backbone centred under its blocks, classifier right-aligned
    yl = -S.em(0.36) / ax.get_window_extent().height * fig.dpi
    ax.text(0.5 * (nb - 1), yl, "backbone block", transform=tr, ha="center", va="top", fontsize=S.fs())
    ax.text(len(layers) - 0.5, yl, "classifier block", transform=tr, ha="right", va="top", fontsize=S.fs())
    S.top_legend(ax, hs + [Line2D([], [], color="#bbbbbb", lw=6, alpha=0.5)],
                 [s[2] for s in sets] + [r"central 68\% of jets"], ncol=2)
    fig.tight_layout(pad=0.3)
    S.save(fig, args.out, "probes_s_few")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
