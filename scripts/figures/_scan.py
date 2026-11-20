"""Shared drawing code of the three pT-scan figures (shift_vs_pt_photon_scan, shift_vs_neutral_hadron_pt,
shift_vs_charged_hadron_pt): mean |dF_W| for W and QCD jets against the pT of one added particle, over the pT
distribution of that particle type in the clean jets (grey, right axis); dotted line: its first percentile.
Input: figures/small/soft_particle_scans/original_model.json / .npz (scripts/analysis/soft_particle_scan.py)."""
import json

import numpy as np

import style as S


def draw(plt, results, fam: str, name: str, out) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = S.fig_dir(results) / "soft_particle_scans"
    res = json.loads((d / "original_model.json").read_text())
    a = np.load(d / "original_model.npz")
    HB = np.asarray(res["pt_bins_GeV"])
    x = np.sqrt(HB[1:] * HB[:-1])
    fams = res["families"]
    arr = lambda f, c, k: np.array([np.nan if v is None else v for v in fams[f][c][k]])  # noqa: E731
    # common y range of the three scans, over all three families, with 12% headroom
    ymax = 1.12 * np.nanmax(np.r_[tuple(arr(f, c, "mean_abs_dF") for f in ("photon", "neutral hadron", "charged hadron")
                                        if f in fams for c in ("W", "QCD"))])
    fig, ax = plt.subplots(figsize=(0.49 * S.TEXTWIDTH, S.H2))
    S.twin_hist(ax, a[f"real_pt_{fam}"], HB, S.ylab(fam))
    hs, ls = [], []
    for lab, c in (("W", S.BLUE), ("QCD", S.MAGENTA)):
        hs.append(ax.errorbar(x, arr(fam, lab, "mean_abs_dF"), arr(fam, lab, "sem_abs_dF"), color=c, lw=1.6,
                              elinewidth=1.0))
        ls.append(f"{lab} jets")
    q01 = fams[fam]["data"]["pt_q01"]
    ax.axvline(q01, color="k", lw=1.2, ls=":")
    hs.append(Line2D([], [], color="k", lw=1.2, ls=":"))
    ls.append("1st percentile")
    hs.append(Patch(facecolor=S.GREY, alpha=0.25, lw=0))
    ls.append(f"{fam}s in JetClass")
    S.log_x(ax, HB[0], HB[-1])
    ax.set_ylim(0, ymax)
    ax.set_xlabel(rf"$p_T$ of the added {fam} [GeV]")
    ax.set_ylabel(r"mean $|\Delta F_W|$")
    S.top_legend(ax, hs, ls, ncol=2)
    S.finish(fig, S.H2)
    S.keep_inside(fig, ax)
    S.save(fig, out, name)
    plt.close(fig)
