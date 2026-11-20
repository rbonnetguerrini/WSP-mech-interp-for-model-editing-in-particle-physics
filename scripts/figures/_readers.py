"""Shared drawing code of the App. E reader figures of the medium and large models (app_readers_<m|l>_attention,
app_readers_<m|l>_patching), in the style of attention_head_grid_photon_cls / patching_heads_s_photon.
Inputs: figures/<medium|large>/readers_attention.json (scripts/analysis/attention_heads.py --classifier) and
figures/<medium|large>/readers_patching.json, probe photon_0.3GeV (scripts/analysis/activation_patching.py)."""
import json

import numpy as np

import style as S

NAME = {"m": "medium", "l": "large"}


def draw(plt, results, s: str, which: str, out) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    d = S.fig_dir(results, s)
    if which == "attention":
        M, st = S.grid(d / "readers_attention.json")
        cmap = LinearSegmentedColormap.from_list("wm", ["#ffffff", S.MAGENTA])
        vmax, lab, title, thr = M.max(), "mean attention", r"+ photon, $f=10^{-3}$", 0.1
    else:
        pres = json.loads((d / "readers_patching.json").read_text())
        Pm, st = S.patching_grid(pres, "photon_0.3GeV")
        M = np.clip(Pm, 0, None)
        cmap = LinearSegmentedColormap.from_list("wb", ["#ffffff", S.BLUE])
        vmax = max(0.5, np.nanmax(Pm)) if s == "m" else float(np.nanmax(Pm))
        lab, title, thr = r"share of $|\Delta F_W|$ removed", r"+ photon, $0.3$ GeV", 0.09
    # half-width panels for both sizes; the large model (32 blocks x 32 heads) without the numbers in the cells
    # (colour only) and with every fourth block / head labelled
    h, w = S.H2 + 0.25, 0.49 * S.TEXTWIDTH
    fig, ax = plt.subplots(figsize=(w, h))
    im = S.heat(plt, ax, M, st, cmap, vmax, "bracket", thr=thr)
    if s == "m":  # 16 heads: every second tick, smaller numbers
        ax.set_xticks(range(0, M.shape[1], 2))
        for t in ax.texts:
            if t.get_text().startswith("."):
                t.set_fontsize(S.fs(-6))
    else:
        for t in list(ax.texts):
            if t.get_text().startswith("."):
                t.remove()
        ax.set_xticks(range(0, M.shape[1], 4))
        nb = sum(x.startswith("body") for x in st)
        ax.set_yticklabels([x.split(".")[-1] if (i < nb and i % 4 == 0) or (i >= nb and (i - nb) in (0, 3))
                            else "" for i, x in enumerate(st)])
    ax.set_title(S.tx(f"{NAME[s]} model, {title}"), fontsize=S.fs(-2), fontweight="normal")
    S.cbar(fig, ax, im, lab)
    S.finish(fig, h)
    S.save(fig, out, f"app_readers_{s}_{which}")
    plt.close(fig)
    print(s, which, "max", round(float(np.nanmax(M)), 2), flush=True)
