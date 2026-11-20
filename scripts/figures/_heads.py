"""Shared drawing code of the attention and patching heat maps of the small model (attention_head_grid_photon_cls,
patching_heads_s_photon, patching_heads_s_nh): blocks x heads, backbone and classifier blocks.
Inputs: figures/small/readers_attention_<photon|neutral_hadron>.json (scripts/analysis/attention_heads.py
--classifier) and figures/small/readers_patching.json (scripts/analysis/activation_patching.py)."""
import json

import numpy as np

import style as S

TITLE = {"photon": r"+ photon, $f=10^{-3}$", "neutral_hadron": r"+ neutral hadron, $f=10^{-3}$"}
PROBE = {"photon": "soft1e-3", "neutral_hadron": "soft7_1e-3"}  # one photon / neutral hadron at f = 1e-3


def cmap(c):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("w" + c, ["#ffffff", c])


def attention(plt, results, particle: str, name: str, out) -> None:
    G, sites = S.grid(S.fig_dir(results) / f"readers_attention_{particle}.json")
    fig, ax = plt.subplots(figsize=(0.49 * S.TEXTWIDTH, S.H2 + 0.25))
    im = S.heat(plt, ax, G, sites, cmap(S.MAGENTA), G.max(), "plain")
    ax.set_title(S.tx(TITLE[particle]), fontsize=S.fs(-2), fontweight="normal")
    S.cbar(fig, ax, im, "mean attention")
    S.finish(fig, S.H2 + 0.25)
    S.save(fig, out, name)
    plt.close(fig)


def patching(plt, results, particle: str, name: str, out) -> None:
    pres = json.loads((S.fig_dir(results) / "readers_patching.json").read_text())
    P, psites = S.patching_grid(pres, PROBE[particle])
    fig, ax = plt.subplots(figsize=(0.49 * S.TEXTWIDTH, S.H2 + 0.25))
    im = S.heat(plt, ax, np.clip(P, 0, None), psites, cmap(S.BLUE), max(0.5, np.nanmax(P)), "plain")
    ax.set_title(S.tx(TITLE[particle]), fontsize=S.fs(-2), fontweight="normal")
    S.cbar(fig, ax, im, r"share of $|\Delta F_W|$ removed")
    S.finish(fig, S.H2 + 0.25)
    S.save(fig, out, name)
    plt.close(fig)
    print(PROBE[particle], "min removed", float(np.nanmin(P)), "max", float(np.nanmax(P)))
