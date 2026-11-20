"""Common style and helpers of the paper figures (numpy + matplotlib).

Each figure is drawn at its width in the paper (textwidth = 455.24 pt) with all font sizes FONTSIZE / SCALE
(18 / 2: axis labels 9 pt, tick labels 7.5 pt, legends 8 pt in the PDF); line widths are absolute (paper) points.
Panels that sit next to each other have the same height and the same top/bottom margins.

Text is set with LaTeX and the Charter font of the paper (mathdesign package). Without a LaTeX installation
(--no-tex, or automatically when `latex` is not found) matplotlib's mathtext is used instead, with the Charter
font if it is installed and the default serif font otherwise; the figures then differ from the paper in the font.

Every figure script takes:
  --results DIR   folder with the cached results (the data record's results/ folder, or RESULTS_DIR)
  --out DIR       output folder (default: <results>/paper_figures); writes <name>.pdf and <name>.png
  --no-tex        mathtext instead of LaTeX
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

FONTSIZE, SCALE = 18, 2.0
TEXTWIDTH = 455.24411 / 72.27  # inches
H2 = 2.6  # height of the two-panel figures (inches)
BLUE, MAGENTA, GREY = "#004aad", "#9f1267", "#888888"
BAND = "#6e6e6e"  # right-axis (data distribution) colour
ORANGE, GREEN, LIGHTGREY = "#e09a2c", "#2f8f4e", "#c4c4c4"
USETEX = True
SIZE_NAME = {"s": "small", "m": "medium", "l": "large"}


def fs(offset: float = 0) -> float:
    """Font size in the PDF (pt) of rc size FONTSIZE + offset."""
    return (FONTSIZE + offset) / SCALE


def em(x: float) -> float:
    """A length that scales with the text (given for 13 pt labels)."""
    return x * fs() / 13


def light(c: str, t: float = 0.55) -> str:
    """Mix a colour with white (t = share of white)."""
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{int(round(v + (255 - v) * t)):02x}" for v in (r, g, b))


def tx(s: str) -> str:
    """A label written for LaTeX, adapted for mathtext when LaTeX is not used."""
    if USETEX:
        return s
    s = s.replace(r"\%", "%")
    return re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)


def style(usetex: bool = True):
    global USETEX
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    USETEX = usetex
    plt.rc("font", family="serif", size=fs(), serif="Charter")
    plt.rc("axes", titlesize=fs(1), labelsize=fs(), titleweight="bold", labelweight="bold")
    plt.rc("xtick", labelsize=fs(-3))
    plt.rc("ytick", labelsize=fs(-3))
    plt.rcParams["legend.fontsize"] = fs(-2)
    if usetex:
        plt.rc("text.latex", preamble=r"""
\usepackage{amsmath}
\usepackage[bitstream-charter]{mathdesign}
\DeclareSymbolFont{usualmathcal}{OMS}{cmsy}{m}{n}
\DeclareSymbolFontAlphabet{\mathcal}{usualmathcal}
""")
        plt.rc("text", usetex=True)
    else:  # mathtext, with the Charter font if installed
        from matplotlib import font_manager as fm
        charter = False
        for d in (Path.home() / ".local/share/fonts", Path.home() / ".fonts"):
            for f in d.glob("Charter*.otf"):
                fm.fontManager.addfont(str(f))
                charter = True
        if charter:
            plt.rcParams.update({"font.family": "Charter", "mathtext.fontset": "custom", "mathtext.rm": "Charter",
                                 "mathtext.it": "Charter:italic", "mathtext.bf": "Charter:bold"})
        else:
            plt.rcParams.update({"font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"})
    plt.rcParams.update({"lines.linewidth": 1.3, "axes.linewidth": 0.6, "legend.frameon": False,
                         "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6,
                         "ytick.minor.size": 1.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "xtick.major.pad": em(5), "ytick.major.pad": em(4), "legend.handlelength": 1.6,
                         "legend.columnspacing": 1.0, "legend.borderaxespad": 0.2, "savefig.dpi": 300,
                         "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})
    return plt


def setup(doc: str):
    """Command line of a figure script: returns (args, plt) with args.results and args.out."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=os.environ.get("RESULTS_DIR"),
                    help="folder with the cached results (default: RESULTS_DIR)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <results>/paper_figures)")
    ap.add_argument("--no-tex", action="store_true", help="mathtext instead of LaTeX (font differs from the paper)")
    args = ap.parse_args()
    if args.results is None:
        ap.error("give --results or set RESULTS_DIR")
    args.out = args.out or args.results / "paper_figures"
    args.out.mkdir(parents=True, exist_ok=True)
    usetex = not args.no_tex and shutil.which("latex") is not None
    if not usetex and not args.no_tex:
        print("latex not found: falling back to mathtext (--no-tex)", flush=True)
    return args, style(usetex)


def fig_dir(results: Path, size: str = "s") -> Path:
    return results / "figures" / SIZE_NAME[size]


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    print("wrote", out / f"{name}.pdf", flush=True)


def finish(fig, height: float, top: float | None = None, bottom: float | None = None) -> None:
    """tight_layout for left/right, then the same top/bottom margins (inches) for every panel of a figure: room for a
    2-row legend or a title above, ticks and label below."""
    top, bottom = em(0.56) if top is None else top, em(0.52) if bottom is None else bottom
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=1 - top / height, bottom=bottom / height)


def keep_inside(fig, ax, margin_px: float = 2) -> None:
    """If the x-axis label sticks out of the figure on the left (a long label under narrow axes), move the axes'
    left edge to the right until it fits; the same for the y-axis labels."""
    fig.canvas.draw()
    bb = ax.xaxis.label.get_window_extent(fig.canvas.get_renderer())
    over = margin_px - bb.x0
    if over > 0:
        sp = fig.subplotpars
        fig.subplots_adjust(left=sp.left + 2 * over / fig.bbox.width)  # label is centred: shift by twice
    fig.canvas.draw()
    x0 = min(a.yaxis.label.get_window_extent(fig.canvas.get_renderer()).x0 for a in fig.axes)
    if margin_px - x0 > 0:
        fig.subplots_adjust(left=fig.subplotpars.left + (margin_px - x0) / fig.bbox.width)


def cbar(fig, ax, im, label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="6%", pad=0.07)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(tx(label))
    cb.ax.tick_params(direction="in", length=2, right=True, left=False)
    return cb


def top_legend(ax, handles=None, labels=None, ncol=3, **kw):
    kw = dict(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, **kw)
    if labels is not None:
        labels = [tx(s) for s in labels]
    return ax.legend(handles, labels, **kw) if handles is not None else ax.legend(**kw)


def twin_hist(ax, values, bins, label):
    """Distribution of the real particles in the clean jets on a grey right axis (fraction per bin)."""
    ax2 = ax.twinx()
    inr = values[(values >= bins[0]) & (values <= bins[-1])]
    ax2.hist(inr, bins, weights=np.full(len(inr), 1 / len(inr)), color=GREY, alpha=0.25, lw=0)
    ax2.set_ylabel(label, color=BAND)
    ax2.tick_params(axis="y", which="both", colors=BAND, direction="in", right=True, labelright=True)
    ax.tick_params(axis="y", which="both", right=False)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.15)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    return ax2


def ylab(fam: str) -> str:
    """Right-axis label of the shift / attention plots: the distribution of that particle type in clean jets,
    normalised to one (fraction per bin)."""
    return f"fraction of {fam}s"


def log_x(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.tick_params(axis="x", which="major", pad=em(6))  # room for the superscripts of 10^-1 etc.


def heat(plt, ax, G, sites, cmap, vmax, ylab_style="bracket", thr=0.1):
    """Blocks x heads heat map; backbone and classifier blocks are labelled once each. Cells >= thr are annotated."""
    im = ax.imshow(G, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    for i, j in zip(*np.nonzero(G >= thr)):
        ax.text(j, i, f"{G[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=fs(-3),
                color="white" if G[i, j] > 0.45 * vmax else "k")
    ax.set_xticks(range(G.shape[1]))
    ax.set_xlabel("attention head")
    ax.tick_params(which="both", top=False, right=False, length=0)
    nb = sum(s.startswith("body") for s in sites)
    ax.set_yticks(range(len(sites)))
    ticks = [s.split(".")[-1] for s in sites]
    if ylab_style == "inline" and nb < len(sites):
        ticks[0], ticks[nb] = f"backbone {ticks[0]}", f"classifier {ticks[nb]}"
    ax.set_yticklabels(ticks)
    if nb == len(sites):
        ax.set_ylabel("backbone block")
        return im
    ax.axhline(nb - 0.5, color="k", lw=0.8)
    if ylab_style == "inline":
        return im
    groups = (("backbone", -0.5, nb - 0.5), ("classifier", nb - 0.5, len(sites) - 0.5))
    tr = ax.get_yaxis_transform()  # x in axes coordinates, y in data
    xt = -0.17
    for name, y0, y1 in groups:
        ax.text(xt - (0.05 if ylab_style == "bracket" else 0), 0.5 * (y0 + y1), name, transform=tr, rotation=90,
                ha="right", va="center", fontsize=fs(), fontweight="bold")
        if ylab_style == "bracket":
            m = 0.12
            ax.plot([xt, xt], [y0 + m, y1 - m], transform=tr, color="k", lw=0.8, clip_on=False)
    return im


def grid(f: Path):
    """Attention grid (blocks x heads) and its site names from an attention_heads.py JSON."""
    r = json.loads(f.read_text())
    return np.array(r["grid"]), r.get("sites") or [f"body.{b}" for b in range(len(r["grid"]))]


def patching_grid(pres: dict, probe: str) -> tuple[np.ndarray, list]:
    """Share of |dF_W| removed per head ("readers" positions) of one probe, sites x heads."""
    r = pres["probes"][probe]["removed"]
    psites = pres["sites"]
    P = np.array([[r.get(f"{s}.h{h}.readers", np.nan) for h in range(pres["heads"][s])] for s in psites])
    return P, psites
