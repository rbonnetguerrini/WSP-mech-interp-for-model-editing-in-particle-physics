#!/usr/bin/env python
"""Fig. 1: infrared-safety sketch. The same jet without (left, tagged as W jet) and with one added soft photon (right,
tagged as QCD jet). Needs no data.

  python scripts/figures/ir_safety_sketch.py [--out DIR]   -> DIR/ir_safety_sketch.pdf / .png
"""
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, FancyArrowPatch

BLUE, MAGENTA, DARK, ORANGE = "#004aad", "#9f1267", "#0a0a0a", "#1A05A2"
TH = np.deg2rad(50.0)          # jet axis
L, A, B = 5.2, 1.95, 0.62      # cone length, cap semi-axes
# (angle offset from the axis in degrees, length): the same particles in both panels
PARTS = [(11.0, 4.5), (2.7, 5.6), (-8.0, 5.4), (-18.0, 4.1), (5.7, 2.6), (-13.6, 2.4), (-2.8, 1.8)]
PHOTON = (19.5, 2.5)


def blend(col, f):
    c = np.array(matplotlib.colors.to_rgb(col))
    return tuple(1 - f * (1 - c))


def unit(a):
    return np.array([np.cos(a), np.sin(a)])


def cone(ax, x0, col):
    o = np.array([x0, 0.0])
    c = o + L * unit(TH)
    n = unit(TH + np.pi / 2)
    ax.add_patch(Polygon([o, c + A * n, c - A * n], closed=True, fc=blend(col, 0.15), ec="none", zorder=1))
    ax.add_patch(Polygon([c + A * n, o, c - A * n], closed=False, fc="none", ec=col, lw=1.6, zorder=2))
    ax.add_patch(Ellipse(c, 2 * A, 2 * B, angle=np.rad2deg(TH) + 90, fc=blend(col, 0.36), ec="none", zorder=2))
    ax.add_patch(Ellipse(c, 2 * A, 2 * B, angle=np.rad2deg(TH) + 90, fc="none", ec=col, lw=1.6, zorder=5))
    for da, ln in PARTS:
        e = o + ln * unit(TH + np.deg2rad(da))
        ax.add_patch(FancyArrowPatch(o, e, arrowstyle="-|>,head_length=0.6,head_width=0.17", mutation_scale=14, lw=1.3, color=DARK, zorder=4,
                                     joinstyle="miter", capstyle="butt",
                                     shrinkA=0, shrinkB=0))
    return o


def photon(ax, o):
    da, ln = PHOTON
    d, n = unit(TH + np.deg2rad(da)), unit(TH + np.deg2rad(da) + np.pi / 2)
    s = np.linspace(0.0, ln - 0.22, 300)
    xy = o + np.outer(s, d) + np.outer(0.075 * np.sin(2 * np.pi * s / 0.30) * np.minimum(1, s / 0.3), n)
    ax.plot(xy[:, 0], xy[:, 1], color=ORANGE, lw=2.0, solid_capstyle="round", zorder=6)
    tip = o + ln * d
    ax.add_patch(FancyArrowPatch(o + (ln - 0.24) * d, tip, arrowstyle="-|>,head_length=0.6,head_width=0.2",
                                 mutation_scale=14, lw=1.6, color=ORANGE, zorder=6, shrinkA=0, shrinkB=0))



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: RESULTS_DIR/paper_figures, else the current folder)")
    args = ap.parse_args()
    out = args.out or (Path(os.environ["RESULTS_DIR"]) / "paper_figures" if os.environ.get("RESULTS_DIR") else Path("."))
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    dx = 10.2
    o1 = cone(ax, 0.0, BLUE)
    o2 = cone(ax, dx, MAGENTA)
    photon(ax, o2)
    ya = 2.45
    ax.add_patch(FancyArrowPatch((5.9, ya), (8.9, ya), arrowstyle="-|>", mutation_scale=20, lw=2.0, color=DARK))
    t = ax.text(5.72, ya + 0.22, "+ ", ha="left", va="bottom", fontsize=13, color=DARK)
    ax.annotate("one soft photon", xy=(1, 0), xycoords=t, ha="left", va="bottom", fontsize=13, color=ORANGE,
                fontweight="bold")
    xc = L * np.cos(TH) / 2 + 0.6
    for o, col, lab in ((o1, BLUE, "tagged as W jet"), (o2, MAGENTA, "tagged as QCD jet")):
        ax.text(o[0] + xc, -0.6, lab, ha="center", va="top", fontsize=15, fontweight="bold", color=col)
    ax.set_xlim(-0.6, dx + 5.6)
    ax.set_ylim(-1.4, 6.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out / "ir_safety_sketch.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out / "ir_safety_sketch.png", bbox_inches="tight", pad_inches=0.05, dpi=160)
    print("wrote", out / "ir_safety_sketch.pdf")


if __name__ == "__main__":
    main()
