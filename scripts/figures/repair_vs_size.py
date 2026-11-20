#!/usr/bin/env python
"""Fig. frontier (repair_vs_size.pdf): remaining response against the number of changed parameters for all the
repair variants tested on the small model. Each WSP point is one edit of one run; the selected variant of each
method (Tab. comparison) has a black outline.

Inputs: results/scores/small/ (fine-tuning, SAE edits) and the run records runs/*/small/run*/scores/ (WSP).
  python scripts/figures/repair_vs_size.py --results RESULTS [--out DIR] [--no-tex] [--linear]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _frontier as F  # noqa: E402
import style as S  # noqa: E402


def main() -> int:
    linear = "--linear" in sys.argv
    if linear:
        sys.argv.remove("--linear")
    args, plt = S.setup(__doc__)
    w, h = 0.6 * S.TEXTWIDTH, S.H2 + 0.45
    fig, ax = plt.subplots(figsize=(w, h))
    F.draw(ax, args.results, "small", F.wsp_rows())
    ax.set_xscale("log")
    ax.set_xlim(6, 4e5)
    if linear:
        ax.set_ylim(0, 0.8)
    else:
        ax.set_yscale("log")
        ax.set_ylim(0.012, 1.0)
    ax.tick_params(axis="x", which="major", pad=S.em(6))
    ax.set_xlabel("changed parameters")
    ax.set_ylabel("remaining response")
    S.top_legend(ax, F.handles(), F.LABELS, ncol=2, handletextpad=0.2)
    S.finish(fig, h, top=S.em(0.80))
    S.save(fig, args.out, "repair_vs_size_linear" if linear else "repair_vs_size")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
