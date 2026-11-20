#!/usr/bin/env python
"""Fig. app_frontier: remaining response against the number of changed parameters for all the repair variants
tested on the medium model (left) and the large model (right). Each WSP point is one edit of one run; the selected
variant of each method (Tab. comparison_sizes) has a black outline.

Inputs: results/scores/{medium,large}/ (fine-tuning, SAE edits) and the run records runs/*/{medium,large}/ (WSP).
  python scripts/figures/app_frontier.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _frontier as F  # noqa: E402
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    h = S.H2 + 0.35
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXTWIDTH, h), sharey=True)
    rows = F.wsp_rows()
    for ax, size in zip(axs, ("medium", "large")):
        F.draw(ax, args.results, size, rows)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(30, 4e6)
        ax.set_ylim(0.012, 1.0)
        ax.set_xlabel("changed parameters")
        ax.tick_params(axis="x", which="major", pad=S.em(6))
        ax.text(0.96, 0.94, f"{size} model", transform=ax.transAxes, ha="right", va="top", fontsize=S.fs(-2))
    axs[0].set_ylabel("remaining response")
    fig.legend(F.handles(), F.LABELS, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2,
               columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=1 - S.em(0.62) / h, bottom=S.em(0.52) / h, wspace=0.06)
    S.save(fig, args.out, "app_frontier")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
