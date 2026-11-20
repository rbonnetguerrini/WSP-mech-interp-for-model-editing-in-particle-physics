#!/usr/bin/env python
"""App. A Fig. attention left: mean attention of the four summary tokens on one added neutral hadron at f = 1e-3, for
every head of the eight backbone blocks (OmniLearned-s).

Input: figures/small/readers_attention_neutral_hadron_body.json (scripts/analysis/attention_heads.py --soft-pid 7).

  python scripts/figures/attention_head_grid_nh.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _heads  # noqa: E402
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    G, sites = S.grid(S.fig_dir(args.results) / "readers_attention_neutral_hadron_body.json")
    fig, ax = plt.subplots(figsize=(0.47 * S.TEXTWIDTH, S.H2))
    im = S.heat(plt, ax, G, sites, _heads.cmap(S.MAGENTA), G.max())
    ax.set_ylabel("backbone block")
    ax.set_title(S.tx(_heads.TITLE["neutral_hadron"]), fontsize=S.fs(-2), fontweight="normal")
    S.cbar(fig, ax, im, "mean attention")
    S.finish(fig, S.H2)
    S.save(fig, args.out, "attention_head_grid_nh")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
