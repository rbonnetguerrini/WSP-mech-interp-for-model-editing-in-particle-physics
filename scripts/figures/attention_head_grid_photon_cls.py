#!/usr/bin/env python
"""Fig. 3 left: mean attention of the summary tokens on one added photon at f = 1e-3, for every head of the
backbone and of the classifier (OmniLearned-s).

Input: figures/small/readers_attention_photon.json (scripts/analysis/attention_heads.py --classifier).

  python scripts/figures/attention_head_grid_photon_cls.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _heads  # noqa: E402
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    _heads.attention(plt, args.results, "photon", "attention_head_grid_photon_cls", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
