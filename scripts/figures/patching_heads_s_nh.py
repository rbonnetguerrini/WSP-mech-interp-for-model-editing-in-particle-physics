#!/usr/bin/env python
"""App. A Fig. attention right: share of |dF_W| removed when the output of one head is restored to its value on the
clean jet, for one added neutral hadron at f = 1e-3 (OmniLearned-s).

Input: figures/small/readers_patching.json, probe soft7_1e-3 (scripts/analysis/activation_patching.py).

  python scripts/figures/patching_heads_s_nh.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _heads  # noqa: E402
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    _heads.patching(plt, args.results, "neutral_hadron", "patching_heads_s_nh", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
