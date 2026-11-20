#!/usr/bin/env python
"""App. E Fig. app_readers, large model, patching: share of |dF_W| removed when the output of one head is restored to its value on the clean jet, for one added photon at 0.3 GeV.

Input: figures/large/readers_patching.json, probe photon_0.3GeV (scripts/analysis/activation_patching.py --size l).

  python scripts/figures/app_readers_l_patching.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _readers  # noqa: E402
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    _readers.draw(plt, args.results, "l", "patching", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
