#!/usr/bin/env python
"""App. E Fig. app_readers, large model, attention: mean attention of the summary tokens on one added photon at f = 1e-3, for every head of the backbone and of the classifier.

Input: figures/large/readers_attention.json (scripts/analysis/attention_heads.py --size l --classifier).

  python scripts/figures/app_readers_l_attention.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _readers  # noqa: E402
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    _readers.draw(plt, args.results, "l", "attention", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
