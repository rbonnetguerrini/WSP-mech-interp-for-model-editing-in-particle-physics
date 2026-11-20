#!/usr/bin/env python
"""App. A Fig. app_types left: mean change of F_W for W and QCD jets when one neutral hadron is added, against its pT; grey: the pT
distribution of the neutral hadrons in the clean JetClass jets; dotted line: its first percentile.

Input: figures/small/soft_particle_scans/original_model.json / .npz (scripts/analysis/soft_particle_scan.py).

  python scripts/figures/shift_vs_neutral_hadron_pt.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _scan  # noqa: E402
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    _scan.draw(plt, args.results, "neutral hadron", "shift_vs_neutral_hadron_pt", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
