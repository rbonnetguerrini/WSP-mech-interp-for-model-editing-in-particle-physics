#!/usr/bin/env python
"""Fig. ir, left panel (ir_response_neutral_hadron.pdf): mean |dF_W| of W jets for one added neutral hadron against its pT
(1 MeV - 10 GeV), for the original model and the models repaired by fine-tuning, the SAE edit and the selected WSP
edit; lower panel: difference to one added charged hadron of the same pT. Inputs and details: _ir.py.

  python scripts/figures/ir_response_neutral_hadron.py --results RESULTS [--out DIR] [--no-tex]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ir  # noqa: E402
import style as S  # noqa: E402


def main() -> int:
    args, plt = S.setup(__doc__)
    _ir.draw(plt, args.results, "neutral hadron", "ir_response_neutral_hadron", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
