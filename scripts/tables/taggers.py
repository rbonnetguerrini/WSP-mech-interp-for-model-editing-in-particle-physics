#!/usr/bin/env python
"""Tab. taggers: signal efficiency (%) of the nine JetClass taggers of the small model after one photon at
f = 1e-3, for the original model and the model repaired by the selected WSP edit (target informed, run 2,
edit_large). Each tagger's cut keeps 80% of its unmodified signal jets.

  python scripts/tables/taggers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import _common as C  # noqa: E402

ORDER = ("H->bb", "H->cc", "H->gg", "H->4q", "H->lvqq", "Z->qq", "W->qq", "t->bqq", "t->blv")
SELECTED_WSP = C.wsp_name("target_informed", 2, "edit_large")


def main() -> int:
    rows = []
    for label, name in (("Original", "original_model"), ("Repaired", SELECTED_WSP)):
        t = C.taggers("small", name, prefer_rerun=False)["taggers"]
        rows.append([label] + [f"{100 * t[c]['photon_f0.001']['eff_sig']:.0f}" for c in ORDER])
    C.print_table([""] + list(ORDER), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
