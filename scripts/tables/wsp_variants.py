#!/usr/bin/env python
"""Tab. wsp_variants: weight-space programming variants on the small model, median [min, max] over the three runs
of the edit each run keeps (scripts/evaluation/select_wsp_edits.py: lowest remaining error among the edits that
pass the limits, without the generator limit). Reads the score files of the run records (runs/).

  python scripts/tables/wsp_variants.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import _common as C  # noqa: E402
import select_wsp_edits as S  # noqa: E402


def main() -> int:
    sel = S.select(S.all_edits())
    rows = []
    for v in S.VARIANTS:
        s = [x for x in sel if x["variant"] == v and x["size"] == "small" and x["edit"]]
        col = lambda k: [x[k] for x in s]  # noqa: E731
        rows.append([S.VARIANT_NAMES[v] + (" (selected)" if v == "target_informed" else ""),
                     C.cell(col("remaining_error"), C.f2), C.cell(col("remaining_response"), C.f2),
                     C.cell(col("delta_accuracy"), C.acc4), C.cell(col("phys_resp"), C.f2),
                     C.cell(col("changed_parameters"), C.count)])
    C.print_table(["Variant", "Remaining error", "Remaining response", "d acc (clean)", "Phys. resp.",
                   "Changed param."], rows)
    print("\nedits kept: " + ", ".join(f"{x['variant']} run{x['run']} {x['edit']}" for x in sel
                                       if x["size"] == "small"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
