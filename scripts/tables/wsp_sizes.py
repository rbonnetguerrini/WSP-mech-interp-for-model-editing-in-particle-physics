#!/usr/bin/env python
"""Tab. wsp_sizes: weight-space programming on the three model sizes, median [min, max] over three runs of the
remaining error, remaining response and changed parameters of the edit each run keeps
(scripts/evaluation/select_wsp_edits.py).

  python scripts/tables/wsp_sizes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import _common as C  # noqa: E402
import select_wsp_edits as S  # noqa: E402


def main() -> int:
    sel = S.select(S.all_edits())
    rows = []
    for size in ("small", "medium", "large"):
        for v in S.VARIANTS:
            s = [x for x in sel if x["variant"] == v and x["size"] == size and x["edit"]]
            col = lambda k: [x[k] for x in s]  # noqa: E731
            rows.append([size.capitalize() if v == S.VARIANTS[0] else "", S.VARIANT_NAMES[v],
                         C.cell(col("remaining_error"), C.f2), C.cell(col("remaining_response"), C.f2),
                         C.cell(col("changed_parameters"), C.count)])
    C.print_table(["Model", "Variant", "Remaining error", "Remaining response", "Changed param."], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
