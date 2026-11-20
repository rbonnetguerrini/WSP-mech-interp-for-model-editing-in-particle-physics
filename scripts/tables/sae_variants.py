#!/usr/bin/env python
"""Tab. sae_variants: SAE-guided edits on the small model (one, two, three readers; three readers at half strength).

  python scripts/tables/sae_variants.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import _common as C  # noqa: E402

ROWS = (("One reader", "sae_one_reader_b7-6"),
        ("Two readers", "sae_two_readers_b7-6_b4-0"),
        ("Three readers (selected)", "sae_three_readers_b7-6_b4-0_c1-2"),
        ("Three readers, half strength", "sae_three_readers_b7-6_b4-0_c1-2_half_strength"))


def main() -> int:
    rows = []
    for label, name in ROWS:
        s = C.score("small", name)
        p, which = C.phys_resp("small", name)
        rows.append([label, C.f2(C.remaining_response(s)), C.acc4(s["clean"]["d_acc"]), f"{C.f2(p)} ({which})",
                     C.count(C.changed_parameters(s))])
    C.print_table(["Variant", "Remaining response", "d acc (clean)", "Phys. resp.", "Changed param."], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
