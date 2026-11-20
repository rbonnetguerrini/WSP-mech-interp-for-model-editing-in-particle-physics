#!/usr/bin/env python
"""Tab. ft_variants: fine-tuning variants on the small model (remaining response, accuracy change, Phys. resp.,
changed parameters), from the cached scores in $RESULTS_DIR (scores/, scores_removal_rule/,
fragmentation_after_removal_rule/).

  python scripts/tables/ft_variants.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import _common as C  # noqa: E402

ROWS = (("Attention only, both neutral types (selected)", "finetuning_attention_only_both_neutral_types_seed0"),
        ("Attention only, photons only", "finetuning_attention_only_photons_only"),
        ("Whole body and classifier, both neutral types", "finetuning_whole_body_and_classifier_both_neutral_types"))


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
