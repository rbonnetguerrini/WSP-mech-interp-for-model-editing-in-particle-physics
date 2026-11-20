#!/usr/bin/env python
"""Tab. comparison: the selected variant of each method on the small model (remaining response, W efficiency after
one photon at f = 1e-3, accuracy change, Phys. resp., changed parameters). WSP: median over the three runs of the
target informed variant (edit kept by scripts/evaluation/select_wsp_edits.py).

  python scripts/tables/comparison.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import _common as C  # noqa: E402
import select_wsp_edits as S  # noqa: E402

SIZE = "small"
FT = "finetuning_attention_only_both_neutral_types_seed0"
SAE = "sae_three_readers_b7-6_b4-0_c1-2"


def weff_note(name: str) -> tuple[str, str]:
    """W efficiency of the 10 000-jet sample if present, else the stored one (with its number of jets)."""
    for n in (f"{name}_10000_jets", name):
        f = C.rpath("w_efficiency", SIZE, f"{n}.json")
        if f.exists():
            d = C.read_json(f)
            return C.pct(d["wp80"]["f1e-3_W"]["eff"]), f"{d.get('n_per_class', '?')} jets per class"
    return "-", "missing"


def main() -> int:
    rows = [["Original model", "1", weff_note("original_model")[0], "-", "-", "-", ""]]
    s = C.score(SIZE, "removal_rule_on_input")
    p, _ = C.phys_resp(SIZE, "removal_rule_on_input")
    rows.append(["Input removal rule", C.f2(C.remaining_response(s)), weff_note("removal_rule_on_input")[0],
                 C.acc4(s["clean"]["d_acc"]), C.f2(p), C.count(C.changed_parameters(s)), ""])
    for label, name in (("Fine-tuning", FT), ("SAE edit", SAE)):
        s = C.score(SIZE, name)
        p, which = C.phys_resp(SIZE, name)
        w, note = weff_note(name)
        rows.append([label, C.f2(C.remaining_response(s)), w, C.acc4(s["clean"]["d_acc"]), C.f2(p),
                     C.count(C.changed_parameters(s)), f"W eff: {note}; phys: {which}"])
    sel = [x for x in S.select(S.all_edits()) if x["variant"] == "target_informed" and x["size"] == SIZE]
    med = lambda k: C.median_range([x[k] for x in sel])[0]  # noqa: E731
    weffs = [C.w_eff(SIZE, C.wsp_name("target_informed", x["run"], x["edit"])) for x in sel]
    have = [w for w in weffs if w is not None]
    rows.append(["WSP", C.f2(med("remaining_response")), C.pct(C.median_range(have)[0]) if have else "-",
                 C.acc4(med("delta_accuracy")), C.f2(med("phys_resp")), C.count(med("changed_parameters")),
                 f"median of 3 runs; W eff from {len(have)} run(s): "
                 + ", ".join("-" if w is None else f"{100 * w:.1f}%" for w in weffs)])
    C.print_table(["Method", "Remaining response", "W eff. (+1 photon)", "d acc (clean)", "Phys. resp.",
                   "Changed param.", "note"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
