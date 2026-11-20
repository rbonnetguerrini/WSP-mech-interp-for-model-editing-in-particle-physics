#!/usr/bin/env python
"""Tab. comparison_sizes: the comparison of Tab. comparison for the medium and large models. WSP: median over the
three runs of the target informed variant. The W efficiency of the SAE edits on these sizes is taken from the tagger
test (W tagger, photon at f = 1e-3) when no W-efficiency file exists (marked in the note column).

  python scripts/tables/comparison_sizes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import _common as C  # noqa: E402
import select_wsp_edits as S  # noqa: E402

FT = "finetuning_attention_only_both_neutral_types_seed0"
SAE = {"medium": "sae_three_readers_b7-0_b11-7_b11-0", "large": "sae_four_readers_b19-11_b18-17_c0-2_c3-24"}


def weff(size: str, name: str) -> tuple[str, str]:
    w = C.w_eff(size, name)
    if w is not None:
        return C.pct(w), ""
    t = C.taggers(size, name, prefer_rerun=False)["taggers"]["W->qq"]["photon_f0.001"]["eff_sig"]
    return C.pct(t), "W eff from the tagger test"


def main() -> int:
    sel = S.select(S.all_edits())
    rows = []
    for size in ("medium", "large"):
        rows.append([size.capitalize(), "Original model", "1", weff(size, "original_model")[0], "-", "-", "-", ""])
        for label, name in (("Fine-tuning", FT), ("SAE edit", SAE[size])):
            s = C.score(size, name)
            p, which = C.phys_resp(size, name)
            w, note = weff(size, name)
            rows.append(["", label, C.f2(C.remaining_response(s)), w, C.acc4(s["clean"]["d_acc"]), C.f2(p),
                         C.count(C.changed_parameters(s)), "; ".join(x for x in (note, f"phys: {which}") if x)])
        t = [x for x in sel if x["variant"] == "target_informed" and x["size"] == size and x["edit"]]
        med = lambda k: C.median_range([x[k] for x in t])[0]  # noqa: E731
        ws = [w for w in (C.w_eff(size, C.wsp_name("target_informed", x["run"], x["edit"])) for x in t) if w]
        rows.append(["", "WSP", C.f2(med("remaining_response")), C.pct(C.median_range(ws)[0]) if ws else "-",
                     C.acc4(med("delta_accuracy")), C.f2(med("phys_resp")), C.count(med("changed_parameters")),
                     f"median of {len(t)} runs (W eff of {len(ws)})"])
    C.print_table(["Model", "Method", "Remaining response", "W eff. (+1 photon)", "d acc (clean)", "Phys. resp.",
                   "Changed param.", "note"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
