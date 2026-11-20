"""Shared code of repair_vs_size (small model) and app_frontier (medium, large): remaining response against the
number of changed parameters for every repair variant. WSP: every edit handed in by every run (score files of the run
records, runs/); fine-tuning and SAE edits: results/scores/<size>/. The selected variant of each method has a black
outline (for WSP the edit of Tab. comparison / comparison_sizes named in SELECTED_WSP)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import select_wsp_edits as SW  # noqa: E402
import style as S  # noqa: E402

MARKERS = {"flaw_informed": "v", "target_informed": "o", "physics_origin_informed": "^"}
LABELS = ["WSP, flaw informed", "WSP, target informed", "WSP, physics origin informed", "SAE edit", "fine-tuning",
          "selected variant"]
# (variant, run, edit) outlined as the selected WSP point of each size
SELECTED_WSP = {"small": ("target_informed", 2, "edit_large"), "medium": ("target_informed", 1, "edit_large"),
                "large": ("target_informed", 1, "edit_large")}
FT = {"small": [("finetuning_attention_only_both_neutral_types_seed0", True),
                ("finetuning_attention_only_photons_only", False),
                ("finetuning_whole_body_and_classifier_both_neutral_types", False)],
      "medium": [("finetuning_attention_only_both_neutral_types_seed0", True),
                 ("finetuning_attention_only_photons_only", False),
                 ("finetuning_whole_body_and_classifier_photons_only", False)],
      "large": [("finetuning_attention_only_both_neutral_types_seed0", True)]}
SAE = {"small": [("sae_one_reader_b7-6", False), ("sae_two_readers_b7-6_b4-0", False),
                 ("sae_three_readers_b7-6_b4-0_c1-2", True), ("sae_three_readers_b7-6_b4-0_c1-2_half_strength", False)],
       "medium": [("sae_one_reader_b7-0", False), ("sae_three_readers_b7-0_b11-7_b11-0", True)],
       "large": [("sae_one_reader_b19-11", False), ("sae_two_readers_b19-11_b18-17", False),
                 ("sae_four_readers_b19-11_b18-17_c0-2_c3-24", True)]}


def score_point(results: Path, size: str, name: str) -> tuple[int, float]:
    d = json.loads((results / "scores" / size / f"{name}.json").read_text())
    s = d["size"]
    return int(s.get("changed_parameters", s.get("dof"))), float(d.get("remaining_response", d.get("R_sub")))


def draw(ax, results: Path, size: str, wsp_rows: list) -> None:
    sel = SELECTED_WSP[size]
    for x in wsp_rows:
        if x["size"] != size:
            continue
        is_sel = (x["variant"], x["run"], x["edit"]) == sel
        ax.scatter(x["changed_parameters"], x["remaining_response"], s=30 if is_sel else 20,
                   marker=MARKERS[x["variant"]], color=S.BLUE, alpha=1 if is_sel else 0.75,
                   edgecolor="k" if is_sel else "white", lw=1.0 if is_sel else 0.4, zorder=4 if is_sel else 3)
    for lst, mk, col, sz in ((SAE[size], "s", S.ORANGE, 22), (FT[size], "D", S.MAGENTA, 24)):
        for name, is_sel in lst:
            d, v = score_point(results, size, name)
            ax.scatter(d, v, s=sz * (1.4 if is_sel else 1), marker=mk, color=col, edgecolor="k" if is_sel else "white",
                       lw=1.0 if is_sel else 0.4, zorder=4 if is_sel else 3)
            print(f"{size} {name}: {d} changed parameters, remaining response {v:.3f}", flush=True)


def wsp_rows() -> list:
    return SW.all_edits()


def handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker="v", ls="", color=S.BLUE, ms=4.5), Line2D([], [], marker="o", ls="", color=S.BLUE, ms=4.5),
            Line2D([], [], marker="^", ls="", color=S.BLUE, ms=4.5), Line2D([], [], marker="s", ls="", color=S.ORANGE, ms=4.5),
            Line2D([], [], marker="D", ls="", color=S.MAGENTA, ms=4.5),
            Line2D([], [], marker="o", ls="", mfc="none", mec="k", mew=1.0, ms=5)]
