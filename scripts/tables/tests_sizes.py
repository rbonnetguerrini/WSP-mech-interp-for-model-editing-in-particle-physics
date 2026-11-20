#!/usr/bin/env python
"""Tab. tests_sizes: the tests of Sec. 4 for the three methods and model sizes, after one added photon.

  Nine taggers efficiency  min-max over the nine taggers (photon at f = 1e-3); WSP: over the three runs
  JetClass-II probability  W jets (photon at f = 1e-3); WSP: median over the three runs
  Regression head [GeV]    mean |change| of the predicted jet mass (ridge head, photon at f = 1e-3)
  Generator update         relative change of the generator update for W jets at t = 0.05 (photon of 0.3 GeV)
  Control                  original model with a photon of 3 GeV
Cells with a range: median [min, max] over the three fine-tuning seeds or SAE seeds (small model) or the three WSP
runs; the taggers and JetClass-II columns of fine-tuning and the SAE edit use the selected edit. The regression
column uses the re-run with the published regression heads (results/regression_head/) where it exists.

  python scripts/tables/tests_sizes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import _common as C  # noqa: E402

FT = "finetuning_attention_only_both_neutral_types_seed{}"
SAE = {"small": "sae_three_readers_b7-6_b4-0_c1-2", "medium": "sae_three_readers_b7-0_b11-7_b11-0",
       "large": "sae_four_readers_b19-11_b18-17_c0-2_c3-24"}
PH = "photon_f0.001"


def tag_range(names, size):
    e = [x[PH]["eff_sig"] for n in names for x in C.taggers(size, n, prefer_rerun=False)["taggers"].values()]
    return f"{100 * min(e):.0f}-{100 * max(e):.0f}%"


def jetclass2_probability(names, size):
    return C.f2(C.median_range([C.taggers(size, n, prefer_rerun=False)["jetclass2_share"][PH]["W->qq"]
                                for n in names])[0])


def regression(names, size, which=PH):
    v = [C.taggers(size, n)["mass_head"]["ridge"][which]["mean_abs_dm_pred_GeV"] for n in names]
    return C.cell(v, C.f1) if len(v) > 1 else C.f1(v[0])


def generator(names, size, which="photon_0.3GeV"):
    v = [100 * C.generator_update(size, n, which) for n in names]
    fmt = lambda x: f"{x:.1f}"  # noqa: E731
    return (C.cell(v, fmt) if len(v) > 1 else fmt(v[0])) + "%"


def main() -> int:
    rows = []
    for size in ("small", "medium", "large"):
        ft_sel = [FT.format(0)]
        ft = [FT.format(k) for k in (0, 1, 2)] if size == "small" else ft_sel
        sae = [SAE[size]]
        sae_gen = [f"{SAE[size]}_sae_seed{k}" for k in (0, 1, 2)] if size == "small" else sae
        wsp = [C.wsp_name("target_informed", k) for k in (1, 2, 3)]
        o = ["original_model"]
        rows += [[size.capitalize(), "Original model", tag_range(o, size), jetclass2_probability(o, size), regression(o, size),
                  generator(o, size)],
                 ["", "Fine-tuning", tag_range(ft_sel, size), jetclass2_probability(ft_sel, size), regression(ft, size),
                  generator(ft, size)],
                 ["", "SAE edit", tag_range(sae, size), jetclass2_probability(sae, size), regression(sae, size),
                  generator(sae_gen, size)],
                 ["", "WSP", tag_range(wsp, size), jetclass2_probability(wsp, size), regression(wsp, size), generator(wsp, size)],
                 ["", "Control", "-", "-", regression(o, size, "photon_3GeV (control)"),
                  generator(o, size, "photon_3GeV_control")]]
    C.print_table(["Model", "Method", "Nine taggers efficiency", "JetClass-II probability", "Regression head [GeV]",
                   "Generator update"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
