#!/usr/bin/env python
"""Selection of the weight-space-programming edit of each run (paper App. C, "Scoring and selection").

Each run hands in up to three edits (edit_small, edit_medium, edit_large). An edit passes if, on the test jets,
  delta accuracy >= -0.010, the largest per-class accuracy drop <= 0.020, delta NLL <= 0.030, and
  |ln rho| <= 0.15 for the angular spread and momentum scale changes and for the fragmentation change after the
  removal rule.
The generator limit named in the task description is NOT used (App. C). Among the passing edits, the run keeps the
one with the lowest remaining error. Phys. resp. is the rho (angular spread, momentum scale, fragmentation after the
removal rule) furthest from one.

Inputs: the score files of the run records, runs/<variant>/<size>/run<k>/scores/edit_<e>.remaining_error.json
(removal-rule scorer: remaining error, rho for fragmentation after the removal rule) and
edit_<e>.remaining_response.json (remaining response, accuracy, NLL, rho for angular spread / momentum scale,
changed parameters).

  python scripts/evaluation/select_wsp_edits.py [--csv out.csv]     prints the selection per run and the
                                                                     median [min, max] per variant and size
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

VARIANTS = ("flaw_informed", "target_informed", "physics_origin_informed")
VARIANT_NAMES = {"flaw_informed": "Flaw informed", "target_informed": "Target informed",
                 "physics_origin_informed": "Physics origin informed"}
EDITS = ("small", "medium", "large")
LIMITS = {"d_acc": -0.010, "class_drop": 0.020, "d_nll": 0.030, "log_rho": 0.15}


def _get(d: dict, *keys):
    """First present key (the record files keep the key names of the code that wrote them)."""
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(keys)


def edit_row(run_dir: Path, edit: str) -> dict | None:
    fe = run_dir / "scores" / f"edit_{edit}.remaining_error.json"
    fr = run_dir / "scores" / f"edit_{edit}.remaining_response.json"
    if not (fe.exists() and fr.exists()):
        return None
    e, r = C.read_json(fe), C.read_json(fr)
    rule = e["rule"]
    c = r["clean"]
    rho = {"angular_spread": r["rho"]["D"], "momentum_scale": r["rho"]["J"],
           "fragmentation": _get(e["rho"], "P_reco", "P_removal")}
    k = max(rho, key=lambda k: abs(math.log(rho[k])))
    fails = []
    if c["d_acc"] < LIMITS["d_acc"]:
        fails.append("accuracy")
    if c["class_drop"] > LIMITS["class_drop"]:
        fails.append("class accuracy")
    if c["d_nll"] > LIMITS["d_nll"]:
        fails.append("NLL")
    fails += [f"rho {n}" for n, v in rho.items() if abs(math.log(v)) > LIMITS["log_rho"]]
    return {"edit": f"edit_{edit}", "remaining_error": _get(rule, "E_rule", "remaining_error"),
            "remaining_error_held_out": rule.get("E_rule_held", rule.get("remaining_error_held_out")),
            "remaining_response": _get(r, "R_sub", "remaining_response"), "delta_accuracy": c["d_acc"],
            "class_drop": c["class_drop"], "delta_nll": c["d_nll"], **{f"rho_{n}": v for n, v in rho.items()},
            "phys_resp": rho[k], "phys_resp_change": k,
            "changed_parameters": _get(r["size"], "dof", "changed_parameters"), "passes": not fails,
            "fails": ",".join(fails)}


def all_edits(runs: Path | None = None) -> list[dict]:
    runs = runs or C.runs_root()
    rows = []
    for v in VARIANTS:
        for size in ("small", "medium", "large"):
            for k in (1, 2, 3):
                for e in EDITS:
                    x = edit_row(runs / v / size / f"run{k}", e)
                    if x:
                        rows.append({"variant": v, "size": size, "run": k, **x})
    return rows


def select(rows: list[dict]) -> list[dict]:
    """Per run: the passing edit with the lowest remaining error (None if no edit passes)."""
    out = []
    for v in VARIANTS:
        for size in ("small", "medium", "large"):
            for k in (1, 2, 3):
                ok = [x for x in rows if (x["variant"], x["size"], x["run"]) == (v, size, k) and x["passes"]]
                out.append(min(ok, key=lambda x: x["remaining_error"]) if ok else
                           {"variant": v, "size": size, "run": k, "edit": None})
    return out


COLUMNS = ("remaining_error", "remaining_response", "delta_accuracy", "phys_resp", "changed_parameters")


def summary(sel: list[dict]) -> dict:
    """(variant, size) -> column -> (median, min, max) over the three runs."""
    out = {}
    for v in VARIANTS:
        for size in ("small", "medium", "large"):
            s = [x for x in sel if x["variant"] == v and x["size"] == size and x["edit"]]
            out[(v, size)] = {c: (st.median([x[c] for x in s]), min(x[c] for x in s), max(x[c] for x in s))
                              for c in COLUMNS} if s else {}
            out[(v, size)]["runs"] = len(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=None, help="run records (default: runs/ of this repository)")
    ap.add_argument("--csv", type=Path, default=None, help="write every edit with its pass flag and selection")
    a = ap.parse_args()
    rows = all_edits(a.runs)
    sel = select(rows)
    chosen = {(x["variant"], x["size"], x["run"], x["edit"]) for x in sel}
    for x in rows:
        x["selected"] = (x["variant"], x["size"], x["run"], x["edit"]) in chosen
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{'variant':24s} {'size':6s} run  {'selected':11s} {'rem.err':>7s} {'rem.resp':>8s} {'d acc':>8s} "
          f"{'phys':>6s} {'changed':>9s}")
    for x in sel:
        if not x["edit"]:
            print(f"{x['variant']:24s} {x['size']:6s} {x['run']:3d}  none passes")
            continue
        print(f"{x['variant']:24s} {x['size']:6s} {x['run']:3d}  {x['edit']:11s} {x['remaining_error']:7.3f} "
              f"{x['remaining_response']:8.3f} {x['delta_accuracy']:+8.4f} {x['phys_resp']:6.3f} "
              f"{x['changed_parameters']:9d}")
    print("\nmedian [min, max] over the three runs")
    for (v, size), d in summary(sel).items():
        if d["runs"]:
            print(f"{VARIANT_NAMES[v]:24s} {size:6s} " + "  ".join(
                f"{c} {m:.4g} [{lo:.4g}, {hi:.4g}]" for c, (m, lo, hi) in ((c, d[c]) for c in COLUMNS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
