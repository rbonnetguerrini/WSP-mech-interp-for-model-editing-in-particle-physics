#!/usr/bin/env python
"""Score any edit of a pretrained OmniLearned model on the test jets (definitions: paper App. B).

An edit is an .npz (or .pt) partial state dict of full replacement tensors (core.load_candidate); tensors that are
not in the file keep their pretrained values. Without --edit the original model is scored.

Output JSON (all on the 20,480 test jets unless stated):
  remaining_response               mean of the 4 ratios below
  response_ratios                  r_a(edited) / r_a(original) for one photon / one neutral hadron at f = 1e-3, 1e-4
  w_efficiency                     W efficiency at the 80% working point after one photon at f = 1e-3
                                   (JetClass W / QCD test files, rows 60000.., --weff-jets per class)
  delta_accuracy                   change of the ten-class JetClass accuracy on unmodified jets
  response_to_physical_changes     rho for angular spread (D), momentum scale (J) and fragmentation after the
                                   removal rule (P), and the one furthest from one (largest |log rho|)
  changed_parameters               sum over changed tensors: min(N, r (m + n - r)) per matrix, changed entries otherwise
  remaining_error                  mean of e_v(edited) / e_v(original) over the sets of modified jets, against the
                                   target predictions f_0(removal rule(x))
  passes_limits                    the selection limits of App. C (accuracy, per-class accuracy, NLL, |log rho| <= 0.15)
  details                          the full outputs of scoring.score_loaded and removal_rule.rule_quantities

  python scripts/evaluation/score_edit.py --edit EDIT.npz --size s --out score.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edit", type=Path, default=None, help="edit file (.npz partial state dict); omit for the original")
    ap.add_argument("--size", default="s", choices=("s", "m", "l"), help="OmniLearned model size")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("--name", default=None, help="name recorded in the output (default: edit file stem)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weff-jets", type=int, default=10000, help="W and QCD jets for the W efficiency")
    ap.add_argument("--no-weff", action="store_true", help="skip the W efficiency")
    ap.add_argument("--no-remaining-error", action="store_true",
                    help="skip the remaining error (rho for fragmentation after the removal rule is then omitted too)")
    ap.add_argument("--generator", action="store_true", help="also measure the generator AUC change (slow)")
    ap.add_argument("--fused-attention", action="store_true",
                    help="use the fused attention kernels (default: math kernel, as for the stored scores)")
    args = ap.parse_args()

    os.environ["MODEL_SIZE"] = args.size
    from softrepair import core as W
    from softrepair import removal_rule as RR
    from softrepair import repair_eval as B
    from softrepair import scoring as S

    if not args.fused_attention:
        W.use_math_sdp()
    if args.size == "l" and not os.environ.get("SOFTREPAIR_MAX_BATCH"):
        os.environ["SOFTREPAIR_MAX_BATCH"] = "32"  # the large model does not fit 512 jets at once
    elif args.size == "m" and not os.environ.get("SOFTREPAIR_MAX_BATCH"):
        os.environ["SOFTREPAIR_MAX_BATCH"] = "128"
    if args.generator and not os.environ.get("SOFTREPAIR_GEN_SEEDS"):
        os.environ["SOFTREPAIR_GEN_SEEDS"] = "3"  # as for the stored scores (s and m)
    name = args.name or (args.edit.stem if args.edit else "original")
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)  # noqa: E731

    ad = W.load_adapter(args.device, args.size)
    orig = W.original_state(ad)
    ir = S.score_loaded(ad, orig, args.edit, name, gen=args.generator)
    log("response and side effects done")
    res = {"name": name, "edit": str(args.edit), "model_size": args.size,
           "remaining_response": ir["remaining_response"],
           "response_ratios": {S.RESPONSE_NAMES[k]: ir["ratio"][k] for k in S.R_SUB},
           "delta_accuracy": ir["clean"]["d_acc"],
           "class_drop": ir["clean"]["class_drop"],
           "delta_nll": ir["clean"]["d_nll"],
           "changed_parameters": ir["size"]["changed_parameters"]}
    rho = {"angular_spread": ir["rho"]["D"], "momentum_scale": ir["rho"]["J"],
           "fragmentation_without_removal": ir["rho"]["P"]}
    details = {"scoring": ir}
    if not args.no_remaining_error:
        rq = RR.rule_quantities(ad, orig, args.edit)
        log("remaining error done")
        rho["fragmentation"] = rq["rho_P_removal"]
        res["remaining_error"] = rq["rule"]["remaining_error"]
        res["remaining_error_held_out"] = rq["rule"].get("remaining_error_held_out")
        details["removal_rule"] = rq
        phys = {k: rho[k] for k in ("angular_spread", "momentum_scale", "fragmentation")}
        k = max(phys, key=lambda k: abs(math.log(phys[k])))
        rho["furthest_from_one"] = {"change": k, "rho": phys[k]}
        c = ir["clean"]
        res["passes_limits"] = bool(c["d_acc"] >= B.GATE["d_acc"] and c["class_drop"] <= B.GATE["class_drop"]
                                    and c["d_nll"] <= B.GATE["d_nll"]
                                    and abs(math.log(phys[k])) <= B.GATE["log_rho"])
    res["response_to_physical_changes"] = rho
    if args.generator:
        res["generator_dauc"] = ir.get("generator_dauc")
    if not args.no_weff:
        W.set_state(ad, W.load_candidate(args.edit, orig) if args.edit else orig)
        we = S.w_efficiency(ad, n=args.weff_jets, soft_pid=6)
        W.set_state(ad, orig)
        log("W efficiency done")
        w = we["wp80"]
        res["w_efficiency"] = {"working_point": 0.8, "photon_f1e-3": w["f1e-3_W"]["eff"],
                               "photon_f1e-3_err": w["f1e-3_W"]["err"], "nominal": w["nominal_W"]["eff"],
                               "qcd_mistag_nominal": w["nominal_QCD"]["eff"],
                               "qcd_mistag_photon_f1e-3": w["f1e-3_QCD"]["eff"], "jets_per_class": args.weff_jets}
        details["w_efficiency"] = we
    res["details"] = details
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    summary = {k: v for k, v in res.items() if k != "details"}
    print(json.dumps(summary, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
