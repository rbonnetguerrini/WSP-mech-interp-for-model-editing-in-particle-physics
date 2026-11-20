#!/usr/bin/env python
"""The removal rule and the remaining error: distance of an edited model to the target predictions.

The removal rule (core.reco_fix, thresholds only): a photon (pid 6) with pT < 0.80 GeV and E < 1.80 GeV, or a neutral
hadron (pid 7) with E < 2.20 GeV, is never reconstructed by JetClass's detector simulation; drop it. Every real
JetClass particle lies above these thresholds, so clean jets are unchanged.

Target predictions: f_target(x) = f_0(rule(x)), f_0 = original weights. An edit f is a set of weights only.
  e_v(f)                   = mean over valid rows ||c_f(x_v) - c_0(rule(x_v))|| / S_0    (c: centred 10 JetClass logits)
  remaining_error          = mean over RULE_VIS of e_v(f) / e_v(f_0)    (0: rule reproduced exactly; 1: original model)
  remaining_error_held_out = the same over RULE_HELD
  e_clean, e_split2, e_soft3e-3 are reported (the rule leaves these inputs (almost) unchanged).
Limits: those of repair_eval.GATE, with the fragmentation response measured on P_removal = rule(P x), the softer
fragmentation as JetClass's detector would record it; rho_P on the raw P is reported only.
The rule itself (f_0 o rule) passes these limits by construction (rho_P_removal = 1.000).

  python -m softrepair.removal_rule score --cand X.npz --name N [--no-gen]   -> RESULTS_DIR/removal_rule/score/N.json
  python -m softrepair.removal_rule score --reco --name rule                 (the rule as preprocessing, the reference)
  python -m softrepair.removal_rule extra --cand X.npz --base <repair_eval score json> --name N
      adds only the removal-rule quantities to an existing repair_eval score (same code path)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from softrepair import core as W
from softrepair import repair_eval as B

RULE_VIS = ("soft1e-3", "soft1e-4", "P")
RULE_HELD = ("soft7_1e-3",)
REPORT = ("clean", "split2", "soft3e-3")
PHYS_RULE = ("D", "J", "P_removal")


def apply(raw: np.ndarray) -> np.ndarray:
    """The removal rule on a (n, 150, 9) jet array (thresholds only)."""
    return W.reco_fix(raw)


def with_reco(data: dict, variants: tuple[str, ...]) -> dict:
    """Add P_removal (and its ok mask) and the rule's copy of every variant ('reco:<v>')."""
    out = dict(data)
    out["P_removal"] = W.reco_fix(np.asarray(data["P"]))
    out["P_removal:ok"] = data["P:ok"]
    for v in variants:
        out["reco:" + v] = W.reco_fix(np.asarray(data[v]))
    return out


def targets(ad, orig: dict, data: dict, variants: tuple[str, ...]) -> dict:
    """Original model's logits on the rule's inputs (the target predictions) and on the raw inputs."""
    W.set_state(ad, orig)
    T = {v: W.logits(ad, np.asarray(data["reco:" + v])) for v in variants}
    Y0 = {v: W.logits(ad, np.asarray(data[v])) for v in variants}
    return {"T": T, "Y0": Y0}


def rule_metrics(Y: dict, ref: dict, data: dict, spread0: float, variants: tuple[str, ...]) -> dict:
    e, e0 = {}, {}
    for v in variants:
        ok = data[v + ":ok"] if v != "clean" else np.ones(len(data["labels"]), bool)
        d = np.linalg.norm(W.centred10(Y[v]) - W.centred10(ref["T"][v]), axis=1) / spread0
        d0 = np.linalg.norm(W.centred10(ref["Y0"][v]) - W.centred10(ref["T"][v]), axis=1) / spread0
        e[v], e0[v] = float(d[ok].mean()), float(d0[ok].mean())
    res = {"e": e, "e_orig": e0}
    res["remaining_error"] = float(np.mean([e[v] / e0[v] for v in RULE_VIS if v in e]))
    if all(v in e for v in RULE_HELD):
        res["remaining_error_held_out"] = float(np.mean([e[v] / e0[v] for v in RULE_HELD]))
    return res


def limits_removal(res: dict) -> dict:
    """repair_eval limits with rho_P_removal in place of rho_P."""
    rho = res["rho"]
    lr = max(abs(np.log(rho[k])) for k in PHYS_RULE)
    c = res["clean"]
    ok = bool(c["d_acc"] >= B.GATE["d_acc"] and c["class_drop"] <= B.GATE["class_drop"]
              and c["d_nll"] <= B.GATE["d_nll"] and lr <= B.GATE["log_rho"])
    return {"max_log_rho_removal": float(lr), "passes_limits_removal_no_generator": ok}


def surgical(ad, state: dict, orig: dict, practice: dict, cache: dict) -> dict:
    """Evaluation-tool reply on the practice jets: repair metrics + P_removal + remaining errors."""
    variants = B.VIS + B.PHYS + ("P_removal",)
    rv = B.VIS + ("P",)  # errors on every probe and on P
    if "data" not in cache:
        cache["data"] = with_reco(practice, rv + ("clean",))
        W.set_state(ad, orig)
        cache["Y0"] = B.raw_eval(ad, cache["data"], variants)
        cache["ref"] = targets(ad, orig, cache["data"], rv + ("clean",))
        cache["spread0"] = W.clean_metrics(cache["Y0"]["clean"], practice["labels"])["spread"]
    data = cache["data"]
    W.set_state(ad, state)
    Y = B.raw_eval(ad, data, variants)
    res = B.metrics({k: v for k, v in Y.items() if k != "P_removal"}, cache["Y0"], data, B.VIS)
    a = np.linalg.norm(W.centred10(Y["P_removal"]) - W.centred10(Y["clean"]), axis=1).mean()
    b = np.linalg.norm(W.centred10(cache["Y0"]["P_removal"]) - W.centred10(cache["Y0"]["clean"]), axis=1).mean()
    res["rho"]["P_removal"] = float(a / b)
    res.update(limits_removal(res))
    res["rule"] = rule_metrics(Y, cache["ref"], data, cache["spread0"], rv + ("clean",))
    return res


def rule_quantities(ad, orig: dict, cand: Path | None, reco: bool = False) -> dict:
    """rho_P_removal and the remaining error of one edit on the test jets (original weights restored after).

    Returns {"rho_P_removal": float, "rule": rule_metrics(...)}. reco=True scores the rule itself as preprocessing.
    """
    test = W.load_set("test", B.VIS + B.HELD + B.PHYS)
    rv = B.VIS + ("P",) + RULE_HELD + ("clean",)
    data = with_reco(test, rv)
    ref = targets(ad, orig, data, rv)
    W.set_state(ad, orig)
    Y0c = W.logits(ad, np.asarray(data["clean"]))
    Y0p = W.logits(ad, np.asarray(data["P_removal"]))
    spread0 = W.clean_metrics(Y0c, test["labels"])["spread"]
    state = W.load_candidate(cand, orig) if cand else dict(orig)
    W.set_state(ad, state)
    tf = (lambda a: W.reco_fix(a)) if reco else (lambda a: a)  # noqa: E731
    Y = {v: W.logits(ad, tf(np.asarray(data[v]))) for v in rv + ("P_removal",)}
    W.set_state(ad, orig)
    a = np.linalg.norm(W.centred10(Y["P_removal"]) - W.centred10(Y["clean"]), axis=1).mean()
    b = np.linalg.norm(W.centred10(Y0p) - W.centred10(Y0c), axis=1).mean()
    return {"rho_P_removal": float(a / b), "rule": rule_metrics(Y, ref, data, spread0, rv)}


def score(cand: Path | None, name: str, gen: bool, reco: bool) -> dict:
    """Test-set score: repair_eval.score (probes, limits, size, swap-back, generator) + P_removal + remaining error."""
    ad = W.load_adapter(os.environ.get("SOFTREPAIR_DEVICE", "cuda"))
    orig = W.original_state(ad)
    base = B.score(cand, f"removal_rule_{name}", gen=gen and not reco, reco={"merge_dr": 0.0} if reco else None)
    q = rule_quantities(ad, orig, cand, reco)
    res = dict(base)
    res["rho"]["P_removal"] = q["rho_P_removal"]
    res.update(limits_removal(res))
    res["rule"] = q["rule"]
    res["passes_limits_removal"] = bool(res["passes_limits_removal_no_generator"]
                                        and res.get("generator_dauc", 0.0) <= B.GATE["generator_dauc"])
    out = W.OUT / "removal_rule" / "score"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    r = res["rule"]
    print(f"[removal_rule] {name}: remaining_error {r['remaining_error']:.3f} "
          f"held_out {r.get('remaining_error_held_out', float('nan')):.3f} "
          f"response_probes {res['response_probes']:.3f} rho_P_removal {res['rho']['P_removal']:.3f} "
          f"rho_P {res['rho']['P']:.3f} changed_parameters {res['size']['changed_parameters']} "
          f"generator {res.get('generator_dauc', float('nan')):+.3f} passes {res['passes_limits_removal']}", flush=True)
    return res


def extra(cand: Path, base_path: Path, name: str) -> dict:
    """Add the removal-rule quantities to an existing repair_eval score of the same edit, without recomputing it."""
    base = json.loads(Path(base_path).read_text())
    ad = W.load_adapter(os.environ.get("SOFTREPAIR_DEVICE", "cuda"))
    orig = W.original_state(ad)
    q = rule_quantities(ad, orig, cand)
    res = dict(base)
    res["rho"] = dict(base["rho"])
    res["rho"]["P_removal"] = q["rho_P_removal"]
    res.update(limits_removal(res))
    res["rule"] = q["rule"]
    gen_dauc = res.get("generator_dauc")
    res["generator_measured"] = gen_dauc is not None and W.model_size() != "l"
    res["passes_limits_removal"] = bool(res["passes_limits_removal_no_generator"]
                                        and (not res["generator_measured"] or gen_dauc <= B.GATE["generator_dauc"]))
    res["source"] = {"base": str(base_path), "cand": str(cand)}
    out = W.OUT / "removal_rule" / "score"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    r = res["rule"]
    print(f"[removal_rule extra] {name}: remaining_error {r['remaining_error']:.3f} "
          f"rho_P_removal {res['rho']['P_removal']:.3f} rho_P {res['rho']['P']:.3f} "
          f"passes {res['passes_limits_removal']}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("score", "extra"))
    ap.add_argument("--cand", type=Path, default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", type=Path, default=None, help="extra: existing repair_eval score JSON of the same edit")
    ap.add_argument("--reco", action="store_true", help="score the rule itself as preprocessing (reference)")
    ap.add_argument("--no-gen", action="store_true")
    args = ap.parse_args()
    if args.mode == "extra":
        if args.cand is None or args.base is None:
            ap.error("extra needs --cand and --base")
        extra(args.cand, args.base, args.name)
    else:
        score(args.cand, args.name, gen=not args.no_gen, reco=args.reco)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
