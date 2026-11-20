#!/usr/bin/env python
"""Shared repair metrics for every method (agent edits, fine-tuning, SAE edits) on the practice and test jets.

Probes: r_P for split2, soft3e-3, soft1e-3 and soft1e-4 (PROBES); held out soft7_1e-3, split3, split2c (HELD).
  response_probes / response_held_out = mean over probes of r_P(edited) / r_P(original).
Side effects (must stay small):
  clean      d_acc, max per-class accuracy drop, d_nll, mean KL(orig || edited) of the 10-class softmax
  physics    response ratio rho_op = mean ||c(op x) - c(x)||_edited / same for the original, for three
             physical changes from softrepair.ops at theta = 1: D (angles x1.105), J (pT and E x1.05),
             P (fragmentation exponent x1.105); the model must keep reacting to these
  generator  velocity drift (cheap) and, at final scoring, the real-vs-generated AUC change
Size: weights changed, the numerical rank of each 2-D tensor's change, and the changed parameters
  min(N, r (m + n - r)) per matrix (N changed entries, rank r), changed entries for other tensors.
Localisation (swap-back): revert one tensor group to the original and see how much of the repair is lost.

  python -m softrepair.repair_eval build                     add soft1e-3, D, J, P to the practice and test sets
  python -m softrepair.repair_eval score --cand X --name N   test-set score (RESULTS_DIR/repair_eval/score/N.json)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from softrepair import core as W

VIS = ("split2", "soft3e-3", "soft1e-3", "soft1e-4")
HELD = ("soft7_1e-3", "split3", "split2c")
PHYS = ("D", "J", "P")
GATE = {"d_acc": -0.010, "class_drop": 0.020, "d_nll": 0.030, "log_rho": 0.15, "generator_dauc": 0.030}
MAX_RANK = 8
RANK_TOL = 1e-4  # numerical rank: singular values above 1e-4 x the largest
GROUPS = {
    "input": ("body.token", "body.embed.", "body.interaction.", "body.local_physics.", "body.pid_embed.",
              "body.add_embed.", "body.time_embed.", "body.cond_embed."),
    "body_attn": (".attn.",),
    "body_mlp": (".mlp.",),
    "classifier": ("classifier.",),
}


def group_of(name: str) -> str:
    if name.startswith("generator."):
        return "generator"
    if name.startswith("classifier."):
        return "classifier"
    if name.startswith(GROUPS["input"]):
        return "input"
    if ".attn." in name:
        return "body_attn"
    if ".mlp." in name:
        return "body_mlp"
    return "body_other"


def phys_op(raw: np.ndarray, op: str) -> np.ndarray:
    from softrepair import ops
    t = torch.as_tensor(np.asarray(raw, np.float32))
    eta = torch.as_tensor(ops.jet_eta(np.asarray(raw, np.float32)))
    return ops.apply(t, eta, op, 1.0).numpy()


def build() -> None:
    """Add soft1e-3 and the physical changes D, J, P to the practice and test sets (fixed seeds)."""
    for name in ("practice", "test"):
        d = W.set_dir(name)
        meta = json.loads((d / "meta.json").read_text())
        raw = np.load(d / "clean.npy")
        for k, v in enumerate(("soft1e-3",) + PHYS):
            if v in meta["variants"]:
                continue
            if v in PHYS:
                p, ok = phys_op(raw, v), np.ones(len(raw), bool)
            else:
                p, ok = W.perturb(raw, v, 5000 + 100 * ("practice", "test").index(name) + k)
            np.save(d / f"{v}.npy", p)
            np.save(d / f"{v}.ok.npy", ok)
            meta["variants"].append(v)
            meta["valid"][v] = int(ok.sum())
            print(f"[{name}] {v}: {ok.sum()}/{len(ok)}", flush=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))


def softmax10(y: np.ndarray) -> np.ndarray:
    l = y[:, 2:12] - y[:, 2:12].max(1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(1, keepdims=True)


def raw_eval(ad, data: dict, variants: tuple[str, ...]) -> dict:
    """Logits of every variant (float64) for the current weights."""
    out = {"clean": W.logits(ad, np.asarray(data["clean"]))}
    for v in variants:
        out[v] = W.logits(ad, np.asarray(data[v]))
    return out


def metrics(Y: dict, Y0: dict, data: dict, probes: tuple[str, ...]) -> dict:
    """Repair metrics of logits Y against the original's logits Y0 on the same inputs."""
    lab = data["labels"]
    cm, cm0 = W.clean_metrics(Y["clean"], lab), W.clean_metrics(Y0["clean"], lab)
    p0, p1 = softmax10(Y0["clean"]), softmax10(Y["clean"])
    res = {"clean": {"acc": cm["acc"], "d_acc": cm["acc"] - cm0["acc"], "d_nll": cm["nll"] - cm0["nll"],
                     "class_drop": max(cm0["acc_per_class"][k] - cm["acc_per_class"][k] for k in cm["acc_per_class"]),
                     "kl": float((p0 * (np.log(p0 + 1e-12) - np.log(p1 + 1e-12))).sum(1).mean())}}
    res["r"], ratio = {}, {}
    for v in probes:
        r = W.irc_metrics(Y["clean"], Y[v], data[v + ":ok"], cm["spread"], lab)["r"]
        r0 = W.irc_metrics(Y0["clean"], Y0[v], data[v + ":ok"], cm0["spread"], lab)["r"]
        res["r"][v], ratio[v] = r, r / r0
    res["response_probes"] = float(np.mean([ratio[v] for v in VIS if v in ratio]))
    if all(v in ratio for v in VIS):  # in-range probes vs probes at/below the photon pT floor
        res["response_in_range"] = float(np.mean([ratio["split2"], ratio["soft3e-3"]]))
        res["response_sub_threshold"] = float(np.mean([ratio["soft1e-3"], ratio["soft1e-4"]]))
    if all(v in ratio for v in HELD):
        res["response_held_out"] = float(np.mean([ratio[v] for v in HELD]))
    res["ratio"] = ratio
    res["rho"] = {}
    for op in PHYS:
        if op in Y:
            a = np.linalg.norm(W.centred10(Y[op]) - W.centred10(Y["clean"]), axis=1).mean()
            b = np.linalg.norm(W.centred10(Y0[op]) - W.centred10(Y0["clean"]), axis=1).mean()
            res["rho"][op] = float(a / b)
            dn = W.centred10(Y[op]) - W.centred10(Y["clean"])
            do = W.centred10(Y0[op]) - W.centred10(Y0["clean"])
            res.setdefault("dir_err", {})[op] = float(np.linalg.norm(dn - do, axis=1).mean() / b)  # reported only
    res["max_log_rho"] = float(max(abs(np.log(x)) for x in res["rho"].values())) if res["rho"] else None
    c = res["clean"]
    res["passes_limits_no_generator"] = bool(
        c["d_acc"] >= GATE["d_acc"] and c["class_drop"] <= GATE["class_drop"] and c["d_nll"] <= GATE["d_nll"]
        and (res["max_log_rho"] is None or res["max_log_rho"] <= GATE["log_rho"]))
    return res


def summary_drift(ad, state: dict, orig: dict, data: dict, n: int = 4096) -> float:
    """Relative change of the body.norm summary tokens on clean test jets (reported, not a limit)."""
    raw = np.asarray(data["clean"][:n])
    W.set_state(ad, orig)
    _, a = W.logits(ad, raw, body=True)
    W.set_state(ad, state)
    _, b = W.logits(ad, raw, body=True)
    return float(np.linalg.norm(b - a) / np.linalg.norm(a))


def edit_size(state: dict, orig: dict) -> dict:
    """Changed entries, numerical rank and changed parameters of every changed tensor, and their totals."""
    per, n, worst, n_params = {}, 0, 0, 0
    for k, v in orig.items():
        d = state[k].double() - v.double()
        nz = int((d != 0).sum())
        if not nz:
            continue
        rank = None
        if d.ndim == 2:
            s = torch.linalg.svdvals(d)
            rank = int((s > RANK_TOL * max(float(s[0]), 1e-30)).sum())  # float32 rounding noise sits ~1e-6 below
            worst = max(worst, rank)
        k_params = nz if rank is None else min(nz, rank * (d.shape[0] + d.shape[1] - rank))
        per[k] = {"changed": nz, "rank": rank, "changed_parameters": k_params,
                  "rel_fro": float(d.norm() / (v.double().norm() + 1e-12)), "group": group_of(k)}
        n += nz
        n_params += k_params
    total = sum(v.numel() for v in orig.values())
    return {"n_changed": n, "frac_changed": n / total, "changed_parameters": n_params, "max_rank": worst,
            "rank_ok": worst <= MAX_RANK, "tensors": per}


def swap_back(ad, state: dict, orig: dict, data: dict, Y0: dict, base_R: float) -> dict:
    """Revert each changed group to the original; report response_probes and the share of the repair lost."""
    out = {}
    groups = sorted({group_of(k) for k in orig if not torch.equal(state[k], orig[k])})
    for g in groups:
        s = {k: (orig[k] if group_of(k) == g else state[k]) for k in orig}
        W.set_state(ad, s)
        Y = raw_eval(ad, data, VIS)
        R = metrics(Y, Y0, data, VIS)["response_probes"]
        out[g] = {"response_reverted": R, "repair_lost": float((R - base_R) / max(1 - base_R, 1e-9))}
    W.set_state(ad, orig)
    return out


def generator_seeds() -> int:
    return int(os.environ.get("SOFTREPAIR_GEN_SEEDS", "1"))


def generator_ok(res: dict) -> bool:
    """The generator AUC is a limit on s and m. On l it is reported only: the original l generator already gives
    real-vs-generated AUC 1.00, so an increase cannot be measured."""
    return W.model_size() == "l" or res.get("generator_dauc", 0.0) <= GATE["generator_dauc"]


def gen_ref_seed(ad, orig: dict, seed: int) -> float:
    """Original model's real-vs-generated AUC for one generation seed (cached under RESULTS_DIR/eval)."""
    size = W.model_size()
    p = W.OUT / "eval" / (f"generator_original_{size}.json" if seed == 0 else f"generator_original_{size}_s{seed}.json")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        W.set_state(ad, orig)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(W.generator_quality(ad, W.load_set("test", ()), seed=seed), indent=1))
        tmp.replace(p)
    return json.loads(p.read_text())["auc_mean"]


def score(cand: Path | None, name: str, transform: float | None = None, gen: bool = True,
          reco: dict | None = None, keyclip: Path | None = None) -> dict:
    """Full test-set score of an edit (or of an input transform: pT cut, removal rule, model-internal removal)."""
    ad = W.load_adapter(os.environ.get("SOFTREPAIR_DEVICE", "cuda"))
    orig = W.original_state(ad)
    test = W.load_set("test", VIS + HELD + PHYS)
    tf = (lambda a: W.pt_cut(a, transform)) if transform else (lambda a: a)
    if reco is not None and reco.get("internal"):  # model-internal removal (tools.internal_fix), original weights
        from softrepair import tools as T
        tf = lambda a, tf0=tf: T.internal_fix(ad, tf0(a))  # noqa: E731
        transform = transform or -2.0
    elif reco is not None:  # removal rule as preprocessing (core.reco_fix); scored like the pT-cut baselines
        tf = lambda a, tf0=tf: W.reco_fix(tf0(a), **reco)  # noqa: E731
        transform = transform or -1.0  # treated as an input transform below (no swap-back, no generator run)
    data = {k: (tf(np.asarray(v)) if (k == "clean" or k in VIS + HELD + PHYS) else v) for k, v in test.items()}
    Y0 = raw_eval(ad, test, VIS + HELD + PHYS)
    if keyclip is not None:  # module change (keyclip.py), installed after the original's reference logits
        from softrepair import keyclip as KC
        KC.install(ad, json.loads(Path(keyclip).read_text()))
    state = W.load_candidate(cand, orig) if cand else dict(orig)
    W.set_state(ad, state)
    Y = raw_eval(ad, data, VIS + HELD + PHYS)
    res = {"name": name, "cand": str(cand), "pt_cut": transform, "reco_fix": reco, "keyclip": str(keyclip),
           **metrics(Y, Y0, test, VIS + HELD)}
    res["size"] = edit_size(state, orig)
    res["summary_drift"] = summary_drift(ad, state, orig, test)
    if cand and not transform:
        res["swap_back"] = swap_back(ad, state, orig, test, Y0, res["response_probes"])  # Y0: ORIGINAL weights
    if gen and not transform:
        tst, d = W.load_set("test", ()), []
        for seed in range(generator_seeds()):
            ref = gen_ref_seed(ad, orig, seed)
            W.set_state(ad, state)
            d.append(W.generator_quality(ad, tst, seed=seed)["auc_mean"] - ref)
        res["generator_dauc"] = float(np.mean(d))
        res["generator_dauc_seeds"] = d
    W.set_state(ad, orig)
    res["passes_limits"] = bool(res["passes_limits_no_generator"] and res["size"]["rank_ok"] and generator_ok(res))
    res["passes_limits_any_rank"] = bool(res["passes_limits_no_generator"] and generator_ok(res))
    out = W.OUT / "repair_eval" / "score"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"[score] {name}: response_probes {res['response_probes']:.3f} held_out {res['response_held_out']:.3f} "
          f"rho {res['rho']} d_acc {res['clean']['d_acc']:+.4f} class_drop {res['clean']['class_drop']:.3f} "
          f"generator {res.get('generator_dauc', float('nan')):+.3f} frac {res['size']['frac_changed']:.2e} "
          f"max_rank {res['size']['max_rank']} passes {res['passes_limits']}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "score"))
    ap.add_argument("--cand", type=Path, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--pt-cut", type=float, default=None)
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--reco-fix", action="store_true", help="score with the removal rule as preprocessing")
    ap.add_argument("--merge-dr", type=float, default=0.0)
    ap.add_argument("--no-thresholds", action="store_true")
    ap.add_argument("--internal-fix", action="store_true", help="score with tools.internal_fix (model-internal)")
    ap.add_argument("--keyclip", type=Path, default=None, help="keyclip.py spec to install before scoring")
    args = ap.parse_args()
    if args.mode == "build":
        build()
    else:
        reco = {"merge_dr": args.merge_dr, "thresholds": not args.no_thresholds} if args.reco_fix else None
        if args.internal_fix:
            reco = {"internal": True}
        score(args.cand, args.name or (args.cand.stem if args.cand else "original"), args.pt_cut, not args.no_gen, reco,
              args.keyclip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
