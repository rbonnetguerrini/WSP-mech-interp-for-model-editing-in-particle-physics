#!/usr/bin/env python
"""Test-jet scoring of an edit: remaining response, accuracy, physical changes, changed parameters, W efficiency.

Added-particle sets (one change per jet; r_P and ratios as in repair_eval.metrics):
  practice probes  soft3e-3, soft1e-3, soft1e-4 (one photon, pid 6, at that fraction of the jet's scalar pT sum,
                   uniform in dR < 0.4) and soft7_3e-3, soft7_1e-3, soft7_1e-4 (the same with a neutral hadron, pid 7)
  held out         soft3e-4, soft7_3e-4 (between the practice fractions); softpair_1e-3 (one photon AND one neutral
                   hadron in the same jet, each at 1e-3); soft1e-2, soft7_1e-2 (well inside the data's pT range);
                   soft2e-3 (photon at the transition); rmsoft (REMOVE the softest real neutral of the jet)
  report           split2, split3, split2c (collinear; reported only)
remaining_response        = mean ratio over soft1e-3, soft1e-4, soft7_1e-3, soft7_1e-4 (paper App. B)
response_practice_probes  = mean ratio over the practice probes; response_held_out over the held-out ones;
response_in_range         = mean(soft3e-3, soft7_3e-3); response_in_range_held_out = mean(soft1e-2, soft7_1e-2, rmsoft).
Limits: as repair_eval (clean, physical changes D/J/P, generator AUC; on l the generator AUC is reported only),
plus a size budget of 32 (2d - 1) changed parameters, d = body width (s 8,160; m 32,736; l 65,504).

Also here:
  fragmentation_response  response to the fragmentation change P after the removal rule (P_removal), any pipeline
  w_efficiency            W efficiency at the 80% working point after one added photon (or neutral hadron)

  python -m softrepair.scoring build                      practice_ir / test_ir under DATA_DIR (from practice / test)
  python -m softrepair.scoring score --cand X --name N    test score -> RESULTS_DIR/scoring/score/N.json
  python -m softrepair.scoring score --neutral-cut 0.5 --name ncut0.5   baseline: drop neutrals below 0.5 GeV
  python -m softrepair.scoring fragmentation --cands name=path.npz | name=@reco | name=@reco_merge | name=@internal
      -> RESULTS_DIR/fragmentation/fragmentation<tag>.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from softrepair import core as W
from softrepair import repair_eval as B

VIS = ("soft3e-3", "soft1e-3", "soft1e-4", "soft7_3e-3", "soft7_1e-3", "soft7_1e-4")
HELD = ("soft3e-4", "soft7_3e-4", "softpair_1e-3", "soft1e-2", "soft7_1e-2", "soft2e-3", "rmsoft")
REPORT = ("split2", "split3", "split2c")
PHYS = B.PHYS
R_IN, R_SUB = ("soft3e-3", "soft7_3e-3"), ("soft1e-3", "soft1e-4", "soft7_1e-3", "soft7_1e-4")
INRANGE_HIDDEN = ("soft1e-2", "soft7_1e-2", "rmsoft")
BUDGET_UNITS = 32
SETS = {"practice_ir": ("practice", VIS + PHYS), "test_ir": ("test", VIS + HELD + REPORT + PHYS)}
# paper names of the four sets behind the remaining response
RESPONSE_NAMES = {"soft1e-3": "photon_f1e-3", "soft1e-4": "photon_f1e-4",
                  "soft7_1e-3": "neutral_hadron_f1e-3", "soft7_1e-4": "neutral_hadron_f1e-4"}


def parameter_budget(orig: dict) -> int:
    d = int(orig["body.norm.weight"].shape[0])
    return BUDGET_UNITS * (2 * d - 1)


# ---------------------------------------------------------------- further changes of the jets

def _compact(jet: np.ndarray) -> None:
    keep = jet[:, 2] != 0
    n = int(keep.sum())
    jet[:] = np.concatenate([jet[keep], np.zeros((len(jet) - n, jet.shape[1]), np.float32)])
    jet[n:, 4] = 3.0  # padding pid filler, as in the files


def perturb(raw: np.ndarray, name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """core.perturb plus softpair_<f> (one photon and one neutral hadron) and rmsoft (remove the softest neutral)."""
    if name.startswith("softpair_"):
        frac = float(name.split("_")[1])
        rng = np.random.default_rng(seed)
        out, ok = raw.copy(), np.zeros(len(raw), bool)
        for j in range(len(raw)):
            ok[j] = W._soft(out[j], frac, 6, rng) and W._soft(out[j], frac, 7, rng)
            if not ok[j]:
                out[j] = raw[j]
        return out, ok
    if name == "rmsoft":
        out, ok = raw.copy(), np.zeros(len(raw), bool)
        for j in range(len(raw)):
            r = W._real(out[j])
            neu = r[np.isin(np.rint(out[j][r, 4]).astype(int), W.NEUTRAL)]
            if neu.size == 0 or r.size < 2:
                continue
            out[j][neu[np.argmin(out[j][neu, 2])]] = 0.0
            _compact(out[j])
            ok[j] = True
        return out, ok
    return W.perturb(raw, name, seed)


def neutral_cut(raw: np.ndarray, cut: float) -> np.ndarray:
    """Baseline: drop neutral particles (pid 6, 7) below `cut` GeV at the input."""
    out = raw.copy()
    m = (out[..., 2] != 0) & np.isin(np.rint(out[..., 4]).astype(int), W.NEUTRAL) & (np.exp(out[..., 2]) < cut)
    out[m] = 0.0
    for j in np.flatnonzero(m.any(1)):
        _compact(out[j])
    return out


def build() -> None:
    """Set folders practice_ir / test_ir. Existing variant files of practice / test are linked, missing ones
    generated with fixed seeds (9000 + 100 * set index + position in the variant list)."""
    for name, (src, variants) in SETS.items():
        s, d = W.set_dir(src), W.set_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        smeta = json.loads((s / "meta.json").read_text())
        meta = {"source": src, "files": smeta["files"], "rows": smeta["rows"], "n": smeta["n"],
                "variants": [], "valid": {}}
        for f in ("clean.npy", "labels.npy"):
            if not (d / f).exists():
                os.symlink(s / f, d / f)
        raw = np.load(s / "clean.npy")
        for k, v in enumerate(variants):
            if not (d / f"{v}.npy").exists():
                if v in smeta["variants"]:
                    for f in (f"{v}.npy", f"{v}.ok.npy"):
                        os.symlink(s / f, d / f)
                else:
                    p, ok = perturb(raw, v, 9000 + 100 * list(SETS).index(name) + k)
                    np.save(d / f"{v}.npy", p)
                    np.save(d / f"{v}.ok.npy", ok)
            ok = np.load(d / f"{v}.ok.npy")
            meta["variants"].append(v)
            meta["valid"][v] = int(ok.sum())
            print(f"[{name}] {v}: {ok.sum()}/{len(ok)}", flush=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- metrics and score

def metrics(Y: dict, Y0: dict, data: dict, probes: tuple[str, ...]) -> dict:
    res = B.metrics(Y, Y0, data, probes)
    for k in ("response_probes", "response_in_range", "response_sub_threshold", "response_held_out"):
        res.pop(k, None)
    ratio = res["ratio"]
    mean = lambda ks: float(np.mean([ratio[k] for k in ks])) if all(k in ratio for k in ks) else None  # noqa: E731
    res["response_practice_probes"] = mean(VIS)
    res["response_in_range"], res["remaining_response"] = mean(R_IN), mean(R_SUB)
    res["response_held_out"] = mean(HELD)
    res["response_in_range_held_out"] = mean(INRANGE_HIDDEN)
    res["response_collinear"] = mean(REPORT)
    return res


def size(state: dict, orig: dict) -> dict:
    sz = B.edit_size(state, orig)
    out = {k: sz[k] for k in ("n_changed", "frac_changed", "changed_parameters")}
    out["changed_parameters_budget"] = parameter_budget(orig)
    out["within_budget"] = bool(sz["changed_parameters"] <= out["changed_parameters_budget"])
    out["tensors"] = {k: {"rank": v["rank"], "changed": v["changed"], "changed_parameters": v["changed_parameters"]}
                      for k, v in sz["tensors"].items()}
    return out


def score_loaded(ad, orig: dict, cand: Path | None, name: str, ncut: float | None = None, gen: bool = True) -> dict:
    """Test score of one edit with an already loaded adapter (original weights restored at the end)."""
    allp = VIS + HELD + REPORT + PHYS
    test = W.load_set("test_ir", allp)
    tf = (lambda a: neutral_cut(a, ncut)) if ncut else (lambda a: a)
    data = {k: (tf(np.asarray(v)) if (k == "clean" or k in allp) else v) for k, v in test.items()}
    W.set_state(ad, orig)
    Y0 = B.raw_eval(ad, test, allp)
    state = W.load_candidate(cand, orig) if cand else dict(orig)
    W.set_state(ad, state)
    Y = B.raw_eval(ad, data, allp)
    res = {"name": name, "cand": str(cand), "neutral_cut": ncut, "model_size": W.model_size(),
           **metrics(Y, Y0, test, VIS + HELD + REPORT)}
    res["size"] = size(state, orig)
    res["summary_drift"] = B.summary_drift(ad, state, orig, test)
    if gen and not ncut:
        tst, d = W.load_set("test", ()), []
        for seed in range(B.generator_seeds()):
            ref = B.gen_ref_seed(ad, orig, seed)
            W.set_state(ad, state)
            d.append(W.generator_quality(ad, tst, seed=seed)["auc_mean"] - ref)
        res["generator_dauc"], res["generator_dauc_seeds"] = float(np.mean(d)), d
    W.set_state(ad, orig)
    res["passes_limits"] = bool(res["passes_limits_no_generator"] and B.generator_ok(res))
    res["passes_limits_and_budget"] = bool(res["passes_limits"] and res["size"]["within_budget"])
    return res


def score(cand: Path | None, name: str, ncut: float | None = None, gen: bool = True) -> dict:
    ad = W.load_adapter(os.environ.get("SOFTREPAIR_DEVICE", "cuda"))
    orig = W.original_state(ad)
    res = score_loaded(ad, orig, cand, name, ncut, gen)
    out = W.OUT / "scoring" / "score"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"[scoring] {name}: remaining_response {res['remaining_response']:.3f} "
          f"practice_probes {res['response_practice_probes']:.3f} held_out {res['response_held_out']:.3f} "
          f"in_range {res['response_in_range']:.3f} rho {res['rho']} d_acc {res['clean']['d_acc']:+.4f} "
          f"generator {res.get('generator_dauc', float('nan')):+.3f} "
          f"changed_parameters {res['size']['changed_parameters']}/{res['size']['changed_parameters_budget']} "
          f"passes {res['passes_limits_and_budget']}", flush=True)
    return res


# ---------------------------------------------------------------- fragmentation after the removal rule

def fragmentation_response(ad, cands: list[str]) -> dict:
    """Response to the fragmentation change P with and without the removal rule applied after it.

    For each pipeline f (weights and/or its own input transform):
      rho_P         = mean ||c_f(P x) - c_f(x)|| / mean ||c_0(P x) - c_0(x)||
      rho_P_removal = mean ||c_f(rule(P x)) - c_f(x)|| / mean ||c_0(rule(P x)) - c_0(x)||
    c = centred 10 JetClass logits; c_0 = original model, no transform. Also the direction errors.
    cands: "name=path.npz" | "name=@original" | "name=@reco" | "name=@reco_merge" | "name=@internal" |
    "name=@keyclip:<spec.json>".
    """
    from softrepair import tools as T
    orig = W.original_state(ad)
    test = W.load_set("test", ("P",))
    X = {"clean": np.asarray(test["clean"]), "P": np.asarray(test["P"])}
    X["P_removal"] = W.reco_fix(X["P"])
    sub = T.sub_threshold(X["P"])
    res = {"P_jets_with_subthreshold_neutral": float(sub.any(1).mean()), "P_subthreshold_per_jet": float(sub.sum(1).mean()),
           "candidates": {}}
    Y0 = {k: W.logits(ad, v) for k, v in X.items()}
    c0 = W.centred10(Y0["clean"])
    b = {k: np.linalg.norm(W.centred10(Y0[k]) - c0, axis=1).mean() for k in ("P", "P_removal")}
    res["original_response"] = {k: float(v) for k, v in b.items()}
    for spec in cands:
        name, what = spec.split("=", 1)
        tf = lambda a: a  # noqa: E731
        if what == "@reco":
            tf = lambda a: W.reco_fix(a)  # noqa: E731
        elif what == "@reco_merge":
            tf = lambda a: W.reco_fix(a, merge_dr=1e-3)  # noqa: E731
        elif what == "@internal":
            tf = lambda a: T.internal_fix(ad, a)  # noqa: E731
        elif what.startswith("@keyclip:"):  # keyclip spec; installed for this pipeline only
            from softrepair import keyclip
            clip_mods = keyclip.install(ad, json.loads(Path(what.split(":", 1)[1]).read_text()))
        elif what != "@original":
            W.set_state(ad, W.load_candidate(Path(what), orig))
        Y = {k: W.logits(ad, tf(v)) for k, v in X.items()}
        W.set_state(ad, orig)
        if what.startswith("@keyclip:"):
            for m in clip_mods:  # restore the stock forward
                del m.forward
                del m._keyclip
        c = W.centred10(Y["clean"])
        rec = {}
        for k in ("P", "P_removal"):
            dn, do = W.centred10(Y[k]) - c, W.centred10(Y0[k]) - c0
            rec[f"rho_{k}"] = float(np.linalg.norm(dn, axis=1).mean() / b[k])
            rec[f"dir_err_{k}"] = float(np.linalg.norm(dn - do, axis=1).mean() / b[k])
        rec["passes_P_removal_limit"] = bool(abs(np.log(rec["rho_P_removal"])) <= 0.15)
        res["candidates"][name] = rec
        print(f"{name:28s} rho_P {rec['rho_P']:.3f}  rho_P_removal {rec['rho_P_removal']:.3f}  "
              f"dir_err {rec['dir_err_P']:.3f} / {rec['dir_err_P_removal']:.3f}", flush=True)
    return res


# ---------------------------------------------------------------- W efficiency at a fixed working point

def hook_logodds(ad, n: int = 10000, soft_pid: int = 6) -> dict:
    """Per-jet log-odds F_W = logit_W - logit_QCD for W and QCD test-file jets (rows 60000..60000+n), nominal and
    with one photon (soft_pid 6) or neutral hadron (7) at f = 1e-3 and 3e-3 of the jet's scalar pT sum."""
    from softrepair import heads as H

    def fw(raw):
        return H.logodds_W(W.logits(ad, raw, batch=512))

    arr = {}
    for lab, name in ((H.WJ, "W"), (H.QCD, "QCD")):
        raw = H.load(lab, 60000, n)
        arr[f"F0_{name}"] = fw(raw)
        sp = "soft" if soft_pid == 6 else f"soft{soft_pid}_"
        for v, (x, ok) in {"f1e-3": W.perturb(raw, f"{sp}1e-3", 11 + lab),
                           "f3e-3": W.perturb(raw, f"{sp}3e-3", 21 + lab)}.items():
            arr[f"F_{v}_{name}"], arr[f"ok_{v}_{name}"] = fw(x), ok
    return arr


def w_efficiency(ad, n: int = 10000, soft_pid: int = 6, wps=(0.8, 0.5)) -> dict:
    """Threshold on F_W that keeps a fraction wp of the nominal W jets for the model under test; efficiency
    (and QCD mistag rate) after the added particle. Keys: wp80 -> {cut_FW, <v>_<W|QCD>: {eff, err, pass}}."""
    arr = hook_logodds(ad, n, soft_pid)

    def eff(x, cut):
        k, m = int((x > cut).sum()), len(x)
        p = k / m
        return p, float(np.sqrt(p * (1 - p) / m)), k

    num = {"n_per_class": n, "soft_pid": soft_pid, "rows": [60000, 60000 + n]}
    for wp in wps:
        cut = float(np.quantile(arr["F0_W"], 1 - wp))
        d = {"cut_FW": cut, "cut_score": float(1 / (1 + np.exp(-cut)))}
        for v in ("nominal", "f1e-3", "f3e-3"):
            for name in ("W", "QCD"):
                x = arr["F0_" + name] if v == "nominal" else arr[f"F_{v}_{name}"]
                p, e, k = eff(x, cut)
                d[f"{v}_{name}"] = {"eff": p, "err": e, "pass": k}
        num[f"wp{int(wp * 100)}"] = d
    return num


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "score", "fragmentation"))
    ap.add_argument("--cand", type=Path, default=None)
    ap.add_argument("--cands", nargs="+", default=None, help="fragmentation: name=path | name=@reco | ...")
    ap.add_argument("--name", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--size", default=None, choices=("s", "m", "l"))
    ap.add_argument("--neutral-cut", type=float, default=None)
    ap.add_argument("--no-gen", action="store_true")
    args = ap.parse_args()
    if args.size:
        os.environ["MODEL_SIZE"] = args.size
    if args.mode == "build":
        build()
    elif args.mode == "fragmentation":
        if not args.cands:
            ap.error("fragmentation needs --cands")
        ad = W.load_adapter(os.environ.get("SOFTREPAIR_DEVICE", "cuda"))
        res = fragmentation_response(ad, args.cands)
        out = W.OUT / "fragmentation"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"fragmentation{args.tag}.json").write_text(json.dumps(res, indent=1))
    else:
        score(args.cand, args.name or (args.cand.stem if args.cand else "original"), args.neutral_cut,
              not args.no_gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
