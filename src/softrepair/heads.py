#!/usr/bin/env python
"""IRC sensitivity across heads, and a toy W-tagging analysis. CPU is enough for pretrain_s.

1. Backbone. Does one soft photon or one collinear split move the body's 4 summary tokens, which
   every head reads?  shift = ||s(x') - s(x)|| / median ||s(x_i) - s(x_j)|| over random
   same-class pairs of clean jets (the typical jet-to-jet distance).
2. Heads. Does a NEW head trained on the frozen backbone inherit the sensitivity? Logistic and MLP
   W-vs-QCD heads on the summary tokens of clean jets, next to the pretrained head.
   shift = mean |d log-odds| / (mean log-odds of W jets - mean log-odds of QCD jets).
3. Generator. One evaluation of the flow-matching head at time t (0 = data, 1 = noise), with the
   same noise on every slot. Relative change of the velocity of the UNCHANGED particles when one
   particle is added or split. Reference: swapping the class label W <-> QCD.
4. Toy analysis, also for an IRC-safe reference tagger (boosted tree on jet mass, tau21, tau32,
   d2, girth). W tagger score s = p_W / (p_W + p_QCD) in [0, 1], cut at 50% W efficiency on the
   nominal jets. Pseudo-data = the same jets with one extra photon of fixed pT (or a collinear split).
   Working points 50% and 80% W efficiency. Efficiency, mistag rate, the bias of a cut-and-count W yield, and IRC-safe control observables
   (jet mass, jet pT, tau21) for comparison.

Data: JetClass validation files of the test jets (W: file 5, QCD: file 32), rows no other script
reads: evaluation rows 60000.., fresh-head training rows 80000...

  python -m softrepair.heads --name original_s
  python -m softrepair.heads --name ptcut075_s --pt-cut 0.75     # drop every particle below 0.75 GeV
  python -m softrepair.heads --name edit_s --cand <edit.npz>      # edited weights (core.load_candidate)
  python -m softrepair.heads --name original_m --size m --device cuda

Writes RESULTS_DIR/heads/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from softrepair import core as W
from softrepair import repair_eval as B

QCD, WJ = 2, 9
FILES = {WJ: W.TEST_FILES[WJ], QCD: W.TEST_FILES[QCD]}
PHOTON = 6
PHOTON_PT = (0.2, 0.3, 0.45, 0.6, 0.85, 1.5, 3.0, 6.0)  # GeV; never exactly 1.0 (log pT = 0 is padding)
FRACS = ("1e-2", "3e-3", "1e-3", "1e-4")
SPLITS = {"split2": "hardest neutral -> 2 collinear halves", "split3": "hardest neutral -> 0.5/0.3/0.2",
          "split2c": "hardest charged -> 2 collinear halves"}
PHYS = {"angles_x1.105": "D", "energy_scale_x1.05": "J", "fragmentation_k1.105": "P"}
TOY = ("photon_0.3GeV", "photon_0.45GeV", "photon_0.6GeV", "photon_0.85GeV", "photon_1.5GeV",
       "photon_3.0GeV", "split2") + tuple(PHYS)
CONTROL = ("jet_mass", "jet_pt", "tau21")
SAFE = ("jet_mass", "tau21", "tau32", "d2", "girth")  # IRC-safe inputs of the reference tagger
WPS = (0.5, 0.8)  # W-efficiency working points


def load(lab: int, r0: int, n: int) -> np.ndarray:
    with h5py.File(W.DATA / f"JetClass_{FILES[lab]}.h5", "r") as h:
        raw = h["data"][r0:r0 + n].astype(np.float32)
        assert np.all(h["pid"][r0:r0 + n] == lab)
    return raw


def add_photon(raw: np.ndarray, pt: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One massless photon of fixed pT (GeV), uniform within dR < 0.4 (as core._soft)."""
    rng = np.random.default_rng(seed)
    out, ok = raw.copy(), np.zeros(len(raw), bool)
    for j in range(len(raw)):
        jet = out[j]
        r, slot = W._real(jet), W._free_slot(jet)
        if slot < 0:
            continue
        eta_j = W._jet_eta(jet, r)
        th, rr = rng.uniform(0, 2 * np.pi), 0.4 * np.sqrt(rng.uniform())
        de, dp = rr * np.cos(th), rr * np.sin(th)
        row = np.zeros(jet.shape[1], np.float32)
        row[:5] = de, dp, np.log(pt), np.log(pt * np.cosh(eta_j + de)), PHOTON
        jet[slot] = row
        ok[j] = True
    return out, ok


def variants(raw: np.ndarray, seed0: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    V = {}
    for k, pt in enumerate(PHOTON_PT):
        V[f"photon_{pt}GeV"] = add_photon(raw, pt, seed0 + k)
    for k, f in enumerate(FRACS):
        V[f"photon_frac{f}"] = W.perturb(raw, f"soft{f}", seed0 + 20 + k)
    for k, s in enumerate(SPLITS):
        V[s] = W.perturb(raw, s, seed0 + 30 + k)
    for name, op in PHYS.items():
        V[name] = (B.phys_op(raw, op), np.ones(len(raw), bool))
    return V


def data_check(raw: np.ndarray) -> dict:
    real = raw[..., 2] != 0
    pid = np.rint(raw[..., 4]).astype(int)
    pt = np.exp(raw[..., 2].astype(np.float64))
    ph = real & (pid == PHOTON)
    jet_pt = (pt * real).sum(1)
    return {"photon_share_of_constituents": float(ph.sum() / real.sum()),
            "photon_pt_min": float(pt[ph].min()), "photon_pt_p01": float(np.quantile(pt[ph], 0.01)),
            "photon_pt_p05": float(np.quantile(pt[ph], 0.05)),
            "all_pt_min": float(pt[real].min()), "all_pt_p01": float(np.quantile(pt[real], 0.01)),
            "jet_scalar_pt_median": float(np.median(jet_pt)),
            "pid_counts": {str(k): int(v) for k, v in zip(*np.unique(pid[real], return_counts=True))}}


def forward(ad, raw: np.ndarray, cut: float | None) -> tuple[np.ndarray, np.ndarray]:
    x = W.pt_cut(raw, cut) if cut else raw
    y, s = W.logits(ad, x, batch=512, body=True)
    return y, s.reshape(len(s), -1).astype(np.float64)


def logodds_W(y: np.ndarray) -> np.ndarray:
    return y[:, WJ] - y[:, QCD]


def fresh_heads(S_tr: np.ndarray, lab_tr: np.ndarray, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lin = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(S_tr, lab_tr)
    mlp = make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=1e-3, early_stopping=True,
                                                        max_iter=300, random_state=seed)).fit(S_tr, lab_tr)

    def mlp_logit(S):
        p = np.clip(mlp.predict_proba(S)[:, 1], 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    return {"fresh_linear": lin.decision_function, "fresh_mlp": mlp_logit}


def auc(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))


def gen_check(ad, raw: dict, V: dict, n: int, ts=(0.05, 0.2)) -> dict:
    out = {}
    for t in ts:
        res = {}
        for lab, other in ((WJ, QCD), (QCD, WJ)):
            x0 = raw[lab][:n]
            eps = np.random.default_rng(7).standard_normal(x0.shape[:2] + (4,)).astype(np.float32)
            labs = np.full(n, lab)
            v0 = W.gen_velocity(ad, x0, labs, eps, t)
            real0 = x0[..., 2] != 0

            def rel(v1, keep):
                num = (np.abs(v1 - v0).sum(2) * keep).sum(1)
                den = (np.abs(v0).sum(2) * keep).sum(1)
                return float(np.mean(num / np.maximum(den, 1e-12)))
            r = {"label_swap_reference": rel(W.gen_velocity(ad, x0, np.full(n, other), eps, t), real0)}
            for name in ("photon_0.3GeV", "photon_0.6GeV", "photon_3.0GeV", "split2"):
                x1, ok = V[lab][name][0][:n], V[lab][name][1][:n]
                keep = real0 & np.all(x1 == x0, axis=2) & ok[:, None]
                r[name] = rel(W.gen_velocity(ad, x1, labs, eps, t), keep)
            res[{WJ: "W", QCD: "QCD"}[lab]] = r
        out[f"t={t}"] = res
    return out


def control(raw0: np.ndarray, raw1: np.ndarray) -> dict:
    S0, S1 = W.jet_summaries(raw0), W.jet_summaries(raw1)
    out = {}
    for k in CONTROL:
        i = W.GEN_SUMMARY.index(k)
        a, b = S0[:, i], S1[:, i]
        out[k] = {"median_nominal": float(np.nanmedian(a)), "median_pseudo": float(np.nanmedian(b)),
                  "mean_shift_in_sd": float((np.nanmean(b) - np.nanmean(a)) / (np.nanstd(a) + 1e-12)),
                  "per_jet_rel_change_median": float(np.nanmedian(np.abs(b - a) / (np.abs(a) + 1e-12)))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="s")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cand", type=Path, default=None)
    ap.add_argument("--pt-cut", type=float, default=None)
    ap.add_argument("--n", type=int, default=10000, help="evaluation jets per class")
    ap.add_argument("--n-train", type=int, default=5000, help="fresh-head training jets per class")
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--no-control", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)  # noqa: E731

    ad = W.load_adapter(args.device, args.size)
    if args.cand:
        W.set_state(ad, W.load_candidate(args.cand, W.original_state(ad)))
    raw = {lab: load(lab, 60000, args.n) for lab in (WJ, QCD)}
    tr = {lab: load(lab, 80000, args.n_train) for lab in (WJ, QCD)}
    res = {"name": args.name, "size": args.size, "cand": str(args.cand), "pt_cut": args.pt_cut,
           "n": args.n, "files": FILES, "data_check": data_check(np.concatenate([raw[WJ], raw[QCD]]))}
    log("data", json.dumps(res["data_check"]))
    V = {lab: variants(raw[lab], 5000 + lab * 100) for lab in (WJ, QCD)}
    log("variants built")

    Y, S = {}, {}
    for lab in (WJ, QCD):
        Y[lab, "clean"], S[lab, "clean"] = forward(ad, raw[lab], args.pt_cut)
        for v, (x, _) in V[lab].items():
            Y[lab, v], S[lab, v] = forward(ad, x, args.pt_cut)
        log("forward done", lab)
    S_tr = np.concatenate([forward(ad, tr[lab], args.pt_cut)[1] for lab in (WJ, QCD)])
    lab_tr = np.r_[np.ones(args.n_train), np.zeros(args.n_train)]
    heads = {"pretrained": lambda lab, v: logodds_W(Y[lab, v])}
    for hname, f in fresh_heads(S_tr, lab_tr).items():
        cache = {}
        heads[hname] = (lambda f, cache: lambda lab, v: cache.setdefault((lab, v), f(S[lab, v])))(f, cache)
    log("fresh heads trained")

    if not args.no_control:
        # IRC-safe reference tagger: boosted tree on IRC-safe substructure observables
        from sklearn.ensemble import HistGradientBoostingClassifier
        idx = [W.GEN_SUMMARY.index(k) for k in SAFE]
        summ = lambda x: W.jet_summaries(x)[:, idx]  # noqa: E731
        Xs = np.concatenate([summ(tr[WJ]), summ(tr[QCD])])
        bdt = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xs, lab_tr)
        safe_cache = {}

        def safe_head(lab, v):
            if (lab, v) not in safe_cache:
                x = raw[lab] if v == "clean" else V[lab][v][0]
                x = W.pt_cut(x, args.pt_cut) if args.pt_cut else x
                p = np.clip(bdt.predict_proba(summ(x))[:, 1], 1e-7, 1 - 1e-7)
                safe_cache[lab, v] = np.log(p / (1 - p))
            return safe_cache[lab, v]
        heads["irc_safe_bdt"] = safe_head
        log("IRC-safe reference tagger trained")

    # 1. backbone shift
    rng = np.random.default_rng(0)
    body = {}
    for lab in (WJ, QCD):
        s0 = S[lab, "clean"]
        i, j = rng.permutation(len(s0)), rng.permutation(len(s0))
        ref = float(np.median(np.linalg.norm(s0[i] - s0[j], axis=1)[i != j]))
        body[str(lab)] = {"jet_to_jet_median": ref}
        for v in V[lab]:
            ok = V[lab][v][1]
            d = np.linalg.norm(S[lab, v] - s0, axis=1)[ok] / ref
            body[str(lab)][v] = {"median": float(np.median(d)), "mean": float(np.mean(d)), "valid": int(ok.sum())}
    res["backbone_shift"] = body

    # 2. head shifts
    hs = {}
    for hname, h in heads.items():
        hw, hq = h(WJ, "clean"), h(QCD, "clean")
        sep = float(hw.mean() - hq.mean())
        hs[hname] = {"auc_clean": auc(hw, hq), "separation": sep}
        for v in V[WJ]:
            d = np.r_[np.abs(h(WJ, v) - hw)[V[WJ][v][1]], np.abs(h(QCD, v) - hq)[V[QCD][v][1]]]
            hs[hname][v] = {"mean_abs_shift": float(d.mean()), "shift_over_separation": float(d.mean() / sep)}
    res["head_shift"] = hs
    log("checks 1-2 done")

    # 4. toy analysis
    toy = {}
    for hname, h in heads.items():
        for wp in WPS:
            hw, hq = h(WJ, "clean"), h(QCD, "clean")
            cut = float(np.quantile(hw, 1 - wp))
            eS0, eB0 = float((hw > cut).mean()), float((hq > cut).mean())
            t = {"cut_logodds": cut, "cut_score": float(1 / (1 + np.exp(-cut))), "eff_W": eS0, "mistag_QCD": eB0,
                 "n_QCD_pass": int((hq > cut).sum())}
            for v in TOY:
                pw, pq = h(WJ, v) > cut, h(QCD, v) > cut
                eS1, eB1 = float(pw.mean()), float(pq.mean())
                t[v] = {"eff_W": eS1, "mistag_QCD": eB1, "sf_W": eS1 / eS0, "sf_QCD": eB1 / max(eB0, 1e-12),
                        "W_in": int((pw & ~(hw > cut)).sum()), "W_out": int((~pw & (hw > cut)).sum()),
                        "QCD_in": int((pq & ~(hq > cut)).sum()), "QCD_out": int((~pq & (hq > cut)).sum()),
                        "median_abs_score_shift": float(np.median(np.abs(1 / (1 + np.exp(-h(WJ, v))) - 1 / (1 + np.exp(-hw))))),
                        # cut-and-count W yield with MC efficiencies; B/S = QCD/W jets before the tagger
                        "yield_bias": {str(bs): float((eS1 - eS0) / eS0 + bs * (eB1 - eB0) / eS0) for bs in (1, 10, 100)}}
            toy[f"{hname}@{wp:.0%}"] = t
    res["toy"] = toy
    if not args.no_control:
        res["toy_control_observables"] = {
            v: {str(lab): control(raw[lab], V[lab][v][0]) for lab in (WJ, QCD)}
            for v in ("photon_0.3GeV", "photon_0.6GeV", "photon_3.0GeV", "split2") + tuple(PHYS)}
    log("toy done")

    # 3. generator
    if args.n_gen > 0:
        res["generator_velocity"] = gen_check(ad, {k: (W.pt_cut(v, args.pt_cut) if args.pt_cut else v) for k, v in raw.items()},
                                              {lab: {k: ((W.pt_cut(x, args.pt_cut) if args.pt_cut else x), ok) for k, (x, ok) in V[lab].items()}
                                               for lab in (WJ, QCD)}, args.n_gen)
        log("generator done")

    out = W.OUT / "heads"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=1))
    log("wrote", out / f"{args.name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
