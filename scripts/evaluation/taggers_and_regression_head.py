#!/usr/bin/env python
"""The nine JetClass taggers, the JetClass-II probability and a new jet mass regression head (paper Sec. 4,
Tab. taggers, Tab. tests_sizes).

1. Taggers F_c = logit(c) - logit(QCD) for the nine JetClass signal classes c (global labels 3..11); the cut keeps
   80% of the unmodified c jets of the model under test. Per added particle: signal efficiency, QCD acceptance,
   mean dF of signal and QCD jets, and mean |dF| over signal + QCD jets divided by the clean separation.
2. Jet mass regression head: ridge (RidgeCV, 13 alphas) and MLP (256, 256) regressors from the 4 summary tokens
   after the body (standardised) to the jet mass computed from the constituents, trained on unmodified jets of all
   ten classes (rows 80000.., --n-train per class) of the same model. Per added particle: mean |d m_pred| against
   mean |d m_true|, signed mean d m_pred per class, |d m_pred| / clean resolution, and the share of W (top) jets
   inside a 65-95 GeV (140-200 GeV) window of predicted mass.
3. JetClass-II probability: mean softmax share of the JetClass-II classes (12-199) of the 200 jet classes.

Added particles (evaluation rows 60000.., seed 11000 + 100 k + label for the k-th entry of PERTURBATIONS, the same
for every model): one photon or neutral hadron at f = 1e-3 of the jet pT, one photon at 0.3 GeV, and one photon at
3 GeV (above the JetClass thresholds: control).

Several edits can be evaluated in one call (the jets are built once):
  python scripts/evaluation/taggers_and_regression_head.py --size s \
      --edit original_model --edit finetuning=$EDITS_DIR/finetuning/small/attention_only_both_neutral_types_seed0.npz \
      --out-dir $RESULTS_DIR/taggers_and_regression_head/small --heads-dir $RESULTS_DIR/regression_heads/small
  -> <out-dir>/<name>.json and, with --heads-dir, the fitted heads <heads-dir>/<head_name(name)>.npz
     (finetuning_<v> -> finetuning/<v>.npz, sae_<v> -> sae/<v>.npz, agents_<variant>_run<k>_<edit> ->
     agents/<variant>/run<k>_<edit>.npz; read them with load_head and predict)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PERTURBATIONS = ("photon_f0.001", "neutral_hadron_f0.001", "photon_0.3GeV", "photon_3GeV (control)")


def perturb(name: str, raw: np.ndarray, seed: int):
    from softrepair import heads as H
    if name == "photon_f0.001":
        return C.soft(raw, 1e-3, 6, seed)
    if name == "neutral_hadron_f0.001":
        return C.soft(raw, 1e-3, 7, seed)
    if name == "photon_0.3GeV":
        return H.add_photon(raw, 0.3, seed)
    if name == "photon_3GeV (control)":
        return H.add_photon(raw, 3.0, seed)
    raise KeyError(name)


# ---------------------------------------------------------------- saved heads

def head_name(name: str) -> str:
    """File name of a saved head: finetuning_<v> -> finetuning/<v>, sae_<v> -> sae/<v>,
    agents_<variant>_run<k>_<edit> -> agents/<variant>/run<k>_<edit>; other names unchanged."""
    import re
    if "/" in name:
        return name
    m = re.fullmatch(r"agents_(.+)_(run\d+_edit_\w+)", name)
    if m:
        return f"agents/{m.group(1)}/{m.group(2)}"
    for p in ("finetuning_", "sae_"):
        if name.startswith(p):
            return f"{p[:-1]}/{name[len(p):]}"
    return name


def save_heads(path: Path, sc, mu: float, sd: float, heads: dict, meta: dict) -> None:
    """Scaler, target standardisation, ridge and MLP weights in one .npz (no pickles)."""
    arr = {"scaler_mean": sc.mean_, "scaler_scale": sc.scale_, "target_mean": np.float64(mu),
           "target_std": np.float64(sd), "ridge_coef": heads["ridge"].coef_,
           "ridge_intercept": np.float64(heads["ridge"].intercept_), "ridge_alpha": np.float64(heads["ridge"].alpha_),
           "mlp_n_layers": np.int64(len(heads["mlp"].coefs_)), "meta": np.array(json.dumps(meta))}
    for i, (w, b) in enumerate(zip(heads["mlp"].coefs_, heads["mlp"].intercepts_)):
        arr[f"mlp_W{i}"], arr[f"mlp_b{i}"] = w, b
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arr)


def load_head(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def predict(head: dict, features: np.ndarray, kind: str = "ridge") -> np.ndarray:
    """Predicted jet mass (GeV) of a saved head from the summary-token features (n, 4 * width)."""
    x = (features - head["scaler_mean"]) / head["scaler_scale"]
    if kind == "ridge":
        y = x @ head["ridge_coef"] + head["ridge_intercept"]
    else:
        n = int(head["mlp_n_layers"])
        for i in range(n):
            x = x @ head[f"mlp_W{i}"] + head[f"mlp_b{i}"]
            if i < n - 1:
                x = np.maximum(x, 0.0)  # ReLU hidden layers, identity output (MLPRegressor defaults)
        y = x[:, 0]
    return y * head["target_std"] + head["target_mean"]


# ---------------------------------------------------------------- one model

def evaluate(ad, X: dict, tr: dict, labs: tuple, perts: tuple, args, log) -> tuple[dict, dict]:
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from softrepair import core as W

    Y, S, M = {}, {}, {}
    for v in X:
        Y[v], S[v], M[v] = {}, {}, {}
        for lab in labs:
            y, s = W.logits(ad, X[v][lab][0], body=True)
            Y[v][lab], S[v][lab] = y, s.reshape(len(s), -1).astype(np.float64)
            M[v][lab] = C.jet_mass(X[v][lab][0])
        log("forward", v)
    S_tr = np.concatenate([W.logits(ad, tr[lab], body=True)[1].reshape(args.n_train, -1) for lab in labs]).astype(np.float64)
    m_tr = np.concatenate([C.jet_mass(tr[lab]) for lab in labs])
    log("training set forward")

    res = {"n": args.n, "n_train": args.n_train, "rows_eval": [60000, 60000 + args.n],
           "rows_train": [80000, 80000 + args.n_train], "taggers": {}, "jetclass2_share": {}, "mass_head": {}}

    # 1. taggers
    for c in C.SIGNALS:
        F = {v: {lab: Y[v][lab][:, c] - Y[v][lab][:, C.QCD] for lab in (c, C.QCD)} for v in X}
        thr = float(np.quantile(F["clean"][c], 0.20))
        sep = float(F["clean"][c].mean() - F["clean"][C.QCD].mean())
        t = {"thr": thr, "separation": sep, "clean": {"eff_sig": 0.8, "acc_QCD": float(np.mean(F["clean"][C.QCD] > thr))}}
        for v in perts:
            r, d_all = {}, []
            for lab, key in ((c, "sig"), (C.QCD, "QCD")):
                ok = X[v][lab][1]
                f1, f0 = F[v][lab][ok], F["clean"][lab][ok]
                r["eff_sig" if key == "sig" else "acc_QCD"] = float(np.mean(f1 > thr))
                r[f"mean_dF_{key}"] = float(np.mean(f1 - f0))
                d_all.append(np.abs(f1 - f0))
            r["shift_over_separation"] = float(np.concatenate(d_all).mean() / sep)
            t[v] = r
        res["taggers"][C.NAMES[c]] = t
    for v in X:
        res["jetclass2_share"][v] = {C.NAMES[lab]: C.shares(Y[v][lab][X[v][lab][1]])["JetClass-II (12-199)"]
                                     for lab in labs}
    log("taggers done")

    # 2. jet mass regression head on the frozen body
    sc = StandardScaler().fit(S_tr)
    mu, sd = m_tr.mean(), m_tr.std()
    heads = {"ridge": RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(S_tr), (m_tr - mu) / sd),
             "mlp": MLPRegressor(hidden_layer_sizes=(256, 256), early_stopping=True, max_iter=300,
                                 random_state=0).fit(sc.transform(S_tr), (m_tr - mu) / sd)}
    for hn, h in heads.items():
        P = {v: {lab: h.predict(sc.transform(S[v][lab])) * sd + mu for lab in labs} for v in X}
        err = np.concatenate([P["clean"][lab] - M["clean"][lab] for lab in labs])
        reso = float(np.sqrt(np.mean(err ** 2)))
        rec = {"clean": {"rms_error_GeV": reso, "mean_abs_error_GeV": float(np.abs(err).mean()),
                         "median_pred_by_class": {C.NAMES[lb]: float(np.median(P["clean"][lb])) for lb in labs}}}
        for c, (lo, hi) in C.WINDOWS.items():
            rec["clean"][f"window_{C.NAMES[c]}"] = {"pred": float(np.mean((P["clean"][c] > lo) & (P["clean"][c] < hi))),
                                                    "true": float(np.mean((M["clean"][c] > lo) & (M["clean"][c] < hi)))}
        for v in perts:
            ok = {lab: X[v][lab][1] for lab in labs}
            dp = {lab: P[v][lab][ok[lab]] - P["clean"][lab][ok[lab]] for lab in labs}
            dt = {lab: M[v][lab][ok[lab]] - M["clean"][lab][ok[lab]] for lab in labs}
            allp, allt = np.concatenate(list(dp.values())), np.concatenate(list(dt.values()))
            r = {"mean_abs_dm_pred_GeV": float(np.abs(allp).mean()), "mean_abs_dm_true_GeV": float(np.abs(allt).mean()),
                 "mean_abs_dm_pred_over_resolution": float(np.abs(allp).mean() / reso),
                 "mean_dm_pred_by_class": {C.NAMES[lb]: float(dp[lb].mean()) for lb in labs},
                 "mean_dm_true_by_class": {C.NAMES[lb]: float(dt[lb].mean()) for lb in labs}}
            for c, (lo, hi) in C.WINDOWS.items():
                pc, mc = P[v][c][ok[c]], M[v][c][ok[c]]
                r[f"window_{C.NAMES[c]}"] = {"pred": float(np.mean((pc > lo) & (pc < hi))),
                                             "true": float(np.mean((mc > lo) & (mc < hi)))}
            rec[v] = r
        res["mass_head"][hn] = rec
        log("mass head", hn, "resolution", round(reso, 2), "GeV")
    fitted = {"scaler": sc, "mu": float(mu), "sd": float(sd), "heads": heads,
              "check": {"features": S["clean"][C.QCD][:64], "pred": {hn: h.predict(sc.transform(S["clean"][C.QCD][:64]))
                                                                     * sd + mu for hn, h in heads.items()}}}
    return res, fitted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--edit", action="append", default=None,
                    help="name=path.npz (repeatable); a bare name (e.g. original_model) is the unedited model")
    ap.add_argument("--n", type=int, default=3000, help="evaluation jets per class")
    ap.add_argument("--n-train", type=int, default=2000, help="regression-head training jets per class")
    ap.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS), choices=PERTURBATIONS,
                    help="added particles to evaluate (default all; the seeds do not depend on the selection)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--heads-dir", type=Path, default=None, help="save the fitted regression heads here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-batch", type=int, default=32, help="forward batch cap (32 for all stored results)")
    a = ap.parse_args()
    C.setup_device(a.size, a.max_batch)
    from softrepair import core as W

    torch.set_num_threads(a.threads)
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)  # noqa: E731
    perts = tuple(p for p in PERTURBATIONS if p in a.perturbations)

    labs = tuple(C.NAMES)
    raw = {lab: C.load(lab, 60000, a.n) for lab in labs}
    X = {"clean": {lab: (raw[lab], np.ones(a.n, bool)) for lab in labs}}
    for k, name in enumerate(PERTURBATIONS):
        if name in perts:
            X[name] = {lab: perturb(name, raw[lab], 11000 + 100 * k + lab) for lab in labs}
    tr = {lab: C.load(lab, 80000, a.n_train) for lab in labs}
    log("jets built")

    ad = W.load_adapter(a.device, a.size)
    orig = W.original_state(ad)
    for name, path in C.parse_edits(a.edit):
        W.set_state(ad, W.load_candidate(path, orig) if path else orig)
        res, fitted = evaluate(ad, X, tr, labs, perts, a, log)
        res = {"name": name, "model_size": a.size, "edit": path.name if path else None, **res}
        C.dump(a.out_dir / f"{name}.json", res)
        if a.heads_dir:
            f = a.heads_dir / f"{head_name(name)}.npz"
            meta = {"model_size": a.size, "edit": res["edit"], "features": "4 summary tokens after the body, flattened",
                    "target": "jet mass (GeV) from the constituents", "rows_train": res["rows_train"],
                    "n_train_per_class": a.n_train, "ridge_alphas": "logspace(-2, 4, 13)",
                    "mlp": "MLPRegressor(256, 256), early_stopping, max_iter 300, random_state 0"}
            save_heads(f, fitted["scaler"], fitted["mu"], fitted["sd"], fitted["heads"], meta)
            h = load_head(f)
            for kind in ("ridge", "mlp"):  # the saved arrays reproduce the fitted heads
                d = np.abs(predict(h, fitted["check"]["features"], kind) - fitted["check"]["pred"][kind]).max()
                assert d < 1e-6, (kind, d)
        mh = res["mass_head"]
        log(name, "W eff (photon f=1e-3)",
            round(res["taggers"]["W->qq"].get("photon_f0.001", {}).get("eff_sig", float("nan")), 4),
            "| regression |dm| ridge", {v: round(mh["ridge"][v]["mean_abs_dm_pred_GeV"], 3) for v in perts})
        W.set_state(ad, orig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
