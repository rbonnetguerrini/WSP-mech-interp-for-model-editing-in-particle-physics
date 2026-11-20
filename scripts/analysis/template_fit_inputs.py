#!/usr/bin/env python
"""Simulation and pseudo-data scores for the W-fraction template fit of Sec. 2 (input of template_fit.py).

Simulation:  W file 5 / QCD file 32 (test-jet files), rows 60000-69999, 10,000 jets per class, nominal.
Pseudo-data: rows 70000.. of the same files (disjoint jets): 2,500 W + 7,500 QCD (true W fraction 0.25).
             Scenarios: nominal; below the threshold ("below_floor": every jet gets one photon, pT uniform in
             0.2-0.6 GeV); in range ("in_range": every jet gets one photon, pT uniform in 1-3 GeV).
Taggers:     F_W of the pretrained head ("original"); the same with every particle below 0.75 GeV dropped
             ("ptcut"); an IRC-safe boosted tree on jet mass, tau21, tau32, D2, girth ("ircsafe", trained on rows
             80000.., 5,000 per class; its log-odds); with --edit an edited model.
Also: cut-and-count extraction of the W fraction at the cuts keeping 80% / 50% of the simulated W jets.

  python scripts/analysis/template_fit_inputs.py [--device cuda] [--edit EDIT.npz --name NAME]
      -> RESULTS_DIR/analysis/template_fit_inputs_s[_NAME].npz / .json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

N_MC, N_PD_W, N_PD_Q = 10000, 2500, 7500
F_TRUE = 0.25  # = N_PD_W / (N_PD_W + N_PD_Q)
SCEN = {"nominal": None, "below_floor": (0.2, 0.6), "in_range": (1.0, 3.0)}
WPS = (0.8, 0.5)


def add_random_photon(raw, lo, hi, seed):
    from softrepair import heads as H
    rng = np.random.default_rng(seed)
    out = raw.copy()
    pts = rng.uniform(lo, hi, len(raw))
    pts = np.where(np.abs(pts - 1.0) < 1e-3, 1.001, pts)
    for j in range(len(raw)):
        out[j] = H.add_photon(raw[j:j + 1], float(pts[j]), int(rng.integers(1 << 31)))[0][0]
    return out


def compute(args) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from softrepair import core as W
    from softrepair import heads as H
    torch.set_num_threads(args.threads)
    ad = W.load_adapter(args.device, "s")
    if args.edit:
        W.set_state(ad, W.load_candidate(args.edit, W.original_state(ad)))
    mc = {lab: H.load(lab, 60000, N_MC) for lab in (H.WJ, H.QCD)}
    pd0 = np.concatenate([H.load(H.WJ, 70000, N_PD_W), H.load(H.QCD, 70000, N_PD_Q)])
    pd_lab = np.r_[np.ones(N_PD_W), np.zeros(N_PD_Q)].astype(np.int8)
    pd = {s: (pd0 if r is None else add_random_photon(pd0, *r, 7 + k)) for k, (s, r) in enumerate(SCEN.items())}
    idx = [W.GEN_SUMMARY.index(k) for k in H.SAFE]
    summ = lambda x: W.jet_summaries(x)  # noqa: E731
    arr = {"pd_label": pd_lab}
    tr = {lab: H.load(lab, 80000, 5000) for lab in (H.WJ, H.QCD)}
    S_tr = np.concatenate([summ(tr[H.WJ])[:, idx], summ(tr[H.QCD])[:, idx]])
    bdt = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(S_tr, np.r_[np.ones(5000), np.zeros(5000)])

    def bdt_lo(S):
        p = np.clip(bdt.predict_proba(S[:, idx])[:, 1], 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    for name, lab in (("W", H.WJ), ("QCD", H.QCD)):
        S = summ(mc[lab])
        arr[f"mc_obs_{name}"] = S[:, [W.GEN_SUMMARY.index(k) for k in ("jet_mass", "jet_pt", "tau21")]].astype(np.float32)
        arr[f"mc_{name}_original"] = H.logodds_W(W.logits(ad, mc[lab]))
        arr[f"mc_{name}_ptcut"] = H.logodds_W(W.logits(ad, W.pt_cut(mc[lab], 0.75)))
        arr[f"mc_{name}_ircsafe"] = bdt_lo(S)
    for s, x in pd.items():
        S = summ(x)
        arr[f"pd_obs_{s}"] = S[:, [W.GEN_SUMMARY.index(k) for k in ("jet_mass", "jet_pt", "tau21")]].astype(np.float32)
        arr[f"pd_{s}_original"] = H.logodds_W(W.logits(ad, x))
        arr[f"pd_{s}_ptcut"] = H.logodds_W(W.logits(ad, W.pt_cut(x, 0.75)))
        arr[f"pd_{s}_ircsafe"] = bdt_lo(S)
        print("scenario done", s, flush=True)
    num = {"f_true": F_TRUE, "n_mc_per_class": N_MC, "n_pd": [N_PD_W, N_PD_Q],
           "edit": args.edit.name if args.edit else None,
           "scenarios": {k: v for k, v in SCEN.items()}, "fraction_of_pd_jets_with_photon": 1.0}
    for tg in ("original", "ptcut", "ircsafe"):
        for wp in WPS:
            cut = float(np.quantile(arr[f"mc_W_{tg}"], 1 - wp))
            e = float((arr[f"mc_W_{tg}"] > cut).mean()); m = float((arr[f"mc_QCD_{tg}"] > cut).mean())
            se, sm = np.sqrt(e * (1 - e) / N_MC), np.sqrt(max(m * (1 - m), 1e-12) / N_MC)
            d = {"cut": cut, "eff_mc": e, "mistag_mc": m}
            for s in SCEN:
                x = arr[f"pd_{s}_{tg}"] > cut
                r = float(x.mean()); n = len(x)
                f = (r - m) / (e - m)
                sr = np.sqrt(r * (1 - r) / n)
                # propagate: df/dr = 1/(e-m); df/de = -(r-m)/(e-m)^2; df/dm = (r-e)/(e-m)^2
                sf = float(np.sqrt((sr / (e - m)) ** 2 + (se * (r - m) / (e - m) ** 2) ** 2 + (sm * (r - e) / (e - m) ** 2) ** 2))
                d[s] = {"pass_rate": r, "f_extracted": f, "f_err": sf, "bias": f - F_TRUE, "bias_rel": (f - F_TRUE) / F_TRUE,
                        "significance": (f - F_TRUE) / sf,
                        "eff_W_pd": float(x[pd_lab == 1].mean()), "mistag_pd": float(x[pd_lab == 0].mean())}
            num[f"{tg}@{int(wp * 100)}"] = d
    stem = C.analysis_dir() / ("template_fit_inputs_s" + (f"_{args.name}" if args.name else ""))
    np.savez_compressed(f"{stem}.npz", **arr)
    Path(f"{stem}.json").write_text(json.dumps(num, indent=1))
    print(json.dumps(num, indent=1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--edit", type=Path, default=None)
    ap.add_argument("--name", default="")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--fused-attention", action="store_true", help="fused attention kernels (default: math kernel)")
    args = ap.parse_args()
    C.set_size("s")
    C.attention_kernel(args.fused_attention)
    compute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
