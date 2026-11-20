#!/usr/bin/env python
"""W-tagger score F_W = logit_W - logit_QCD on clean jets and on copies with one added particle (Sec. 2, Fig. 2 left).

Per jet: F_W of the nominal jet and of perturbed copies: one photon at a fraction f = 1e-3 and 3e-3 of the jet pT
(core.perturb soft1e-3 / soft3e-3), a collinear split of the hardest neutral (split2), a random reordering of the
particles, and a pT scan (one photon, pT log-uniform in 0.05-10 GeV, 2 draws per jet). With --soft-pid 7 the
f = 1e-3 / 3e-3 copies get one neutral hadron instead of a photon.
Numbers: mean / median change of F_W per perturbation, and the W efficiency and QCD acceptance at the cuts that keep
80% and 50% of the nominal W jets.

Jets: JetClass validation, W file 5 and QCD file 32 (the test-jet files), rows 60000..60000+n.

  python scripts/analysis/w_tagger_response.py [--size s] [--n 10000] [--edit EDIT.npz] [--soft-pid 7] [--out STEM]
      -> STEM.npz (per-jet arrays) and STEM_summary.json
      default STEM: RESULTS_DIR/figures/<size>/w_tagger_response[_neutral_hadron]
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


def reorder(raw: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = raw.copy()
    for j in range(len(out)):
        out[j] = out[j][rng.permutation(out.shape[1])]
    return out


def photon_scan(raw: np.ndarray, seed: int, draws: int = 2) -> tuple[list[np.ndarray], np.ndarray]:
    from softrepair import heads as H
    rng = np.random.default_rng(seed)
    outs, pts = [], []
    for d in range(draws):
        pt = np.exp(rng.uniform(np.log(0.05), np.log(10.0), len(raw)))
        pt = np.where(np.abs(pt - 1.0) < 1e-3, 1.001, pt)  # log pT = 0 would read as padding
        out = raw.copy()
        for j in range(len(raw)):
            o, ok = H.add_photon(raw[j:j + 1], float(pt[j]), int(rng.integers(1 << 31)))
            out[j] = o[0]
        outs.append(out)
        pts.append(pt)
    return outs, np.concatenate(pts)


def eff(x, cut):
    k, n = int((x > cut).sum()), len(x)
    p = k / n
    return p, float(np.sqrt(p * (1 - p) / n)), k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--n", type=int, default=10000, help="jets per class")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--no-scan", dest="scan", action="store_false")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--edit", type=Path, default=None, help="edited weights (core.load_candidate)")
    ap.add_argument("--soft-pid", type=int, default=6, choices=(6, 7),
                    help="particle added in the f = 1e-3 / 3e-3 copies: 6 photon, 7 neutral hadron")
    ap.add_argument("--out", type=Path, default=None, help="output stem (default under RESULTS_DIR/figures/<size>)")
    ap.add_argument("--fused-attention", action="store_true", help="fused attention kernels (default: math kernel)")
    args = ap.parse_args()
    C.set_size(args.size)
    from softrepair import core as W
    from softrepair import heads as H
    C.attention_kernel(args.fused_attention)

    torch.set_num_threads(args.threads)
    ad = W.load_adapter(args.device, args.size)
    if args.edit:
        W.set_state(ad, W.load_candidate(args.edit, W.original_state(ad)))

    def fw(raw):
        return H.logodds_W(W.logits(ad, raw, batch=512))

    arr, num = {}, {"size": args.size, "n_per_class": args.n, "rows": [60000, 60000 + args.n],
                    "files": {"W": H.FILES[H.WJ], "QCD": H.FILES[H.QCD]}}
    for lab, name in ((H.WJ, "W"), (H.QCD, "QCD")):
        raw = H.load(lab, 60000, args.n)
        real = raw[..., 2] != 0
        pid = np.rint(raw[..., 4]).astype(int)
        arr[f"photon_pt_{name}"] = np.exp(raw[..., 2][real & (pid == H.PHOTON)]).astype(np.float32)
        arr[f"jet_pt_{name}"] = (np.exp(raw[..., 2]) * real).sum(1).astype(np.float32)
        arr[f"F0_{name}"] = fw(raw)
        sp = "soft" if args.soft_pid == 6 else f"soft{args.soft_pid}_"  # 7: one neutral hadron instead of a photon
        for v, (x, ok) in {"f1e-3": W.perturb(raw, f"{sp}1e-3", 11 + lab), "f3e-3": W.perturb(raw, f"{sp}3e-3", 21 + lab),
                           "split2": W.perturb(raw, "split2", 31 + lab),
                           "reorder": (reorder(raw, 41 + lab), np.ones(len(raw), bool))}.items():
            arr[f"F_{v}_{name}"], arr[f"ok_{v}_{name}"] = fw(x), ok
        if args.scan:
            outs, pts = photon_scan(raw, 51 + lab)
            arr[f"scan_pt_{name}"] = pts
            arr[f"scan_dF_{name}"] = np.concatenate([fw(o) for o in outs]) - np.tile(arr[f"F0_{name}"], len(outs))
        print(f"[{args.size}] {name} done", flush=True)
    for name in ("W", "QCD"):
        F0 = arr[f"F0_{name}"]
        for v in ("f1e-3", "f3e-3", "split2", "reorder"):
            ok = arr[f"ok_{v}_{name}"]
            d = (arr[f"F_{v}_{name}"] - F0)[ok]
            num[f"{v}_{name}"] = {"n": int(ok.sum()), "signed_mean": float(d.mean()), "signed_mean_err": float(d.std() / np.sqrt(len(d))),
                                  "mean_abs": float(np.abs(d).mean()), "median": float(np.median(d)),
                                  "frac_negative": float((d < 0).mean()), "std": float(d.std())}
    num["median_gap_W_minus_QCD"] = float(np.median(arr["F0_W"]) - np.median(arr["F0_QCD"]))
    num["mean_gap_W_minus_QCD"] = float(arr["F0_W"].mean() - arr["F0_QCD"].mean())
    for wp in (0.8, 0.5):
        cut = float(np.quantile(arr["F0_W"], 1 - wp))
        d = {"cut_FW": cut, "cut_score": float(1 / (1 + np.exp(-cut)))}
        for v in ("nominal", "f1e-3", "f3e-3", "split2"):
            for name in ("W", "QCD"):
                x = arr["F0_" + name] if v == "nominal" else arr[f"F_{v}_{name}"]
                p, e, k = eff(x, cut)
                d[f"{v}_{name}"] = {"eff": p, "err": e, "pass": k}
        num[f"wp{int(wp * 100)}"] = d
    ph = np.concatenate([arr["photon_pt_W"], arr["photon_pt_QCD"]])
    num["photon_pt"] = {"min": float(ph.min()), "p01": float(np.quantile(ph, 0.01)), "p001": float(np.quantile(ph, 0.001)),
                        "n_photons": int(len(ph))}
    jp = np.concatenate([arr["jet_pt_W"], arr["jet_pt_QCD"]])
    num["jet_scalar_pt"] = {"min": float(jp.min()), "median": float(np.median(jp)), "max": float(jp.max())}
    num["edit"] = args.edit.name if args.edit else None
    num["soft_pid"] = args.soft_pid
    stem = args.out or C.out_dir(args.size) / ("w_tagger_response" + ("_neutral_hadron" if args.soft_pid == 7 else ""))
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{stem}.npz", **arr)
    Path(f"{stem}_summary.json").write_text(json.dumps(num, indent=1))
    print(json.dumps(num, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
