#!/usr/bin/env python
"""Generator head test (paper Sec. 4 "The generator head", Tab. tests_sizes).

  --mode update           Prediction of the generator (flow-matching velocity) on a lightly noised jet (t = 0.05,
                          0.2, 0.5; fixed noise), with and without one added particle, on the particles the two
                          inputs share: mean over jets of sum |v1 - v0| / sum |v0| (relative change of the update).
                          Added particles: photon of 0.3 GeV (below the JetClass thresholds), photon of 3 GeV
                          (control), photon and neutral hadron at f = 1e-3. Reference: the same change when the
                          class label is swapped (W <-> QCD) instead.
  --mode generated_mass   The noised W jet (t0 = 0.05, 0.2) is run back to t = 0 with the generator ODE (midpoint,
                          the step of the 128-step sampler), once without and once with the added particle (same
                          noise on the shared particles); relative change of the mass of the shared particles (and
                          of their log pT and positions). Reference: label swap.

Jets: JetClass W (and QCD) test files, rows 60000..60000+n (n = 1500 small, 1000 medium, 500 large in the paper).
  python scripts/evaluation/generator_test.py --mode update --size s [--edit EDIT.npz] --name NAME --out-dir DIR
      -> <out-dir>/update_<NAME>.json  (paper cell: ["t=0.05"]["W"]["photon_0.3GeV"])
  python scripts/evaluation/generator_test.py --mode generated_mass --size s [--edit EDIT.npz] --name NAME --out-dir DIR
      -> <out-dir>/generated_mass_<NAME>.json  (paper: ["t0=0.05"]["photon_0.3GeV"]["dmass_rel"]["mean"])
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

WJ, QCD = 9, 2
NAMES = {2: "QCD", 9: "W"}


def update(ad, W, H, raw: dict, res: dict) -> None:
    for t in (0.05, 0.2, 0.5):
        out = {}
        for lab, other in ((WJ, QCD), (QCD, WJ)):
            x0 = raw[lab]
            eps = np.random.default_rng(7).standard_normal(x0.shape[:2] + (4,)).astype(np.float32)
            labs = np.full(len(x0), lab)
            v0 = W.gen_velocity(ad, x0, labs, eps, t)
            real0 = x0[..., 2] != 0

            def rel(v1, keep):
                num = (np.abs(v1 - v0).sum(2) * keep).sum(1)
                den = (np.abs(v0).sum(2) * keep).sum(1)
                return float(np.mean(num / np.maximum(den, 1e-12)))
            r = {"label_swap_reference": rel(W.gen_velocity(ad, x0, np.full(len(x0), other), eps, t), real0)}
            P = {"photon_0.3GeV": H.add_photon(x0, 0.3, 11 + lab), "photon_3GeV_control": H.add_photon(x0, 3.0, 21 + lab),
                 "photon_f1e-3": W.perturb(x0, "soft1e-3", 31 + lab),
                 "neutral_hadron_f1e-3": W.perturb(x0, "soft7_1e-3", 41 + lab)}
            for pn, (x1, ok) in P.items():
                keep = real0 & np.all(x1 == x0, axis=2) & ok[:, None]
                r[pn] = rel(W.gen_velocity(ad, x1, labs, eps, t), keep)
            out[NAMES[lab]] = r
        res[f"t={t}"] = out
        print(t, json.dumps(out), flush=True)


def generated_mass(ad, W, H, x0: np.ndarray, res: dict, t0s, batch: int) -> None:
    from torchdiffeq import odeint
    eps = np.random.default_rng(7).standard_normal(x0.shape[:2] + (4,)).astype(np.float32)

    def finish(raw, lab, t0):
        out = []
        ts = torch.linspace(t0, 0.0, max(int(round(127 * t0)), 2) + 1, device=ad.device)
        with torch.no_grad():
            for s in range(0, len(raw), batch):
                b = ad.preprocess({"data": torch.as_tensor(np.asarray(raw[s:s + batch], np.float32))})
                X = b["X"].to(ad.device); pid = b["pid"].to(ad.device); add = b["add_info"].to(ad.device)  # noqa: E702
                mask = (X[:, :, 2:3] != 0).float()
                y = torch.full((X.shape[0],), lab, device=ad.device).int()
                z = ((1 - t0) * X + t0 * torch.as_tensor(eps[s:s + batch], device=ad.device)) * mask

                def f(t, xt):
                    tt = t * torch.ones((xt.shape[0],), device=ad.device)
                    return ad.model.generator(ad.model.body(xt * mask, None, pid, add, tt), y).float() * mask
                out.append(odeint(f, z, ts, method="midpoint")[-1].float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def mass(f, keep):
        pt = np.exp(f[..., 2]) * keep
        px, py, pz, E = pt * np.cos(f[..., 1]), pt * np.sin(f[..., 1]), pt * np.sinh(f[..., 0]), pt * np.cosh(f[..., 0])
        return np.sqrt(np.maximum(E.sum(1) ** 2 - px.sum(1) ** 2 - py.sum(1) ** 2 - pz.sum(1) ** 2, 0))

    def stat(v, ok):
        v = v[ok]
        return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))), "median": float(np.median(v))}

    def compare(f0, f1, keep, ok):
        k = keep.astype(np.float64)
        n = np.maximum(k.sum(1), 1)
        w = np.exp(x0[..., 2].astype(np.float64)) * k
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        dl = np.abs(f1[..., 2] - f0[..., 2])
        dr = np.hypot(f1[..., 0] - f0[..., 0], f1[..., 1] - f0[..., 1])
        m0, m1 = mass(f0, k), mass(f1, k)
        return {"dlogpt": stat((dl * k).sum(1) / n, ok), "dlogpt_ptw": stat((dl * w).sum(1), ok),
                "dR": stat((dr * k).sum(1) / n, ok), "dR_ptw": stat((dr * w).sum(1), ok),
                "dmass_rel": stat(np.abs(m1 - m0) / np.maximum(m0, 1e-9), ok),
                "dmass_signed_rel": stat((m1 - m0) / np.maximum(m0, 1e-9), ok), "n_jets": int(ok.sum())}

    res["class"] = "W"
    real0 = x0[..., 2] != 0
    allok = np.ones(len(x0), bool)
    P = {"photon_0.3GeV": H.add_photon(x0, 0.3, 11 + WJ), "photon_3GeV_control": H.add_photon(x0, 3.0, 21 + WJ),
         "neutral_hadron_f1e-3": W.perturb(x0, "soft7_1e-3", 41 + WJ)}
    for t0 in t0s:
        f0 = finish(x0, WJ, t0)
        r = {"regeneration_vs_real": compare(x0[..., :4].astype(np.float64), f0, real0, allok),
             "label_swap_reference": compare(f0, finish(x0, QCD, t0), real0, allok)}
        for pn, (x1, ok) in P.items():
            keep = real0 & np.all(x1 == x0, axis=2) & ok[:, None]
            r[pn] = compare(f0, finish(x1, WJ, t0), keep, ok)
        res[f"t0={t0}"] = r
        print(t0, json.dumps({k: {m: round(v[m]["mean"], 5) for m in ("dlogpt", "dR", "dmass_rel")}
                              for k, v in r.items()}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("update", "generated_mass"))
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--edit", type=Path, default=None)
    ap.add_argument("--name", default=None, help="output name (default: original_model or the edit file stem)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None, help="jets per class (default 1500 update / 1000 generated_mass)")
    ap.add_argument("--t0", default="0.05,0.2", help="generated_mass: start times")
    ap.add_argument("--batch", type=int, default=250, help="generated_mass: ODE batch")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    C.setup_device(a.size, {"s": None, "m": 128, "l": 32}[a.size])
    from softrepair import core as W
    from softrepair import heads as H
    torch.set_num_threads(8)
    n = a.n or (1500 if a.mode == "update" else 1000)
    ad = W.load_adapter(a.device, a.size)
    orig = W.original_state(ad)
    if a.edit:
        W.set_state(ad, W.load_candidate(a.edit, orig))
    name = a.name or (a.edit.stem if a.edit else "original_model")
    res = {"name": name, "model_size": a.size, "edit": a.edit.name if a.edit else None, "n": n}
    if a.mode == "update":
        update(ad, W, H, {lab: C.load(lab, 60000, n) for lab in (WJ, QCD)}, res)
    else:
        generated_mass(ad, W, H, C.load(WJ, 60000, n), res, [float(t) for t in a.t0.split(",")], a.batch)
    C.dump(a.out_dir / f"{a.mode}_{name}.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
