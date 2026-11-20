#!/usr/bin/env python
"""Single neurons and residual coordinates against the SAE latents (paper App. B.2, "Robustness checks").

Units tested, on particle tokens of the small model:
- the 128 residual-stream coordinates at each reader's key input (norm1 output of body blocks 7 and 4; body output
  for classifier block 1), both signs;
- the MLP hidden neurons (output of the MLP activation) of every body block;
- the identified SAE latent of each reader (encoder pre-activation), for comparison.
Positives: sub-threshold neutrals (photon 0.3 GeV and neutral hadron 0.5 GeV added to JetClass jets, and the
sub-threshold neutrals of JetClass-II). Negatives: real JetClass neutrals; also all real JetClass particles.
Per unit: AUC (positives vs real JetClass neutrals), and coverage = fraction of positives above the unit's maximum
on all real JetClass particle tokens (a threshold with no false positive on real JetClass particles).

  python scripts/sae/neuron_comparison.py [--seed 0] [--sae-dir DIR] [--device cpu]
      -> $RESULTS_DIR/sae/small/neuron_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from softrepair import core as W
from softrepair import keyclip as K
from softrepair import tools as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

READERS = ("b7-6", "b4-0", "c1-2")


def auc_rows(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUC of every column (unit) for pos > neg, via ranks."""
    X = np.concatenate([pos, neg])
    r = X.argsort(0).argsort(0).astype(np.float64) + 1
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def collect(ad, raw, mods, p0):
    cap, out = {}, {k: [] for k in mods}
    hs = [m.register_forward_hook(lambda mod, i, o, k=k: cap.__setitem__(k, o.detach())) for k, m in mods.items()]
    try:
        with torch.no_grad():
            for s0 in range(0, len(raw), 256):
                x = np.asarray(raw[s0:s0 + 256], np.float32)
                ad.forward(ad.preprocess({"data": torch.as_tensor(x)}))
                for k in mods:
                    out[k].append(cap[k][:, p0:p0 + x.shape[1]].float().cpu().numpy())
    finally:
        for h in hs:
            h.remove()
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="JetClass jets per class")
    ap.add_argument("--seed", type=int, default=0, help="seed of the SAEs taken from --sae-dir")
    ap.add_argument("--saes", nargs=3, type=Path, default=None, help="SAE checkpoints of b7-6, b4-0, c1-2")
    ap.add_argument("--sae-dir", type=Path, default=None, help="SAE folder (default: <EDITS_DIR>/../sae)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    C.set_size_env("s")
    torch.set_num_threads(args.threads)
    ad = W.load_adapter(args.device, "s")
    body = ad.model.body
    p0 = ad.token_layout().particle_start
    site = {"b7-6": "residual coordinates at the input of b7-6", "b4-0": "residual coordinates at the input of b4-0",
            "c1-2": "residual coordinates at the input of c1-2 (body output)"}
    mods = {site["b7-6"]: body.in_blocks[7].norm1, site["b4-0"]: body.in_blocks[4].norm1, site["c1-2"]: body.norm}
    for b, blk in enumerate(body.in_blocks):
        mods[f"MLP neurons, body block {b}"] = blk.mlp.act
    jc, _ = T.jetclass_jets(args.n, 30000)
    jcii, _ = T.jetclass2_jets(1500, 10000)
    base = jc[:2000]
    A_jc = collect(ad, jc, mods, p0)
    A_jcii = collect(ad, jcii, mods, p0)
    adds = []
    for pid, pt, seed in ((6, 0.3, 41), (7, 0.5, 42)):
        x, ok, slot = T.add_one(base, np.full(len(base), pt), pid, seed)
        A = collect(ad, x, mods, p0)
        idx = np.flatnonzero(ok)
        adds.append({k: v[idx, slot[idx]] for k, v in A.items()})
    code = np.rint(jc[..., 4]).astype(int)
    real = jc[..., 2] != 0
    neu = real & np.isin(code, (6, 7))
    sub2 = T.sub_threshold(jcii)
    root = C.sae_root(args.sae_dir)
    saes = dict(zip(READERS, args.saes)) if args.saes else {r: C.sae_path(root, "s", r, args.seed) for r in READERS}
    sae_of = {site[r]: saes[r] for r in READERS}
    res = {"saes": {r: p.name for r, p in saes.items()}}
    for k in mods:
        pos = np.concatenate([adds[0][k], adds[1][k], A_jcii[k][sub2]])
        negN, negAll = A_jc[k][neu], A_jc[k][real]
        units = {"unit": (pos, negN, negAll)}
        if k.startswith("residual"):  # both signs of a coordinate
            units = {"unit (+)": (pos, negN, negAll), "unit (-)": (-pos, -negN, -negAll)}
        rec = {"n_units": int(pos.shape[1]), "n_pos": int(len(pos)), "n_neg_neutral": int(len(negN))}
        best = []
        for sign, (P, N, NA) in units.items():
            auc = auc_rows(P[:4000], N[:20000])
            cov = (P > NA.max(0)).mean(0)
            for i in np.argsort(-auc)[:3]:
                best.append({"unit": int(i), "sign": sign, "auc": float(auc[i]), "coverage_above_real_max": float(cov[i])})
            for i in np.argsort(-cov)[:2]:
                best.append({"unit": int(i), "sign": sign, "auc": float(auc[i]), "coverage_above_real_max": float(cov[i])})
        rec["best_units"] = sorted({(b["unit"], b["sign"]): b for b in best}.values(), key=lambda b: -b["auc"])[:5]
        rec["max_coverage_of_tested_units"] = max(b["coverage_above_real_max"] for b in best)
        if k in sae_of:
            lat = C.best_latents(C.identification(sae_of[k]))
            enc, b0, _ = K.latent_affine(sae_of[k], lat)
            z = lambda a: a @ enc.T + b0  # noqa: E731
            zp, zn, za = z(pos)[:, 0], z(negN)[:, 0], z(negAll)[:, 0]
            rec["sae_latent"] = {"latent": lat[0], "auc": float(auc_rows(zp[:4000, None], zn[:20000, None])[0]),
                                 "coverage_above_real_max": float((zp > za.max()).mean())}
        res[k] = rec
        b0_ = rec["best_units"][0]
        print(f"{k:58s} best unit {b0_['unit']:4d}{b0_['sign'][4:]} AUC {b0_['auc']:.4f} cov {b0_['coverage_above_real_max']:.3f}"
              f" | max cov {rec['max_coverage_of_tested_units']:.3f}"
              + (f" | SAE latent {rec['sae_latent']['latent']} AUC {rec['sae_latent']['auc']:.4f} cov "
                 f"{rec['sae_latent']['coverage_above_real_max']:.3f}" if "sae_latent" in rec else ""), flush=True)
    out = args.out or (W.OUT / "sae" / "small" / "neuron_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
