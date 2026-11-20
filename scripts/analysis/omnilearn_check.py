#!/usr/bin/env python
"""Soft-particle (infrared) test of the original OmniLearn (Mikuni & Nachman, arXiv:2404.16091 and arXiv:2502.14652),
which was pretrained on JetClass only (Sec. 2: "We also tested OmniLearn ... and found no infrared sensitivity to
soft neutral particles").

Model: PET (TensorFlow 2.14), checkpoint PET_jetclass_8_local_layer_scale_token_baseline_all.weights.h5 from
github.com/ViniciusMikuni/OmniLearn (commit 6e55210). Its 10 JetClass outputs are used directly, as for OmniLearned:
F_W = logit(Wqq) - logit(QCD). Class order (OmniLearn dataloader): QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl.

Jets: raw JetClass test ROOT files (test_20M; OmniLearn trained on train_100M), the first --n jets of
WToQQ_106 and ZJetsToNuNu_114, and --n-class jets of one file per class for a 10-class sanity check. Features exactly
as OmniLearn's preprocessing/dataloader.read_file + utils.DataLoader.preprocess (13 particle, 4 jet features).
Perturbation as for OmniLearned (core._soft / tools.add_one): one massless neutral, uniform in dR < 0.4 of the
jet axis, at pT = f * jet pT or at a fixed pT; the jet four-vector and multiplicity include it.
Needs tensorflow 2.14, uproot and awkward (not in requirements.txt).

  python scripts/analysis/omnilearn_check.py --omnilearn <OmniLearn repo> --root <dir with test_20M ROOT files> --out <dir>
      -> <dir>/omnilearn_soft_particles.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

FILES = ["ZJetsToNuNu_114", "HToBB_114", "HToCC_108", "HToGG_109", "HToWW4Q_110", "HToWW2Q1L_104", "ZToQQ_114",
         "WToQQ_106", "TTBar_119", "TTBarLep_118"]          # in OmniLearn's class order
I_QCD, I_W = 0, 7
MEAN_PART = np.array([0.0, 0.0, -0.0278, 1.8999407, -0.027, 2.244736, 0.0, 0, 0, 0, 0, 0, 0], np.float32)
STD_PART = np.array([0.215, 0.215, 0.070, 1.2212526, 0.069, 1.2334691, 1.0, 1, 1, 1, 1, 1, 1], np.float32)
MEAN_JET = np.array([6.18224920e+02, 0.0, 1.2064709e+02, 3.94133173e+01], np.float32)
STD_JET = np.array([106.71761, 0.88998157, 40.196922, 15.096386], np.float32)
NP = 150
PART = ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi", "part_charge",
        "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon"]
JET = ["jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_nparticles"]


def load_root(path: Path, n: int) -> dict:
    import awkward as ak
    import uproot
    t = uproot.open(path)["tree"].arrays(PART + JET, entry_stop=n)
    out = {k: ak.to_numpy(ak.fill_none(ak.pad_none(t[k], NP, clip=True), 0.0)).astype(np.float64) for k in PART}
    out.update({k: ak.to_numpy(t[k]).astype(np.float64) for k in JET})
    out["mask"] = np.arange(NP)[None, :] < np.minimum(ak.to_numpy(ak.num(t["part_px"])), NP)[:, None]
    return out


def add_soft(d: dict, kind: str, seed: int, frac: float | None = None, pt: float | None = None,
             keep_mult: bool = False) -> dict:
    """Add one massless photon or neutral hadron in the first free slot (jets with 150 particles are left as is)."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in d.items()}
    n = len(d["jet_pt"])
    th, rr = rng.uniform(0, 2 * np.pi, n), 0.4 * np.sqrt(rng.uniform(size=n))
    de, dp = rr * np.cos(th), rr * np.sin(th)
    p = frac * d["jet_pt"] if frac is not None else np.full(n, pt)
    eta, phi = d["jet_eta"] + de, d["jet_phi"] + dp
    px, py, pz, E = p * np.cos(phi), p * np.sin(phi), p * np.sinh(eta), p * np.cosh(eta)
    free = (~d["mask"]).argmax(1)
    ok = ~d["mask"].all(1)
    j, s = np.flatnonzero(ok), free[ok]
    for k, v in (("part_px", px), ("part_py", py), ("part_pz", pz), ("part_energy", E), ("part_deta", de),
                 ("part_dphi", dp)):
        out[k][j, s] = v[ok]
    for k in ("part_charge", "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron",
              "part_isMuon"):
        out[k][j, s] = 0.0
    out["part_isPhoton" if kind == "photon" else "part_isNeutralHadron"][j, s] = 1.0
    out["mask"][j, s] = True
    # jet four-vector and multiplicity include the new particle
    jpx, jpy = d["jet_pt"] * np.cos(d["jet_phi"]), d["jet_pt"] * np.sin(d["jet_phi"])
    jpz = d["jet_pt"] * np.sinh(d["jet_eta"])
    jpx, jpy, jpz, jE = jpx + px * ok, jpy + py * ok, jpz + pz * ok, d["jet_energy"] + E * ok
    out["jet_pt"], out["jet_phi"] = np.hypot(jpx, jpy), np.arctan2(jpy, jpx)
    out["jet_eta"], out["jet_energy"] = np.arcsinh(jpz / out["jet_pt"]), jE
    out["jet_nparticles"] = d["jet_nparticles"] + (0 if keep_mult else ok)
    return out


def features(d: dict):
    """OmniLearn read_file + DataLoader.preprocess."""
    m = d["mask"]
    pt = np.hypot(d["part_px"], d["part_py"])
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = [d["part_deta"] * np.sign(d["jet_eta"])[:, None], d["part_dphi"],
                np.log(1e-6 + 1.0 - pt / d["jet_pt"][:, None]), np.log(pt),
                np.log(1e-6 + 1.0 - d["part_energy"] / d["jet_energy"][:, None]), np.log(d["part_energy"]),
                np.hypot(d["part_deta"], d["part_dphi"]), d["part_charge"], d["part_isChargedHadron"],
                d["part_isNeutralHadron"], d["part_isPhoton"], d["part_isElectron"], d["part_isMuon"]]
    x = np.stack(cols, -1)
    x[~m] = 0.0
    x = (m[..., None] * (x - MEAN_PART) / STD_PART).astype(np.float32)
    jm = np.sqrt(np.maximum(d["jet_energy"] ** 2 - (d["jet_pt"] * np.cosh(d["jet_eta"])) ** 2, 0.0))
    jet = np.stack([d["jet_pt"], d["jet_eta"], jm, d["jet_nparticles"]], 1)
    return x, m.astype(np.float32), ((jet - MEAN_JET) / STD_JET).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omnilearn", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-class", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--keep-mult", action="store_true",
                    help="control: leave the jet-multiplicity input unchanged when the particle is added")
    ap.add_argument("--scan", choices=("full", "short"), default="full")
    ap.add_argument("--only-class", action="store_true", help="stop after the 10-class check")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.omnilearn / "scripts"))
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    from PET import PET

    model = PET(num_feat=13, num_jet=4, num_classes=10, local=True, num_layers=8, simple=False, layer_scale=True,
                talking_head=False, mode="all", class_activation=None)
    ck = a.omnilearn / "checkpoints" / "PET_jetclass_8_local_layer_scale_token_baseline_all.weights.h5"
    # The checkpoint stores its metric trackers under other names; they are not model weights, so skip them.
    tf.keras.metrics.Metric.load_own_variables = lambda self, store: None
    model.load_weights(str(ck))

    def logits(d):
        x, m, j = features(d)
        out = []
        for s in range(0, len(x), a.batch):
            b = slice(s, s + a.batch)
            o = model.classifier([x[b], x[b, :, :2], m[b, :, None], j[b], np.zeros((len(x[b]), 1), np.float32)],
                                 training=False)
            out.append(o[0].numpy())
        return np.concatenate(out).astype(np.float64)

    res: dict = {"checkpoint": ck.name, "files": FILES, "n": a.n, "n_class": a.n_class}
    t0 = time.time()
    acc, nll = [], []
    for k, f in enumerate(FILES if a.n_class > 0 else []):
        l = logits(load_root(a.root / f"{f}.root", a.n_class))
        acc.append(float(np.mean(l.argmax(1) == k)))
        z = l - l.max(1, keepdims=True)
        nll.append(float(np.mean(np.log(np.exp(z).sum(1)) - z[:, k])))
    res["ten_class"] = {"acc_per_class": acc, "acc": float(np.mean(acc)) if acc else None,
                        "nll": float(np.mean(nll)) if nll else None}
    if acc:
        print("10-class acc", np.round(acc, 3), "mean", round(float(np.mean(acc)), 4), flush=True)
    if a.only_class:
        (a.out / "omnilearn_soft_particles.json").write_text(json.dumps(res, indent=1))
        return 0

    sets = {"W": load_root(a.root / f"{FILES[I_W]}.root", a.n), "QCD": load_root(a.root / f"{FILES[I_QCD]}.root", a.n)}
    # the softest photon / neutral hadron in these clean jets (the JetClass reconstruction floor)
    floor = {}
    for name, d in sets.items():
        pt = np.hypot(d["part_px"], d["part_py"])
        floor[name] = {"photon_pt_min": float(pt[d["mask"] & (d["part_isPhoton"] > 0)].min()),
                       "nh_E_min": float(d["part_energy"][d["mask"] & (d["part_isNeutralHadron"] > 0)].min())}
    res["clean_floor"] = floor
    F = {name: (lambda l: l[:, I_W] - l[:, I_QCD])(logits(d)) for name, d in sets.items()}
    thr = float(np.quantile(F["W"], 0.20))
    from sklearn.metrics import roc_auc_score
    res["working_point"] = {"thr": thr, "eff_W": 0.8, "acc_QCD": float(np.mean(F["QCD"] > thr)),
                            "auc": float(roc_auc_score(np.r_[np.ones(len(F["W"])), np.zeros(len(F["QCD"]))],
                                                       np.r_[F["W"], F["QCD"]])),
                            "F_sep": float(F["W"].mean() - F["QCD"].mean())}
    print("clean floor", floor, "\nworking point", res["working_point"], flush=True)

    perts = {}
    res["keep_mult"] = a.keep_mult
    for f in ((1e-4, 1e-3) if a.scan == "short" else (1e-4, 3e-4, 1e-3, 3e-3)):
        perts[f"photon_f{f:g}"] = dict(kind="photon", frac=f)
        perts[f"nh_f{f:g}"] = dict(kind="nh", frac=f)
    for p in ((0.1, 0.3, 0.85, 3.0) if a.scan == "short" else
              (0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.85, 1.05, 1.3, 1.7, 2.2, 3.0, 5.0, 10.0)):
        perts[f"photon_pt{p:g}"] = dict(kind="photon", pt=p)
        perts[f"nh_pt{p:g}"] = dict(kind="nh", pt=p)
    res["perturbations"] = {}
    for pname, kw in perts.items():
        row = {}
        for name, d in sets.items():
            dp = add_soft(d, kw["kind"], seed=zlib.crc32(pname.encode()), frac=kw.get("frac"), pt=kw.get("pt"),
                         keep_mult=a.keep_mult)
            l = logits(dp)
            Fp = l[:, I_W] - l[:, I_QCD]
            dF = Fp - F[name]
            row[name] = {"mean_dF": float(dF.mean()), "median_dF": float(np.median(dF)),
                         "mean_abs_dF": float(np.abs(dF).mean()), "pass": float(np.mean(Fp > thr))}
        res["perturbations"][pname] = row
        print(f"{pname:15s} W: dF {row['W']['mean_dF']:+.3f} |dF| {row['W']['mean_abs_dF']:.3f} "
              f"eff {row['W']['pass']:.3f}   QCD: dF {row['QCD']['mean_dF']:+.3f} acc {row['QCD']['pass']:.4f}",
              flush=True)
    res["seconds"] = time.time() - t0
    (a.out / "omnilearn_soft_particles.json").write_text(json.dumps(res, indent=1))
    print("wrote", a.out / "omnilearn_soft_particles.json", f"{res['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
