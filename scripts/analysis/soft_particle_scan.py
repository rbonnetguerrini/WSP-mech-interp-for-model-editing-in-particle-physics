#!/usr/bin/env python
"""Response of the W tagger to one added particle of each type, as a function of its pT (Sec. 2, App. A).

For each particle family in JetClass (particle codes: photon 6, neutral hadron 7, charged hadron {0, 8},
electron {2, 4}, muon {1, 5}):
1. data: the pT distribution of the real particles of that family in the clean jets (min, 0.1 / 1 / 5% quantiles);
2. scan: add one particle of that family at pT log-uniform in [0.05, 10] GeV, uniform within dR < 0.4 of the jet
   axis (massless, E = pT cosh(eta)); charge sign drawn 50/50 between the two codes. Impact parameters (columns
   5-8): 0 for neutrals; for charged types copied from a random real charged particle of the same jet.
   Record dF_W = F_W(perturbed) - F_W(clean), F_W = logit_W - logit_QCD.
3. impact-parameter swaps: the same particle code with the impact-parameter columns set the other way round
   ("photon, track IPs", "charged hadron, zero IPs", ...), and the finer swaps of the values (columns 5, 7) or the
   uncertainties (columns 6, 8) only (e.g. "photon, track IP sigmas only").

Jets: W file 5 and QCD file 32 (test-jet files), rows 60000..60000+n.

  python scripts/analysis/soft_particle_scan.py [--n 5000] [--draws 2] [--edit EDIT.npz] [--name NAME]
        [--pt-cut 0.75] [--families photon "neutral hadron" ...] [--attn] [--pt-min 0.001 --pt-max 10]
      -> RESULTS_DIR/figures/<size>/soft_particle_scans/<NAME>.npz / .json (NAME default: original_model or the
         edit file stem)
  --pt-min: scan from that pT (same bin density, so more bins: use a larger --n);
  --edit: edited weights (core.load_candidate); --pt-cut: drop every particle below this pT (GeV) before the model,
  in clean and perturbed jets alike; --attn: also record the summary-token attention of backbone block 7 / head 6
  (b7/6, the main reader of OmniLearned-s; other heads with --heads) on the added particle.
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

FAMILIES = {"photon": (6,), "neutral hadron": (7,), "charged hadron": (0, 8), "electron": (2, 4), "muon": (1, 5)}
CHARGED = (0, 8, 1, 5, 2, 4)
# impact-parameter swap test: same particle code, impact-parameter columns (5-8: tanh d0, sigma d0, tanh dz,
# sigma dz) set the other way round. "track" = copied from a random real charged particle of the same jet.
SWAPS = {"charged hadron, zero IPs": ((0, 8), "zero"), "photon, track IPs": ((6,), "track"),
         "neutral hadron, track IPs": ((7,), "track")}
# finer swaps: which of the four columns matters, the IP values (5, 7) or their uncertainties (6, 8)?
SWAPS_FINE = {"photon, track IP values only": ((6,), "track_vals"),   # sigma stays 0
              "photon, track IP sigmas only": ((6,), "track_sigmas"),  # values stay 0
              "charged hadron, zero IP sigmas": ((0, 8), "zero_sigmas"),
              "charged hadron, zero IP values": ((0, 8), "zero_vals")}
HB = np.geomspace(0.05, 10, 41)  # pT bins (GeV)


def add_particle(raw: np.ndarray, codes: tuple[int, ...], rng: np.random.Generator, ip: str | None = None):
    """ip: None = natural (track IPs for charged codes, zero for neutral), "zero" or "track" to force."""
    from softrepair import core as W
    out, ok = raw.copy(), np.zeros(len(raw), bool)
    etas = np.zeros(len(raw))
    pts = np.exp(rng.uniform(np.log(HB[0]), np.log(HB[-1]), len(raw)))
    for j in range(len(raw)):
        jet = out[j]
        r, slot = W._real(jet), W._free_slot(jet)
        if slot < 0:
            continue
        eta_j = W._jet_eta(jet, r)
        th, rr = rng.uniform(0, 2 * np.pi), 0.4 * np.sqrt(rng.uniform())
        de, dp = rr * np.cos(th), rr * np.sin(th)
        code = codes[rng.integers(len(codes))]
        row = np.zeros(jet.shape[1], np.float32)
        row[:5] = de, dp, np.log(pts[j]), np.log(pts[j] * np.cosh(eta_j + de)), code
        ch = r[np.isin(np.rint(jet[r, 4]).astype(int), CHARGED)]
        trk = jet[rng.choice(ch), 5:] if ch.size else np.zeros(4, np.float32)
        if (ip == "track") or (ip is None and code in CHARGED):
            row[5:] = trk
        elif ip == "track_vals":
            row[5], row[7] = trk[0], trk[2]
        elif ip == "track_sigmas":
            row[6], row[8] = trk[1], trk[3]
        elif ip == "zero_sigmas":
            row[5], row[7] = trk[0], trk[2]
        elif ip == "zero_vals":
            row[6], row[8] = trk[1], trk[3]
        etas[j] = eta_j + de
        jet[slot] = row
        ok[j] = True
    add_particle.last_eta = etas
    return out, ok, pts


def compute(args) -> None:
    global HB
    from softrepair import core as W
    from softrepair import heads as H
    if args.pt_min is not None:  # wider scan (e.g. down to 1 MeV) at the same bin density as the default
        per_dec = (len(HB) - 1) / np.log10(HB[-1] / HB[0])
        HB = np.geomspace(args.pt_min, args.pt_max, int(round(per_dec * np.log10(args.pt_max / args.pt_min))) + 1)
    torch.set_num_threads(args.threads)
    ad = W.load_adapter(args.device, args.size)
    heads = [tuple(int(v) for v in h.split(":")) for h in args.heads]
    if args.edit:
        W.set_state(ad, W.load_candidate(args.edit, W.original_state(ad)))
    cut = (lambda x: W.pt_cut(x, args.pt_cut)) if args.pt_cut else (lambda x: x)  # noqa: E731
    res, arr = {"n_per_class": args.n, "draws": args.draws, "size": args.size, "heads": args.heads,
                "pt_bins_GeV": HB.tolist(), "families": {}}, {}
    t0 = time.time()
    clean, raws = {}, {}
    for lab, name in ((H.WJ, "W"), (H.QCD, "QCD")):
        raws[name] = H.load(lab, 60000, args.n)
        clean[name] = H.logodds_W(W.logits(ad, cut(raws[name]), batch=512))
    allraw = np.concatenate(list(raws.values()))
    real = allraw[..., 2] != 0
    pid, pt = np.rint(allraw[..., 4]).astype(int), np.exp(allraw[..., 2].astype(np.float64))
    for name in ("W", "QCD"):  # collinear-split references: hardest neutral, hardest charged
        for sp in ("split2", "split2c"):
            pert, ok = W.perturb(raws[name], sp, 500 + (name == "QCD"))
            d = (H.logodds_W(W.logits(ad, cut(pert), batch=512)) - clean[name])[ok]
            res.setdefault("splits", {})[f"{sp}_{name}"] = {"mean_abs": float(np.abs(d).mean()), "n": int(ok.sum())}
    todo = {f: (c, None) for f, c in FAMILIES.items()} | (SWAPS if args.swaps else {}) | SWAPS_FINE
    if args.families:
        todo = {k: v for k, v in todo.items() if k in args.families}
    res.update({"edit": args.edit.name if args.edit else None, "pt_cut": args.pt_cut})
    for fam, (codes, ipmode) in todo.items():
        sel = real & np.isin(pid, codes)
        p = pt[sel]
        dat = {"n_real": int(sel.sum()), "share_of_constituents": float(sel.sum() / real.sum())}
        if p.size:
            dat.update({"pt_min": float(p.min()), "pt_q001": float(np.quantile(p, 0.001)),
                        "pt_q01": float(np.quantile(p, 0.01)), "pt_q05": float(np.quantile(p, 0.05)),
                        "pt_median": float(np.median(p))})
        arr[f"real_pt_{fam}"] = p.astype(np.float32)
        arr[f"real_E_{fam}"] = np.exp(allraw[..., 3].astype(np.float64))[sel].astype(np.float32)
        rec = {"data": dat}
        for name in ("W", "QCD"):
            pts_all, d_all, att_all, eta_all, jet_all = [], [], [], [], []
            att_more = {h: [] for h in heads[1:]}
            for k in range(args.draws):
                rng = np.random.default_rng(1000 * k + 17 * len(fam) + (0 if name == "W" else 1))
                pert, ok, pts = add_particle(raws[name], codes, rng, ipmode)
                F = H.logodds_W(W.logits(ad, cut(pert), batch=512))
                pts_all.append(pts[ok]); d_all.append((F - clean[name])[ok]); jet_all.append(np.flatnonzero(ok))
                eta_all.append(add_particle.last_eta[ok])
                if args.attn:  # head attention on the added slot (the slot add_particle filled)
                    slot = np.array([W._free_slot(r) for r in raws[name]])
                    xin = cut(pert) if args.pt_cut else pert
                    for hi, (bb, hh) in enumerate(heads):
                        w6 = C.head_weights(ad, xin, block=bb, head=hh)
                        a6 = w6[np.arange(len(pert)), :, np.clip(slot, 0, None)].mean(1)[ok & (slot >= 0)]
                        (att_all if hi == 0 else att_more[(bb, hh)]).append(a6)
            pts_all, d_all = np.concatenate(pts_all), np.concatenate(d_all)
            att = np.concatenate(att_all) if att_all else None
            arr[f"pt_{fam}_{name}"], arr[f"dF_{fam}_{name}"] = pts_all.astype(np.float32), d_all.astype(np.float32)
            arr[f"eta_{fam}_{name}"] = np.concatenate(eta_all).astype(np.float32)
            arr[f"jet_{fam}_{name}"] = np.concatenate(jet_all).astype(np.int32)  # index into F0_<name>
            arr[f"F0_{name}"] = clean[name].astype(np.float32)  # clean F_W per jet (working point, separation)
            idx = np.digitize(pts_all, HB)
            if att is not None and len(att) == len(pts_all):
                ia = np.digitize(pts_all, HB)
                arr[f"att_{fam}_{name}"] = att.astype(np.float32)
                att_rec = [float(np.median(att[ia == i])) if (ia == i).sum() > 5 else None for i in range(1, len(HB))]
            else:
                att_rec = None
            more_rec, ia = {}, np.digitize(pts_all, HB)
            for (bb, hh), v in att_more.items():
                if v:
                    a2 = np.concatenate(v)
                    arr[f"att_b{bb}h{hh}_{fam}_{name}"] = a2.astype(np.float32)
                    more_rec[f"median_attention_b{bb}h{hh}"] = [float(np.median(a2[ia == i])) if (ia == i).sum() > 5
                                                               else None for i in range(1, len(HB))]
            rec[name] = {"median_head6_attention": att_rec,
                         "mean_abs_dF": [float(np.abs(d_all[idx == i]).mean()) if (idx == i).sum() > 5 else None
                                         for i in range(1, len(HB))],
                         "sem_abs_dF": [float(np.abs(d_all[idx == i]).std() / np.sqrt((idx == i).sum()))
                                        if (idx == i).sum() > 5 else None for i in range(1, len(HB))],
                         "mean_signed_dF": [float(d_all[idx == i].mean()) if (idx == i).sum() > 5 else None
                                            for i in range(1, len(HB))], **more_rec}
            for lo, hi in ((0.001, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.45), (1.0, 3.0), (3.0, 10.0)):
                m = (pts_all >= lo) & (pts_all < hi)
                rec[name][f"mean_abs_dF_{lo}-{hi}GeV"] = float(np.abs(d_all[m]).mean()) if m.any() else None
        res["families"][fam] = rec
        print(f"[{time.time() - t0:6.0f}s] {fam}: floor {dat.get('pt_min', float('nan')):.3f} GeV, "
              f"q01 {dat.get('pt_q01', float('nan')):.3f}; |dF_W| W 0.2-0.45 GeV "
              f"{rec['W']['mean_abs_dF_0.2-0.45GeV']:.2f}, 1-3 GeV {rec['W']['mean_abs_dF_1.0-3.0GeV']:.2f}",
              flush=True)
    name = args.name or (args.edit.stem if args.edit else "original_model")
    out = args.out_dir or C.out_dir(args.size, "soft_particle_scans")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", **arr)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / f"{name}.json", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000, help="jets per class")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-swaps", dest="swaps", action="store_false")
    ap.add_argument("--edit", type=Path, default=None, help="edited weights (core.load_candidate)")
    ap.add_argument("--name", default=None, help="output name (default: original_model or the edit file stem)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--pt-cut", type=float, default=None)
    ap.add_argument("--pt-min", type=float, default=None, help="scan range (GeV); default 0.05-10")
    ap.add_argument("--pt-max", type=float, default=10.0)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--heads", nargs="+", default=["7:6"],
                    help="backbone block:head whose summary-query attention on the added particle is recorded "
                         "(the first one under the keys att_<family>_<W|QCD>); medium model: 7:0 11:7")
    ap.add_argument("--fused-attention", action="store_true", help="fused attention kernels (default: math kernel)")
    args = ap.parse_args()
    C.set_size(args.size)
    C.attention_kernel(args.fused_attention)
    compute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
