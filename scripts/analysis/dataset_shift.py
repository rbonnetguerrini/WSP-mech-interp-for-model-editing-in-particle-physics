#!/usr/bin/env python
"""Does one soft neutral move the jet out of the JetClass classes into the JetClass-II classes? (Sec. 2,
"A hypothesis for the origin of the sensitivity": 99% -> 1.5% JetClass, 98.5% JetClass-II.)

OmniLearned was pretrained on ATLAS, Aspen, JetClass, JetClass-II, H1, CMS QCD and CMS BSM jets. Labels: ATLAS 0-1,
JetClass 2-11, JetClass-II 12-199 (the 200-way jet head); Aspen / CMS QCD / CMS BSM 200-202. On held-out JetClass
W and QCD jets (OmniLearned-s, CPU):

1. Dataset share. Softmax over the 200 jet logits; summed probability of the ATLAS, JetClass and JetClass-II
   blocks, clean and with one added neutral (photon 0.3 GeV, neutral hadron 0.5 GeV, photon 3 GeV, split2).
2. Fill. b7/6 attention and |dF_W| for a 0.3 GeV photon carrying JetClass's neutral impact-parameter fill
   (0, 0, 0, 0), Aspen's neutral fill (-0.76, -1, -0.76, -1), or a mixed one.
3. In-support control (--jetclass2 FILE): the same additions to real JetClass-II jets, where soft zero-IP neutrals
   are normal. Mean change of the centred JetClass-II logits in units of the jet-to-jet distance, and whether the
   200-way argmax changes.

Jets: W file 5 and QCD file 32 (test-jet files), rows 60000..60000+n (the stored result uses --n 500).

  python scripts/analysis/dataset_shift.py [--n 500] [--threads 2] [--jetclass2 JETCLASS2_DIR/JetClass2_9.h5]
      -> RESULTS_DIR/analysis/dataset_shift_s.json
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

FILLS = {"JetClass fill (0, 0, 0, 0)": (0.0, 0.0, 0.0, 0.0), "Aspen fill (-0.76, -1, -0.76, -1)": (-0.76, -1.0, -0.76, -1.0),
         "sigmas only (0, -1, 0, -1)": (0.0, -1.0, 0.0, -1.0), "values only (-0.76, 0, -0.76, 0)": (-0.76, 0.0, -0.76, 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--jetclass2", type=Path, default=None, help="a JetClass-II h5 file for the in-support control")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    C.set_size("s")
    from softrepair import core as W
    from softrepair import heads as H
    torch.set_num_threads(args.threads)
    ad = W.load_adapter("cpu", "s")
    res = {"n_per_class": args.n, "blocks": {k: list(v) for k, v in C.BLOCKS.items()}, "classes": {}}
    for lab, cname in ((H.WJ, "W"), (H.QCD, "QCD")):
        raw = H.load(lab, 60000, args.n)
        y0 = W.logits(ad, raw)
        F0 = H.logodds_W(y0)
        rec = {"clean": C.shares(y0)}
        for k, (pname, pid, pt) in enumerate((("photon_0.3GeV", 6, 0.3), ("neutral_hadron_0.5GeV", 7, 0.5),
                                               ("photon_3GeV", 6, 3.0))):
            pert, ok = C.add_fixed(raw, pt, pid, 9200 + k)
            y = W.logits(ad, pert)
            rec[pname] = {**C.shares(y[ok]), "mean_abs_dF_W": float(np.abs(H.logodds_W(y) - F0)[ok].mean())}
        pert, ok = W.perturb(raw, "split2", 9210)
        y = W.logits(ad, pert)
        rec["split2"] = {**C.shares(y[ok]), "mean_abs_dF_W": float(np.abs(H.logodds_W(y) - F0)[ok].mean())}
        # fills on a 0.3 GeV photon
        base, ok = C.add_fixed(raw, 0.3, 6, 9220)
        slot = np.array([W._free_slot(r) for r in raw])
        fills = {}
        for fname, f in FILLS.items():
            x = base.copy()
            x[np.flatnonzero(ok), slot[ok], 5:9] = np.array(f, np.float32)
            F = H.logodds_W(W.logits(ad, x))
            w = C.head_weights(ad, x)
            att = w[np.arange(len(x)), :, np.clip(slot, 0, None)].mean(1)[ok]
            fills[fname] = {"mean_abs_dF_W": float(np.abs(F - F0)[ok].mean()), "head6_attention_median": float(np.median(att))}
        rec["photon_0.3GeV_fills"] = fills
        res["classes"][cname] = rec
        print(cname, json.dumps(rec, indent=1), flush=True)
    if args.jetclass2:  # in-support control: the same additions to real JetClass-II jets
        import h5py
        with h5py.File(args.jetclass2, "r") as h:
            raw = h["data"][:2 * args.n].astype(np.float32)
        y0 = W.logits(ad, raw)
        c0 = y0[:, 12:200] - y0[:, 12:200].mean(1, keepdims=True)  # centred JetClass-II logits
        S = float(np.sqrt(((c0[None, :256] - c0[:256, None]) ** 2).sum(-1).mean()))  # jet-to-jet RMS distance
        rec = {"file": args.jetclass2.name, "clean": C.shares(y0)}
        for k, (pname, pid, pt) in enumerate((("photon_0.3GeV", 6, 0.3), ("neutral_hadron_0.5GeV", 7, 0.5),
                                               ("photon_3GeV", 6, 3.0))):
            pert, ok = C.add_fixed(raw, pt, pid, 9300 + k)
            y = W.logits(ad, pert)
            c = y[:, 12:200] - y[:, 12:200].mean(1, keepdims=True)
            rec[pname] = {**C.shares(y[ok]), "r_JetClass2_logits": float(np.linalg.norm(c - c0, axis=1)[ok].mean() / S),
                          "argmax_unchanged": float((y[ok, :200].argmax(1) == y0[ok, :200].argmax(1)).mean())}
        res["JetClass-II jets"] = rec
        print("JetClass-II", json.dumps(rec, indent=1), flush=True)
    out = args.out or C.analysis_dir() / "dataset_shift_s.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
