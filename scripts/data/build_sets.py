#!/usr/bin/env python
"""Build the practice, test and fine-tuning training jet sets with their added-particle draws and physical changes.

Every set is a folder DATA_DIR/<set>/ with clean.npy (n, 150, 9 float32), labels.npy (global JetClass label),
<variant>.npy (the modified jets), <variant>.ok.npy (rows where the change could be applied) and meta.json.
All draws use fixed seeds, so the sets equal the published ones.

  practice     JetClass validation rows 0-1024 of the practice files, 10 classes (10,240 jets)
  test         rows 50000-52048 of the test files (20,480 jets)
  train        rows 2048-22048 of the practice files (fine-tuning training jets, 200,000 jets)
  practice_ir  / test_ir   the added-particle sets used for scoring (links to practice / test files + new draws)

Steps (in this order, each skips what already exists):
  1. sets      practice / test / train with split2, soft3e-3, soft1e-4 (+ soft7_1e-3, split3, split2c on test);
               seed 1000 * (set index + 1) + variant index
  2. physical  soft1e-3 and the physical changes D, J, P on practice / test (repair_eval.build);
               seed 5000 + 100 * set index + variant index
  3. scoring   practice_ir / test_ir (scoring.build); seed 9000 + 100 * set index + variant index
  4. evaluate  (optional, GPU) metrics of the original model on practice / test and the pT-cut input baselines,
               written to RESULTS_DIR/eval/{original,ptcut}.json
               (original_<size>.json for m and l, read by the evaluation tool)

  python scripts/data/build_sets.py                 steps 1-3 (CPU; reads JETCLASS_DIR, writes DATA_DIR)
  python scripts/data/build_sets.py --skip-train    without the 200,000 fine-tuning jets
  python scripts/data/build_sets.py --evaluate      also step 4
"""

from __future__ import annotations

import argparse
import json
import time

import h5py
import numpy as np

from softrepair import core as W
from softrepair import repair_eval as B
from softrepair import scoring as S

CUTS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)


def build(name: str, variants: tuple[str, ...]) -> None:
    files, r0, r1 = W.SETS[name]
    d = W.set_dir(name)
    if (d / "meta.json").exists():
        print(f"[{name}] exists, skipped", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    raws, labels = [], []
    for lab, f in files.items():
        with h5py.File(W.DATA / f"JetClass_{f}.h5", "r") as h:
            raw = h["data"][r0:r1].astype(np.float32)
            pid = h["pid"][r0:r1]
        assert np.all(pid == lab), (f, np.unique(pid))
        raws.append(raw)
        labels.append(np.full(len(raw), lab, np.int64))
    raw, labels = np.concatenate(raws), np.concatenate(labels)
    np.save(d / "clean.npy", raw)
    np.save(d / "labels.npy", labels)
    counts = {}
    for k, v in enumerate(variants):
        t = time.time()
        seed = 1000 * (list(W.SETS).index(name) + 1) + k
        p, ok = W.perturb(raw, v, seed)
        np.save(d / f"{v}.npy", p)
        np.save(d / f"{v}.ok.npy", ok)
        counts[v] = int(ok.sum())
        print(f"[{name}] {v}: {ok.sum()}/{len(ok)} valid ({time.time() - t:.0f}s)", flush=True)
    meta = {"files": files, "rows": [r0, r1], "variants": list(variants), "valid": counts, "n": int(len(raw))}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def evaluate(device: str) -> None:
    """Metrics of the original model and of the pT-cut input baselines on practice and test."""
    ad = W.load_adapter(device)
    ev = W.OUT / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, variants in (("practice", W.VISIBLE), ("test", W.VISIBLE + W.HELDOUT)):
        out[name] = W.evaluate_set(ad, W.load_set(name), variants)
        print(name, json.dumps({k: (v["r"] if "r" in v else v["acc"]) for k, v in out[name].items()}), flush=True)
    size = W.model_size()
    (ev / ("original.json" if size == "s" else f"original_{size}.json")).write_text(json.dumps(out, indent=1))

    # Padding filler check (pt_cut writes pid 3 into dropped rows).
    raw = W.load_set("practice", ())["clean"][:2000]
    pad = raw[:, :, 2] == 0
    print("padding pid values:", np.unique(raw[:, :, 4][pad]), "real pT quantiles:",
          np.quantile(np.exp(raw[:, :, 2][~pad]), [0.001, 0.01, 0.1, 0.5]), flush=True)

    cuts = {}
    for c in CUTS:
        tf = lambda a, c=c: W.pt_cut(a, c)  # noqa: E731
        rec = {}
        for name, variants in (("practice", W.VISIBLE), ("test", W.VISIBLE + W.HELDOUT)):
            res = W.evaluate_set(ad, W.load_set(name), variants, raw_transform=tf)
            # response of the cut pipeline relative to the ORIGINAL pipeline; accuracy vs original.
            res["ratio_visible"] = W.ratios(res, out[name], W.VISIBLE)
            if name == "test":
                res["ratio_heldout"] = W.ratios(res, out[name], W.HELDOUT)
            res["d_acc"] = res["clean"]["acc"] - out[name]["clean"]["acc"]
            res["d_nll"] = res["clean"]["nll"] - out[name]["clean"]["nll"]
            rec[name] = res
        cuts[str(c)] = rec
        t = rec["test"]
        print(f"cut {c}: ratio_visible {t['ratio_visible']['mean']:.3f} ratio_heldout {t['ratio_heldout']['mean']:.3f} "
              f"dacc {t['d_acc']:+.4f} dnll {t['d_nll']:+.4f}", flush=True)
    (ev / ("ptcut.json" if size == "s" else f"ptcut_{size}.json")).write_text(json.dumps(cuts, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true", help="do not build the fine-tuning training set")
    ap.add_argument("--train-only", action="store_true", help="build only the fine-tuning training set")
    ap.add_argument("--evaluate", action="store_true", help="also evaluate the original model (GPU)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.train_only:
        build("train", W.VISIBLE)
        return 0
    build("practice", W.VISIBLE)
    build("test", W.VISIBLE + W.HELDOUT)
    if not args.skip_train:
        build("train", W.VISIBLE)
    B.build()
    S.build()
    if args.evaluate:
        evaluate(args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
