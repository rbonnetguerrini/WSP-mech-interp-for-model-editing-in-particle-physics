#!/usr/bin/env python
"""Rebuild the practice / test variants in a fresh DATA_DIR from existing clean sets (no JetClass files needed).

The clean sets (clean.npy, labels.npy, meta.json) are copied from --src (a folder with practice/ and test/ built
by build_sets.py). The variants are regenerated with exactly the seeds build_sets.py uses, so the result equals
the published sets; check it by re-scoring an edit (scripts/evaluation/score_edit.py) or by comparing files.

  DATA_DIR=<new dir> python scripts/data/rebuild_variants.py --src <old DATA_DIR>
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from softrepair import core as W
from softrepair import repair_eval as B
from softrepair import scoring as S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--no-scoring-sets", action="store_true", help="do not build practice_ir / test_ir")
    args = ap.parse_args()
    for name, variants in (("practice", W.VISIBLE), ("test", W.VISIBLE + W.HELDOUT)):
        d = W.set_dir(name)
        if (d / "meta.json").exists():
            print(f"[{name}] exists, skipped", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        for f in ("clean.npy", "labels.npy"):
            shutil.copy2(args.src / name / f, d / f)
        meta = json.loads((args.src / name / "meta.json").read_text())
        files, r0, r1 = W.SETS[name]
        assert meta["rows"] == [r0, r1] and {int(k): v for k, v in meta["files"].items()} == files, meta
        raw = np.load(d / "clean.npy")
        counts = {}
        for k, v in enumerate(variants):
            t = time.time()
            seed = 1000 * (list(W.SETS).index(name) + 1) + k  # as build_sets.build
            p, ok = W.perturb(raw, v, seed)
            np.save(d / f"{v}.npy", p)
            np.save(d / f"{v}.ok.npy", ok)
            counts[v] = int(ok.sum())
            print(f"[{name}] {v}: {ok.sum()}/{len(ok)} valid ({time.time() - t:.0f}s)", flush=True)
        meta.update({"variants": list(variants), "valid": counts})
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
    B.build()  # soft1e-3 and the physical changes D, J, P, with their own seeds
    if not args.no_scoring_sets:
        S.build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
