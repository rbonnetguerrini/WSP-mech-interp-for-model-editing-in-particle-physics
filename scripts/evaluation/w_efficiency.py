#!/usr/bin/env python
"""W efficiency at a fixed working point after one added photon (paper Sec. 2.1, Tab. comparison, App. B).

The cut on F_W = logit(W) - logit(QCD) keeps 80% (and 50%) of the unmodified W jets of the model under test
(JetClass W and QCD test files, rows 60000..60000+n); the efficiency is the fraction of W jets (and QCD jets) that
pass it after one photon (or neutral hadron) at f = 1e-3 and 3e-3 of the jet pT (scoring.w_efficiency).

  python scripts/evaluation/w_efficiency.py --size s --n 10000 \
      --edit original_model --edit wsp=$EDITS_DIR/agents/target_informed/small/run1/edit_large.npz \
      --out-dir $RESULTS_DIR/w_efficiency/small          -> <out-dir>/<name>[_neutral_hadron].json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--edit", action="append", default=None, help="name=path.npz (repeatable); bare name = original")
    ap.add_argument("--n", type=int, default=10000, help="W and QCD jets")
    ap.add_argument("--neutral-hadron", action="store_true", help="add a neutral hadron instead of a photon")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    C.setup_device(a.size, {"s": None, "m": 128, "l": 32}[a.size])
    from softrepair import core as W
    from softrepair import scoring as S

    t0 = time.time()
    ad = W.load_adapter(a.device, a.size)
    orig = W.original_state(ad)
    pid = 7 if a.neutral_hadron else 6
    for name, path in C.parse_edits(a.edit):
        W.set_state(ad, W.load_candidate(path, orig) if path else orig)
        res = {"name": name, "model_size": a.size, "edit": path.name if path else None,
               **S.w_efficiency(ad, n=a.n, soft_pid=pid)}
        W.set_state(ad, orig)
        C.dump(a.out_dir / (name + ("_neutral_hadron" if a.neutral_hadron else "") + ".json"), res)
        w = res["wp80"]
        print(f"[{time.time() - t0:5.0f}s] {name}: W eff (80% WP) after one {'neutral hadron' if pid == 7 else 'photon'}"
              f" at f=1e-3: {w['f1e-3_W']['eff']:.4f} +- {w['f1e-3_W']['err']:.4f}; QCD {w['nominal_QCD']['eff']:.4f}"
              f" -> {w['f1e-3_QCD']['eff']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
