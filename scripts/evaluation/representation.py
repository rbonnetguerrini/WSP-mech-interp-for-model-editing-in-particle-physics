#!/usr/bin/env python
"""The repaired representation (paper Sec. 4, Fig. latent): for each jet, the share of JetClass-II jets among its 15
nearest neighbours in the unmodified jets (W, QCD, JetClass-II), on the input tokens of the classifier (the four
summary tokens after the body), for the original model and a repaired one.

The embeddings and neighbour shares are computed by scripts/analysis/latent_umap.py (run once per model, see
slurm/evaluation_representation.sbatch); this script prints the comparison quoted in the paper.

  python scripts/evaluation/representation.py [--size s] [--repaired agents_target_informed_run2_edit_large]
      reads RESULTS_DIR/figures/<size>/representation/{original_model,<repaired>}.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

SETS = ("W", "QCD", "JetClass-II", "W + photon", "QCD + photon", "W + photon 3 GeV (control)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--repaired", default="agents_target_informed_run2_edit_large")
    ap.add_argument("--layer", default="body.norm", help="summary-token site (default: after the body)")
    a = ap.parse_args()
    d = C.rpath("figures", C.SIZES[a.size], "representation")
    res = {n: C.read_json(C.need(d / f"{n}.json"))["layers"][a.layer] for n in ("original_model", a.repaired)}
    print(f"share of JetClass-II jets among the {15} nearest unmodified neighbours ({a.layer}):")
    rows = [[k, C.pct(res["original_model"][k]["share_JetClass-II_neighbours"]),
             C.pct(res[a.repaired][k]["share_JetClass-II_neighbours"])] for k in SETS]
    C.print_table(["jets", "original model", "repaired model"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
