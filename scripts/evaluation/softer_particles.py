#!/usr/bin/env python
"""The response to softer particles (paper Sec. 4, Fig. ir): mean |dF_W| of W jets for one added photon, neutral
hadron or charged hadron between 1 MeV and 10 GeV, for the original model and the repaired models.

The scans are computed by scripts/analysis/soft_particle_scan.py (--pt-min 0.001 --pt-max 10 --families photon
"neutral hadron" "charged hadron", 5000 W and QCD jets, two draws; slurm/evaluation_softer_particles.sbatch); this
script prints the numbers quoted in the text, in the bins of the figure (5 per decade, scripts/figures/_ir.py).

  python scripts/evaluation/softer_particles.py      reads RESULTS_DIR/figures/small/ir_response/<model>.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

MODELS = {"original": "original_model", "fine-tuning": "finetuning_attention_only_both_neutral_types_seed0",
          "SAE edit": "sae_three_readers_b7-6_b4-0_c1-2", "WSP": "agents_target_informed_run2_edit_large"}
BINS = np.geomspace(1e-3, 10, 21)
X = np.sqrt(BINS[1:] * BINS[:-1])


def curve(a, fam: str) -> np.ndarray:
    pt, dF = a[f"pt_{fam}_W"], np.abs(a[f"dF_{fam}_W"].astype(np.float64))
    idx = np.digitize(pt, BINS) - 1
    return np.array([dF[idx == i].mean() if (idx == i).sum() > 20 else np.nan for i in range(len(X))])


def main() -> int:
    d = C.rpath("figures", "small", "ir_response")
    rows = []
    for lab, name in MODELS.items():
        a = np.load(C.need(d / f"{name}.npz"))
        c = {f: curve(a, f) for f in ("photon", "neutral hadron", "charged hadron")}
        below = X < 0.45  # sub-threshold bins (the first percentile of the photon pT in JetClass is 0.85 GeV)
        rows.append([lab, f"{c['photon'][np.argmin(np.abs(X - 1.26))]:.2f}", f"{np.nanmax(c['photon']):.2f} at "
                     f"{X[np.nanargmax(c['photon'])]:.2f} GeV", f"{c['photon'][0]:.2f}", f"{c['charged hadron'][0]:.2f}",
                     "{:.2f}-{:.2f}".format(*(lambda v: (np.nanmin(v), np.nanmax(v)))(
                         np.r_[c["photon"][below], c["neutral hadron"][below]])),
                     f"{c['photon'][-1]:.2f}"])
    print(f"mean |dF_W| of W jets, bins of 5 per decade (centres {X[0] * 1e3:.1f} MeV ... {X[-1]:.1f} GeV)"
          "; 'below threshold' = bins with centres under 0.45 GeV")
    C.print_table(["model", "photon 1.3 GeV", "photon maximum", "photon 1 MeV", "charged hadron 1 MeV",
                   "photon and neutral hadron below threshold", f"photon {X[-1]:.1f} GeV"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
