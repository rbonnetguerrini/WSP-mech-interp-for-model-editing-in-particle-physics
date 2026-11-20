#!/usr/bin/env python
"""Print the in-text numbers of Sec. 2, App. A and the App. E readers from the cached results (no model needed).

Reads (under --results, default RESULTS_DIR; a missing input skips its lines):
  figures/small/w_tagger_response_summary.json, w_tagger_response_neutral_hadron_summary.json  (w_tagger_response.py)
  analysis/template_fit.json                                                                 (template_fit.py)
  figures/small/readers_attention_photon.json                                                (attention_heads.py)
  figures/small/soft_particle_scans/original_model.json / .npz, original_model_head_b7-6_attention.json
                                                                                             (soft_particle_scan.py)
  figures/small/readers_patching.json, figures/{medium,large}/readers_patching.json          (activation_patching.py)
  figures/small/probes.json                                                                  (probes.py)
  analysis/dataset_shift_s.json                                                              (dataset_shift.py)
  analysis/head_attention_threshold.json                                                     (head_attention_threshold.py)

  python scripts/analysis/print_numbers.py [--results DIR]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load(p: Path):
    if not p.exists():
        print(f"  (missing {p})")
        return None
    return np.load(p) if p.suffix == ".npz" else json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=os.environ.get("RESULTS_DIR"))
    a = ap.parse_args()
    if a.results is None:
        ap.error("give --results or set RESULTS_DIR")
    s, an = a.results / "figures" / "small", a.results / "analysis"

    print("W tagger at the 80% working point (Sec. 2)")
    for f, what in (("w_tagger_response_summary.json", "photon"), ("w_tagger_response_neutral_hadron_summary.json",
                                                                   "neutral hadron")):
        r = load(s / f)
        if r:
            w = r["wp80"]
            print(f"  + {what}, f = 1e-3: W efficiency {w['nominal_W']['eff']:.4f} -> {w['f1e-3_W']['eff']:.4f}, "
                  f"QCD acceptance {w['nominal_QCD']['eff']:.4f} -> {w['f1e-3_QCD']['eff']:.4f}")

    r = load(an / "template_fit.json")
    if r:
        o = r["fits"]["original"]
        print("Template fit of the W fraction (true 0.25), OmniLearned F_W")
        for sc in ("nominal", "below_floor", "in_range"):
            print(f"  {sc:12s} f = {o[sc]['f']:.4f}  chi2/ndf = {o[sc]['chi2']:.1f}/{o[sc]['ndf']} = {o[sc]['chi2_ndf']:.2f}")

    print("Attention of the summary tokens on one added photon at f = 1e-3 (Sec. 2)")
    r = load(s / "readers_attention_photon.json")
    if r:
        G, sites = np.array(r["grid"]), r["sites"]
        nb = sum(x.startswith("body") for x in sites)
        body = G[:nb].copy()
        print(f"  b7/6 {G[7, 6]:.4f}")
        body[7, 6] = -1
        i, j = np.unravel_index(np.argmax(body), body.shape)
        print(f"  largest other backbone head: b{i}/{j} {body[i, j]:.4f}")
    r = load(s / "soft_particle_scans" / "original_model_head_b7-6_attention.json")
    if r:
        hb = np.asarray(r["pt_bins_GeV"])
        k = int(np.digitize(0.3, hb) - 1)
        m = r["families"]["photon"]["W"]["median_head6_attention"][k]
        print(f"  b7/6 median attention on a photon at pT = 0.3 GeV (bin {hb[k]:.3f}-{hb[k + 1]:.3f} GeV), W jets: {m:.4f}")
    r, z = load(s / "soft_particle_scans" / "original_model.json"), load(s / "soft_particle_scans" / "original_model.npz")
    if r and z is not None:
        n_jets = 2 * r["n_per_class"]
        mult = sum(r["families"][f]["data"]["n_real"] for f in ("photon", "neutral hadron", "charged hadron",
                                                                 "electron", "muon")) / n_jets
        print(f"  mean particle multiplicity {mult:.1f}: pT-independent share of one added particle 1/(N+1) = "
              f"{1 / (mult + 1):.4f}")
        print("Impact-parameter check: mean |dF_W|, W jets, added photon with pT < 0.45 GeV")
        for fam in ("photon", "photon, track IP sigmas only", "photon, track IPs"):
            pt, dF = z[f"pt_{fam}_W"], np.abs(z[f"dF_{fam}_W"])
            print(f"  {fam:30s} {dF[pt < 0.45].mean():.4f}")

    print("Activation patching: share of |dF_W| removed (readers positions)")
    for size, probe in (("small", "soft1e-3"), ("small", "soft7_1e-3"), ("medium", "photon_0.3GeV"),
                        ("large", "photon_0.3GeV")):
        r = load(a.results / "figures" / size / "readers_patching.json")
        if r and probe in r["probes"]:
            rm = r["probes"][probe]["removed"]
            top = sorted(((v, k) for k, v in rm.items() if ".h" in k and k.endswith("readers")), reverse=True)[:6]
            print(f"  {size:6s} {probe:12s} " + ", ".join(f"{k.replace('.readers', '')} {v:.3f}" for v, k in top))

    r = load(an / "dataset_shift_s.json")
    if r:
        c = r["classes"]["W"]
        print("Dataset class shares of W jets (200-way head)")
        print(f"  clean: JetClass {c['clean']['JetClass (2-11)']:.4f}; + photon 0.3 GeV: JetClass "
              f"{c['photon_0.3GeV']['JetClass (2-11)']:.4f}, JetClass-II {c['photon_0.3GeV']['JetClass-II (12-199)']:.4f}")

    r = load(s / "probes.json")
    if r:
        print("Linear probes (App. A)")
        print(f"  held-out JetClass-II jets with a sub-threshold neutral: {r['jc2_jets_with_subthreshold_neutrals']:.4f}")
        for L in ("body block 6", "body block 7", "body.norm"):
            st = r["layers"][L]["stats"]["JetClass + photon 0.3 GeV"]
            print(f"  {L:12s} JetClass + photon 0.3 GeV: mean P(JetClass-II) "
                  f"{r['layers'][L]['JetClass + photon 0.3 GeV']:.4f}, 16% quantile {st['q16']:.4f}")

    r = load(an / "head_attention_threshold.json")
    if r:
        print("b7/6 attention: half-maximum point and softest real particle, per |eta| bin")
        for fam, bins in r.items():
            for lab, v in bins.items():
                print(f"  {fam:14s} {lab}: E_half {v['att_half_edge_E']:.2f} GeV (softest E {v['data_floor_E']:.2f}), "
                      f"pT_half {v['att_half_edge_pT']:.2f} GeV (softest pT {v['data_floor_pT']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
