#!/usr/bin/env python
"""Seed and dictionary-size robustness of the identified latent (paper App. B.2, "Robustness checks").

For every dictionary setting (latents N, active k) of one reader, with SAEs trained with several seeds
(train_sae.py; the data record holds b7-6 with 512 to 8192 latents, seeds 0 and 1):
- per seed: held-out explained variance, dead fraction, final training NMSE, whether a latent is identified, the
  selected latent's firing rates, selectivity and key alignment, and its latent-subtraction test;
- across the first two seeds: the largest decoder cosine of seed A's selected latent with any latent of seed B,
  and the reverse; "replicated" = both >= 0.9.
Selected latent (--rule): "tie_break" = the identification rule of train_sae.py (most selective, ties within 0.02
broken by key alignment); "most_selective" = the most selective latent only. The published summary
(dictionary_size_sweep.json in the data record) was made with "most_selective"; with it, one seed of the two
smallest dictionaries (512, 1024 latents) selects a different latent. With "tie_break" both seeds agree for every
dictionary size.

  python scripts/sae/dictionary_sweep.py --size s --reader b7-6 [--sae-dir DIR] [--out sweep.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ADDED = ("added photon 0.3 GeV", "added neutral hadron 0.5 GeV")


def load(p: Path):
    res = json.loads(p.read_text())
    pt = torch.load(p.with_suffix(".pt"), map_location="cpu", weights_only=False)
    Wd = pt["state"]["W_d"].float()
    return res, Wd / Wd.norm(dim=0, keepdim=True)


RULE = "tie_break"


def best(res) -> int:
    """The selected latent: most selective, ties within 0.02 broken by key alignment (RULE "tie_break"), or the
    most selective latent (RULE "most_selective")."""
    if RULE == "most_selective":
        return int(max(res["top_latents"], key=lambda x: x["selectivity"])["latent"])
    if "best_latent_rule" in res:
        return int(res["best_latent_rule"]["best"])
    t = res["top_latents"]
    ties = [x for x in t if x["selectivity"] >= t[0]["selectivity"] - 0.02]
    return int(max(ties, key=lambda x: x["key_logit_per_unit"])["latent"])


def _causal(c: dict) -> dict:
    """Latent-subtraction results, with the key names of train_sae.py (older records: per-head key names)."""
    ren = {"head6_attention_before": "head_attention_before", "head6_attention_after": "head_attention_after",
           "abs_dF_before": "abs_dF_W_before", "abs_dF_after": "abs_dF_W_after"}
    return {k: {ren.get(kk, kk): round(vv, 3) for kk, vv in v.items()} for k, v in c.items()}


def row(res) -> dict:
    b = best(res)
    t = next((x for x in res["top_latents"] if x["latent"] == b), {})
    return {"n_latents": res.get("n_latents", res["args"]["m"]), "k": res["args"]["k"],
            "fve_heldout_all": res.get("sae_fve_heldout_all"), "fve_real_neutrals": res.get("sae_fve_real_neutrals"),
            "dead_fraction": res.get("dead_fraction_heldout"),
            "final_nmse": res["train_log"][-1]["nmse"] if res.get("train_log") else None,
            "verdict": res["verdict"], "n_found": len(res["found"]), "best_latent": b,
            "rate_added_photon": t.get("rate: " + ADDED[0]), "rate_added_nh": t.get("rate: " + ADDED[1]),
            "rate_real_neutrals": t.get("rate: real JetClass neutrals"),
            "rate_jetclass2_sub": t.get("rate: JetClass-II neutrals below JetClass threshold"),
            "rate_photon_3GeV": t.get("rate: added photon 3 GeV"),
            "selectivity": t.get("selectivity"), "null_p99": res["null_selectivity_p99"],
            "key_logit_per_unit": t.get("key_logit_per_unit"), "causal": _causal(res.get("causal", {}))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--reader", default="b7-6")
    ap.add_argument("--sae-dir", type=Path, default=None, help="SAE folder (default: <EDITS_DIR>/../sae)")
    ap.add_argument("--rule", default="tie_break", choices=("tie_break", "most_selective"),
                    help="how the latent of each SAE is selected (see above)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default $RESULTS_DIR/sae/<size name>/<reader>/dictionary_size_sweep.json)")
    args = ap.parse_args()
    global RULE
    RULE = args.rule
    d = C.sae_root(args.sae_dir) / C.SIZE_NAMES[args.size] / args.reader
    runs = defaultdict(dict)
    for p in sorted(d.glob("latents*_top*_seed*.json")):
        m = re.fullmatch(r"(latents\d+_top\d+)_seed(\d+)", p.stem)
        if m and p.with_suffix(".pt").exists():
            runs[m.group(1)][int(m.group(2))] = p
    out = {}
    for setting, seeds in sorted(runs.items(), key=lambda kv: int(re.search(r"\d+", kv[0]).group())):
        loaded = {s: load(p) for s, p in seeds.items()}
        out[setting] = {"seeds": {s: row(r) for s, (r, _) in loaded.items()}}
        if len(loaded) >= 2:
            (ra, A), (rb, B) = [loaded[s] for s in sorted(loaded)[:2]]
            Cm = A.T @ B  # (mA, mB) decoder cosines
            ab, ba = float(Cm[best(ra)].max()), float(Cm[:, best(rb)].max())
            out[setting]["cross_seed"] = {"best_A_in_B": round(ab, 3), "best_B_in_A": round(ba, 3),
                                          "replicated": bool(min(ab, ba) >= 0.9),
                                          "found_A_matched_in_B": round(float((Cm[ra["found"]].max(1).values >= 0.9).float().mean()), 3)
                                          if ra["found"] else None,
                                          "dictionary_median_best_cos": round(float(Cm.max(1).values.median()), 3)}
        print(setting, json.dumps(out[setting].get("cross_seed")),
              [(s, r["verdict"], r["best_latent"]) for s, r in out[setting]["seeds"].items()], flush=True)
    if args.out is None:
        from softrepair import paths
        args.out = paths.results_dir() / "sae" / C.SIZE_NAMES[args.size] / args.reader / "dictionary_size_sweep.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
