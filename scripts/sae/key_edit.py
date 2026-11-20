#!/usr/bin/env python
"""Rank-one key edit from SAE latents (paper Sec. 3.2, App. B.2): W_K -> W_K (I - alpha u u^T).

u = unit decoder direction of the reader's identified latent, in the space of the reader's attention-block input.
The reader's key rows of the fused query/key/value weights become W_K (I - alpha u u^T): the head can no longer
distinguish particles along u. Queries, values and the other heads are unchanged. For the small model this is
511 changed parameters per reader (rank one on a 16 x 128 block of rows).

Several readers are edited one after the other (--readers ... --saes ...), each with its own SAE; the result is one
partial state dict of full replacement tensors.

  python scripts/sae/key_edit.py --size s --readers b7-6 b4-0 c1-2 --seed 0 --out three_readers.npz
      (SAEs from --sae-dir, default the data record's sae/ folder; or give them with --saes)
  python scripts/sae/key_edit.py --size s --readers b7-6 --saes sae.pt --latent 536 --alpha 0.5 --out e.npz
  python scripts/sae/key_edit.py --size s --scale-edit three_readers.npz --alpha 0.5 --out half.npz
      (half strength from an existing full-strength edit: orig + alpha (edit - orig), exact for one rank-one
       projection per tensor; used for the published half-strength edit, whose SAE checkpoints are not public)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from softrepair import core as W

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def project_out(edits: dict, orig: dict, ad, reader: str, sae_pt: Path, latents: list[int], alpha: float) -> dict:
    """Apply W_K (I - alpha U U^T) to the reader's key rows (U: orthonormal basis of the latents' decoder
    directions); edits is updated in place. Returns a description of the change."""
    _, head = C.parse_reader(reader)
    key = C.key_tensor(reader)
    ck = torch.load(sae_pt, map_location="cpu", weights_only=False)
    D_ = ck["state"]["W_d"][:, latents].double().numpy()
    U, _ = np.linalg.qr(D_)  # orthonormal basis of the latents' decoder directions (width x k)
    Wt = (edits[key].astype(np.float64) if key in edits else orig[key].double().numpy()).copy()
    attn = ad.model.get_submodule(key.rsplit(".", 1)[0])  # fused q|k|v rows: keys start at embed_dim
    E, dh = attn.embed_dim, attn.embed_dim // attn.num_heads
    rows = slice(E + dh * head, E + dh * (head + 1))
    Wk = Wt[rows].copy()
    Wt[rows] = Wk - alpha * (Wk @ U) @ U.T
    edits[key] = Wt.astype(np.float32)
    return {"reader": reader, "sae": sae_pt.name, "latents": latents, "alpha": alpha, "tensor": key,
            "rows": [rows.start, rows.stop],
            "rel_change_of_head_keys": float(np.linalg.norm(Wt[rows] - Wk) / np.linalg.norm(Wk)),
            "norm_Wk_U": float(np.linalg.norm(Wk @ U))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"), help="OmniLearned model size")
    ap.add_argument("--readers", nargs="+", default=None, help="readers to edit, in order, e.g. b7-6 b4-0 c1-2")
    ap.add_argument("--edit", default=None, help="a published edit name (sets --readers), e.g. three_readers_b7-6_b4-0_c1-2")
    ap.add_argument("--saes", nargs="+", type=Path, default=None, help="SAE checkpoint per reader (same order)")
    ap.add_argument("--sae-dir", type=Path, default=None, help="SAE folder (default: <EDITS_DIR>/../sae)")
    ap.add_argument("--seed", type=int, default=0, help="SAE seed when the checkpoints come from --sae-dir")
    ap.add_argument("--latent", type=int, nargs="+", default=None,
                    help="latent per reader (default: the identified latent of each SAE's .json record)")
    ap.add_argument("--top-k", type=int, default=1,
                    help="project out the k most selective identified latents of each SAE (rank-k change)")
    ap.add_argument("--alpha", type=float, default=1.0, help="strength: 1 = full projection, 0.5 = half strength")
    ap.add_argument("--base", type=Path, default=None, help="earlier edit .npz to add this edit to")
    ap.add_argument("--scale-edit", type=Path, default=None,
                    help="instead of SAEs: scale an existing edit, orig + alpha (edit - orig) (float64)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    C.set_size_env(args.size)
    ad = W.load_adapter("cpu", args.size)
    orig = W.original_state(ad)
    if args.scale_edit is not None:
        full = dict(np.load(args.scale_edit))
        edits = {}
        for k, v in full.items():
            o = orig[k].double().numpy()
            edits[k] = (o + args.alpha * (v.astype(np.float64) - o)).astype(np.float32)
        info = {"scaled_from": args.scale_edit.name, "alpha": args.alpha, "tensors": sorted(edits)}
    else:
        readers = args.readers or (list(C.EDITS[args.size][args.edit]) if args.edit else None)
        if not readers:
            ap.error("give --readers, --edit or --scale-edit")
        if args.saes is not None:
            saes = list(args.saes)
        else:
            root = C.sae_root(args.sae_dir)
            saes = [C.sae_path(root, args.size, r, args.seed) for r in readers]
        if len(saes) != len(readers) or (args.latent is not None and len(args.latent) != len(readers)):
            ap.error("one SAE (and one --latent, if given) per reader")
        edits = dict(np.load(args.base)) if args.base else {}
        info = {"base": args.base.name if args.base else None, "steps": []}
        for i, (r, p) in enumerate(zip(readers, saes)):
            lat = [args.latent[i]] if args.latent is not None else C.best_latents(C.identification(p), args.top_k)
            step = project_out(edits, orig, ad, r, p, lat, args.alpha)
            info["steps"].append(step)
            print(json.dumps(step), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **edits)
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print("[edit] wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
