#!/usr/bin/env python
"""Attention of the summary tokens on one added soft particle, for every attention head (Sec. 2 Fig. 3 left,
App. A Fig. attention left, App. E Fig. app_readers left).

For every backbone block and head (and with --classifier also the two classifier blocks): the mean attention
weight that the 4 summary-token queries (averaged) give the slot of one added photon at a fraction f = 1e-3 of the
jet pT (core.perturb soft1e-3; --soft-pid 7: one neutral hadron, soft7_1e-3). Mean over jets, W and QCD pooled.
Weights are re-computed with need_weights=True from the same nn.MultiheadAttention call (forward pre-hook); the
model's output is unchanged. For the small model also, for b7/6: per-jet attention share of the summary queries on
the summary tokens themselves (clean jets) and on the added particle.

Jets: W file 5 and QCD file 32 (test-jet files), rows 60000..60000+n.

  python scripts/analysis/attention_heads.py [--n 1000] [--size s] [--classifier] [--soft-pid 7] [--edit X]
        [--name NAME]
      -> RESULTS_DIR/figures/<size>/<NAME>.npz / .json
         (NAME default: readers_attention_<photon|neutral_hadron>, with suffix _body without --classifier)
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


def attn_blocks(ad, classifier: bool = False) -> list:
    """(name, block) of the backbone blocks, and with classifier=True also of the classifier's attention blocks
    (the same sites as activation_patching.Patcher; the classifier uses the same token layout)."""
    out = [(f"body.{b}", blk) for b, blk in enumerate(ad.model.body.in_blocks)]
    if classifier:
        out += [(f"classifier.{c}", blk) for c, blk in enumerate(getattr(ad.model.classifier, "in_blocks", []))
                if getattr(blk, "use_attn", True) and hasattr(blk, "attn")]
    return out


def all_weights(ad, raw: np.ndarray, batch: int = 128, classifier: bool = False):
    """Per block (index into attn_blocks): (n, heads, 4, T) attention of the 4 summary queries on all tokens."""
    blocks = [blk for _, blk in attn_blocks(ad, classifier)]
    caps, hooks = {}, []
    for b, blk in enumerate(blocks):
        def pre(mod, args, kwargs, b=b):
            caps[b] = (args, dict(kwargs))
        hooks.append(blk.attn.register_forward_pre_hook(pre, with_kwargs=True))
    out = {b: [] for b in range(len(blocks))}
    try:
        with torch.no_grad():
            for s0 in range(0, len(raw), batch):
                ad.forward(ad.preprocess({"data": torch.as_tensor(np.asarray(raw[s0:s0 + batch], np.float32))}))
                for b, blk in enumerate(blocks):
                    a, kw = caps[b]
                    kw.update(need_weights=True, average_attn_weights=False)
                    _, w = blk.attn(*a, **kw)
                    out[b].append(w[:, :, :4, :].float().cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
    return {b: np.concatenate(v) for b, v in out.items()}


def compute(args) -> None:
    from softrepair import core as W
    from softrepair import heads as H
    torch.set_num_threads(args.threads)
    ad = W.load_adapter(args.device, args.size)
    if args.edit:
        W.set_state(ad, W.load_candidate(args.edit, W.original_state(ad)))
    start = ad.token_layout().particle_start
    grid, per = [], {}
    for lab, name in ((H.WJ, "W"), (H.QCD, "QCD")):
        raw = H.load(lab, 60000, args.n)
        slot = np.array([W._free_slot(r) for r in raw])
        sp = "soft" if args.soft_pid == 6 else f"soft{args.soft_pid}_"  # 7: one neutral hadron instead of a photon
        pert, ok = W.perturb(raw, f"{sp}1e-3", 11 + lab)  # same placement as w_tagger_response.py
        ok &= slot >= 0
        w0, w1 = all_weights(ad, raw, classifier=args.classifier), all_weights(ad, pert, classifier=args.classifier)
        # backbone blocks: keys = all tokens (summary, time, particles), particle j at start + j; classifier blocks:
        # queries = summary tokens, keys = the other tokens (time, particles), particle j at start - n_summary + j
        nsum = ad.token_layout().n_summary
        names = [n for n, _ in attn_blocks(ad, args.classifier)]
        off = {b: start - (nsum if names[b].startswith("classifier") else 0) for b in w1}
        g = np.stack([w1[b][np.arange(len(raw)), :, :, off[b] + np.clip(slot, 0, None)].mean(-1)[ok].mean(0)
                      for b in sorted(w1)])  # (blocks, heads)
        grid.append(g)
        b7 = 7
        if args.size != "s":  # b7/6 is the reader of the small model; for medium / large only the grid
            print(name, "done", flush=True)
            continue
        per[f"photon_share_{name}"] = w1[b7][np.arange(len(raw)), 6, :, start + np.clip(slot, 0, None)].mean(-1)[ok]
        per[f"self_share_clean_{name}"] = w0[b7][:, 6, :, :4].sum(-1).mean(-1)
        per[f"self_share_photon_{name}"] = w1[b7][:, 6, :, :4].sum(-1).mean(-1)[ok]
        per[f"max_particle_clean_{name}"] = w0[b7][:, 6, :, start:].max(-1).mean(-1)
        print(name, "done", flush=True)
    G = np.mean(grid, 0)
    particle = "neutral_hadron" if args.soft_pid == 7 else "photon"
    name = args.name or f"readers_attention_{particle}" + ("" if args.classifier else "_body")
    out = args.out_dir or C.out_dir(args.size)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", grid=G, grid_W=grid[0], grid_QCD=grid[1], **per)
    i, j = np.unravel_index(np.argmax(G), G.shape)
    srt = np.sort(G.ravel())[::-1]
    num = {"n_per_class": args.n, "f": 1e-3, "soft_pid": args.soft_pid,
           "definition": "mean over jets (W and QCD pooled) of the attention weight "
           "that the 4 summary-token queries (averaged) give the added photon's slot", "argmax_block_head": [int(i), int(j)],
           "max": float(G[i, j]), "second": float(srt[1]), "median_cell": float(np.median(G)),
           "b7h6": {k: {"mean": float(v.mean()), "median": float(np.median(v)), "p10": float(np.quantile(v, .1)),
                        "p90": float(np.quantile(v, .9))} for k, v in per.items()},
           "top5": [[int(k // G.shape[1]), int(k % G.shape[1]), float(G.ravel()[k])] for k in np.argsort(G.ravel())[::-1][:5]],
           "size": args.size, "sites": [n for n, _ in attn_blocks(ad, args.classifier)], "grid": G.tolist()}
    (out / f"{name}.json").write_text(json.dumps(num, indent=1))
    print(json.dumps({k: v for k, v in num.items() if k != "grid"}, indent=1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--edit", type=Path, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--classifier", action="store_true", help="also the classifier's attention blocks")
    ap.add_argument("--soft-pid", type=int, default=6, choices=(6, 7), help="6 photon, 7 neutral hadron")
    ap.add_argument("--fused-attention", action="store_true", help="fused attention kernels (default: math kernel)")
    args = ap.parse_args()
    C.set_size(args.size)
    C.attention_kernel(args.fused_attention)
    compute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
