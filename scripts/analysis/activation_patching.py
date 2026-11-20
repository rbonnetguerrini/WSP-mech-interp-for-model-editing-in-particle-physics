#!/usr/bin/env python
"""Activation patching of every attention head: how much of the change of F_W does each head carry?
(Sec. 2 Fig. 3 right, App. A Fig. attention right, App. E Fig. app_readers right.)

For a clean jet x and its perturbed copy x' (one added soft particle, or a collinear split), with
F = logit_W - logit_QCD of the pretrained 210-way head:
- noising ("removed"): run x', but replace one head's output (its attention output, before the output projection)
  with that head's output on x. removed = 1 - mean|F_patched - F(x)| / mean|F(x') - F(x)|.
  1 = this head alone carries the whole shift; 0 = none of it.
- denoising ("recreated"): run x, but insert that head's output from x'.
  recreated = mean|F_inserted - F(x)| / mean|F(x') - F(x)|. 1 = this head alone is sufficient.
Positions:
- "readers": every token whose input row is identical in x and x' (summary tokens, time token, all unchanged
  particles). The perturbed tokens keep their own value, so this blocks only what the head passes from the
  perturbed tokens to everything else.
- "summary": only the summary tokens (the classifier reads them).
Also every whole block (all heads at once, "readers"). Sites: the backbone blocks and the classifier blocks.

Attention is recomputed by an exact re-implementation of nn.MultiheadAttention (checked against the model's own
output at start), which exposes the per-head outputs.

Probes: photon_0.3GeV, neutral_hadron_0.5GeV, photon_3GeV (one neutral of fixed pT), split2, soft1e-3 and
soft7_1e-3 (one photon / neutral hadron at f = 1e-3, core.perturb).
Jets: W file 5 and QCD file 32 (test-jet files), rows 60000..60000+n (the stored results use --n 400).

  python scripts/analysis/activation_patching.py [--n 400] [--size s] [--device cuda] [--compute-probes ...]
      -> RESULTS_DIR/figures/<size>/readers_patching.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PROBES = {"photon_0.3GeV": (6, 0.3), "neutral_hadron_0.5GeV": (7, 0.5), "photon_3GeV": (6, 3.0), "split2": None,
          "soft1e-3": None, "soft7_1e-3": None}  # last two: one photon / neutral hadron at f = 1e-3 (core._soft)


class Patcher:
    """Replaces the forward of each nn.MultiheadAttention by an exact per-head re-implementation."""

    def __init__(self, ad):
        self.sites = {}
        for b, blk in enumerate(ad.model.body.in_blocks):
            self.sites[f"body.{b}"] = blk.attn
        for c, blk in enumerate(getattr(ad.model.classifier, "in_blocks", [])):
            if getattr(blk, "use_attn", True) and hasattr(blk, "attn"):
                self.sites[f"classifier.{c}"] = blk.attn
        self.mode, self.store, self.patch = None, {}, None
        for name, mod in self.sites.items():
            mod.forward = self._make(name, mod)

    def _make(self, name, mod):
        def fwd(query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False, **kw):
            assert mod.batch_first
            B, T, D = query.shape
            S, Hh = key.shape[1], mod.num_heads
            d = D // Hh
            Wq, Wk, Wv = mod.in_proj_weight.chunk(3, 0)
            bq = bk = bv = None
            if mod.in_proj_bias is not None:
                bq, bk, bv = mod.in_proj_bias.chunk(3, 0)
            q = Fn.linear(query, Wq, bq).view(B, T, Hh, d).transpose(1, 2)
            k = Fn.linear(key, Wk, bk).view(B, S, Hh, d).transpose(1, 2)
            v = Fn.linear(value, Wv, bv).view(B, S, Hh, d).transpose(1, 2)
            sc = q @ k.transpose(-1, -2) / math.sqrt(d)
            if attn_mask is not None:
                am = attn_mask.view(B, Hh, T, S) if attn_mask.dim() == 3 else attn_mask
                sc = sc.masked_fill(am, float("-inf")) if am.dtype == torch.bool else sc + am
            if key_padding_mask is not None:
                sc = sc.masked_fill(key_padding_mask[:, None, None, :].bool(), float("-inf"))
            o = torch.softmax(sc, -1) @ v  # (B, H, T, d)
            if self.mode == "capture":
                self.store[name] = o.detach().clone()
            elif self.mode == "patch" and self.patch is not None and name in self.patch["sites"]:
                src, pos, heads = self.patch["src"][name], self.patch["pos"], self.patch["heads"]
                o = o.clone()
                for h in heads:
                    o[:, h] = torch.where(pos[:, :T, None], src[:, h], o[:, h])
            out = o.transpose(1, 2).reshape(B, T, D)
            out = Fn.linear(out, mod.out_proj.weight, mod.out_proj.bias)
            return out, None
        return fwd


def fwd_F(ad, raw: np.ndarray) -> np.ndarray:
    from softrepair import heads as H
    with torch.no_grad():
        o = ad.forward(ad.preprocess({"data": torch.as_tensor(np.asarray(raw, np.float32))}))
    return H.logodds_W(o["y_pred"].double().cpu().numpy())


def compute(args) -> None:
    from softrepair import core as W
    from softrepair import heads as H
    torch.set_num_threads(args.threads)
    ad = W.load_adapter(args.device, args.size)
    raw_all = np.concatenate([H.load(H.WJ, 60000, args.n), H.load(H.QCD, 60000, args.n)])
    # exactness check of the re-implementation, before patching is installed
    chk = W._trunc(raw_all[:64])
    F_ref = fwd_F(ad, chk)
    P = Patcher(ad)
    F_new = fwd_F(ad, chk)
    err = float(np.abs(F_new - F_ref).max())
    print(f"[patch] re-implementation max |dF| = {err:.2e}", flush=True)
    assert err < 1e-3, err
    lay = ad.token_layout()
    p0, nsum = lay.particle_start, lay.n_summary
    sites = list(P.sites)
    nh = {s: P.sites[s].num_heads for s in sites}
    res = {"n_per_class": args.n, "sites": sites, "heads": nh, "particle_start": p0, "probes": {}}
    out_file = args.out or C.out_dir(args.size) / "readers_patching.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.compute_probes:  # add to an existing result (same jets: rows 60000.., same --n) instead of redoing all
        old = json.loads(out_file.read_text()) if out_file.exists() else {}
        assert not old or old.get("n_per_class") == args.n, "different --n than the existing result"
        res["probes"] = old.get("probes", {})
    for pi, (pname, spec) in enumerate(PROBES.items()):
        if args.compute_probes and pname not in args.compute_probes:
            continue
        if spec is None:
            pert, ok = W.perturb(raw_all, pname, 9100 + pi)
        else:
            pert, ok = C.add_fixed(raw_all, spec[1], spec[0], 9100 + pi)
        idx = np.flatnonzero(ok)
        acc = {"base": [], "noise": {}, "insert": {}}
        for s0 in range(0, len(idx), args.batch):
            ii = idx[s0:s0 + args.batch]
            c, p = raw_all[ii], pert[ii]
            last = int(max(np.max(np.where((c[:, :, 2] != 0).any(0))[0]),
                           np.max(np.where((p[:, :, 2] != 0).any(0))[0]))) + 1
            c, p = c[:, :last], p[:, :last]
            same = np.all(c == p, axis=2)  # (n, last) unchanged particle rows
            P.mode = "capture"
            Fc = fwd_F(ad, c); sc = dict(P.store)
            Fp = fwd_F(ad, p); sp = dict(P.store)
            T = next(iter(sc.values())).shape[2]
            readers = torch.ones(len(ii), T, dtype=torch.bool, device=ad.device)
            readers[:, p0:p0 + last] = torch.as_tensor(same, device=ad.device)
            summary = torch.zeros(len(ii), T, dtype=torch.bool, device=ad.device)
            summary[:, :nsum] = True
            acc["base"].append(np.abs(Fp - Fc))
            P.mode = "patch"
            jobs = []
            for s in sites:
                for h in range(nh[s]):
                    jobs += [("noise", f"{s}.h{h}.readers", [s], [h], readers),
                             ("noise", f"{s}.h{h}.summary", [s], [h], summary),
                             ("insert", f"{s}.h{h}.readers", [s], [h], readers)]
                jobs += [("noise", f"{s}.all.readers", [s], list(range(nh[s])), readers),
                         ("insert", f"{s}.all.readers", [s], list(range(nh[s])), readers)]
            for kind, key, ss, hs, pos in jobs:
                src = sc if kind == "noise" else sp
                P.patch = {"sites": set(ss), "heads": hs, "pos": pos, "src": src}
                Fx = fwd_F(ad, p if kind == "noise" else c)
                acc[kind].setdefault(key, []).append(np.abs(Fx - Fc))
            P.patch, P.mode = None, None
            print(f"[{time.time() - t0:6.0f}s] {pname}: {s0 + len(ii)}/{len(idx)} jets", flush=True)
        base = np.concatenate(acc["base"])
        m0 = float(base.mean())
        rec = {"n": int(len(base)), "mean_abs_dF": m0, "removed": {}, "recreated": {}}
        for key, v in acc["noise"].items():
            rec["removed"][key] = 1.0 - float(np.concatenate(v).mean()) / m0
        for key, v in acc["insert"].items():
            rec["recreated"][key] = float(np.concatenate(v).mean()) / m0
        top = sorted(((v, k) for k, v in rec["removed"].items() if ".h" in k and k.endswith("readers")), reverse=True)[:6]
        rec["top_removed_readers"] = [[k, round(v, 3)] for v, k in top]
        res["probes"][pname] = rec
        print(f"{pname}: mean|dF| {m0:.3f}; top heads (removed, readers): {rec['top_removed_readers']}", flush=True)
        out_file.write_text(json.dumps(res, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="jets per class (W, QCD)")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None, help="output JSON (default RESULTS_DIR/figures/<size>/readers_patching.json)")
    ap.add_argument("--compute-probes", nargs="*", default=None,
                    help="compute only these probes and add them to the existing output file")
    ap.add_argument("--fused-attention", action="store_true", help="fused attention kernels (default: math kernel)")
    args = ap.parse_args()
    C.set_size(args.size)
    C.attention_kernel(args.fused_attention)
    compute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
