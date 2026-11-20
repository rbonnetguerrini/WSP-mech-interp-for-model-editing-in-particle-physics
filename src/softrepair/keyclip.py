#!/usr/bin/env python
"""Clipped-key repair: switch an SAE feature off in one head's keys only above what real JetClass tokens ever show.

For each reader (attention block, head) and an orthonormal basis e_1..e_k of SAE feature directions at its key
input x, the head's keys become
    k_h = W_K,h (x - sum_j relu(x . e_j - tau_j) e_j),    tau_j = max of x . e_j over real JetClass key tokens.
Real JetClass tokens are untouched by construction (c_j <= tau_j); tokens beyond JetClass's range (the sub-threshold
neutrals that signal JetClass-II) are clipped back to the edge. Queries, values and the other heads are unchanged.
This is a small module change (one ReLU per direction and head), not a pure weight edit.

  python -m softrepair.keyclip build --out spec.json --sites body.7:6:SAE.pt:1 body.4:0:SAE.pt:1 classifier.1:2:SAE.pt:1
  (repair_eval score --keyclip spec.json scores it; install(ad, spec) applies it to a loaded adapter)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from softrepair import core as W
from softrepair import tools as T


def basis(sae_path: Path, top_k: int) -> tuple[np.ndarray, list[int]]:
    js = json.loads(Path(str(sae_path).replace(".pt", ".json")).read_text())
    found = set(js["found"])
    ranked = [int(l["latent"]) for l in js["top_latents"] if int(l["latent"]) in found][:top_k]
    ck = torch.load(sae_path, map_location="cpu", weights_only=False)
    U, _ = np.linalg.qr(ck["state"]["W_d"][:, ranked].double().numpy())
    return U, ranked


def key_tokens(cap, raw):
    """Real key tokens at the head's key input: summary + time + real particles (body) or time + particles
    (classifier token block)."""
    acts, _, _ = cap.run(raw, attention=False)
    a = acts["norm1"]
    m = np.zeros(a.shape[:2], bool)
    m[:, cap.p0:cap.p0 + raw.shape[1]] = raw[..., 2] != 0
    if cap.token_block:
        m[:, cap.p0 - 1] = True  # time token
    else:
        m[:, :cap.p0] = True
    return a[m]


def latent_affine(sae_path: Path, latents: list[int]):
    """Raw-activation form of the SAE encoder pre-activation and decoder of the given latents:
    z = x @ enc.T + b0 (z = W_e (x_n - b_d) + b_e with x_n = (x - mu) / scale) and dec = scale * W_d[:, i]."""
    ck = torch.load(sae_path, map_location="cpu", weights_only=False)
    st, mu, scale = ck["state"], np.asarray(ck["mu"], np.float64), float(ck["scale"])
    We, be, bd, Wd = (st[k].double().numpy() for k in ("W_e", "b_e", "b_d", "W_d"))
    enc = We[latents] / scale
    b0 = be[latents] - We[latents] @ (mu / scale + bd)
    dec = (Wd[:, latents] * scale).T
    return enc, b0, dec


def build(args) -> None:
    ad = W.load_adapter(args.device, "s")
    jc, _ = T.jetclass_jets(args.n, 0)
    base = jc[:3000]
    spec = []
    for s in args.sites:
        where, head, path, k = s.split(":")
        cap = T.Capture(ad, sites=("norm1",), where=where, head=int(head))
        real = key_tokens(cap, jc)
        if args.clip == "coordinate":
            U, lat = basis(Path(path), int(k))
            c = real @ U  # (tokens, k)
            tau = c.max(0)
            spec.append({"where": where, "head": int(head), "latents": lat, "U": U.T.tolist(), "tau": tau.tolist(),
                         "tau_quantiles_999": np.quantile(c, 0.999, axis=0).tolist(), "n_tokens": int(len(c))})
        else:  # latent: clip the SAE encoder pre-activation above its maximum on real JetClass key tokens
            _, lat = basis(Path(path), int(k))
            enc, b0, dec = latent_affine(Path(path), lat)
            z = real @ enc.T + b0
            tau = np.quantile(z, args.tau_quantile, axis=0) if args.tau_quantile < 1 else z.max(0)
            # coverage on added sub-threshold neutrals (photon 0.3 GeV)
            x, ok, slot = T.add_one(base, np.full(len(base), 0.3), 6, 5)
            a = cap.run(x, attention=False)[0]["norm1"]
            idx = np.flatnonzero(ok)
            za = a[idx, cap.p0 + slot[idx]] @ enc.T + b0
            spec.append({"mode": "latent", "where": where, "head": int(head), "latents": lat, "enc": enc.tolist(),
                         "b0": b0.tolist(), "dec": dec.tolist(), "tau": tau.tolist(),
                         "z_real_max": z.max(0).tolist(), "z_real_q9999": np.quantile(z, 0.9999, axis=0).tolist(),
                         "z_added_photon_median": np.median(za, 0).tolist(),
                         "added_photon_frac_above_tau": (za > tau).mean(0).tolist(), "n_tokens": int(len(z))})
            c = z
        print(where, head, lat, "tau", np.round(tau, 3), {k2: v for k2, v in spec[-1].items()
                                                          if k2.startswith(("z_", "added"))}, flush=True)
    Path(args.out).write_text(json.dumps(spec, indent=1))


def install(ad, spec) -> list:
    """Replace the forward of each listed nn.MultiheadAttention by an exact re-implementation with the clipped keys
    of one head. Returns the modules (for uninstall)."""
    mods = []
    for e in spec:
        mod = T.attn_block(ad, e["where"]).attn
        dev = mod.in_proj_weight.device
        f32 = lambda v: torch.as_tensor(np.array(v), dtype=torch.float32, device=dev)  # noqa: E731
        h = e["head"]
        if e.get("mode") == "latent":  # (enc, b0, dec): key input x - relu(x enc^T + b0 - tau) dec
            item = (h, f32(e["enc"]), f32(e["b0"]), f32(e["dec"]), f32(e["tau"]))
        else:  # coordinate: enc = dec = U, b0 = 0
            U = f32(e["U"])
            item = (h, U, torch.zeros(U.shape[0], device=dev), U, f32(e["tau"]))
        prev = getattr(mod, "_keyclip", [])
        mod._keyclip = prev + [item]
        if not prev:
            def fwd(query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False,
                    average_attn_weights=True, mod=mod, **kw):
                B, Tq, D = query.shape
                S, Hh = key.shape[1], mod.num_heads
                d = D // Hh
                Wq, Wk, Wv = mod.in_proj_weight.chunk(3, 0)
                q = Fn.linear(query, Wq).view(B, Tq, Hh, d).transpose(1, 2)
                k = Fn.linear(key, Wk).view(B, S, Hh, d).transpose(1, 2).clone()
                v = Fn.linear(value, Wv).view(B, S, Hh, d).transpose(1, 2)
                for hh, enc, b0, dec, tc in mod._keyclip:
                    c = key @ enc.T + b0  # (B, S, k)
                    kin = key - torch.relu(c - tc) @ dec
                    k[:, hh] = Fn.linear(kin, Wk[16 * hh:16 * (hh + 1)]) if d == 16 else \
                        Fn.linear(kin, Wk[d * hh:d * (hh + 1)])
                sc = q @ k.transpose(-1, -2) / math.sqrt(d)
                if attn_mask is not None:
                    am = attn_mask.view(B, Hh, Tq, S) if attn_mask.dim() == 3 else attn_mask
                    sc = sc.masked_fill(am, float("-inf")) if am.dtype == torch.bool else sc + am
                if key_padding_mask is not None:
                    sc = sc.masked_fill(key_padding_mask[:, None, None, :].bool(), float("-inf"))
                w = torch.softmax(sc, -1)
                o = (w @ v).transpose(1, 2).reshape(B, Tq, D)
                o = Fn.linear(o, mod.out_proj.weight, mod.out_proj.bias)
                if need_weights:
                    return o, (w.mean(1) if average_attn_weights else w)
                return o, None
            mod.forward = fwd
        mods.append(mod)
    return mods


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build",))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", nargs="+", required=True, help="where:head:sae.pt:k")
    ap.add_argument("--n", type=int, default=1000, help="JetClass jets per class for tau")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip", default="latent", choices=("latent", "coordinate"))
    ap.add_argument("--tau-quantile", type=float, default=1.0, help="1 = max over real JetClass key tokens")
    args = ap.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
