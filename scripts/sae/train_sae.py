#!/usr/bin/env python
"""TopK sparse autoencoder at the input of one reader, latent identification and the latent-subtraction test
(paper Sec. 3.2, App. B.2).

Site: particle tokens at the input of the reader's attention block (norm1 output of a body block; body output for
a classifier block). Training tokens: JetClass jets (all ten classes, --n-jetclass per class from row 0) and JetClass-II
jets (--n-jetclass2 per file from row 30000), as they are (JetClass-II supplies sub-threshold neutrals). The defaults
give about 2.5 million tokens. Activations are centred and divided by their RMS distance from the mean; the SAE
minimises the reconstruction error normalised by the total activation variance.
Control dictionary: the same encoder/decoder with frozen random unit directions (seed 7).

Evaluation tokens (not used in training): real JetClass neutrals and charged particles; JetClass-II neutrals below /
above the JetClass thresholds; one neutral added to JetClass jets below threshold (photon 0.3 GeV, neutral hadron
0.5 GeV) and above (photon 3 GeV).
Latent identification: a latent fires when its activation is positive. selectivity = mean firing rate on the two
added sub-threshold neutrals - firing rate on real JetClass neutrals. Identified ("found") latents: rate >= 0.5 on
the added neutrals, <= 0.05 on real JetClass neutrals, selectivity above the 99th percentile of the control
dictionary. Latents are ranked by selectivity; among the 15 highest, those within 0.02 of the highest are compared
by the contribution of their decoder direction to the reader's attention logit for the mean clean query
(key alignment), and the largest is selected (`best_latent_rule`).
Latent-subtraction test: subtract z_l(h) d_l from the added particle's activation at the block input and measure the
reader's attention on it and |dF_W|; control: one other active latent of the same token, drawn at random.

  python scripts/sae/train_sae.py --size s --reader b7-6 --seed 0
      -> $RESULTS_DIR/sae/small/b7-6/latents4096_top32_seed0.{pt,json}
  python scripts/sae/train_sae.py --size s --reader b7-6 --checkpoint PATH.pt   (identification only, no training)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from softrepair import core as W
from softrepair import heads as H
from softrepair import sae_model as D
from softrepair import tools as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ADDED = ("added photon 0.3 GeV", "added neutral hadron 0.5 GeV")


def particle_tokens(cap, raw):
    acts, y, att = cap.run(raw)
    a = acts["norm1"][:, cap.p0:cap.p0 + raw.shape[1]]
    return a, y, att


def rates(model, X, dev):
    with torch.no_grad():
        out = []
        for s in range(0, len(X), 32768):
            out.append((model.encode(torch.as_tensor(X[s:s + 32768], device=dev)) > 0).float().mean(0))
        w = np.array([min(32768, len(X) - s) for s in range(0, len(X), 32768)], float)
        return (torch.stack(out) * torch.as_tensor(w / w.sum(), device=dev, dtype=torch.float32)[:, None]).sum(0).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"), help="OmniLearned model size")
    ap.add_argument("--reader", default="b7-6", help="reader head, e.g. b7-6, b4-0, c1-2 (b = body, c = classifier)")
    ap.add_argument("--latents", type=int, default=None, help="dictionary size (default 32 x token width)")
    ap.add_argument("--k", type=int, default=None, help="active latents per token (default: the reader's, 32 or 64)")
    ap.add_argument("--queries", default=None, choices=("summary", "all"),
                    help="whose attention is measured: the summary queries, or summary + real-particle queries "
                         "(default: the reader's setting)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-jetclass", type=int, default=3000, help="training jets per JetClass class")
    ap.add_argument("--n-jetclass2", type=int, default=10000, help="training jets per JetClass-II file")
    ap.add_argument("--seed", type=int, default=0, help="SAE initialisation seed")
    ap.add_argument("--batch-seed", type=int, default=None,
                    help="seed of the training-batch order (default 1 + 1000 seed; the published SAEs used seed + 1)")
    ap.add_argument("--cap-batch", type=int, default=None, help="jets per forward pass when capturing (s/m 256, l 32)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="existing SAE checkpoint: skip training, redo the identification and the subtraction test")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path without suffix (default $RESULTS_DIR/sae/<size name>/<reader>/<sae name>)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    C.set_size_env(args.size)
    dev = args.device
    where, head = C.parse_reader(args.reader)
    q_default, k_default = C.reader_settings(args.size, args.reader)
    prior = {}
    if args.checkpoint is not None and args.checkpoint.with_suffix(".json").exists():
        prior = json.loads(args.checkpoint.with_suffix(".json").read_text()).get("args", {})
    queries = args.queries or prior.get("queries", q_default)
    k = args.k or prior.get("k") or k_default
    cap_batch = args.cap_batch or C.CAPTURE_BATCH[args.size]
    ad = W.load_adapter(dev, args.size)
    cap = T.Capture(ad, sites=("norm1", "queries"), where=where, head=head, queries=queries)
    _run = cap.run  # every forward pass in cap_batch jets; per-jet results unchanged
    cap.run = lambda raw, batch=cap_batch, attention=True: _run(raw, batch=batch, attention=attention)
    res = {"reader": args.reader, "where": where, "head": head}
    if args.checkpoint is None:
        # ---- training tokens
        jc, _ = T.jetclass_jets(args.n_jetclass, 0)
        jcii, _ = T.jetclass2_jets(args.n_jetclass2, 30000)
        Xs = []
        cap_tr = T.Capture(ad, sites=("norm1",), where=where, head=head, queries=queries)
        for raw in (jc, jcii):  # in chunks of 2048 jets (memory on the large model); per-jet results unchanged
            for c0 in range(0, len(raw), 2048):
                r = raw[c0:c0 + 2048]
                a = cap_tr.run(r, batch=cap_batch, attention=False)[0]["norm1"][:, cap_tr.p0:cap_tr.p0 + r.shape[1]]
                Xs.append(a[r[..., 2] != 0].copy())
                del a
        X = np.concatenate(Xs).astype(np.float32)
        mu, scale = X.mean(0), float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))
        width = X.shape[1]
        norm = lambda a: ((a - mu) / scale).astype(np.float32)  # noqa: E731
        Xt = torch.as_tensor(norm(X), device=dev)
        print(f"[sae] {len(X)} training tokens", flush=True)
        m = args.latents or C.EXPANSION * width
        print(f"[sae] size {args.size}, reader {args.reader}, width {width}, {m} latents, k {k}, seed {args.seed}",
              flush=True)
        sae = D.TopKSAE(width, m, k, seed=args.seed).to(dev)
        bseed = args.batch_seed if args.batch_seed is not None else 1 + 1000 * args.seed
        log = D.train(sae, Xt, args.steps, args.batch, device=dev, seed=bseed)
        res.update({"n_train_tokens": int(len(X)), "train_log": log})
        del Xt
    else:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        mu, scale = ck["mu"], float(ck["scale"])
        width, m = ck["state"]["W_d"].shape
        sae = D.TopKSAE(width, m, k).to(dev)
        sae.load_state_dict(ck["state"])
        norm = lambda a: ((a - mu) / scale).astype(np.float32)  # noqa: E731
        res["checkpoint"] = args.checkpoint.name
        print(f"[sae] identification only: {args.checkpoint}, width {width}, {m} latents, k {k}", flush=True)
    null = D.TopKSAE(width, m, k, frozen_random=True, seed=7).to(dev)
    with torch.no_grad():
        null.b_d.copy_(torch.zeros(width, device=dev))
    res["args"] = {"size": args.size, "reader": args.reader, "where": where, "head": head, "queries": queries,
                   "m": int(m), "k": int(k), "seed": args.seed, "steps": args.steps, "batch": args.batch,
                   "n_jetclass": args.n_jetclass, "n_jetclass2": args.n_jetclass2, "batch_seed": args.batch_seed, "cap_batch": cap_batch}
    # ---- evaluation tokens
    jc_te, _ = T.jetclass_jets(300, 30000)
    jcii_te, _ = T.jetclass2_jets(2000, 10000)
    groups = {}
    a, _, _ = particle_tokens(cap, jc_te)
    code = np.rint(jc_te[..., 4]).astype(int)
    groups["real JetClass neutrals"] = a[(jc_te[..., 2] != 0) & np.isin(code, (6, 7))]
    groups["real JetClass charged"] = a[(jc_te[..., 2] != 0) & np.isin(code, (0, 8))]
    a, _, _ = particle_tokens(cap, jcii_te)
    code2 = np.rint(jcii_te[..., 4]).astype(int)
    sub = T.sub_threshold(jcii_te)
    neu2 = (jcii_te[..., 2] != 0) & np.isin(code2, (6, 7))
    groups["JetClass-II neutrals below JetClass threshold"] = a[neu2 & sub]
    groups["JetClass-II neutrals above"] = a[neu2 & ~sub]
    base = jc_te[:2000]
    added = {}
    for name, pid, pt in (("added photon 0.3 GeV", 6, 0.3), ("added neutral hadron 0.5 GeV", 7, 0.5),
                          ("added photon 3 GeV", 6, 3.0)):
        x, ok, slot = T.add_one(base, np.full(len(base), pt), pid, 31 + pid + int(pt * 10))
        a, y, att = particle_tokens(cap, x)
        idx = np.flatnonzero(ok)
        groups[name] = a[idx, slot[idx]]
        added[name] = (x, ok, slot, y, att)
    res["groups"] = {g: int(len(v)) for g, v in groups.items()}
    R = {}
    for dname, model in (("sae", sae), ("null", null)):
        R[dname] = {g: rates(model, norm(v), dev) for g, v in groups.items()}
        R[dname]["selectivity"] = 0.5 * (R[dname][ADDED[0]] + R[dname][ADDED[1]]) - R[dname]["real JetClass neutrals"]
        with torch.no_grad():
            v = torch.as_tensor(norm(groups["real JetClass neutrals"]), device=dev)
            h = torch.cat([model(v[c0:c0 + 8192])[0] for c0 in range(0, len(v), 8192)])  # chunks (memory on m, l)
            res[f"{dname}_fve_real_neutrals"] = float(1 - ((h - v) ** 2).sum() / ((v - v.mean(0)) ** 2).sum())
    with torch.no_grad():  # in chunks: (tokens x latents) does not fit on the GPU for m, l
        V = norm(np.concatenate([g for g in groups.values()]))
        vm = torch.as_tensor(V.mean(0), device=dev)
        sse = sst = 0.0
        fired = torch.zeros(m, dtype=torch.bool, device=dev)
        for c0 in range(0, len(V), 8192):
            v = torch.as_tensor(V[c0:c0 + 8192], device=dev)
            h, code = sae(v)
            sse += float(((h - v) ** 2).sum())
            sst += float(((v - vm) ** 2).sum())
            fired |= (code > 0).any(0)
        res["sae_fve_heldout_all"] = 1 - sse / sst
        res["dead_fraction_heldout"] = float((~fired).float().mean())
        res["n_latents"] = int(m)
        del V, v, h, code, fired
    thr = float(np.quantile(R["null"]["selectivity"], 0.99))
    sub_rate = 0.5 * (R["sae"][ADDED[0]] + R["sae"][ADDED[1]])
    found = np.flatnonzero((sub_rate >= 0.5) & (R["sae"]["real JetClass neutrals"] <= 0.05) & (R["sae"]["selectivity"] > thr))
    # key alignment: the reader's key read-out for the mean query on clean jets
    attn = cap.blk.attn
    Wq, Wk, _ = attn.in_proj_weight.detach().chunk(3, 0)
    dh = attn.embed_dim // attn.num_heads
    hs = slice(dh * head, dh * (head + 1))
    acts, _, _ = cap.run(jc_te[:1000], attention=False)
    if cap.queries == "summary":
        qx = acts["queries"][:, :4].reshape(-1, width)
    else:
        qx = np.concatenate([acts["norm1"][:, :4].reshape(-1, width),
                             acts["norm1"][:, cap.p0:cap.p0 + jc_te.shape[1]][jc_te[:1000, :, 2] != 0]])
    qm = (torch.as_tensor(qx, device=dev) @ Wq[hs].T).mean(0)
    dirs = sae.W_d.detach() * scale  # back in activation units
    key_logit = (qm @ (Wk[hs] @ dirs)).cpu().numpy() / float(np.sqrt(dh))
    order = np.argsort(-R["sae"]["selectivity"])[:15]
    # tie-break: among latents within 0.02 of the top selectivity, the one with the largest key alignment
    sel = R["sae"]["selectivity"]
    ties = [int(i) for i in order if sel[i] >= sel[order[0]] - 0.02]
    top = max(ties, key=lambda i: key_logit[i])
    order = np.array([top] + [int(i) for i in order if int(i) != top])
    res["best_latent_rule"] = {"ties": ties, "best": top}
    res["null_selectivity_p99"] = thr
    res["found"] = [int(i) for i in found]
    res["verdict"] = "found" if len(found) else "not found"
    res["top_latents"] = [{"latent": int(i), "selectivity": float(R["sae"]["selectivity"][i]),
                           "key_logit_per_unit": float(key_logit[i]),
                           **{f"rate: {g}": float(R["sae"][g][i]) for g in groups}} for i in order]
    # ---- latent-subtraction test of the selected latent
    causal = {}
    for name in ADDED:
        x, ok, slot, y0, att0 = added[name]
        idx = np.flatnonzero(ok)[:1000]
        F_clean = H.logodds_W(cap.run(base[idx], attention=False)[1])
        F_pert = H.logodds_W(y0[idx])
        a_tok = torch.as_tensor(norm(groups[name][:len(idx)]), device=dev)
        with torch.no_grad():
            codes = sae.encode(a_tok)
        for label, pick in (("best latent", lambda c: torch.full((len(c),), int(order[0]), device=dev)),
                            ("random other active latent", None)):
            if pick is None:
                g = torch.Generator(device=dev).manual_seed(3)
                act = (codes > 0).float()
                act[:, int(order[0])] = 0
                pick_idx = torch.multinomial(act + 1e-9, 1, generator=g)[:, 0]
            else:
                pick_idx = pick(codes)
            delta = codes[torch.arange(len(codes)), pick_idx][:, None] * dirs[:, pick_idx].T  # (n, width) activation units
            sl = torch.as_tensor(cap.p0 + slot[idx], device=dev)
            state = {"i": 0}

            def hook(mod, inp, out):
                s0 = state["i"]
                n = out.shape[0]
                o = out.clone()
                o[torch.arange(n, device=dev), sl[s0:s0 + n]] -= delta[s0:s0 + n]
                state["i"] += n
                return o

            hk = cap.mods["norm1"].register_forward_hook(hook)
            try:
                _, y, att = cap.run(x[idx], batch=cap_batch)
            finally:
                hk.remove()
            F = H.logodds_W(y)
            a_on = att[np.arange(len(idx)), slot[idx]]
            causal[f"{name} | {label}"] = {
                "head_attention_before": float(np.median(att0[idx, slot[idx]])),
                "head_attention_after": float(np.median(a_on)),
                "abs_dF_W_before": float(np.abs(F_pert - F_clean).mean()),
                "abs_dF_W_after": float(np.abs(F - F_clean).mean())}
    res["causal"] = causal
    print(json.dumps({k_: v_ for k_, v_ in res.items() if k_ != "train_log"}, indent=1), flush=True)
    if args.out is not None:
        out = args.out
    elif args.checkpoint is not None:
        out = W.OUT / "sae" / C.SIZE_NAMES[args.size] / args.reader / (args.checkpoint.stem + "_identification")
    else:
        out = W.OUT / "sae" / C.SIZE_NAMES[args.size] / args.reader / C.sae_name(m, k, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    if args.checkpoint is None:
        torch.save({"state": sae.state_dict(), "mu": mu, "scale": scale, "where": where, "head": head},
                   out.with_suffix(".pt"))
    print("[sae] wrote", out.with_suffix(".json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
