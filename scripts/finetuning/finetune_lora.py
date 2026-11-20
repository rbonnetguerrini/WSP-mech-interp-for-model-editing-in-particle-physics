#!/usr/bin/env python
"""Fine-tuning repair with low-rank weight updates (paper Sec. 3.1, App. B.1).

Every selected weight matrix W gets W + A B (A: out x r, zero init; B: r x in, random init), so each change has
rank <= r. Selected vectors (biases, normalization gains) change freely. The generator is frozen.

Loss on a batch x of practice jets (y = current logits, y0 = original logits, c = centred ten JetClass logits,
S0 = spread of the original centred logits over clean jets):
  L_hold = mean_k w_k (y_k(x) - y0_k(x))^2 / mean_k w_k           w = 1 on JetClass logits, 0.1 elsewhere
  L_IC   = mean over the modifications a(x) of ||c(y(a x)) - c(y(x))||^2 / S0^2
           a = one soft neutral at pT fraction f ~ logU[1e-4, 1e-2], uniform in dR < 0.4 (photon, or photon or
               neutral hadron with equal probability per jet), and the hardest neutral split into two collinear
               halves; a new random draw at every step.
           --both-per-jet: one soft photon AND one soft neutral hadron per jet, each its own modification.
  L_phys = mean over T in (angular spread, momentum scale, fragmentation) of
           ||(c(y(T x)) - c(y(x))) - (c(y0(T x)) - c(y0(x)))||^2 / S0^2
  L_FT   = L_hold + lambda L_IC + mu L_phys
Adam, 100-step linear warm-up, then cosine decay.

The edit is written as a partial state dict of full replacement tensors (the changed tensors only):
  $RESULTS_DIR/edits/finetuning/<size>/<name>.npz  and  <name>.train_log.json

  python scripts/finetuning/finetune_lora.py --size s --variant attention_only_both_neutral_types --seed 0
  python scripts/finetuning/finetune_lora.py --size s --scope attention --particles both --rank 8 --name my_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from softrepair import core as W
from softrepair import repair_eval as B
from softrepair.adapters.omnilearned import OmniLearnedAdapter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

SCOPE_RULES = {
    "whole_body_and_classifier": lambda k: k.startswith(("body.", "classifier.")),
    "attention": lambda k: k.startswith("body.") and (".attn." in k or k.startswith("body.interaction.")),
}


def augment(raw: np.ndarray, rng: np.random.Generator, pids: tuple[int, ...] = (6,),
            both_per_jet: bool = False) -> list[np.ndarray]:
    """The L_IC modifications of one batch: soft-neutral addition(s) and the hardest-neutral split.
    both_per_jet=False: one soft addition whose particle type is drawn per jet from pids.
    both_per_jet=True: one soft addition per particle type in pids."""
    softs, split = [raw.copy() for _ in (pids if both_per_jet else pids[:1])], raw.copy()
    for j in range(len(raw)):
        if both_per_jet:
            for s, pid in zip(softs, pids):
                W._soft(s[j], float(10 ** rng.uniform(-4, -2)), pid, rng)
        else:
            pid = pids[0] if len(pids) == 1 else int(rng.choice(pids))
            W._soft(softs[0][j], float(10 ** rng.uniform(-4, -2)), pid, rng)
        W._split(split[j], (0.5, 0.5), charged=False)
    return softs + [split]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"), help="OmniLearned model size")
    ap.add_argument("--variant", default=C.SELECTED, choices=tuple(C.VARIANTS),
                    help="paper variant (sets --scope, --particles, --both-per-jet, --mu, --rank); explicit options "
                         "below override it")
    ap.add_argument("--name", default=None, help="output name (default: the variant, with _seed<s> if --seed > 0)")
    ap.add_argument("--scope", default=None, choices=C.SCOPES,
                    help="attention: body attention modules + pairwise-interaction network; "
                         "whole_body_and_classifier: every body and classifier tensor")
    ap.add_argument("--particles", default=None, choices=tuple(C.PARTICLES),
                    help="type of the added soft neutral in L_IC: photons, or both (photon or neutral hadron)")
    ap.add_argument("--both-per-jet", action="store_true", default=None,
                    help="with --particles both: one photon AND one neutral hadron per jet")
    ap.add_argument("--rank", type=int, default=None, help="rank of each matrix update (default 8; 0 = full update)")
    ap.add_argument("--lam", type=float, default=30.0, help="lambda, weight of L_IC")
    ap.add_argument("--mu", type=float, default=None, help="mu, weight of L_phys (default 10; 0 = without L_phys)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="Adam learning rate (default 1e-3; 3e-5 for --rank 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="accumulate gradients over chunks of this many jets (same update as the whole batch when "
                         "it divides --batch, less GPU memory); 0 = whole batch. Default per size: s 0, m 32, l 8")
    ap.add_argument("--math-attention", action=argparse.BooleanOptionalAction, default=None,
                    help="math attention kernel instead of the fused kernels (default per size: s off, m/l on)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output folder (default $RESULTS_DIR/edits/finetuning/<size name>)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    v = C.VARIANTS[args.variant]
    scope = args.scope or v["scope"]
    particles = args.particles or v["particles"]
    both_per_jet = args.both_per_jet if args.both_per_jet is not None else v.get("both_per_jet", False)
    rank = args.rank if args.rank is not None else v.get("rank", 8)
    mu = args.mu if args.mu is not None else v.get("mu", 10.0)
    sd = C.SIZE_DEFAULTS[args.size]
    micro_batch = args.micro_batch if args.micro_batch is not None else sd["micro_batch"]
    math_attention = args.math_attention if args.math_attention is not None else sd["math_attention"]
    if sd["max_batch"] and not os.environ.get("SOFTREPAIR_MAX_BATCH"):
        os.environ["SOFTREPAIR_MAX_BATCH"] = str(sd["max_batch"])
    os.environ["MODEL_SIZE"] = args.size
    name = args.name or (args.variant + (f"_seed{args.seed}" if args.seed else ""))
    PIDS = C.PARTICLES[particles]
    if both_per_jet and len(PIDS) < 2:
        ap.error("--both-per-jet needs --particles both")

    if math_attention:
        W.use_math_sdp()
    lr = args.lr or (3e-5 if rank == 0 else 1e-3)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = args.out_dir or (W.OUT / "edits" / "finetuning" / C.SIZE_NAMES[args.size])
    out.mkdir(parents=True, exist_ok=True)
    ad = W.load_adapter(args.device, args.size)
    dev, model = ad.device, ad.model
    orig = W.original_state(ad)
    data = W.load_set("practice", B.PHYS, mmap=False)
    n = len(data["labels"])

    named = dict(model.named_parameters())
    for p in named.values():
        p.requires_grad_(False)
    sel = [k for k in named if SCOPE_RULES[scope](k)]
    lora, free = {}, {}
    for k in sel:
        Wt = named[k]
        if Wt.ndim == 2 and rank > 0:
            r = min(rank, *Wt.shape)
            A = torch.zeros(Wt.shape[0], r, device=dev, requires_grad=True)
            Bm = (torch.randn(r, Wt.shape[1], device=dev) / Wt.shape[1] ** 0.5).requires_grad_(True)
            lora[k] = (A, Bm)
        else:
            free[k] = torch.zeros_like(Wt, requires_grad=True)
    params = [t for ab in lora.values() for t in ab] + list(free.values())
    print(f"[ft] {name}: size {args.size}, scope {scope}, particles {particles}"
          f"{' (both per jet)' if both_per_jet else ''}, rank {rank}, lambda {args.lam}, mu {mu}, "
          f"{len(lora)} low-rank + {len(free)} free tensors, {sum(p.numel() for p in params)} trainable numbers, "
          f"lr {lr}, micro-batch {micro_batch or 'off'}, math attention {math_attention}", flush=True)

    def current() -> dict:
        d = {}
        for k, (A, Bm) in lora.items():
            d[k] = named[k] + A @ Bm
        for k, D in free.items():
            d[k] = named[k] + D
        return d

    def fwd(raw: np.ndarray, over: dict) -> torch.Tensor:
        b = OmniLearnedAdapter.preprocess({"data": torch.as_tensor(W._trunc(np.asarray(raw, np.float32)))})
        X = b["X"].to(dev)
        bo = {k[5:]: v for k, v in over.items() if k.startswith("body.")}
        co = {k[11:]: v for k, v in over.items() if k.startswith("classifier.")}
        xb = functional_call(model.body, bo, (X, None, b["pid"].to(dev), b["add_info"].to(dev),
                                              torch.zeros(X.shape[0], device=dev)), strict=False)
        return functional_call(model.classifier, co, (xb,), strict=False)

    with torch.no_grad():
        Y0 = {v: torch.as_tensor(np.concatenate([W.logits(ad, np.asarray(data[v][s:s + 512]))
                                                 for s in range(0, n, 512)]), dtype=torch.float32, device=dev)
              for v in ("clean",) + B.PHYS}
    c0 = W.centred10(Y0["clean"].double().cpu().numpy())
    S0sq = float(2 * c0.var(0).sum())
    wlog = torch.full((210,), 0.1, device=dev)
    wlog[2:12] = 1.0

    def c10(y):
        l = y[:, 2:12]
        return l - l.mean(1, keepdim=True)

    opt = torch.optim.Adam(params, lr=lr)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, args.steps - warm))))
    log, t0 = [], time.time()
    micro = micro_batch or args.batch
    for step in range(args.steps):
        idx = np.sort(rng.choice(n, args.batch, replace=False))
        raw = np.asarray(data["clean"][idx])
        augs = augment(raw, rng, PIDS, both_per_jet)  # drawn for the whole batch: independent of the micro-batch
        opt.zero_grad(set_to_none=True)
        tot = {"L_FT": 0.0, "L_hold": 0.0, "L_IC": 0.0, "L_phys": 0.0}
        # every term is a mean over jets, so chunk losses weighted by their share give the whole-batch gradient
        for s0 in range(0, args.batch, micro):
            sl = slice(s0, min(s0 + micro, args.batch))
            f = (sl.stop - sl.start) / args.batch
            ii = idx[sl]
            over = current()
            y = fwd(raw[sl], over)
            hold = ((((y - Y0["clean"][ii]) ** 2) * wlog).sum(1) / wlog.sum()).mean()
            ic = sum(((c10(fwd(a[sl], over)) - c10(y)) ** 2).sum(1).mean() for a in augs) / (len(augs) * S0sq)
            phys = 0.0
            if mu > 0:
                for op in B.PHYS:
                    d_new = c10(fwd(np.asarray(data[op][ii]), over)) - c10(y)
                    d_old = c10(Y0[op][ii]) - c10(Y0["clean"][ii])
                    phys = phys + ((d_new - d_old) ** 2).sum(1).mean() / S0sq
                phys = phys / len(B.PHYS)
            loss = hold + args.lam * ic + mu * phys
            (f * loss).backward()
            for k, v_ in (("L_FT", loss), ("L_hold", hold), ("L_IC", ic), ("L_phys", phys)):
                tot[k] += f * float(v_)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            rec = {"step": step, **tot}
            log.append(rec)
            print(f"[ft] {rec} ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        final = {k: v_.detach().float().cpu().numpy() for k, v_ in current().items()}
    path = out / f"{name}.npz"
    np.savez(path, **final)
    state = dict(orig)
    state.update({k: torch.as_tensor(v_).to(orig[k].dtype) for k, v_ in final.items()})
    size = B.edit_size(state, orig)
    print(f"[ft] changed parameters: {size['changed_parameters']} (max rank {size['max_rank']})", flush=True)
    settings = {"variant": args.variant, "name": name, "size": args.size, "scope": scope, "particles": particles,
                "both_per_jet": both_per_jet, "rank": rank, "lambda": args.lam, "mu": mu, "steps": args.steps,
                "batch": args.batch, "lr": lr, "warmup_steps": warm, "seed": args.seed,
                "micro_batch": micro_batch, "math_attention": math_attention,
                "trainable_numbers": int(sum(p.numel() for p in params)), "training_jets": int(n)}
    (out / f"{name}.train_log.json").write_text(json.dumps(
        {"settings": settings, "changed_parameters": size["changed_parameters"], "max_rank": size["max_rank"],
         "log": log}, indent=1))
    print("[ft] wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
