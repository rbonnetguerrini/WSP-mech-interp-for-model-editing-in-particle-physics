"""Helpers shared by the analysis scripts of Sec. 2 and App. A (soft-particle response of OmniLearned).

  add_fixed      one massless neutral of fixed pT added to every jet
  head_weights   attention of the four summary tokens of one backbone head on every particle slot
  shares         summed probability of the ATLAS, JetClass and JetClass-II class blocks of the 200-way head
  features       the four summary tokens after every backbone block, the final norm and every classifier block
  out_dir        default output folder RESULTS_DIR/figures/<small|medium|large>

The scripts import the library (softrepair) for the model, the jets and the added particles.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from softrepair import core as W
from softrepair import paths

SIZE_NAME = {"s": "small", "m": "medium", "l": "large"}
BLOCK, HEAD = 7, 6  # the main reader of OmniLearned-s, b7/6
BLOCKS = {"ATLAS (0-1)": (0, 2), "JetClass (2-11)": (2, 12), "JetClass-II (12-199)": (12, 200)}


def out_dir(size: str, sub: str | None = None) -> Path:
    d = paths.results_dir() / "figures" / SIZE_NAME[size]
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def analysis_dir() -> Path:
    d = paths.results_dir() / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def attention_kernel(fused: bool) -> None:
    """Math attention kernel unless fused=True (the stored GPU results were computed with the math kernel)."""
    if not fused:
        W.use_math_sdp()


def set_size(size: str) -> None:
    os.environ["MODEL_SIZE"] = size
    if size == "l" and not os.environ.get("SOFTREPAIR_MAX_BATCH"):
        os.environ["SOFTREPAIR_MAX_BATCH"] = "32"


def add_fixed(raw: np.ndarray, pt: float, pid: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One massless neutral of fixed pT (GeV), uniform within dR < 0.4, impact parameters 0."""
    rng = np.random.default_rng(seed)
    out, ok = raw.copy(), np.zeros(len(raw), bool)
    for j in range(len(raw)):
        jet = out[j]
        r, slot = W._real(jet), W._free_slot(jet)
        if slot < 0:
            continue
        eta_j = W._jet_eta(jet, r)
        th, rr = rng.uniform(0, 2 * np.pi), 0.4 * np.sqrt(rng.uniform())
        de, dp = rr * np.cos(th), rr * np.sin(th)
        row = np.zeros(jet.shape[1], np.float32)
        row[:5] = de, dp, np.log(pt), np.log(pt * np.cosh(eta_j + de)), pid
        jet[slot] = row
        ok[j] = True
    return out, ok


def head_weights(ad, raw: np.ndarray, batch: int = 256, block: int = BLOCK, head: int = HEAD) -> np.ndarray:
    """(n, 4, 150) attention of the summary queries of a backbone block / head (default b7/6) on the particle
    slots. The weights come from re-running the same nn.MultiheadAttention call with need_weights=True (forward
    pre-hook); the model's own output is not changed."""
    attn = ad.model.body.in_blocks[block].attn
    start = ad.token_layout().particle_start
    out, cap = [], {}

    def pre(mod, args, kwargs):
        cap["kw"] = kwargs

    h = attn.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        with torch.no_grad():
            for s0 in range(0, len(raw), batch):
                ad.forward(ad.preprocess({"data": torch.as_tensor(np.asarray(raw[s0:s0 + batch], np.float32))}))
                kw = dict(cap["kw"])
                kw.update(need_weights=True, average_attn_weights=False)
                _, w = attn(**kw)  # (B, heads, T, T)
                out.append(w[:, head, :4, start:start + raw.shape[1]].float().cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(out)


def shares(y: np.ndarray) -> dict:
    """Softmax over the 200 jet logits; mean summed probability of each dataset block, and the five classes outside
    JetClass with the largest mean probability."""
    z = y[:, :200] - y[:, :200].max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    out = {k: float(p[:, a:b].sum(1).mean()) for k, (a, b) in BLOCKS.items()}
    out["top_non_JetClass_classes"] = [int(i) for i in np.argsort(-np.concatenate([p[:, :2], np.zeros((len(p), 10)),
                                                                                    p[:, 12:]], 1).mean(0))[:5]]
    return out


def features(ad, raw: np.ndarray, batch: int = 250) -> tuple[dict, np.ndarray]:
    """The four summary tokens (concatenated) after each backbone block ("body block <b>"), after the final norm
    ("body.norm") and after each classifier block ("classifier block <c>"); and the 210 logits."""
    body = ad.model.body
    mods = {f"body block {b}": blk for b, blk in enumerate(body.in_blocks)}
    mods["body.norm"] = body.norm
    for c, blk in enumerate(getattr(ad.model.classifier, "in_blocks", [])):
        mods[f"classifier block {c}"] = blk
    cap, feats, ys = {}, {k: [] for k in mods}, []
    hs = [m.register_forward_hook(lambda mod, i, o, k=k: cap.__setitem__(k, o[:, :4].reshape(len(o), -1)))
          for k, m in mods.items()]
    try:
        with torch.no_grad():
            for s0 in range(0, len(raw), batch):
                x = W._trunc(np.asarray(raw[s0:s0 + batch], np.float32))
                o = ad.forward(ad.preprocess({"data": torch.as_tensor(x)}))
                ys.append(o["y_pred"].double().cpu().numpy())
                for k in mods:
                    feats[k].append(cap[k].float().cpu().numpy())
    finally:
        for h in hs:
            h.remove()
    return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(ys)
