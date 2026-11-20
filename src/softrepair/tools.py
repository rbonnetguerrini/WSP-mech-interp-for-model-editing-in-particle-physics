"""Shared helpers for the JetClass / JetClass-II analyses: jet loading, activation and attention capture at one
attention head, adding one neutral particle, JetClass-II scores, and the model-internal removal."""

from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from softrepair import core as W
from softrepair import heads as H
from softrepair import paths

JETCLASS2_FILES = ("JetClass2_9.h5", "JetClass2_99.h5", "JetClass2_998.h5")  # QCD, QCD, 3-4 prong resonances
W_LIKE_JETCLASS2 = (15, 17, 19, 20)  # global index = JetClass-II label + 12: X->qq, cs, cq, sq
QCD_JETCLASS2 = tuple(range(173, 200))  # JetClass-II QCD, 27 classes
BLOCK, HEAD = 7, 6


def jetclass_jets(n_per_class: int, r0: int, classes=None) -> tuple[np.ndarray, np.ndarray]:
    """JetClass validation jets, n per class from row r0 (rows >= 60000 are the held-out test rows)."""
    xs, ys = [], []
    for lab in classes or sorted(W.TEST_FILES):
        with h5py.File(W.DATA / f"JetClass_{W.TEST_FILES[lab]}.h5", "r") as h:
            xs.append(h["data"][r0:r0 + n_per_class].astype(np.float32))
        ys.append(np.full(len(xs[-1]), lab))
    return np.concatenate(xs), np.concatenate(ys)


def jetclass2_jets(n_per_file: int, r0: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for f in JETCLASS2_FILES:
        with h5py.File(paths.jetclass2_dir() / f, "r") as h:
            xs.append(h["data"][r0:r0 + n_per_file].astype(np.float32))
            ys.append(h["pid"][r0:r0 + n_per_file].astype(int))
    return np.concatenate(xs), np.concatenate(ys)


def sub_threshold(raw: np.ndarray) -> np.ndarray:
    """(n, 150) bool: neutrals below every JetClass particle of their type (core.RECO_THRESHOLDS)."""
    t = W.RECO_THRESHOLDS
    pt, E = np.exp(raw[..., 2].astype(np.float64)), np.exp(raw[..., 3].astype(np.float64))
    code = np.rint(raw[..., 4]).astype(int)
    real = raw[..., 2] != 0
    return real & (((code == 6) & (pt < t["photon_pt"]) & (E < t["photon_E"])) | ((code == 7) & (E < t["neutral_hadron_E"])))


def attn_block(ad, where: str):
    """'body.7' -> body.in_blocks[7]; 'classifier.1' -> classifier.in_blocks[1]."""
    part, i = where.split(".")
    return (ad.model.body if part == "body" else ad.model.classifier).in_blocks[int(i)]


class Capture:
    """Forward hooks on named modules of a loaded adapter; run(raw) returns {site: (n, T, D) float32}, logits and
    the attention of one head (default block 7 / head 6) on every particle slot (n, 150): mean over the summary
    queries (queries="summary") or over the summary and real-particle queries (queries="all").
    Site "norm1" is the norm1 output of that same block (the head's input)."""

    def __init__(self, ad, sites=("b7_norm1",), where: str = f"body.{BLOCK}", head: int = HEAD,
                 queries: str = "summary"):
        self.ad, body = ad, ad.model.body
        self.blk, self.head, self.queries = attn_block(ad, where), head, queries
        # classifier TokenAttBlock: the 4 class tokens (raw, no norm) query the body.norm output after the tokens
        self.token_block = not hasattr(self.blk, "norm1")
        self.p0 = ad.token_layout().particle_start
        mods = {"b7_norm1": body.in_blocks[BLOCK].norm1, "body_norm": body.norm}
        if self.token_block:
            c = int(where.split(".")[1])
            mods["norm1"] = body.norm  # the keys' input
            mods["queries"] = body.norm if c == 0 else ad.model.classifier.in_blocks[c - 1]
            self.kp0 = self.p0 - self.blk.num_tokens  # particle slot s is key index kp0 + s
            self.queries = "summary"
        else:
            mods["norm1"] = self.blk.norm1
            mods["queries"] = self.blk.norm1
            self.kp0 = self.p0
        for b, blk in enumerate(body.in_blocks):
            mods[f"block{b}"] = blk
        self.mods = {s: mods[s] for s in sites}

    def run(self, raw: np.ndarray, batch: int = 256, attention: bool = True):
        batch = int(os.environ.get("SOFTREPAIR_CAPTURE_BATCH", batch))  # small on a shared GPU
        cap, out = {}, {s: [] for s in self.mods}
        att, ys = [], []
        hs = [m.register_forward_hook(lambda mod, i, o, s=s: cap.__setitem__(s, o.detach()))
              for s, m in self.mods.items()]
        attn = self.blk.attn
        kwcap = {}
        hs.append(attn.register_forward_pre_hook(lambda m, a, kw: kwcap.__setitem__("kw", kw), with_kwargs=True))
        try:
            with torch.no_grad():
                for s0 in range(0, len(raw), batch):
                    x = np.asarray(raw[s0:s0 + batch], np.float32)
                    o = self.ad.forward(self.ad.preprocess({"data": torch.as_tensor(x)}))
                    ys.append(o["y_pred"].double().cpu().numpy())
                    for s in self.mods:
                        out[s].append(cap[s].float().cpu().numpy())
                    if attention:
                        kw = dict(kwcap["kw"])
                        kw.update(need_weights=True, average_attn_weights=False)
                        _, w = attn(**kw)
                        wh = w[:, self.head, :, self.kp0:self.kp0 + x.shape[1]]
                        if self.queries == "summary":
                            att.append(wh[:, :4].mean(1).float().cpu().numpy())
                        else:
                            qm = torch.zeros(wh.shape[:2], device=wh.device)
                            qm[:, :4] = 1.0
                            qm[:, self.p0:self.p0 + x.shape[1]] = torch.as_tensor(x[..., 2] != 0, device=wh.device).float()
                            att.append(((wh * qm[..., None]).sum(1) / qm.sum(1, keepdim=True)).float().cpu().numpy())
        finally:
            for h in hs:
                h.remove()
        res = {s: np.concatenate(v) for s, v in out.items()}
        return res, np.concatenate(ys), (np.concatenate(att) if att else None)


def add_one(raw: np.ndarray, pt: np.ndarray, pid: int, seed: int, log_E: np.ndarray | None = None):
    """One neutral per jet at the given pT (array, GeV), uniform in dR < 0.4; E physical unless log_E is given.
    Returns (jets, ok, slot)."""
    rng = np.random.default_rng(seed)
    out, ok, slot = raw.copy(), np.zeros(len(raw), bool), np.full(len(raw), -1)
    for j in range(len(raw)):
        jet = out[j]
        r, s = W._real(jet), W._free_slot(jet)
        if s < 0:
            continue
        eta_j = W._jet_eta(jet, r)
        th, rr = rng.uniform(0, 2 * np.pi), 0.4 * np.sqrt(rng.uniform())
        de, dp = rr * np.cos(th), rr * np.sin(th)
        row = np.zeros(jet.shape[1], np.float32)
        le = np.log(pt[j] * np.cosh(eta_j + de)) if log_E is None else log_E[j]
        row[:5] = de, dp, np.log(pt[j]), le, pid
        jet[s] = row
        ok[j], slot[j] = True, s
    return out, ok, slot


def scores(y: np.ndarray) -> dict:
    """F_W (JetClass only) and the dataset-marginalised Bayes score F_Bayes."""
    z = y[:, :200] - y[:, :200].max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    pw = p[:, H.WJ] + p[:, list(W_LIKE_JETCLASS2)].sum(1)
    pq = p[:, H.QCD] + p[:, list(QCD_JETCLASS2)].sum(1)
    return {"F_W": y[:, H.WJ] - y[:, H.QCD], "F_Bayes": np.log(pw + 1e-300) - np.log(pq + 1e-300),
            "p_jetclass2": p[:, 12:200].sum(1)}


def internal_fix(ad, raw: np.ndarray, tau: float = 0.5, max_drop: int = 3) -> np.ndarray:
    """Model-internal repair, no hand-set thresholds. The jets are known to be JetClass, so p(JetClass-II) = 0 a
    priori: when the model puts > tau of its 200-way probability on JetClass-II classes, the evidence must come from
    a particle the JetClass detector cannot produce. Drop the particle that block 7 / head 6's summary queries attend
    to most, and re-evaluate; up to max_drop times. Uses the weights currently loaded in ad."""
    cap = Capture(ad, sites=())
    out = raw.copy()
    active = np.arange(len(raw))
    dropped = np.zeros(len(raw), int)
    for _ in range(max_drop):
        if active.size == 0:
            break
        _, y, att = cap.run(out[active])
        z = y[:, :200] - y[:, :200].max(1, keepdims=True)
        p = np.exp(z)
        pj2 = p[:, 12:200].sum(1) / p.sum(1)
        flag = pj2 > tau
        if not flag.any():
            break
        for j, a in zip(active[flag], att[flag]):
            real = out[j, :, 2] != 0
            a = np.where(real[:a.shape[0]], a, -1.0)
            s = int(np.argmax(a))
            out[j, s] = 0.0
            out[j, s, 4] = 3.0
            dropped[j] += 1
        active = active[flag]
    internal_fix.last_dropped = dropped
    print(f"[internal_fix] {len(raw)} jets: dropped 1/2/3 particles in "
          f"{(dropped == 1).mean():.4f}/{(dropped == 2).mean():.4f}/{(dropped >= 3).mean():.4f}", flush=True)
    return out
