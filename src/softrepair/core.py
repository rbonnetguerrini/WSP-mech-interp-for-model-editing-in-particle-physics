"""Shared code for every repair method: jet sets, added-particle changes, model loading, edits, metrics.

One evaluator for every edit (agent edits, fine-tuning, SAE edits, input cuts), so all numbers are paired on
identical, precomputed inputs.

Raw columns: 0 deta, 1 dphi, 2 log_pt, 3 log_E, 4 pid, 5-8 impact parameters.
Neutral pids are {6, 7} (no impact-parameter measurement); 6 is the photon, 7 the neutral hadron.

Variant names of the prepared jet sets (files <DATA_DIR>/<set>/<variant>.npy):
  soft<f>         one photon (pid 6) added at pT fraction f of the jet's scalar pT sum, uniform in dR < 0.4
  soft7_<f>       the same with one neutral hadron (pid 7)
  softpair_<f>    one photon and one neutral hadron, each at f
  rmsoft          the softest real neutral removed
  split2 / split3 the hardest neutral split into 2 (0.5/0.5) or 3 (0.5/0.3/0.2) collinear pieces
  split2c         the hardest charged particle split into 2 collinear halves
  D / J / P       physical changes (softrepair.ops at theta = 1): angular spread, momentum scale, fragmentation

Metrics on the 10 JetClass logits l (global indices 2..11), centred c = l - mean(l):
  spread S    = sqrt(2 * sum_k Var_jets(c_k)) on clean jets  (RMS distance of two random jets)
  r_P         = mean_j ||c(P x_j) - c(x_j)|| / S            (scale-free response to the change P)
  acc, nll    = 10-class accuracy and cross-entropy on clean jets
  dF_P        = mean |F(P x) - F(x)|, F = logsumexp(l_W, l_Z) - l_QCD

Environment (all optional):
  MODEL_SIZE                model size used when none is passed (s, m, l; default s)
  SOFTREPAIR_MATH_SDP=1     use the math attention kernel (all stored scores were made with it)
  SOFTREPAIR_MAX_BATCH      cap on every forward batch (needed for the large model)
  SOFTREPAIR_EVAL_BATCH     cap on classifier batches only; without it the cap is 128 on GPUs with < 40 GB
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from softrepair import paths

# JetClass validation files per global label. Practice and test jets come from disjoint files, except Z (8) and
# W (9), which have one file each: practice and test use disjoint rows of it.
PRACTICE_FILES = {2: 23, 3: 7, 4: 1, 5: 14, 6: 9, 7: 4, 8: 29, 9: 5, 10: 31, 11: 3}
TEST_FILES = {2: 32, 3: 16, 4: 11, 5: 15, 6: 22, 7: 18, 8: 29, 9: 5, 10: 39, 11: 12}
SETS = {
    "practice": (PRACTICE_FILES, 0, 1024),
    "test": (TEST_FILES, 50000, 52048),
    "train": (PRACTICE_FILES, 2048, 22048),  # fine-tuning training jets
}
VISIBLE = ("split2", "soft3e-3", "soft1e-4")
HELDOUT = ("soft7_1e-3", "split3", "split2c")
NEUTRAL = (6, 7)
CLASSES = tuple(range(2, 12))
N_SLOTS = 150


def __getattr__(name: str):
    """Lazy path constants: DATA (JetClass dir), CKPT_DIR, CKPT (small model), OUT (results), SIZE."""
    if name == "DATA":
        return paths.jetclass_dir()
    if name == "CKPT_DIR":
        return paths.checkpoints_dir()
    if name == "CKPT":
        return paths.checkpoint("s")
    if name == "OUT":
        return paths.results_dir()
    if name == "SIZE":
        return model_size()
    raise AttributeError(name)


def model_size() -> str:
    return os.environ.get("MODEL_SIZE", "s")


def _cap(batch: int) -> int:
    cap = int(os.environ.get("SOFTREPAIR_MAX_BATCH", "0")) or None
    return min(batch, cap) if cap else batch


def use_math_sdp() -> None:
    """Disable the fused attention kernels (numerically equivalent math kernel only).

    The fused kernels failed with "CUDA error: invalid configuration argument" on some masked batches; every
    stored score was computed with the math kernel."""
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)


if os.environ.get("SOFTREPAIR_MATH_SDP") == "1":
    use_math_sdp()


# ---------------------------------------------------------------- added particles and splits

def _free_slot(jet: np.ndarray) -> int:
    free = np.flatnonzero(jet[:, 2] == 0)
    return int(free[0]) if free.size else -1


def _real(jet: np.ndarray) -> np.ndarray:
    return np.flatnonzero(jet[:, 2] != 0)


def _jet_eta(jet: np.ndarray, r: np.ndarray) -> float:
    pt, E = np.exp(jet[r, 2].astype(np.float64)), np.exp(jet[r, 3].astype(np.float64))
    a = np.arccosh(np.maximum(E / pt, 1.0))
    cands = [np.median(s * a - jet[r, 0]) for s in (1, -1)]
    return float(cands[int(np.argmin([np.std(np.abs(c + jet[r, 0]) - a) for c in cands]))])


def _split(jet: np.ndarray, fracs: tuple[float, ...], charged: bool) -> bool:
    """Replace the hardest neutral (or charged) particle by collinear pieces with pT fractions."""
    r = _real(jet)
    is_neu = np.isin(np.rint(jet[r, 4]).astype(int), NEUTRAL)
    cand = r[~is_neu] if charged else r[is_neu]
    free = np.flatnonzero(jet[:, 2] == 0)
    if cand.size == 0 or free.size < len(fracs) - 1:
        return False
    i = cand[np.argmax(jet[cand, 2])]
    src = jet[i].copy()
    for k, f in enumerate(fracs):
        slot = i if k == 0 else free[k - 1]
        jet[slot] = src
        jet[slot, 2] = src[2] + np.log(f)
        jet[slot, 3] = src[3] + np.log(f)
    return True


def _soft(jet: np.ndarray, frac: float, pid: int, rng: np.random.Generator) -> bool:
    """Add one massless neutral at pT fraction frac of the jet, uniform within dR < 0.4."""
    r, slot = _real(jet), _free_slot(jet)
    if slot < 0:
        return False
    eta_j = _jet_eta(jet, r)
    th, rr = rng.uniform(0, 2 * np.pi), 0.4 * np.sqrt(rng.uniform())
    de, dp = rr * np.cos(th), rr * np.sin(th)
    p_new = frac * np.exp(jet[r, 2].astype(np.float64)).sum()
    row = np.zeros(jet.shape[1], np.float32)
    row[0], row[1], row[2], row[3], row[4] = de, dp, np.log(p_new), np.log(p_new * np.cosh(eta_j + de)), pid
    jet[slot] = row
    return True


def perturb(raw: np.ndarray, name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (modified copy, valid mask). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    out, ok = raw.copy(), np.zeros(len(raw), bool)
    for j in range(len(raw)):
        if name == "split2":
            ok[j] = _split(out[j], (0.5, 0.5), charged=False)
        elif name == "split3":
            ok[j] = _split(out[j], (0.5, 0.3, 0.2), charged=False)
        elif name == "split2c":
            ok[j] = _split(out[j], (0.5, 0.5), charged=True)
        elif name.startswith("soft"):
            tag, frac = name[4:].split("_") if "_" in name else ("6", name[4:])
            ok[j] = _soft(out[j], float(frac), int(tag), rng)
        else:
            raise KeyError(name)
        if not ok[j]:
            out[j] = raw[j]
    return out, ok


def pt_cut(raw: np.ndarray, cut: float) -> np.ndarray:
    """Input baseline: drop constituents with pT < cut (same units as exp(log_pt))."""
    out = raw.copy()
    real = out[:, :, 2] != 0
    drop = real & (np.exp(out[:, :, 2]) < cut)
    out[drop] = 0.0
    out[drop, 4] = 3.0  # padding pid filler, as in the files
    return out


# ---------------------------------------------------------------- removal rule (JetClass reconstruction thresholds)

RECO_THRESHOLDS = {"photon_pt": 0.80, "photon_E": 1.80, "neutral_hadron_E": 2.20}  # just below JetClass's softest


def reco_fix(raw: np.ndarray, merge_dr: float = 0.0, thresholds: bool = True) -> np.ndarray:
    """The removal rule: make an input consistent with JetClass's reconstruction (no weights changed).

    thresholds: drop neutrals that JetClass's detector simulation never reconstructs (photons with pT < 0.80 GeV
      and E < 1.80 GeV; neutral hadrons with E < 2.20 GeV; every real JetClass particle lies above these).
    merge_dr > 0: recombine same-pid particles closer than merge_dr in (deta, dphi) into the harder one (pT and E
      summed, pT-weighted direction, the harder one's pid and impact parameters); an exact collinear split is
      restored exactly. Dropped or merged rows become padding (zeros, pid 3), as pt_cut does.
    """
    out = raw.copy()
    real = out[:, :, 2] != 0
    if thresholds:
        pt, E = np.exp(out[:, :, 2].astype(np.float64)), np.exp(out[:, :, 3].astype(np.float64))
        code = np.rint(out[:, :, 4]).astype(int)
        t = RECO_THRESHOLDS
        drop = real & (((code == 6) & (pt < t["photon_pt"]) & (E < t["photon_E"])) |
                       ((code == 7) & (E < t["neutral_hadron_E"])))
        out[drop] = 0.0
        out[drop, 4] = 3.0
        real = real & ~drop
    if merge_dr > 0:
        for j in range(len(out)):
            r = np.flatnonzero(real[j])
            if r.size < 2:
                continue
            x = out[j, r]
            d2 = (x[:, None, 0] - x[None, :, 0]) ** 2 + (x[:, None, 1] - x[None, :, 1]) ** 2
            same = np.rint(x[:, None, 4]) == np.rint(x[None, :, 4])
            close = np.triu((d2 < merge_dr ** 2) & same, 1)
            if not close.any():
                continue
            order = np.argsort(-x[:, 2])  # hardest first: merge softer partners into it
            gone = np.zeros(len(r), bool)
            for a in order:
                if gone[a]:
                    continue
                partners = [b for b in np.flatnonzero(close[a] | close[:, a]) if not gone[b] and b != a]
                if not partners:
                    continue
                grp = [a] + partners
                pts, Es = np.exp(x[grp, 2].astype(np.float64)), np.exp(x[grp, 3].astype(np.float64))
                row = out[j, r[a]]
                row[0] = float((pts * x[grp, 0]).sum() / pts.sum())
                row[1] = float((pts * x[grp, 1]).sum() / pts.sum())
                row[2], row[3] = np.log(pts.sum()), np.log(Es.sum())
                for b in partners:
                    gone[b] = True
                    out[j, r[b]] = 0.0
                    out[j, r[b], 4] = 3.0
    return out


# ---------------------------------------------------------------- jet sets on disk

def set_dir(name: str) -> Path:
    return paths.data_dir() / name


def load_set(name: str, variants: tuple[str, ...] | None = None, mmap: bool = True) -> dict:
    """{"labels", "clean", "meta", <variant>, <variant>:ok} of one prepared set (memory-mapped by default)."""
    d = set_dir(name)
    meta = json.loads((d / "meta.json").read_text())
    mode = "r" if mmap else None
    out = {"labels": np.load(d / "labels.npy"), "clean": np.load(d / "clean.npy", mmap_mode=mode), "meta": meta}
    for v in variants if variants is not None else meta["variants"]:
        out[v] = np.load(d / f"{v}.npy", mmap_mode=mode)
        out[v + ":ok"] = np.load(d / f"{v}.ok.npy")
    return out


# ---------------------------------------------------------------- model

def load_adapter(device: str = "cuda", size: str | None = None):
    """The pretrained OmniLearned model of one size (default: MODEL_SIZE, else s), frozen, in eval mode."""
    from softrepair.adapters.omnilearned import OmniLearnedAdapter
    size = size or model_size()
    return OmniLearnedAdapter(paths.checkpoint(size), device=device, compute_checksum=False)


def original_state(ad) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in ad.model.state_dict().items()}


def load_candidate(path: str | Path, orig: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """An edit is an .npz or .pt mapping state_dict keys to FULL replacement tensors.

    Keys not present stay at their original value. Raises on unknown keys or shape mismatch.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            over = {k: torch.as_tensor(np.asarray(z[k])) for k in z.files}
    else:
        over = torch.load(path, map_location="cpu", weights_only=True)
    state = dict(orig)
    for k, v in over.items():
        if k not in orig:
            raise KeyError(f"unknown state_dict key {k!r}")
        if tuple(v.shape) != tuple(orig[k].shape):
            raise ValueError(f"{k}: shape {tuple(v.shape)} != {tuple(orig[k].shape)}")
        if not torch.isfinite(v.float()).all():
            raise ValueError(f"{k}: non-finite values")
        state[k] = v.to(orig[k].dtype)
    return state


def set_state(ad, state: dict[str, torch.Tensor]) -> None:
    ad.model.load_state_dict(state, strict=True)
    ad.model.eval()


def _trunc(raw: np.ndarray) -> np.ndarray:
    real = raw[:, :, 2] != 0
    last = int(np.max(np.where(real.any(0))[0])) + 1 if real.any() else 1
    return raw[:, :last]


def _eval_batch(batch: int) -> int:
    """Cap inference batches on small GPUs (e.g. 24 GB L4: batch 512 needs > 22 GB). Results do not depend on
    the batch size. SOFTREPAIR_EVAL_BATCH overrides; otherwise the cap is 128 on GPUs with < 40 GB."""
    cap = os.environ.get("SOFTREPAIR_EVAL_BATCH")
    if cap:
        return min(batch, int(cap))
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 40e9:
        return min(batch, 128)
    return batch


def logits(ad, raw: np.ndarray, batch: int = 512, body: bool = False):
    """All 210 logits (float64 numpy); optionally also the 4 summary tokens of the body output."""
    ys, xb = [], []
    batch = _eval_batch(_cap(batch))
    with torch.no_grad():
        for s0 in range(0, len(raw), batch):
            chunk = _trunc(np.asarray(raw[s0:s0 + batch], np.float32))
            o = ad.forward(ad.preprocess({"data": torch.as_tensor(chunk)}))
            ys.append(o["y_pred"].double().cpu().numpy())
            if body:
                xb.append(o["x_body"][:, :4].float().cpu().numpy())  # 4 summary tokens
    y = np.concatenate(ys)
    return (y, np.concatenate(xb)) if body else y


def gen_drift_inputs(raw: np.ndarray, seed: int = 7, t: float = 0.5):
    """Fixed noised inputs for the generator-drift proxy (same for every edit)."""
    g = np.random.default_rng(seed)
    return g.standard_normal(raw.shape[:2] + (4,)).astype(np.float32), t


def gen_velocity(ad, raw: np.ndarray, labels: np.ndarray, eps: np.ndarray, t: float, batch: int = 256):
    """Generator-head prediction on z = alpha x + sigma eps at fixed time t (no RNG inside)."""
    batch = _cap(batch)
    from omnilearned.diffusion import get_logsnr_alpha_sigma
    outs = []
    with torch.no_grad():
        for s0 in range(0, len(raw), batch):
            b = ad.preprocess({"data": torch.as_tensor(np.asarray(raw[s0:s0 + batch], np.float32))})
            X = b["X"].to(ad.device)
            mask = (X[:, :, 2:3] != 0).float()
            tt = torch.full((X.shape[0],), t, device=ad.device)
            _, alpha, sigma = get_logsnr_alpha_sigma(tt)
            e = torch.as_tensor(eps[s0:s0 + batch], device=ad.device) * mask
            z = alpha * X + sigma * e
            xb = ad.model.body(z, None, b["pid"].to(ad.device), b["add_info"].to(ad.device), tt)
            y = torch.as_tensor(labels[s0:s0 + batch], device=ad.device)
            outs.append((ad.model.generator(xb, y.int()) * mask).float().cpu().numpy())
    return np.concatenate(outs)


# ---------------------------------------------------------------- metrics

def centred10(y: np.ndarray) -> np.ndarray:
    l = y[:, 2:12]
    return l - l.mean(1, keepdims=True)


def Fscore(y: np.ndarray) -> np.ndarray:
    return np.logaddexp(y[:, 8], y[:, 9]) - y[:, 2]


def clean_metrics(y: np.ndarray, labels: np.ndarray) -> dict:
    l = y[:, 2:12]
    lab = labels.astype(int) - 2
    lse = np.logaddexp.reduce(l, axis=1)
    c = centred10(y)
    return {
        "acc": float((l.argmax(1) == lab).mean()),
        "nll": float((lse - l[np.arange(len(l)), lab]).mean()),
        "spread": float(np.sqrt(2 * c.var(0).sum())),
        "acc_per_class": {str(k): float((l[lab == k - 2].argmax(1) == k - 2).mean()) for k in CLASSES},
    }


def irc_metrics(y0: np.ndarray, yp: np.ndarray, ok: np.ndarray, spread: float, labels: np.ndarray) -> dict:
    d = np.linalg.norm(centred10(yp) - centred10(y0), axis=1) / spread
    dF = np.abs(Fscore(yp) - Fscore(y0))
    lab = labels.astype(int)
    return {
        "r": float(d[ok].mean()),
        "r_q90": float(np.quantile(d[ok], 0.9)),
        "dF": float(dF[ok].mean()),
        "n": int(ok.sum()),
        "r_per_class": {str(k): float(d[ok & (lab == k)].mean()) for k in CLASSES if (ok & (lab == k)).any()},
    }


def evaluate_set(ad, data: dict, variants: tuple[str, ...], raw_transform=None) -> dict:
    """Full metric block on one set. raw_transform (e.g. a pT cut) is applied to every input."""
    tf = raw_transform or (lambda a: a)
    y0 = logits(ad, tf(np.asarray(data["clean"])))
    res = {"clean": clean_metrics(y0, data["labels"])}
    for v in variants:
        yp = logits(ad, tf(np.asarray(data[v])))
        res[v] = irc_metrics(y0, yp, data[v + ":ok"], res["clean"]["spread"], data["labels"])
    return res


def ratios(res: dict, ref: dict, variants: tuple[str, ...]) -> dict:
    rr = {v: res[v]["r"] / ref[v]["r"] for v in variants}
    return {"per_variant": rr, "mean": float(np.mean(list(rr.values())))}


# ---------------------------------------------------------------- generator quality

GEN_SUMMARY = ("tau21", "tau32", "d2", "zg", "rg", "subjet_z", "subjet_dr", "girth", "z_max", "em_frac", "nh_frac",
               "jet_pt", "jet_mass", "n_constituents", "n_charged")
GEN_CLASSES = (9, 2)  # W, QCD
GEN_N = 2048


def _concepts(raw, glob_):
    from softrepair.observables.restricted import continuous_concepts
    return {k: np.asarray(v, dtype=np.float64) for k, v in continuous_concepts(raw, glob_).items()}


def jet_summaries(raw: np.ndarray, workers: int = 8) -> np.ndarray:
    """(n, 15) jet summaries (GEN_SUMMARY) used by the generator tests."""
    from multiprocessing import Pool
    from softrepair.observables.substructure import constituent_four_vectors
    v = constituent_four_vectors(raw)
    px, py, pz, E = (v[k].sum(1) for k in ("px", "py", "pz", "E"))
    pt = np.hypot(px, py)
    m = np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 1e-6))
    g = np.column_stack([np.log(pt), np.log(m), (raw[..., 2] != 0).sum(1) / 100.0])
    idx = np.array_split(np.arange(len(raw)), max(1, len(raw) // 500))
    with Pool(workers) as pool:
        parts = pool.starmap(_concepts, [(raw[i], g[i]) for i in idx])
    c = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return np.column_stack([c[k] for k in GEN_SUMMARY])


def generate_like(ad, raw: np.ndarray, label: int, seed: int = 0, nsteps: int = 128, batch: int = 128,
                  edit: dict | None = None) -> np.ndarray:
    """Generate jets with the real jets' multiplicity, pid and impact-parameter conditioning.

    edit (optional): {"site": adapter site, "vector": (D,), "scale": float, "pids": [..] | None}
    adds scale * vector at the chosen particle slots of that site at EVERY ODE network evaluation
    (slots: real particles of the conditioning jet, restricted to the listed pids if given).
    """
    batch = _cap(batch)
    from omnilearned.diffusion import generate
    b = ad.preprocess({"data": torch.as_tensor(np.asarray(raw, np.float32))})
    n = (raw[..., 2] != 0).sum(1)
    out = []
    handle, cur = None, {}
    if edit is not None:
        module = ad._site_modules()[edit["site"]]
        vec = torch.as_tensor(np.asarray(edit["vector"], np.float32), device=ad.device) * float(edit["scale"])
        start = 0 if edit["site"] == "embed" else ad.token_layout().particle_start
        real = raw[..., 2] != 0
        sel = real & np.isin(np.rint(raw[..., 4]).astype(int), edit["pids"]) if edit.get("pids") else real

        def hook(mod, inp, output):  # noqa: ANN001
            t = output[0] if isinstance(output, tuple) else output
            m = cur["mask"]
            t = t.clone()
            t[:, start:start + m.shape[1]] += m[..., None] * vec
            return (t, *output[1:]) if isinstance(output, tuple) else t

        handle = module.register_forward_hook(hook)
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            for s0 in range(0, len(raw), batch):
                if edit is not None:
                    cur["mask"] = torch.as_tensor(sel[s0:s0 + batch], dtype=torch.float32, device=ad.device)
                X = b["X"][s0:s0 + batch]
                y = torch.full((X.shape[0],), label, dtype=torch.long, device=ad.device)
                g = generate(ad.model, y, X.shape, cond=None, pid=b["pid"][s0:s0 + batch].to(ad.device),
                             add_info=b["add_info"][s0:s0 + batch].to(ad.device), nsteps=nsteps,
                             multiplicity=torch.as_tensor(n[s0:s0 + batch] / 100.0, dtype=torch.float32),
                             dataset="jetclass", device=ad.device)
                out.append(g.float().cpu().numpy())
    finally:
        if handle is not None:
            handle.remove()
    gen = np.asarray(raw, np.float32).copy()
    real = gen[..., 2] != 0
    gen[..., :4] = np.where(real[..., None], np.concatenate(out), 0.0)
    return gen


def generator_quality(ad, test: dict, seed: int = 0) -> dict:
    """Real-vs-generated separability (held-out AUC of a boosted tree on 15 jet summaries) per class."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    res = {}
    for lab in GEN_CLASSES:
        raw = np.asarray(test["clean"][np.flatnonzero(test["labels"] == lab)[:GEN_N]])
        S_real = jet_summaries(raw)
        gen = generate_like(ad, raw, lab, seed)
        realp = gen[..., 2] != 0
        finite = np.all(np.isfinite(gen.reshape(len(gen), -1)), 1)
        lp = np.where(realp, gen[..., 2], 0.0)
        valid = finite & np.all(~realp | ((lp >= -3) & (lp <= 8)), 1)
        S_gen = jet_summaries(np.nan_to_num(gen))
        valid &= np.isfinite(S_gen[:, 0]) & np.isfinite(S_gen[:, 2])
        half = np.arange(len(raw)) < len(raw) // 2
        Xtr = np.concatenate([S_real[half], S_gen[half & valid]])
        ytr = np.r_[np.zeros(half.sum()), np.ones((half & valid).sum())]
        Xte = np.concatenate([S_real[~half], S_gen[~half & valid]])
        yte = np.r_[np.zeros((~half).sum()), np.ones((~half & valid).sum())]
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xtr, ytr)
        sd = np.nanstd(S_real, 0) + 1e-12
        res[str(lab)] = {
            "auc": float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])),
            "valid_frac": float(valid.mean()),
            "mean_shift_sd": dict(zip(GEN_SUMMARY, ((np.nanmean(S_gen[valid], 0) - np.nanmean(S_real, 0)) / sd).tolist())),
        }
    res["auc_mean"] = float(np.mean([res[str(l)]["auc"] for l in GEN_CLASSES]))
    return res
