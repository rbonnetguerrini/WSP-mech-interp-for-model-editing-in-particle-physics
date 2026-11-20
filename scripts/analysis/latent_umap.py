#!/usr/bin/env python
"""Summary tokens of OmniLearned in 2-d: clean W / QCD jets, the same jets with one soft photon, and clean JetClass-II
jets, after backbone block 6, after the backbone (final norm) and after the last classifier block (Sec. 2 Fig. 4;
with --edit also Sec. 4).

Representation: the 4 summary tokens concatenated (4 x 128), as in probes.py. Three views per layer:
- PCA fitted on the CLEAN jets only (JetClass W, JetClass QCD, JetClass-II), then applied to every set;
- t-SNE on all sets jointly (perplexity 30);
- UMAP (n_neighbors 30, min_dist 0.3, random_state 0) on 50 PCA components, fitted on the CLEAN jets only and
  applied to every set, like the PCA.
Quantified: for each set, among the k = 15 nearest CLEAN reference jets (standardised features, same layer), the
share that are JetClass-II, and among the JetClass ones the share that are W.

Sets (W file 5 and QCD file 32, rows 60000+): W, QCD; W and QCD + one photon at f = 1e-3 of the jet pT
(core._soft); W + one photon at 3 GeV (inside JetClass's photon range: control); clean JetClass-II jets (the first n
of one validation file).

  python scripts/analysis/latent_umap.py --jetclass2 JETCLASS2_DIR/JetClass2_9.h5 [--n 1500] [--threads 8]
        [--size s] [--edit EDIT.npz --name NAME] [--layers "body block 6" body.norm "classifier block 1"]
      -> RESULTS_DIR/figures/<size>/representation/<NAME>.json and <NAME>_embedding.npz
         (NAME default: original_model or the edit file stem; npz keys "<layer>|<set>|<pca|tsne|umap>")
--layers: summary-token sites (names as in _common.features). The default suits the small model (reader b7/6);
for the medium / large model pick the block before their main reader, body.norm and the last classifier block.
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

LAYERS = ("body block 6", "body.norm", "classifier block 1")
METHODS = ("pca", "tsne", "umap")
SETS = ("JetClass-II", "W", "QCD", "W + photon", "QCD + photon")  # the embedded sets (order of the npz keys)


def soft(raw: np.ndarray, frac: float, pid: int, seed: int) -> np.ndarray:
    from softrepair import core as W
    rng = np.random.default_rng(seed)
    out, ok = raw.copy(), np.zeros(len(raw), bool)
    for j in range(len(out)):
        ok[j] = W._soft(out[j], frac, pid, rng)
    return out[ok]


def main() -> int:
    import h5py
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jetclass2", type=Path, required=True, help="a JetClass-II h5 file")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--edit", type=Path, default=None, help="edited weights (.npz/.pt, core.load_candidate)")
    ap.add_argument("--name", default=None, help="output name (default: original_model or the edit file stem)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--layers", nargs="+", default=list(LAYERS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
                    help="umap needs umap-learn; it is skipped with a warning if missing")
    a = ap.parse_args()
    C.set_size(a.size)
    from softrepair import core as W
    from softrepair import heads as H
    methods = list(a.methods)
    if "umap" in methods:
        try:
            import umap  # noqa: F401
        except ImportError:
            print("WARNING: umap-learn not installed, skipping UMAP", flush=True)
            methods.remove("umap")
    torch.set_num_threads(a.threads)
    ad = W.load_adapter("cpu", a.size)
    if a.edit:
        W.set_state(ad, W.load_candidate(a.edit, W.original_state(ad)))
    w, q = H.load(H.WJ, 60000, a.n), H.load(H.QCD, 60000, a.n)
    with h5py.File(a.jetclass2, "r") as h:
        jetclass2 = h["data"][:a.n].astype(np.float32)
    sets = {"W": w, "QCD": q, "JetClass-II": jetclass2,
            "W + photon": soft(w, 1e-3, 6, 101), "QCD + photon": soft(q, 1e-3, 6, 102)}
    x3, ok3 = C.add_fixed(w, 3.0, 6, 103)
    sets["W + photon 3 GeV (control)"] = x3[ok3]
    F = {}
    for k, v in sets.items():
        F[k], _ = C.features(ad, v)
        print(k, len(v), flush=True)

    res = {"n": {k: int(len(v)) for k, v in sets.items()}, "k": a.k, "size": a.size,
           "edit": a.edit.name if a.edit else None, "layers": {}}
    ref = ("W", "QCD", "JetClass-II")
    emb = {}
    for layer in a.layers:
        Xref = np.concatenate([F[k][layer] for k in ref])
        lab = np.concatenate([np.full(len(F[k][layer]), i) for i, k in enumerate(ref)])
        sc = StandardScaler().fit(Xref)
        nn = NearestNeighbors(n_neighbors=a.k + 1).fit(sc.transform(Xref))
        rec = {}
        for k in sets:
            _, idx = nn.kneighbors(sc.transform(F[k][layer]))
            idx = idx[:, 1:] if k in ref else idx[:, :-1]  # a clean jet is its own nearest neighbour
            nl = lab[idx]
            jc = nl < 2
            rec[k] = {"share_JetClass-II_neighbours": float((nl == 2).mean()),
                      "share_W_among_JetClass_neighbours": float((nl[jc] == 0).mean()) if jc.any() else None}
        pca = PCA(2).fit(sc.transform(Xref))
        rec["pca_explained"] = [float(v) for v in pca.explained_variance_ratio_]
        res["layers"][layer] = rec
        keys = list(SETS)
        Xall = np.concatenate([sc.transform(F[k][layer]) for k in keys])
        cut = np.cumsum([0] + [len(F[k][layer]) for k in keys])
        E = {}
        if "pca" in methods:
            E["pca"] = pca.transform(Xall)
        if "tsne" in methods:  # all sets jointly (t-SNE cannot place new points)
            E["tsne"] = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(PCA(50).fit_transform(Xall))
        if "umap" in methods:  # fitted on the clean reference jets only, then applied to every set (like PCA)
            import umap
            red = PCA(50).fit(sc.transform(Xref))
            um = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit(red.transform(sc.transform(Xref)))
            E["umap"] = um.transform(red.transform(Xall))
        emb[layer] = {k: {m: E[m][cut[i]:cut[i + 1]] for m in E} for i, k in enumerate(keys)}
        print(layer, json.dumps(rec), flush=True)

    out = a.out_dir or C.out_dir(a.size, "representation")
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or (a.edit.stem if a.edit else "original_model")
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / f"{name}_embedding.npz", **{f"{l}|{k}|{m}": emb[l][k][m] for l in emb for k in emb[l]
                                                         for m in emb[l][k]})
    print("wrote", out / f"{name}_embedding.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
