"""Helpers shared by the evaluation, table and figure scripts (paper Sec. 4 and App. E).

Jets for the tagger / regression tests come straight from the JetClass test files (rows 60000.. for evaluation,
80000.. for training a new head), as in Sec. 4. Results written by the scripts of this folder and the cached results
of the data record (folder `results/`) share one layout; `results_root()` returns it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from softrepair import core as W

REPO = Path(__file__).resolve().parents[2]
QCD = 2
NAMES = {2: "QCD", 3: "H->bb", 4: "H->cc", 5: "H->gg", 6: "H->4q", 7: "H->lvqq", 8: "Z->qq", 9: "W->qq",
         10: "t->bqq", 11: "t->blv"}
SIGNALS = tuple(range(3, 12))
WINDOWS = {9: (65.0, 95.0), 10: (140.0, 200.0)}  # W and top mass windows of the predicted mass
SIZES = {"s": "small", "m": "medium", "l": "large"}
BLOCKS = {"ATLAS (0-1)": (0, 2), "JetClass (2-11)": (2, 12), "JetClass-II (12-199)": (12, 200)}


# ---------------------------------------------------------------- data

def load(lab: int, r0: int, n: int) -> np.ndarray:
    """Rows r0..r0+n of the JetClass test file of one class (global label lab)."""
    with h5py.File(W.DATA / f"JetClass_{W.TEST_FILES[lab]}.h5", "r") as h:
        raw = h["data"][r0:r0 + n].astype(np.float32)
        assert np.all(h["pid"][r0:r0 + n] == lab)
    return raw


def jet_mass(raw: np.ndarray) -> np.ndarray:
    """Jet mass (GeV) from the massless constituent four-vectors."""
    from softrepair.observables.substructure import constituent_four_vectors
    v = constituent_four_vectors(raw)
    px, py, pz, E = (v[k].sum(1) for k in ("px", "py", "pz", "E"))
    return np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0))


def soft(raw: np.ndarray, frac: float, pid: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One photon (pid 6) or neutral hadron (pid 7) at pT fraction frac of each jet (core._soft)."""
    rng = np.random.default_rng(seed)
    out, ok = raw.copy(), np.zeros(len(raw), bool)
    for j in range(len(out)):
        ok[j] = W._soft(out[j], frac, pid, rng)
    return out, ok


def shares(y: np.ndarray) -> dict:
    """Mean softmax probability of the ATLAS, JetClass and JetClass-II blocks of the 200 jet classes."""
    z = y[:, :200] - y[:, :200].max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    out = {k: float(p[:, a:b].sum(1).mean()) for k, (a, b) in BLOCKS.items()}
    out["top_non_JetClass_classes"] = [int(i) for i in np.argsort(-np.concatenate([p[:, :2], np.zeros((len(p), 10)),
                                                                                    p[:, 12:]], 1).mean(0))[:5]]
    return out


def parse_edits(specs: list[str] | None) -> list[tuple[str, Path | None]]:
    """["name=path.npz", "original", ...] -> [(name, path or None)]; a bare name without "=" is the original."""
    out = []
    for s in specs or ["original_model"]:
        if "=" in s:
            name, path = s.split("=", 1)
            out.append((name, Path(path) if path not in ("", "-") else None))
        else:
            out.append((s, None))
    return out


def setup_device(size: str, max_batch: int | None = None) -> None:
    """Model size and forward batch caps as for the stored results, math attention kernel."""
    os.environ["MODEL_SIZE"] = size
    if max_batch:
        os.environ.setdefault("SOFTREPAIR_MAX_BATCH", str(max_batch))
        os.environ.setdefault("SOFTREPAIR_EVAL_BATCH", str(max_batch))
    W.use_math_sdp()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


# ---------------------------------------------------------------- results layout

def results_root() -> Path:
    """Folder with the cached results: RESULTS_DIR (the data record's `results/` or your own outputs)."""
    from softrepair import paths
    return paths.results_dir()


def runs_root() -> Path:
    """The agent-run records of this repository (runs/<variant>/<size>/run<k>/)."""
    return REPO / "runs"


# Older result files use shorter keys; read them through these aliases.
TAGGER_KEY_ALIASES = {"jc2_share": "jetclass2_share", "nh_f0.001": "neutral_hadron_f0.001"}


def normalise_keys(obj):
    if isinstance(obj, dict):
        return {TAGGER_KEY_ALIASES.get(k, k): normalise_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalise_keys(v) for v in obj]
    return obj


def read_json(path: Path) -> dict:
    return normalise_keys(json.loads(Path(path).read_text()))


def median_range(values) -> tuple[float, float, float]:
    v = np.asarray(list(values), float)
    return float(np.median(v)), float(v.min()), float(v.max())


def need(path: Path) -> Path:
    if not Path(path).exists():
        sys.exit(f"missing result file: {path}")
    return Path(path)


# ---------------------------------------------------------------- readers of the cached results (tables, figures)
# Folder layout under results_root() (the data record's results/ and the outputs of these scripts):
#   scores/<size>/<name>.json                           remaining response, accuracy, rho, changed parameters
#   scores_removal_rule/<size>/<name>.json              remaining error, rho for fragmentation after the removal rule
#   fragmentation_after_removal_rule/<size>/*.json      rho for fragmentation after the removal rule (any edit)
#   w_efficiency/<size>/<name>.json                     W efficiency at the 80% working point
#   taggers_and_regression_head/<size>/<name>.json      nine taggers, JetClass-II probability, regression head
#   regression_head/<size>/<name>.json                  re-run of the above with the published regression heads
#   generator_head/<size>/update_<name>.json            change of the generator update
# Edit names: original_model, finetuning_<variant>, sae_<variant>, agents_<variant>_run<k>_edit_<e>.

def rpath(*parts: str) -> Path:
    return results_root().joinpath(*parts)


def score(size: str, name: str) -> dict:
    return read_json(need(rpath("scores", size, f"{name}.json")))


def remaining_response(d: dict) -> float:
    return d.get("remaining_response", d.get("R_sub"))


def changed_parameters(d: dict) -> int:
    s = d["size"]
    return int(s.get("changed_parameters", s.get("dof")))


def fragmentation_after_rule(size: str, name: str) -> float | None:
    """rho for the fragmentation change after the removal rule: removal-rule scorer, else fragmentation files."""
    f = rpath("scores_removal_rule", size, f"{name}.json")
    if f.exists():
        rho = read_json(f)["rho"]
        return rho.get("P_reco", rho.get("P_removal"))
    for f in sorted(rpath("fragmentation_after_removal_rule", size).glob("*.json")):
        d = read_json(f)
        if d.get("name") == name:
            return d.get("rho_P_reco", d.get("rho_P_removal"))
        c = d.get("candidates", {}).get(name)
        if c is not None:
            return c.get("rho_P_reco", c.get("rho_P_removal")) if isinstance(c, dict) else c
    return None


def phys_resp(size: str, name: str) -> tuple[float, str]:
    """The rho furthest from one among angular spread, momentum scale and fragmentation after the removal rule."""
    import math
    rho = score(size, name)["rho"]
    r = {"angular spread": rho["D"], "momentum scale": rho["J"]}
    p = fragmentation_after_rule(size, name)
    if p is None:
        print(f"  note: no fragmentation-after-removal-rule result for {size}/{name}; Phys. resp. without it",
              file=sys.stderr)
    else:
        r["fragmentation"] = p
    k = max(r, key=lambda k: abs(math.log(r[k])))
    return r[k], k


def w_eff(size: str, name: str) -> float | None:
    f = rpath("w_efficiency", size, f"{name}.json")
    return read_json(f)["wp80"]["f1e-3_W"]["eff"] if f.exists() else None


def taggers(size: str, name: str, prefer_rerun: bool = True) -> dict:
    """Tagger / JetClass-II / regression-head result; the re-run with the published heads if present."""
    f = rpath("regression_head", size, f"{name}.json")
    if not (prefer_rerun and f.exists()):
        f = need(rpath("taggers_and_regression_head", size, f"{name}.json"))
    return read_json(f)


def generator_update(size: str, name: str, which: str = "photon_0.3GeV") -> float:
    d = read_json(need(rpath("generator_head", size, f"update_{name}.json")))
    return float(d["t=0.05"]["W"][which])


def wsp_name(variant: str, run: int, edit: str = "edit_large") -> str:
    return f"agents_{variant}_run{run}_{edit}"


# ---------------------------------------------------------------- number formats of the paper tables

def f2(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def f1(x) -> str:
    return "-" if x is None else f"{x:.1f}"


def acc4(x) -> str:
    """Accuracy change with four decimals; positive values with a sign, zero as 0.0000."""
    if x is None:
        return "-"
    s = f"{x:+.4f}"
    return "0.0000" if s in ("+0.0000", "-0.0000") else (s if s.startswith("+") else s)


def pct(x, digits: int = 0) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def count(n) -> str:
    return "-" if n is None else f"{int(n):,}".replace(",", " ")


def cell(values, fmt) -> str:
    """median [min, max] of three values, each with fmt."""
    m, lo, hi = median_range(values)
    return f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]"


def print_table(head: list[str], rows: list[list[str]]) -> None:
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r))  # noqa: E731
    print(line(head))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))
