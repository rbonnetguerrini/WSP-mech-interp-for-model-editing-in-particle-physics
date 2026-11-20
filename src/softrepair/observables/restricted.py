"""The restricted concept set: a short list of frozen, significance-quantified concepts.

This module freezes one definition per concept *before* any feature matching,
so that no definition can be tuned against a latent afterwards. The version
string below is written into every table built from it.

Continuous measurements are the primary targets. Each "reference-tail" concept
is additionally turned into a binary label by a cut calibrated on QCD jets
(JetClass label 2) alone, in bins of the nuisances that most change the QCD
distribution of that variable. A jet is in the tail when it is unusual *for a
QCD jet of the same pT (and mass, or charged multiplicity)*. The cut is fitted
on a reference sample that is disjoint from every jet the model is tested on,
and never sees an activation or a latent. The 5% reference acceptance is the
working point; 1% and 10% are the sensitivity checks. These are analysis
conventions, not physical boundaries and not purity statements.

What could NOT be implemented as originally specified, and why:

* **Signed impact parameter.** The stored ``tanh_d0`` has no lifetime sign: in
  H->bb jets the fraction of |S| > 3 tracks with positive d0 is 0.500, the same
  as in QCD (0.504). A lifetime sign would skew heavy-flavour jets strongly
  positive. The displacement concept therefore uses |S|, and a jet-probability
  statistic (which calibrates resolution from the negative-sign tail) is not
  computable.
* **Lepton identification.** Muon/electron codes are the Delphes reconstructed
  particle type. There are no identification working points and no conversion
  flags, so "identified lepton" means "carries that particle code".
* **Jet eta** is not stored for JetClass (only constituent offsets from the
  axis); **constituent truncation** never occurs (at most 128 constituents in
  150 slots); **provenance** is a JetClass-versus-Aspen question handled
  separately. None of these is a nuisance column here.

Everything is numpy-only, like the modules it calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from softrepair.observables.shape import (
    PHOTON_PID_VALUE,
    angularity_features,
    composition_features,
    energy_correlation_features,
)
from softrepair.observables.substructure import exclusive_subjets, n_subjettiness, soft_drop
from softrepair.observables.basic import (
    LOG_PT_COLUMN,
    PID_COLUMN,
    constituent_mask,
    jet_observables,
    unsaturated_displacement_features,
)

RESTRICTED_VERSION = "restricted-2026-10-24.1"

#: Particle codes on this data (see softrepair.observables.shape): 1=mu+, 5=mu-, 2=e+, 4=e-.
MUON_PID_VALUES: tuple[int, ...] = (1, 5)
ELECTRON_PID_VALUES: tuple[int, ...] = (2, 4)

#: Soft-Drop grooming is the fixed (z_cut=0.1, beta=0, R=0.8) default of
#: softrepair.observables.substructure. A passing splitting counts as RESOLVED only
#: above this opening angle: 0.1 is the HCAL cell size of the Delphes CMS card,
#: the coarsest granularity a neutral constituent can have. 0.05 is the check.
RG_RESOLUTION_LIMIT = 0.1
RG_RESOLUTION_LIMIT_CHECK = 0.05

REFERENCE_LABEL = 2  # JetClass QCD
ACCEPTANCES: tuple[float, ...] = (0.01, 0.05, 0.10)
PRIMARY_ACCEPTANCE = 0.05

#: tau32 is reported only where its denominator is not vanishingly small:
#: tau2 at or above this quantile of QCD tau2 in the same (pT, mass) cell.
TAU2_FLOOR_QUANTILE = 0.01

#: Number of QCD-quantile bins per calibration variable.
N_BINS: dict[str, int] = {"jet_pt": 5, "jet_mass": 6, "n_charged": 6}

NUISANCES: tuple[str, ...] = ("jet_pt", "jet_mass", "n_constituents", "n_charged")


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _real_pt(data: np.ndarray) -> np.ndarray:
    mask = constituent_mask(data)
    return np.where(mask, np.exp(np.where(mask, data[..., LOG_PT_COLUMN], 0.0)), 0.0)


def _leading_fraction(selector: np.ndarray, pt: np.ndarray, total: np.ndarray) -> np.ndarray:
    leading = np.where(selector, pt, 0.0).max(axis=1)
    return np.divide(leading, total, out=np.zeros_like(leading), where=total > 0)


def continuous_concepts(data: np.ndarray, global_: np.ndarray) -> dict[str, np.ndarray]:
    """(N,) continuous targets, validity flags and nuisances for one batch of jets.

    NaN marks a value that is undefined for that jet (fewer than two subjets,
    no resolved Soft-Drop splitting, fewer than two usable tracks, ...), so an
    undefined jet can never be mistaken for a measured zero.
    """
    jo = jet_observables(data, global_)
    n_const = jo["n_constituents"]
    nsub = n_subjettiness(data)
    ecf = energy_correlation_features(data)
    ang = angularity_features(data)
    comp = composition_features(data)
    unsat = unsaturated_displacement_features(data)
    sd = soft_drop(data, with_rg=True)
    subjets = exclusive_subjets(data, 2)

    out: dict[str, np.ndarray] = {name: np.asarray(jo[name], dtype=np.float64) for name in NUISANCES}

    # -- shape / prong structure ---------------------------------------------
    tau21_ok = (n_const >= 2) & (nsub["tau1"] > 0)
    tau32_ok = (n_const >= 3) & (nsub["tau2"] > 0)
    out["tau21"] = np.where(tau21_ok, nsub["tau21"], np.nan)
    out["tau32"] = np.where(tau32_ok, nsub["tau32"], np.nan)
    out["tau2"] = np.asarray(nsub["tau2"], dtype=np.float64)
    out["d2"] = np.where(np.isfinite(ecf["d2"]) & (ecf["ecf2"] > 0), ecf["d2"], np.nan)

    pt1, pt2 = subjets[:, 0, 0], subjets[:, 1, 0]
    two = pt2 > 0
    total = np.where(two, pt1 + pt2, 1.0)
    out["subjet_z"] = np.where(two, np.minimum(pt1, pt2) / total, np.nan)
    dr12 = np.hypot(subjets[:, 0, 1] - subjets[:, 1, 1], _wrap(subjets[:, 0, 2] - subjets[:, 1, 2]))
    out["subjet_dr"] = np.where(two, dr12, np.nan)

    sd_passed = sd["zg"] > 0
    resolved = sd_passed & (sd["rg"] > RG_RESOLUTION_LIMIT)
    out["resolved_split"] = resolved.astype(np.float64)
    out["resolved_split_check"] = (sd_passed & (sd["rg"] > RG_RESOLUTION_LIMIT_CHECK)).astype(np.float64)
    out["zg"] = np.where(resolved, sd["zg"], np.nan)
    out["rg"] = np.where(resolved, sd["rg"], np.nan)

    out["girth"] = np.asarray(ang["girth"], dtype=np.float64)
    out["z_max"] = np.asarray(ang["z1"], dtype=np.float64)

    # -- flavour / displacement / leptons -------------------------------------
    s2_ok = unsat["n_unsaturated_charged"] >= 2
    out["n_usable_tracks"] = np.asarray(unsat["n_unsaturated_charged"], dtype=np.float64)
    out["s2_unsat"] = np.where(s2_ok, unsat["second_max_abs_d0_significance_unsat"], np.nan)

    mask = constituent_mask(data)
    pid = np.rint(data[..., PID_COLUMN]).astype(np.int64)
    pt = _real_pt(data)
    total_pt = pt.sum(axis=1)
    is_muon = mask & np.isin(pid, MUON_PID_VALUES)
    is_electron = mask & np.isin(pid, ELECTRON_PID_VALUES)
    out["has_muon"] = is_muon.any(axis=1).astype(np.float64)
    out["has_electron"] = is_electron.any(axis=1).astype(np.float64)
    out["muon_pt_frac"] = _leading_fraction(is_muon, pt, total_pt)
    out["electron_pt_frac"] = _leading_fraction(is_electron, pt, total_pt)

    # -- composition -----------------------------------------------------------
    is_em = mask & ((pid == PHOTON_PID_VALUE) | np.isin(pid, ELECTRON_PID_VALUES))
    em_pt = np.where(is_em, pt, 0.0).sum(axis=1)
    out["em_frac"] = np.divide(em_pt, total_pt, out=np.zeros_like(em_pt), where=total_pt > 0)
    out["nh_frac"] = np.asarray(comp["neutral_hadron_frac"], dtype=np.float64)

    return {key: np.asarray(value, dtype=np.float64) for key, value in out.items()}


# ---------------------------------------------------------------------------
# Reference-tail labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailSpec:
    """One reference-tail concept.

    ``undefined_label`` is the label given where ``variable`` is NaN: NaN
    (the jet is excluded from that concept) or 0.0 (the jet provably cannot be
    in the tail, e.g. fewer than two usable tracks cannot be "several displaced
    tracks").
    """

    name: str
    variable: str
    side: str
    bins: tuple[str, ...]
    undefined_label: float = math.nan


TAIL_SPECS: tuple[TailSpec, ...] = (
    TailSpec("two_prong", "tau21", "low", ("jet_pt", "jet_mass")),
    TailSpec("three_prong", "tau32", "low", ("jet_pt", "jet_mass")),
    TailSpec("broad", "girth", "high", ("jet_pt",)),
    TailSpec("leading_dominance", "z_max", "high", ("jet_pt",)),
    TailSpec("multi_displaced", "s2_unsat", "high", ("jet_pt", "n_charged"), 0.0),
    TailSpec("em_rich", "em_frac", "high", ("jet_pt",)),
    TailSpec("nh_rich", "nh_frac", "high", ("jet_pt",)),
)

#: Binary concepts that need no calibration: they are measured content.
DIRECT_LABELS: tuple[str, ...] = ("resolved_split", "has_muon", "has_electron")

#: The continuous primary targets, in the order of the frozen list.
CONTINUOUS_TARGETS: tuple[str, ...] = (
    "tau21",
    "d2",
    "subjet_z",
    "subjet_dr",
    "tau32",
    "zg",
    "rg",
    "girth",
    "z_max",
    "s2_unsat",
    "muon_pt_frac",
    "electron_pt_frac",
    "em_frac",
    "nh_frac",
)


def tail_label_name(spec: TailSpec | str, acceptance: float) -> str:
    name = spec.name if isinstance(spec, TailSpec) else spec
    return f"{name}_tail{int(round(acceptance * 100)):02d}"


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        raise ValueError("calibration variable is constant on the reference sample")
    return edges


def _cell_index(table: Mapping[str, np.ndarray], bins: Sequence[str], edges: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Flat cell index; values beyond the reference range fall into the edge bins."""
    index = np.zeros(np.asarray(table[bins[0]]).shape[0], dtype=np.int64)
    for var in bins:
        e = np.asarray(edges[var], dtype=np.float64)
        b = np.searchsorted(e[1:-1], np.asarray(table[var], dtype=np.float64), side="right")
        index = index * (e.size - 1) + b
    return index


def _n_cells(bins: Sequence[str], edges: Mapping[str, Sequence[float]]) -> int:
    return int(np.prod([len(edges[var]) - 1 for var in bins]))


def fit_reference(
    reference: Mapping[str, np.ndarray],
    *,
    acceptances: Sequence[float] = ACCEPTANCES,
    min_cell: int = 200,
) -> dict[str, Any]:
    """Fit every tail cut on QCD reference jets only. Returns a JSON-able dict.

    A cell with fewer than ``min_cell`` defined reference jets is an error
    rather than a noisy cut: the binning must be coarsened instead.
    """
    out: dict[str, Any] = {
        "version": RESTRICTED_VERSION,
        "reference_label": REFERENCE_LABEL,
        "n_reference": int(np.asarray(reference["jet_pt"]).shape[0]),
        "acceptances": [float(a) for a in acceptances],
        "specs": {},
    }
    for spec in TAIL_SPECS:
        edges = {var: _quantile_edges(np.asarray(reference[var], dtype=np.float64), N_BINS[var]).tolist() for var in spec.bins}
        cells = _cell_index(reference, spec.bins, edges)
        values = np.asarray(reference[spec.variable], dtype=np.float64)
        n_cells = _n_cells(spec.bins, edges)
        floor = None
        if spec.variable == "tau32":
            tau2 = np.asarray(reference["tau2"], dtype=np.float64)
            floor = np.array([np.quantile(tau2[cells == c], TAU2_FLOOR_QUANTILE) for c in range(n_cells)])
            values = np.where(tau2 >= floor[cells], values, np.nan)
        thresholds: dict[str, list[float]] = {}
        counts = []
        for c in range(n_cells):
            v = values[(cells == c) & np.isfinite(values)]
            if v.size < min_cell:
                raise ValueError(f"{spec.name}: reference cell {c} has {v.size} < {min_cell} defined jets")
            counts.append(int(v.size))
        for acc in acceptances:
            q = acc if spec.side == "low" else 1.0 - acc
            thresholds[f"{acc:.2f}"] = [
                float(np.quantile(values[(cells == c) & np.isfinite(values)], q)) for c in range(n_cells)
            ]
        out["specs"][spec.name] = {
            "variable": spec.variable,
            "side": spec.side,
            "bins": list(spec.bins),
            "edges": edges,
            "thresholds": thresholds,
            "reference_cell_counts": counts,
            "undefined_label": None if math.isnan(spec.undefined_label) else spec.undefined_label,
            **({"tau2_floor": floor.tolist()} if floor is not None else {}),
        }
    return out


def apply_reference(table: Mapping[str, np.ndarray], reference: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """(N,) float tail labels (1 in tail, 0 not, NaN excluded) for every spec and acceptance."""
    if reference.get("version") != RESTRICTED_VERSION:
        raise ValueError(f"reference was fitted with {reference.get('version')!r}, code is {RESTRICTED_VERSION!r}")
    out: dict[str, np.ndarray] = {}
    for spec in TAIL_SPECS:
        info = reference["specs"][spec.name]
        cells = _cell_index(table, spec.bins, info["edges"])
        values = np.asarray(table[spec.variable], dtype=np.float64)
        undefined = ~np.isfinite(values)
        excluded = np.zeros_like(undefined)
        if "tau2_floor" in info:
            floor = np.asarray(info["tau2_floor"], dtype=np.float64)
            excluded = np.asarray(table["tau2"], dtype=np.float64) < floor[cells]
        fill = np.nan if info["undefined_label"] is None else float(info["undefined_label"])
        for acc_key, thresholds in info["thresholds"].items():
            thr = np.asarray(thresholds, dtype=np.float64)[cells]
            with np.errstate(invalid="ignore"):
                selected = values < thr if spec.side == "low" else values > thr
            label = selected.astype(np.float64)
            label[undefined] = fill
            label[excluded] = np.nan
            out[tail_label_name(spec, float(acc_key))] = label
    return out
