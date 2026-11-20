"""Jet substructure observables computed from raw JetClass constituents only.

Like ``softrepair.observables.basic``, this module is numpy-only (no torch, no model or
activation dependence) so the physics ground truth it computes can never be
confused with something the network learned, and stays importable anywhere the
raw HDF5 data is importable. It implements N-subjettiness (tau1/tau2/
tau3 and their ratios), kT splitting scales, soft-drop grooming, and the two
colour-flow constructions (the between-prong radiation ratio and the pull
angle). All of it is built on one shared kT sequential-recombination
clustering of a jet's own constituents (``kt_cluster``), whose merge history
also backs ``exclusive_subjets``.

CRITICAL FOUR-VECTOR FINDING (read this before trusting any mass-like number
here). The raw columns are ``delta_eta, delta_phi, log_pt, log_energy, ...``
(see ``softrepair.observables.basic`` for the full layout), and ``delta_eta``/
``delta_phi`` are documented as relative to the jet axis. The naive recipe

    px = pt*cos(delta_phi), py = pt*sin(delta_phi),
    pz = pt*sinh(delta_eta), E = exp(log_energy)

reproduces the stored ``global`` jet pT exactly (relative error ~1e-7, pure
float32 rounding: transverse momentum only needs pt and phi, both of which
are self-consistent) but reproduces the stored jet mass with a median ratio
of ~4.1 -- i.e. the "previous quick attempt" failure mode this module was
specifically asked to avoid, reproduced and diagnosed here. The reason:
``log_energy`` is the particle's actual lab-frame energy, but ``delta_eta``
is a pseudorapidity *difference* from an unknown jet-frame reference. Pairing
a lab-frame energy with a jet-frame ``pz = pt*sinh(delta_eta)`` mixes two
different frames and is not a valid four-vector in either one -- there is no
algebraic reason it should reproduce an invariant mass, and empirically it
does not.

The fix, verified against 20,000 real jets from
the JetClass validation file ``JetClass_0.h5``:
treat every constituent as **massless** and build its four-vector purely from
``(pt, delta_eta, delta_phi)``, discarding ``log_energy`` entirely for any
four-vector or substructure calculation:

    px = pt*cos(delta_phi), py = pt*sin(delta_phi),
    pz = pt*sinh(delta_eta), E  = pt*cosh(delta_eta)

This is exact for a genuinely massless particle in *any* frame related to the
lab frame by a boost along the beam axis, because pseudorapidity is a pure
function of the polar angle and shifts additively under such boosts; summing
these massless four-vectors over a jet's constituents therefore reproduces
the Lorentz-invariant jet mass regardless of what the (unknown) absolute jet
pseudorapidity actually is, up to the real (small, ~100 MeV-GeV-scale) rest
masses of the constituents. Measured on the same 20,000-jet sample: pT
relative error is ~1e-7 (unchanged, as expected), and jet mass relative error
has median 0.56%, 90th percentile 19%, 99th percentile 45% -- "a few percent"
at the median, as required, with a tail driven by jets where genuine hadron
rest masses are a non-negligible fraction of a low jet mass. ``jet_kinematics``
below implements exactly this and is the function to check against
``global[:, 1]`` (see the development tests).
Every clustering, grooming and colour-flow function in this module builds its
own internal four-vectors the same way, for internal consistency (E-scheme
recombination of two massless legs is generally massive, exactly as in a real
kT algorithm, so pseudojet masses are not artificially forced back to zero).

kT clustering convention. ``kt_cluster`` implements the standard generalised-kT
sequential recombination measure

    d_ij = min(pt_i^2, pt_j^2) * dR_ij^2 / R^2,   d_iB = pt_i^2,   R = 0.8

over a jet's own real constituents. Because these constituents were already
assigned to one jet (by whatever upstream algorithm produced this file, at a
radius <= R), d_ij < d_iB for any pair with dR_ij < R: min(pt_i,pt_j)^2 *
(dR_ij/R)^2 < min(pt_i,pt_j)^2 <= pt_i^2. So the "send to beam" step only ever
fires for a constituent that is angularly separated by more than R from every
other still-active object -- a rare, genuine outlier -- and when it does fire
here it is treated as *freezing* that object as its own permanent one-particle
subjet (it stops participating in further mergers but is not discarded: it
still counts as one of the jet's constituent axes). This differs from
inclusive hadron-collider kT jet-finding, where a beam step removes a particle
from the event entirely; that behaviour would be wrong here because every
constituent in this data has already been assigned to this jet, and grooming
it away silently would misrepresent the jet's own momentum. The distinction
only matters for pathological jets; see ``JetCluster`` and the "fragmented
jet" fallback in ``exclusive_subjets`` for exactly what happens then.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from softrepair.observables.basic import (
    DELTA_ETA_COLUMN,
    DELTA_PHI_COLUMN,
    LOG_PT_COLUMN,
    constituent_mask,
)

#: Standard anti-kT-style jet radius used both as the kT distance-measure
#: normalisation and, implicitly, as the scale nothing in this module fits.
DEFAULT_R = 0.8

#: Below this, a would-be denominator (tau1, tau2, a subjet's own pT, the
#: pull-vector magnitude, ...) is treated as exactly zero rather than "small
#: but positive", so ratios take their documented convention instead of
#: amplifying float noise.
EPS = 1e-9

#: soft_drop() defaults (Larkoski et al., "Soft Drop", arXiv:1402.2657).
DEFAULT_ZCUT = 0.1
DEFAULT_BETA = 0.0


def _wrap(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle (or angle difference) into [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Four-vectors and the ground-truth check
# ---------------------------------------------------------------------------


def constituent_four_vectors(data: np.ndarray) -> dict[str, np.ndarray]:
    """(N, P) massless px/py/pz/E/pt per constituent; see module docstring.

    Every component is exactly 0 at padding (pt is 0 there by construction of
    constituent_mask, and every other component is a product with pt).
    """
    mask = constituent_mask(data)
    pt = np.where(mask, np.exp(data[..., LOG_PT_COLUMN]), 0.0)
    deta = data[..., DELTA_ETA_COLUMN]
    dphi = data[..., DELTA_PHI_COLUMN]
    px = pt * np.cos(dphi)
    py = pt * np.sin(dphi)
    pz = pt * np.sinh(deta)
    E = pt * np.cosh(deta)
    return {"px": px, "py": py, "pz": pz, "E": E, "pt": pt}


def jet_kinematics(data: np.ndarray) -> dict[str, np.ndarray]:
    """(N,) jet_pt and jet_mass recomputed from constituents (see module docstring).

    This is the ground-truth check against ``global[:, 0]``/``global[:, 1]``:
    jet_pt should match to float32 precision, jet_mass to a few percent at the
    median. Anything computed elsewhere in this module that needs "the jet's
    own pT" (tau_N normalisation, splitting-scale normalisation, ...) uses
    this jet_pt, not the stored ``global`` value, so the module is fully
    self-contained and importable without ``global`` at all.
    """
    fv = constituent_four_vectors(data)
    jet_px = fv["px"].sum(axis=1)
    jet_py = fv["py"].sum(axis=1)
    jet_pz = fv["pz"].sum(axis=1)
    jet_E = fv["E"].sum(axis=1)
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    m2 = jet_E**2 - jet_px**2 - jet_py**2 - jet_pz**2
    jet_mass = np.sqrt(np.maximum(m2, 0.0))
    return {"jet_pt": jet_pt, "jet_mass": jet_mass}


# ---------------------------------------------------------------------------
# kT sequential recombination clustering
# ---------------------------------------------------------------------------


@dataclass
class JetCluster:
    """The full kT merge history of one jet's real constituents.

    Object ids ``0 .. n_leaves - 1`` are the real constituents themselves, in
    the order they were passed to ``_cluster_jet`` (i.e. after masking out
    padding, but otherwise not reordered -- clustering is order-independent by
    construction, see the permutation-invariance test). Ids ``>= n_leaves``
    are pseudojets created by a merge step, in creation order. Every object's
    own (pt, eta, phi, px, py, pz, E) is fixed at the moment it is created and
    never mutated afterwards -- only its *participation* (active / frozen /
    absorbed into a later merge) changes.

    ``steps`` is the literal merge history: ``("merge", i, j, k, dij,
    total_after)`` records that objects i and j combined into new object k at
    scale dij, leaving ``total_after`` objects representing the jet; ``("beam",
    i, diB, total)`` records that object i was frozen (removed from further
    merging, see the module docstring for why this does not discard it).

    ``snapshot_by_count[n]`` is the list of object ids that jointly represent
    the jet's momentum when there are exactly ``n`` objects total -- exactly
    what ``exclusive_subjets(n)`` needs. Beam events never change this list
    (they change which internal set an id sits in, not the set of ids), so it
    is only recorded when a merge changes the total count. ``final_count`` is
    the smallest count reached (1 for every jet whose constituents all lie
    within angular distance R of some path connecting them; see the module
    docstring for when it can exceed 1).
    """

    n_leaves: int
    pt: list[float] = field(default_factory=list)
    eta: list[float] = field(default_factory=list)
    phi: list[float] = field(default_factory=list)
    px: list[float] = field(default_factory=list)
    py: list[float] = field(default_factory=list)
    pz: list[float] = field(default_factory=list)
    E: list[float] = field(default_factory=list)
    children: list[tuple[int, int] | None] = field(default_factory=list)
    steps: list[tuple] = field(default_factory=list)
    snapshot_by_count: dict[int, list[int]] = field(default_factory=dict)
    final_count: int = 0


def _cluster_jet(pt0: np.ndarray, eta0: np.ndarray, phi0: np.ndarray, R: float) -> JetCluster:
    """kT-cluster one jet's real constituents (see module docstring for the measure)."""
    n0 = int(len(pt0))
    cluster = JetCluster(n_leaves=n0)
    cluster.pt = [float(x) for x in pt0]
    cluster.eta = [float(x) for x in eta0]
    cluster.phi = [float(x) for x in phi0]
    cluster.px = [cluster.pt[i] * math.cos(cluster.phi[i]) for i in range(n0)]
    cluster.py = [cluster.pt[i] * math.sin(cluster.phi[i]) for i in range(n0)]
    cluster.pz = [cluster.pt[i] * math.sinh(cluster.eta[i]) for i in range(n0)]
    cluster.E = [cluster.pt[i] * math.cosh(cluster.eta[i]) for i in range(n0)]
    cluster.children = [None] * n0

    active = list(range(n0))
    frozen: list[int] = []
    total = n0
    cluster.snapshot_by_count[total] = list(range(n0))

    while len(active) >= 2:
        n_active = len(active)
        arr_pt = np.array([cluster.pt[i] for i in active])
        arr_eta = np.array([cluster.eta[i] for i in active])
        arr_phi = np.array([cluster.phi[i] for i in active])

        deta = arr_eta[:, None] - arr_eta[None, :]
        dphi = _wrap(arr_phi[:, None] - arr_phi[None, :])
        dR2 = deta**2 + dphi**2
        pt2 = arr_pt**2
        dij = np.minimum(pt2[:, None], pt2[None, :]) * dR2 / (R * R)
        np.fill_diagonal(dij, np.inf)

        flat = int(np.argmin(dij))
        ii, jj = divmod(flat, n_active)
        min_dij = float(dij[ii, jj])

        kk = int(np.argmin(pt2))
        min_diB = float(pt2[kk])

        if min_dij <= min_diB:
            i_id, j_id = active[ii], active[jj]
            new_px = cluster.px[i_id] + cluster.px[j_id]
            new_py = cluster.py[i_id] + cluster.py[j_id]
            new_pz = cluster.pz[i_id] + cluster.pz[j_id]
            new_E = cluster.E[i_id] + cluster.E[j_id]
            new_pt = math.hypot(new_px, new_py)
            new_phi = math.atan2(new_py, new_px)
            new_eta = math.asinh(new_pz / new_pt) if new_pt > 0 else 0.0

            k_id = len(cluster.pt)
            cluster.pt.append(new_pt)
            cluster.eta.append(new_eta)
            cluster.phi.append(new_phi)
            cluster.px.append(new_px)
            cluster.py.append(new_py)
            cluster.pz.append(new_pz)
            cluster.E.append(new_E)
            cluster.children.append((i_id, j_id))

            active = [x for x in active if x != i_id and x != j_id]
            active.append(k_id)
            total -= 1
            cluster.steps.append(("merge", i_id, j_id, k_id, min_dij, total))
            cluster.snapshot_by_count[total] = sorted(active + frozen)
        else:
            i_id = active[kk]
            active = [x for x in active if x != i_id]
            frozen.append(i_id)
            cluster.steps.append(("beam", i_id, min_diB, total))

    cluster.final_count = total
    return cluster


def kt_cluster(data: np.ndarray, R: float = DEFAULT_R) -> list[JetCluster]:
    """One JetCluster per jet (see module docstring). A per-jet Python loop;
    each jet's clustering is itself numpy-vectorised over its constituents.
    Jets with 0 or 1 real constituents short-circuit trivially inside
    _cluster_jet (the while-loop body never runs) rather than crashing.
    """
    mask = constituent_mask(data)
    pt_all = np.where(mask, np.exp(data[..., LOG_PT_COLUMN]), 0.0)
    deta_all = data[..., DELTA_ETA_COLUMN]
    dphi_all = data[..., DELTA_PHI_COLUMN]

    clusters: list[JetCluster] = []
    for j in range(data.shape[0]):
        m = mask[j]
        clusters.append(_cluster_jet(pt_all[j][m], deta_all[j][m], dphi_all[j][m], R))
    return clusters


# ---------------------------------------------------------------------------
# Exclusive subjets
# ---------------------------------------------------------------------------


def exclusive_subjets(
    data: np.ndarray,
    n: int,
    R: float = DEFAULT_R,
    clusters: list[JetCluster] | None = None,
) -> np.ndarray:
    """(N, n, 3) = [pt, eta, phi] of the n exclusive kT axes per jet.

    Axes are sorted by descending pt within each jet (slot 0 = hardest), for a
    deterministic array layout; this ordering carries no special clustering
    meaning beyond "reading order for a fixed-size array".

    Two fallbacks, both padding with a pt=0 (== invalid) axis rather than
    raising, so this always returns a fixed (N, n, 3) shape:

    * n > n_leaves (more axes requested than there are real constituents):
      the n_leaves real axes are used and the rest are zero-padded.
    * n < final_count (the jet's own clustering never reached as few as n
      objects -- possible only in the rare fragmented-jet case described in
      the module docstring, where some constituent sits beyond R from
      everything else): the final, most-clustered state available is used and
      truncated to the n hardest objects. This under-counts the true "n
      exclusive subjets" for such a jet; it is a documented approximation, not
      a crash.
    """
    if clusters is None:
        clusters = kt_cluster(data, R)
    out = np.zeros((len(clusters), n, 3), dtype=np.float64)
    for idx, c in enumerate(clusters):
        if c.n_leaves == 0:
            continue
        if n in c.snapshot_by_count:
            ids = c.snapshot_by_count[n]
        elif n < c.final_count:
            ids = c.snapshot_by_count[c.final_count]
        else:  # n > n_leaves
            ids = c.snapshot_by_count[c.n_leaves]
        take = sorted(ids, key=lambda i: -c.pt[i])[:n]
        for slot, obj_id in enumerate(take):
            out[idx, slot, 0] = c.pt[obj_id]
            out[idx, slot, 1] = c.eta[obj_id]
            out[idx, slot, 2] = c.phi[obj_id]
    return out


# ---------------------------------------------------------------------------
# N-subjettiness
# ---------------------------------------------------------------------------


def n_subjettiness(data: np.ndarray, max_n: int = 3, R: float = DEFAULT_R) -> dict[str, np.ndarray]:
    """tau1..tau_max_n and the ratios tau21, tau32 (concepts A1-A3).

    tau_N = (1 / (sum_i pt_i * R)) * sum_i pt_i * min_over_axes dR(i, axis),
    with axes = exclusive_subjets(data, N). A padded (pt=0) axis, from the
    "fewer real constituents than N" fallback in exclusive_subjets, is
    excluded from the "nearest axis" search for every constituent rather than
    silently contributing a dR of 0 from an axis that does not really exist.

    Ratio convention (tested, never NaN): tau21 = tau2 / tau1 when tau1 > EPS,
    else 1.0; tau32 = tau3 / tau2 when tau2 > EPS, else 1.0. tau1 is exactly 0
    only when every constituent sits exactly on its nearest axis, which for a
    single real particle (or an empty jet) is trivially true; 1.0 ("this jet
    is no more two-pronged than one-pronged") is the conservative choice so a
    denominator-free jet is never mistaken for an extremely two-pronged one.
    """
    mask = constituent_mask(data)
    pt_all = np.where(mask, np.exp(data[..., LOG_PT_COLUMN]), 0.0)
    deta_all = data[..., DELTA_ETA_COLUMN]
    dphi_all = data[..., DELTA_PHI_COLUMN]
    total_pt = pt_all.sum(axis=1)
    safe_denom = np.where(total_pt > 0, total_pt * R, 1.0)

    clusters = kt_cluster(data, R)
    result: dict[str, np.ndarray] = {}
    for N in range(1, max_n + 1):
        axes = exclusive_subjets(data, N, R=R, clusters=clusters)
        axis_pt = axes[..., 0]
        axis_eta = axes[..., 1]
        axis_phi = axes[..., 2]
        valid = axis_pt > 0  # (n_jets, N)

        d_eta = deta_all[:, :, None] - axis_eta[:, None, :]
        d_phi = _wrap(dphi_all[:, :, None] - axis_phi[:, None, :])
        dR = np.sqrt(d_eta**2 + d_phi**2)
        dR = np.where(valid[:, None, :], dR, np.inf)
        min_dR = dR.min(axis=2)
        min_dR = np.where(mask & np.isfinite(min_dR), min_dR, 0.0)

        num = (pt_all * min_dR).sum(axis=1)
        tau = np.where(total_pt > 0, num / safe_denom, 0.0)
        result[f"tau{N}"] = tau

    if "tau1" in result and "tau2" in result:
        tau1, tau2 = result["tau1"], result["tau2"]
        result["tau21"] = np.where(tau1 > EPS, tau2 / np.where(tau1 > EPS, tau1, 1.0), 1.0)
    if "tau2" in result and "tau3" in result:
        tau2, tau3 = result["tau2"], result["tau3"]
        result["tau32"] = np.where(tau2 > EPS, tau3 / np.where(tau2 > EPS, tau2, 1.0), 1.0)
    return result


# ---------------------------------------------------------------------------
# Splitting scales
# ---------------------------------------------------------------------------


def splitting_scales(data: np.ndarray, R: float = DEFAULT_R) -> dict[str, np.ndarray]:
    """sqrt(d12), sqrt(d23) and their pT-normalised versions (concept A9).

    d12 is the dij of the merge step whose total_after == 1 (the very last
    merge of the jet's own kT tree); d23 is the dij of the step whose
    total_after == 2. A jet with fewer than 2 (resp. 3) real constituents, or
    one that never fully merges down to 1 object (the fragmented-jet case),
    has no such step and gets exactly 0 -- documented, not a crash: there is
    no splitting scale to report, not a vanishingly small one.
    """
    mask = constituent_mask(data)
    pt_all = np.where(mask, np.exp(data[..., LOG_PT_COLUMN]), 0.0)
    jet_pt = pt_all.sum(axis=1)

    clusters = kt_cluster(data, R)
    n_jets = len(clusters)
    d12 = np.zeros(n_jets)
    d23 = np.zeros(n_jets)
    for idx, c in enumerate(clusters):
        for step in c.steps:
            if step[0] != "merge":
                continue
            _, _, _, _, dij, total_after = step
            if total_after == 1:
                d12[idx] = dij
            elif total_after == 2:
                d23[idx] = dij

    sqrt_d12 = np.sqrt(d12)
    sqrt_d23 = np.sqrt(d23)
    safe_pt = np.where(jet_pt > 0, jet_pt, 1.0)
    return {
        "sqrt_d12": sqrt_d12,
        "sqrt_d23": sqrt_d23,
        "sqrt_d12_over_pt": np.where(jet_pt > 0, sqrt_d12 / safe_pt, 0.0),
        "sqrt_d23_over_pt": np.where(jet_pt > 0, sqrt_d23 / safe_pt, 0.0),
    }


# ---------------------------------------------------------------------------
# Soft drop
# ---------------------------------------------------------------------------


def soft_drop(
    data: np.ndarray,
    zcut: float = DEFAULT_ZCUT,
    beta: float = DEFAULT_BETA,
    R: float = DEFAULT_R,
    with_rg: bool = False,
) -> dict[str, np.ndarray]:
    """Groomed jet mass and groomed momentum fraction zg (concept B2).

    Declusters the jet's own kT tree (kt_cluster; NOT exclusive_subjets, which
    is a different observable) from the root down: at each branching into
    children (i, j), compute z = min(pt_i, pt_j) / (pt_i + pt_j) and dR12 =
    deltaR(i, j); if z < zcut * (dR12 / R)**beta the softer child is dropped
    and declustering continues into the harder one, otherwise both children
    are kept and grooming stops there. groomed_mass is the invariant mass of
    the surviving node's own four-vector (which already sums both kept
    children, or is the bare massless leaf if grooming stripped all the way
    down). zg is the *groomed momentum fraction* in the standard soft-drop
    sense -- the z of the splitting where grooming stopped, i.e. of the first
    (hardest, since we go top-down) splitting to pass the condition -- not
    "surviving pT / original pT". If grooming strips down to a single leaf
    without ever finding a passing splitting, there is no such splitting and
    zg is 0 by convention (documented, tested).

    The root is the object with total_after == 1 in the jet's own history
    (the ordinary case). For a fragmented jet (see module docstring) that
    never reaches 1 object, the hardest of the final surviving objects is used
    as the root and the rest of the jet's momentum is simply not considered
    for grooming -- a documented approximation for a rare edge case, not a
    crash.

    ``with_rg=True`` adds ``rg``, the opening angle dR12 of that same passing
    splitting (0 where no splitting passed, like zg). It is opt-in so the
    default key set, which the vocabulary registry checks exactly, is unchanged.
    """
    clusters = kt_cluster(data, R)
    n_jets = len(clusters)
    groomed_mass = np.zeros(n_jets)
    zg = np.zeros(n_jets)
    rg = np.zeros(n_jets)
    groomed_pt = np.zeros(n_jets)
    ungroomed_pt = np.zeros(n_jets)

    for idx, c in enumerate(clusters):
        if c.n_leaves == 0:
            continue
        if 1 in c.snapshot_by_count and len(c.snapshot_by_count[1]) == 1:
            root = c.snapshot_by_count[1][0]
        else:
            ids = c.snapshot_by_count[c.final_count]
            root = max(ids, key=lambda i: c.pt[i])
        ungroomed_pt[idx] = c.pt[root]

        current = root
        while c.children[current] is not None:
            i, j = c.children[current]
            pti, ptj = c.pt[i], c.pt[j]
            if pti + ptj <= 0:
                break
            z = min(pti, ptj) / (pti + ptj)
            deta = c.eta[i] - c.eta[j]
            dphi = _wrap(c.phi[i] - c.phi[j])
            dR12 = math.hypot(deta, dphi)
            if beta == 0.0:
                threshold = zcut
            elif dR12 <= 0.0:
                threshold = 0.0 if beta > 0 else float("inf")
            else:
                threshold = zcut * (dR12 / R) ** beta
            if z < threshold:
                current = i if pti > ptj else j
                continue
            zg[idx] = z
            rg[idx] = dR12
            break
        groomed_pt[idx] = c.pt[current]
        m2 = c.E[current] ** 2 - c.px[current] ** 2 - c.py[current] ** 2 - c.pz[current] ** 2
        groomed_mass[idx] = math.sqrt(max(m2, 0.0))

    out = {
        "groomed_mass": groomed_mass,
        "zg": zg,
        "groomed_pt": groomed_pt,
        "ungroomed_pt": ungroomed_pt,
    }
    if with_rg:
        out["rg"] = rg
    return out


# ---------------------------------------------------------------------------
# Colour ring / between-prong radiation (concept D2)
# ---------------------------------------------------------------------------


def colour_ring(data: np.ndarray, R: float = DEFAULT_R) -> np.ndarray:
    """(N,) momentum-weighted between-prong radiation ratio (concept D2).

    Geometric definition (one specific, documented choice among several in
    the literature -- see the module docstring's discussion for why this one):
    let A, B be the two hardest exclusive subjets (exclusive_subjets(data, 2))
    and dR_AB their angular separation. A constituent k is classified
    "between" the prongs iff

        dR(k, A) < dR_AB  AND  dR(k, B) < dR_AB,

    i.e. k lies in the lens ("vesica") formed by the intersection of the two
    circles of radius dR_AB centred on each subjet axis. This is the standard
    parameter-free way to say "closer to the segment joining A and B than A
    and B are to each other": it is widest at the midpoint and pinches to zero
    width exactly at each axis (a particle sitting exactly on axis A has
    dR(k,B) == dR_AB, which fails the strict inequality, so the immediate
    cores of the two subjets are themselves classified "outside" -- correctly,
    since D2 asks about radiation *between* the prongs, not the prongs
    themselves). Every constituent not classified "between" is "outside".

    ratio = (pT-weighted sum in "between") / (pT-weighted sum "outside").

    Both dR(k,A), dR(k,B) and dR_AB depend only on angular differences, so the
    ratio is invariant to a global rotation in phi; the between-condition is
    manifestly symmetric under swapping A and B (both conjuncts just swap
    labels), so the ratio is swap-invariant, as required.

    Convention for edge cases (0.0, not NaN or inf): fewer than 2 real
    subjets (one of A, B has pt == 0), or a degenerate dR_AB == 0 (the two
    subjets coincide), or no "outside" radiation at all (denominator == 0) --
    in every case there is no meaningful between/outside contrast to report.
    """
    mask = constituent_mask(data)
    pt_all = np.where(mask, np.exp(data[..., LOG_PT_COLUMN]), 0.0)
    deta_all = data[..., DELTA_ETA_COLUMN]
    dphi_all = data[..., DELTA_PHI_COLUMN]

    axes = exclusive_subjets(data, 2, R=R)
    n_jets = data.shape[0]
    ratio = np.zeros(n_jets)

    for idx in range(n_jets):
        if axes[idx, 0, 0] <= 0 or axes[idx, 1, 0] <= 0:
            continue
        A_eta, A_phi = axes[idx, 0, 1], axes[idx, 0, 2]
        B_eta, B_phi = axes[idx, 1, 1], axes[idx, 1, 2]
        dR_AB = math.hypot(A_eta - B_eta, _wrap(A_phi - B_phi))
        if dR_AB <= 0:
            continue

        m = mask[idx]
        deta_k = deta_all[idx][m]
        dphi_k = dphi_all[idx][m]
        pt_k = pt_all[idx][m]

        dR_kA = np.hypot(deta_k - A_eta, _wrap(dphi_k - A_phi))
        dR_kB = np.hypot(deta_k - B_eta, _wrap(dphi_k - B_phi))
        between = (dR_kA < dR_AB) & (dR_kB < dR_AB)

        pt_between = float(pt_k[between].sum())
        pt_outside = float(pt_k[~between].sum())
        if pt_outside > 0:
            ratio[idx] = pt_between / pt_outside
        # else leave at the documented default 0.0

    return ratio


# ---------------------------------------------------------------------------
# Pull angle (concept D3)
# ---------------------------------------------------------------------------


def pull_angle(data: np.ndarray, R: float = DEFAULT_R) -> dict[str, np.ndarray]:
    """Pull magnitude and pull angle of the harder of the two exclusive subjets (D3).

    Pull vector (Gallicchio & Schwartz's "jet pull", specialised to a subjet
    rather than a whole jet): for the harder of the two exclusive subjets
    (larger pT), assign each real constituent to whichever of the two subjet
    axes it is angularly closer to (the same nearest-axis rule n_subjettiness
    uses), then take the pT-weighted mean (delta_eta, delta_phi) displacement
    of the harder subjet's own assigned constituents from its own axis:

        pull_vec = (1 / sum_k pt_k) * sum_k pt_k * (eta_k - eta_axis, phi_k - phi_axis)

    pull_angle is the unsigned angle in [0, pi] between pull_vec and the
    vector from the harder axis to the other (softer) subjet's axis.

    Convention (0.0, never NaN): fewer than 2 real subjets, the harder
    subjet's assigned constituents happen to sum to zero pT (should not occur
    for a jet with real constituents but guarded anyway), a zero pull-vector
    magnitude, or a degenerate zero separation between the two subjet axes --
    in each case there is no well-defined direction to report an angle for.
    """
    mask = constituent_mask(data)
    pt_all = np.where(mask, np.exp(data[..., LOG_PT_COLUMN]), 0.0)
    deta_all = data[..., DELTA_ETA_COLUMN]
    dphi_all = data[..., DELTA_PHI_COLUMN]

    axes = exclusive_subjets(data, 2, R=R)
    n_jets = data.shape[0]
    magnitude = np.zeros(n_jets)
    angle = np.zeros(n_jets)

    for idx in range(n_jets):
        if axes[idx, 0, 0] <= 0 or axes[idx, 1, 0] <= 0:
            continue
        hard, other = (0, 1) if axes[idx, 0, 0] >= axes[idx, 1, 0] else (1, 0)
        A_eta, A_phi = axes[idx, hard, 1], axes[idx, hard, 2]
        B_eta, B_phi = axes[idx, other, 1], axes[idx, other, 2]

        m = mask[idx]
        deta_k = deta_all[idx][m]
        dphi_k = dphi_all[idx][m]
        pt_k = pt_all[idx][m]

        dR_hard = np.hypot(deta_k - A_eta, _wrap(dphi_k - A_phi))
        dR_other = np.hypot(deta_k - B_eta, _wrap(dphi_k - B_phi))
        assigned = dR_hard <= dR_other

        pt_h = pt_k[assigned]
        pt_h_sum = float(pt_h.sum())
        if pt_h_sum <= 0:
            continue

        deta_h = deta_k[assigned] - A_eta
        dphi_h = _wrap(dphi_k[assigned] - A_phi)
        pull_eta = float((pt_h * deta_h).sum() / pt_h_sum)
        pull_phi = float((pt_h * dphi_h).sum() / pt_h_sum)
        mag = math.hypot(pull_eta, pull_phi)
        magnitude[idx] = mag
        if mag <= EPS:
            continue

        to_other_eta = B_eta - A_eta
        to_other_phi = _wrap(B_phi - A_phi)
        norm_to_other = math.hypot(to_other_eta, to_other_phi)
        if norm_to_other <= EPS:
            continue

        cos_angle = (pull_eta * to_other_eta + pull_phi * to_other_phi) / (mag * norm_to_other)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle[idx] = math.acos(cos_angle)

    return {"pull_magnitude": magnitude, "pull_angle": angle}
