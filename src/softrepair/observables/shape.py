"""Jet-shape, kinematic, flavour-adjacent, charge and composition concepts.

This module implements the prong structure / shape, mass and kinematics,
soft-lepton, jet charge and composition families of concepts, computed straight from the raw
JetClass ``data``/``global`` arrays. Like ``softrepair.observables.basic`` this module
is numpy-only (no torch, no omnilearned) so that the concept vocabulary can be
computed anywhere the raw HDF5 files can be read -- these are the ground-truth
labels that probes and causal claims are checked against, so they must not
depend on the model existing at all.

It reuses ``softrepair.observables.basic.constituent_mask`` / ``is_charged`` /
``impact_parameters`` for the padding and charge conventions rather than
reimplementing them, so this module's notion of "real constituent" and
"charged" can never drift from the rest of the project.

Column layout of the ``data`` array (see ``softrepair.observables.basic`` for the
canonical statement)::

    0 delta_eta   1 delta_phi   2 log_pt   3 log_energy   4 pid_int
    5 tanh_d0     6 sigma_d0    7 tanh_dz  8 sigma_dz

PARTICLE-ID CODES on this data (task-supplied, independently sanity-checked
below): 0=pi-, 1=mu+, 2=e+, 3=PADDING, 4=e-, 5=mu-, 6=photon, 7=neutral kaon,
8=pi+. Charged = {0,1,2,4,5,8} (matches ``observables.CHARGED_PID_VALUES``),
neutral = {6,7}. Electric charge: 0->-1, 1->+1, 2->+1, 4->-1, 5->-1, 8->+1,
everything else (photon, neutral kaon, padding) -> 0.

Re-verification of the charge assignment (this module's own check, on
``JetClass_0.h5``, first 20,000 jets): per-jet pT^0.5-weighted-sign charge sum
(see ``jet_charge``) computed with this sign table centres near 0 with
spread ~2.3 in raw (unweighted) net-charge units, and the global count of
positively-assigned versus negatively-assigned real particles has ratio
1.013 -- both reproduce the numbers asserted in the task brief. No
disagreement found; see the development tests for a lightweight jet-level
regression version of this check (all-pi+ / all-pi- / balanced synthetic
jets).

THE FOUR-VECTOR CONVENTION (the load-bearing empirical result of this
module -- see B1/B3/B4). ``global[:, 0]`` is documented as
``log(jet_pt)`` and ``global[:, 1]`` as ``log(jet_mass)``, both computed
upstream from the same constituents this module sees. Two hypotheses were
tested against ``JetClass_0.h5`` (20,000 jets):

* Naive: treat ``log_energy`` as each constituent's absolute energy and
  ``delta_eta`` as its own pseudorapidity, i.e.
  ``pz = pt*sinh(delta_eta)``, ``E = exp(log_energy)``. This reproduces
  ``jet_pt`` exactly (see below) but gives median jet mass ~420 GeV against
  a stored median of ~99 GeV -- the wrong convention flagged in the task
  brief. The bug: ``log_energy`` is each constituent's *absolute* lab-frame
  energy (a large number set mostly by the constituent's own large absolute
  eta, which we do not have -- only delta_eta relative to the jet is
  supplied), while ``delta_eta`` is a *jet-relative* angle. Combining an
  absolute energy with a jet-relative angle mixes two different frames and
  manufactures a spurious longitudinal component.
* Working convention: treat each constituent as **massless** and build its
  four-vector entirely from ``pt = exp(log_pt)`` (confirmed absolute, in
  GeV -- see below) and the two jet-relative angles, both interpreted in a
  single consistent frame centred on the jet axis::

      px = pt * cos(delta_phi)
      py = pt * sin(delta_phi)
      pz = pt * sinh(delta_eta)
      E  = pt * cosh(delta_eta)          # NOT exp(log_energy)

  This is exactly the statement "delta_eta acts as this constituent's
  rapidity in the jet-centred frame", which is self-consistent because a
  shift in rapidity is a symmetry (an active boost along the beam axis) that
  leaves an invariant mass unchanged -- so a massless four-vector built from
  (pt, delta_eta, delta_phi) alone has the same invariant-mass sum as the
  true constituents would, up to the true particles' individual rest masses
  (numerically negligible: pion/kaon/lepton masses are at most ~0.5 GeV
  against constituent pT of O(1-100) GeV) and up to genuine massive-particle
  corrections to how rapidity shifts under a boost (also small at these
  energies). ``log_energy`` is not used by this module at all -- every
  quantity below is built from ``pt`` and the two angles only.

  Measured agreement on 20,000 real jets from ``JetClass_0.h5``:
  ``jet_pt`` matches ``exp(global[:, 0]))`` to float32 rounding (median and
  max relative error ~1e-6 -- expected, since summing
  ``pt_i*(cos, sin)(delta_phi_i)`` is exactly how the upstream pipeline must
  have built the stored jet pT in the first place). ``jet_mass`` matches
  ``exp(global[:, 1])`` with **median relative error 0.56%** (median absolute
  error ~0.5 GeV against a ~99 GeV median mass); the error has a long tail
  from the massless approximation and from second-order boost effects when a
  jet's own eta is large (p90 ~19%, p99 ~45%, max ~176% relative error on
  this sample) -- callers using ``jet_mass`` for anything beyond typical-case
  physics should be aware of this tail. See the development tests for the
  reproduction check with the actual achieved numbers on a larger sample.

ECF3 TRUNCATION. ``ECF3`` sums over all *triples* of constituents and is
O(n^3); at up to 150 constituents per jet this is infeasible to compute
exactly for every jet in a batch (150 choose 3 ~= 551,300 triples/jet). This
module truncates the triple sum to the leading ``nmax_ecf3`` constituents by
pT (default 24; a parameter of every function that touches ECF3/C2/D2).
``ECF1``/``ECF2`` are NOT truncated -- they are only O(n) / O(n^2) and are
computed over every real constituent. Since jets typically have ~20-40 real
constituents (median 27, p99 58, max 92 on a 20,000-jet JetClass sample),
default ``nmax_ecf3 = 24`` already drops the softest tail of a typical jet.

The measured effect is LARGE in absolute terms and should not be understated:
on 500 real jets (JetClass QCD file, beta=1), comparing ``nmax_ecf3=24``
(default) against ``nmax_ecf3=96`` (effectively untruncated -- 96 exceeds the
99th-percentile real multiplicity of 58 on a 20,000-jet sample), the median
relative change in ``D2`` from truncation is **~25%** (mean ~27%, p90 ~59%).
Truncation is not a small correction here -- ``D2``'s absolute scale is not
reliable at ``nmax_ecf3=24`` and should not be compared across different
truncation choices, or to values reported elsewhere in the literature at
full multiplicity, without accounting for this. What DOES survive the
truncation, checked explicitly on 1,500 real jets per class at both
``nmax_ecf3=24`` and ``nmax_ecf3=96``, is the class **ordering** this module's
tests rely on: median D2 at nmax=24 is QCD 1.87 > W->qq 0.98 > H->bb 0.66;
at nmax=96 it is QCD 3.42 > W->qq 1.55 > H->bb 1.33 -- the absolute values
roughly double, but QCD stays highest and H->bb stays lowest at either
truncation. So: use the default for anything relative (class comparisons,
correlation with a model direction), and do not trust the untransformed
magnitude. Truncating to the *hardest* particles is a deliberate choice
despite the large effect: the triples that dominate the (pT-weighted) ECF3
sum are still overwhelmingly built from the hardest constituents (cubic in
pT, only linear-ish in the angular factor), so this is the correct direction
to truncate in even though the magnitude lost is not negligible.

Padding discipline: every quantity here is invariant under appending extra
padding rows, because every pT-weighted sum is weighted by ``exp(log_pt)``
forced to exactly 0 at padding (via ``constituent_mask``), so a padding row
contributes exactly 0 to every product it appears in -- including inside a
pairwise or triple-wise term, where it is enough for *one* factor in the
product to be a padding pT for the whole term to vanish, regardless of
whatever junk the angle/pid columns hold there (see
``softrepair.observables.basic`` docstring: padding is not all-zero in this data).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np

from softrepair.observables.basic import (
    DELTA_ETA_COLUMN,
    DELTA_PHI_COLUMN,
    LOG_PT_COLUMN,
    PID_COLUMN,
    constituent_mask,
    is_charged,
)

logger = logging.getLogger(__name__)

# -- tunable defaults ------------------------------------------------------

#: Default angular exponent for the energy correlation functions (A4/A5).
DEFAULT_ECF_BETA = 1.0

#: Default truncation of the ECF3 triple sum to the leading-pT constituents.
#: See module docstring "ECF3 TRUNCATION" for the measured effect of this
#: choice.
DEFAULT_ECF3_NMAX = 24

#: Default kappa exponent for the pT-weighted jet charge (D1).
DEFAULT_CHARGE_KAPPA = 0.5

#: Angularity exponents making up the lambda_beta family (A8). beta=1.0 is
#: girth (A6) by definition.
ANGULARITY_BETAS: tuple[float, ...] = (0.5, 1.0, 2.0)

#: Default batch chunk size for the O(n^2)/O(nmax^3) shape computations, so
#: that memory stays bounded regardless of how many jets are passed in one
#: call. Tuned so that the largest transient array (the (chunk, 150, 150)
#: pairwise angular-distance matrix used for ECF2) stays under ~100 MB.
DEFAULT_CHUNK_SIZE = 512

# -- particle-code categories (see module docstring for the code table) ----

#: mu+, e+, e-, mu- -- the four lepton codes present in this pid scheme.
LEPTON_PID_VALUES: frozenset[int] = frozenset({1, 2, 4, 5})

#: pi-, pi+ -- the two charged-hadron codes present in this pid scheme.
CHARGED_HADRON_PID_VALUES: frozenset[int] = frozenset({0, 8})

#: photon.
PHOTON_PID_VALUE = 6

#: neutral kaon. In this 9-code scheme this is the *only* neutral-hadron
#: code present, so ``neutral_hadron`` and ``kaon`` are numerically identical
#: categories here (see ``composition_features``) -- kept as separate keys
#: because the concept vocabulary (E1-E4) lists them as separate concepts and
#: a scheme with additional neutral-hadron codes would need to distinguish
#: them.
NEUTRAL_KAON_PID_VALUE = 7

#: pid_int -> electric charge, for codes that carry a definite charge.
#: Everything not in this mapping (photon, neutral kaon, padding) has charge
#: 0. Validated on this data: see module docstring "Re-verification of the
#: charge assignment".
PID_CHARGE: dict[int, int] = {0: -1, 1: 1, 2: 1, 4: -1, 5: -1, 8: 1}


# -- small shared helpers ----------------------------------------------------


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into [-pi, pi] (same convention as observables._wrap_angle)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _real_pt(data: np.ndarray) -> np.ndarray:
    """exp(log_pt), forced to exactly 0 at padding -- see observables._real_pt for why."""
    mask = constituent_mask(data)
    return np.where(mask, np.exp(data[..., LOG_PT_COLUMN]), 0.0)


def _pid_int(data: np.ndarray) -> np.ndarray:
    """Rounded integer pid code, valid only where constituent_mask is True."""
    return np.rint(data[..., PID_COLUMN]).astype(np.int64)


def four_vectors(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(px, py, pz, E), each (N, P), under the massless jet-frame convention.

    See the module docstring "THE FOUR-VECTOR CONVENTION" for the derivation
    and the measured agreement with the stored ``global`` values. All four
    arrays are exactly 0 at padding because ``pt`` (via ``_real_pt``) is 0
    there and every component is proportional to ``pt``.

    ``delta_eta``/``delta_phi`` are masked to exactly 0 at padding BEFORE
    ``sinh``/``cosh``, not just relied on implicitly via the ``pt`` factor:
    padding is not all-zero in this data (see observables.py docstring), and
    an unmasked junk ``delta_eta`` large enough to overflow ``sinh``/``cosh``
    (float32 overflows above ~88) would multiply the correct ``pt == 0`` by
    +-inf, giving ``0 * inf = nan`` under IEEE arithmetic -- silently turning
    every downstream sum for that jet into NaN. Masking first avoids this.
    """
    mask = constituent_mask(data)
    pt = _real_pt(data)
    deta = np.where(mask, data[..., DELTA_ETA_COLUMN], 0.0)
    dphi = np.where(mask, data[..., DELTA_PHI_COLUMN], 0.0)
    px = pt * np.cos(dphi)
    py = pt * np.sin(dphi)
    pz = pt * np.sinh(deta)
    energy = pt * np.cosh(deta)
    return px, py, pz, energy


def _angle_to_axis(data: np.ndarray) -> np.ndarray:
    """(N, P) angular distance dR of each constituent from the jet axis (0, 0).

    The jet axis is the origin of the (delta_eta, delta_phi) frame by
    construction (angles are already jet-axis-relative, per the data
    description), so this is a direct distance rather than a recomputed
    pT-weighted centroid -- the standard definition used by A6/A7/A8/D2.
    delta_phi is wrapped defensively even though it is small in practice.
    """
    deta = data[..., DELTA_ETA_COLUMN]
    dphi = _wrap_angle(data[..., DELTA_PHI_COLUMN])
    return np.sqrt(deta**2 + dphi**2)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """numerator / denominator, 0 wherever denominator <= 0 (rather than NaN/inf)."""
    safe_denom = np.where(denominator > 0, denominator, 1.0)
    return np.where(denominator > 0, numerator / safe_denom, 0.0)


# -- B1/B3/B4: kinematics -----------------------------------------------------


def kinematics_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """(N,) jet_pt, jet_mass, constituent multiplicity and charged multiplicity.

    jet_pt/jet_mass are built from ``four_vectors`` -- see module docstring
    for the convention and its measured agreement with ``global``.
    """
    mask = constituent_mask(data)
    charged = is_charged(data)

    px, py, pz, energy = four_vectors(data)
    px_sum = px.sum(axis=1)
    py_sum = py.sum(axis=1)
    pz_sum = pz.sum(axis=1)
    e_sum = energy.sum(axis=1)

    jet_pt = np.sqrt(px_sum**2 + py_sum**2)
    mass_sq = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
    jet_mass = np.sqrt(np.maximum(mass_sq, 0.0))

    n_constituents = mask.sum(axis=1).astype(np.int64)
    n_charged = (mask & charged).sum(axis=1).astype(np.int64)

    return {
        "jet_pt": jet_pt,
        "jet_mass": jet_mass,
        "n_constituents": n_constituents,
        "n_charged": n_charged,
    }


# -- A6/A7/A8/A10: angularities and shape ------------------------------------


def angularity_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """(N,) girth, pTD, lambda_beta family, and z1/z2 leading momentum fractions.

    All momentum fractions are relative to the total real pT of the jet
    (0 if the jet has no real constituents at all, which should not happen
    but is guarded). ``lambda_1_0`` (beta=1) equals ``girth`` by definition
    (A8 docstring) and the two are computed independently here as a cheap
    internal consistency check (see the development tests).
    """
    mask = constituent_mask(data)
    pt = _real_pt(data)
    total_pt = pt.sum(axis=1)

    dR = _angle_to_axis(data)
    dR = np.where(mask, dR, 0.0)

    girth = _safe_divide((pt * dR).sum(axis=1), total_pt)
    pt_sq_sum = (pt**2).sum(axis=1)
    pTD = _safe_divide(np.sqrt(np.maximum(pt_sq_sum, 0.0)), total_pt)

    z = _safe_divide(pt, total_pt[:, None])
    lambdas: dict[str, np.ndarray] = {}
    for beta in ANGULARITY_BETAS:
        key = f"lambda_beta_{str(beta).replace('.', 'p')}"
        lambdas[key] = (z * dR**beta).sum(axis=1)

    # z1/z2: leading and subleading momentum fraction (A10). Sort descending;
    # padding (pt=0) never outranks a real constituent, so this is safe even
    # for jets with 0 or 1 real constituents.
    sorted_pt = -np.sort(-pt, axis=1)
    n_cols = sorted_pt.shape[1]
    leading_pt = sorted_pt[:, 0] if n_cols >= 1 else np.zeros(data.shape[0])
    sub_pt = sorted_pt[:, 1] if n_cols >= 2 else np.zeros(data.shape[0])
    z1 = _safe_divide(leading_pt, total_pt)
    z2 = _safe_divide(sub_pt, total_pt)

    return {
        "girth": girth,
        "pTD": pTD,
        **lambdas,
        "z1": z1,
        "z2": z2,
    }


# -- A4/A5: energy correlation functions -------------------------------------


def _triple_indices(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static (i, j, k) index arrays for every i < j < k in range(m).

    Precomputed once per distinct m (the actual truncation length, which is
    min(nmax_ecf3, n_slots) and so is fixed for a whole dataset) rather than
    per jet or per chunk -- it does not depend on the data at all.
    """
    if m < 3:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    combos = np.array(list(itertools.combinations(range(m), 3)), dtype=np.int64)
    return combos[:, 0].copy(), combos[:, 1].copy(), combos[:, 2].copy()


def energy_correlation_features(
    data: np.ndarray,
    beta: float = DEFAULT_ECF_BETA,
    nmax_ecf3: int = DEFAULT_ECF3_NMAX,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """(N,) ECF1, ECF2, ECF3 (truncated), C2 and D2.

    Processed in chunks along the batch axis so that the largest transient
    array -- the (chunk, n_slots, n_slots) pairwise angular-distance matrix
    used for ECF2 -- stays bounded regardless of how many jets are passed at
    once; the (n_slots^3) tensor implied by a naive ECF3 is never
    materialised at all (see ``_triple_indices`` -- only the ``C(m, 3)``
    valid triples are ever gathered, not the full cube). See the module
    docstring "ECF3 TRUNCATION" for what ``nmax_ecf3`` trades off.

    C2/D2 are exactly 0.0 (the documented sentinel) wherever ECF2 is 0
    (0 or 1 real constituents -- no pair exists) and are also exactly 0.0
    whenever fewer than 3 real constituents exist, because ECF3 is then
    exactly 0 by construction (no triple exists) regardless of ECF2.
    """
    n_jets, n_slots, _ = data.shape
    m = min(nmax_ecf3, n_slots)
    idx_i, idx_j, idx_k = _triple_indices(m)

    ecf1 = np.zeros(n_jets, dtype=np.float64)
    ecf2 = np.zeros(n_jets, dtype=np.float64)
    ecf3 = np.zeros(n_jets, dtype=np.float64)

    for start in range(0, n_jets, chunk_size):
        stop = min(start + chunk_size, n_jets)
        chunk = data[start:stop]

        pt = _real_pt(chunk)
        deta = chunk[..., DELTA_ETA_COLUMN]
        dphi = chunk[..., DELTA_PHI_COLUMN]

        ecf1[start:stop] = pt.sum(axis=1)

        # ECF2 = sum_{i<j} pt_i pt_j dR_ij^beta = 0.5 * sum_{i,j} (...),
        # since the i==j diagonal has dR=0 and contributes nothing.
        deta_diff = deta[:, :, None] - deta[:, None, :]
        dphi_diff = _wrap_angle(dphi[:, :, None] - dphi[:, None, :])
        dR = np.sqrt(deta_diff**2 + dphi_diff**2)
        weight = pt[:, :, None] * pt[:, None, :]
        ecf2[start:stop] = 0.5 * (weight * dR**beta).sum(axis=(1, 2))

        if m >= 3:
            # Truncate to the leading-pT m constituents, then gather only
            # the valid (i<j<k) triples -- never the full (chunk, m, m, m)
            # cube.
            order = np.argsort(-pt, axis=1)[:, :m]
            pt_top = np.take_along_axis(pt, order, axis=1)
            deta_top = np.take_along_axis(deta, order, axis=1)
            dphi_top = np.take_along_axis(dphi, order, axis=1)

            pt_i = pt_top[:, idx_i]
            pt_j = pt_top[:, idx_j]
            pt_k = pt_top[:, idx_k]

            def _pair_dR(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                ddeta = deta_top[:, a] - deta_top[:, b]
                ddphi = _wrap_angle(dphi_top[:, a] - dphi_top[:, b])
                return np.sqrt(ddeta**2 + ddphi**2)

            dR_ij = _pair_dR(idx_i, idx_j)
            dR_ik = _pair_dR(idx_i, idx_k)
            dR_jk = _pair_dR(idx_j, idx_k)

            term = pt_i * pt_j * pt_k * (dR_ij * dR_ik * dR_jk) ** beta
            ecf3[start:stop] = term.sum(axis=1)

    ecf2_sq = ecf2**2
    ecf1_cubed = ecf1**3
    c2 = np.where(ecf2 > 0, np.divide(ecf3 * ecf1, np.where(ecf2 > 0, ecf2_sq, 1.0)), 0.0)
    d2 = np.where(
        ecf2 > 0, np.divide(ecf3 * ecf1_cubed, np.where(ecf2 > 0, ecf2**3, 1.0)), 0.0
    )

    return {
        "ecf1": ecf1,
        "ecf2": ecf2,
        "ecf3": ecf3,
        "c2": c2,
        "d2": d2,
    }


# -- C4/C5: soft-lepton content -----------------------------------------------


def soft_lepton_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """(N,) soft-lepton count, count fraction, momentum fraction, and leading-lepton pt_rel.

    Two distinct "fraction" quantities are reported because they measure
    different things and disagree sharply for a real physics reason (see
    the development tests physics-sanity section for the reproduced numbers):

    * ``soft_lepton_count_fraction`` = n_leptons / n_constituents (a share of
      particles). This is the quantity that matches the validated per-class
      numbers measured on JetClass (H->lvqq ~4%,
      t->blv ~5%, W->qq ~0.6%) -- consistent with those classes' leptons
      being one particle among ~30, i.e. genuinely a small share of the
      jet's particle content regardless of how much of the *momentum* that
      one particle carries.
    * ``soft_lepton_fraction`` = share of jet **pT** carried by leptons. For
      ``t->blv``/``H->lvqq`` this comes out far larger (tens of percent, not
      ~5%) because in those decays the lepton is not a soft byproduct of a
      heavy-flavour hadron decaying inside an otherwise hadronic jet -- it
      is itself one of the (few) visible primary decay products of the W,
      so it typically carries a large share of the jet's momentum. For
      ``H->bb`` (true soft-lepton tagging: an embedded semileptonic
      B-hadron decay inside an otherwise hadronic jet) the pT fraction is
      small (a few percent), which is the regime C4's "soft-lepton content"
      plain-meaning actually describes.

    ``pt_rel`` is the component of the leading (highest-pT) lepton's momentum
    perpendicular to the jet axis, using the same massless four-vector
    convention as ``kinematics_features``: with the jet axis direction taken
    as the pT-weighted total jet direction (the sum of all real
    constituents' momenta, not just the lepton's own angle), pt_rel is
    ``|p_lepton x p_hat_jet|`` (magnitude of the 3-vector rejection). 0 for
    jets with no lepton.
    """
    mask = constituent_mask(data)
    pid = _pid_int(data)
    is_lepton = mask & np.isin(pid, np.fromiter(LEPTON_PID_VALUES, dtype=np.int64))

    n_constituents = mask.sum(axis=1)
    pt = _real_pt(data)
    total_pt = pt.sum(axis=1)

    n_soft_leptons = is_lepton.sum(axis=1).astype(np.int64)
    soft_lepton_count_fraction = _safe_divide(
        n_soft_leptons.astype(np.float64), n_constituents.astype(np.float64)
    )
    lepton_pt_sum = np.where(is_lepton, pt, 0.0).sum(axis=1)
    soft_lepton_fraction = _safe_divide(lepton_pt_sum, total_pt)

    px, py, pz, energy = four_vectors(data)
    jet_px = px.sum(axis=1)
    jet_py = py.sum(axis=1)
    jet_pz = pz.sum(axis=1)
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)
    safe_jet_p = np.where(jet_p > 0, jet_p, 1.0)
    jet_hat = np.stack([jet_px, jet_py, jet_pz], axis=-1) / safe_jet_p[:, None]

    # pick, per jet, the leading (max pt) lepton slot
    lepton_pt_masked = np.where(is_lepton, pt, -1.0)
    leading_idx = np.argmax(lepton_pt_masked, axis=1)
    has_lepton = n_soft_leptons > 0

    rows = np.arange(data.shape[0])
    lep_px = px[rows, leading_idx]
    lep_py = py[rows, leading_idx]
    lep_pz = pz[rows, leading_idx]
    lep_vec = np.stack([lep_px, lep_py, lep_pz], axis=-1)

    along = np.sum(lep_vec * jet_hat, axis=-1)
    lep_mag_sq = lep_px**2 + lep_py**2 + lep_pz**2
    perp_sq = np.maximum(lep_mag_sq - along**2, 0.0)
    lepton_pt_rel = np.where(has_lepton & (jet_p > 0), np.sqrt(perp_sq), 0.0)

    return {
        "n_soft_leptons": n_soft_leptons,
        "soft_lepton_count_fraction": soft_lepton_count_fraction,
        "soft_lepton_fraction": soft_lepton_fraction,
        "lepton_pt_rel": lepton_pt_rel,
    }


# -- D1: jet charge ------------------------------------------------------------


def jet_charge(data: np.ndarray, kappa: float = DEFAULT_CHARGE_KAPPA) -> np.ndarray:
    """(N,) pT-weighted jet charge: sum(pt_i^kappa * q_i) / sum(pt_i^kappa).

    q_i comes from ``PID_CHARGE`` (0 for photon/neutral-kaon/padding). 0 for
    a jet with no real constituents (should not happen, guarded anyway).
    """
    mask = constituent_mask(data)
    pid = _pid_int(data)
    charge = np.zeros_like(pid, dtype=np.float64)
    for code, q in PID_CHARGE.items():
        charge = np.where(mask & (pid == code), float(q), charge)

    pt = _real_pt(data)
    weight = np.where(mask, pt**kappa, 0.0)
    return _safe_divide((weight * charge).sum(axis=1), weight.sum(axis=1))


# -- E1-E5: composition --------------------------------------------------------


def composition_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """(N,) momentum fractions by particle-type category, plus charged/neutral ratio.

    ``photon_frac + charged_hadron_frac + neutral_hadron_frac + lepton_frac``
    sums to 1 (tested in the development tests) because {photon} u
    {charged hadron} u {neutral hadron} u {lepton} is an exhaustive,
    mutually-exclusive partition of every real pid code in this scheme
    (0,1,2,4,5,6,7,8). ``kaon_frac`` is reported separately per the concept
    vocabulary (E1-E4 lists photon/charged-hadron/neutral-hadron/kaon as four
    distinct concepts) but is numerically identical to ``neutral_hadron_frac``
    on this dataset, because the neutral kaon (pid 7) is the only
    neutral-hadron code this pid scheme has -- see module docstring. It is
    NOT added again when checking the sum-to-1 identity above, to avoid
    double-counting.
    """
    mask = constituent_mask(data)
    pid = _pid_int(data)
    pt = _real_pt(data)
    total_pt = pt.sum(axis=1)

    def _fraction(selector: np.ndarray) -> np.ndarray:
        return _safe_divide(np.where(selector, pt, 0.0).sum(axis=1), total_pt)

    is_photon = mask & (pid == PHOTON_PID_VALUE)
    is_charged_hadron = mask & np.isin(
        pid, np.fromiter(CHARGED_HADRON_PID_VALUES, dtype=np.int64)
    )
    is_neutral_hadron = mask & (pid == NEUTRAL_KAON_PID_VALUE)
    is_lepton = mask & np.isin(pid, np.fromiter(LEPTON_PID_VALUES, dtype=np.int64))
    is_kaon = mask & (pid == NEUTRAL_KAON_PID_VALUE)

    photon_frac = _fraction(is_photon)
    charged_hadron_frac = _fraction(is_charged_hadron)
    neutral_hadron_frac = _fraction(is_neutral_hadron)
    lepton_frac = _fraction(is_lepton)
    kaon_frac = _fraction(is_kaon)

    charged = is_charged(data)
    charged_pt = np.where(mask & charged, pt, 0.0).sum(axis=1)
    neutral_pt = np.where(mask & ~charged, pt, 0.0).sum(axis=1)
    charged_to_neutral_ratio = _safe_divide(charged_pt, neutral_pt)

    return {
        "photon_frac": photon_frac,
        "charged_hadron_frac": charged_hadron_frac,
        "neutral_hadron_frac": neutral_hadron_frac,
        "lepton_frac": lepton_frac,
        "kaon_frac": kaon_frac,
        "charged_to_neutral_ratio": charged_to_neutral_ratio,
    }


# -- top-level vocabulary table -------------------------------------------------


def shape_features(
    data: np.ndarray,
    ecf_beta: float = DEFAULT_ECF_BETA,
    nmax_ecf3: int = DEFAULT_ECF3_NMAX,
    charge_kappa: float = DEFAULT_CHARGE_KAPPA,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """The full flat (N,) concept-vocabulary dict this module provides.

    Combines ``kinematics_features``, ``angularity_features``,
    ``energy_correlation_features``, ``soft_lepton_features``, ``jet_charge``
    and ``composition_features``. Every key is stable and documented at its
    source function; this is the single entry point the rest of the project
    should import. Only ``data`` (the raw (N, n_slots, 9) constituent array)
    is required -- ``global`` is not needed because jet_pt/jet_mass are
    reconstructed from constituents (see module docstring), which is also
    what makes this module usable for a validation check against ``global``.
    """
    out: dict[str, np.ndarray] = {}
    out.update(kinematics_features(data))
    out.update(angularity_features(data))
    out.update(
        energy_correlation_features(
            data, beta=ecf_beta, nmax_ecf3=nmax_ecf3, chunk_size=chunk_size
        )
    )
    out.update(soft_lepton_features(data))
    out["jet_charge"] = jet_charge(data, kappa=charge_kappa)
    out.update(composition_features(data))
    return out
