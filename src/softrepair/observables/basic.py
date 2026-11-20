"""Physics observables and concept labels computed from raw JetClass data only.

Every function here reads the raw HDF5 ``data``/``global`` arrays and nothing
else. In particular nothing in this module touches a model, an activation or
a checkpoint: probes and causal claims need a ground truth that does not
depend on what the network happened to compute, and that ground truth has to
be importable on a machine that cannot load OmniLearned at all (e.g. a laptop
analysing cached activations). Hence numpy-only, no torch, no omnilearned
imports.

Column layout of the ``data`` array, mirrored (not imported, to avoid pulling
in torch) from ``adapters.omnilearned.KINEMATIC_COLUMNS`` / ``ADD_COLUMNS``::

    0 delta_eta   1 delta_phi   2 log_pt   3 log_energy   4 pid_int
    5 tanh_d0     6 sigma_d0    7 tanh_dz  8 sigma_dz

A slot is a real constituent iff ``data[..., 2] != 0`` (non-zero log_pt) --
this is the model's own padding criterion (see
``OmniLearnedAdapter.particle_mask``), so our notion of "real" cannot drift
from the network's.

Empirical findings used below (checked against
the JetClass validation file ``JetClass_0.h5``,
first 50,000 jets; re-verify if the data pipeline changes):

* Padding rows are not all-zero: only ``log_pt`` (and therefore ``log_energy``,
  which the model derives from it) is guaranteed zero. In the sampled file,
  padding carries ``pid_int == 3`` -- a value that never occurs at a real
  (non-padding) slot. Every function below still gates on ``constituent_mask``
  explicitly rather than relying on this, because a filler value is an
  incidental property of one preprocessing version, not a contract.
* Among real constituents, ``pid_int`` takes values ``{0, 1, 2, 4, 5, 6, 7,
  8}``. Values ``{0, 1, 2, 4, 5, 8}`` always carry ``sigma_d0 > 0`` and
  non-zero ``tanh_d0`` (i.e. a real impact-parameter measurement); values
  ``{6, 7}`` have ``sigma_d0 == 0`` and ``tanh_d0 == 0`` identically. This is
  the empirical criterion ``is_charged`` uses. It is PROBABLE (matching the
  task brief's reading of pid_int as pion/lepton/photon/kaon codes) but not
  independently confirmed against a JetClass PDG-code table that charged
  leptons and charged pions are exactly the ``{0,1,2,4,5,8}`` set and
  photons/K0 are exactly ``{6,7}`` -- what is confirmed directly from the data
  is the presence/absence of impact-parameter information, which is the only
  thing this module actually needs.
* ``tanh_d0`` and ``tanh_dz`` saturate at exactly +-1.0 in float32 for a
  non-negligible fraction of real, charged constituents: at a clip epsilon of
  1e-6 (i.e. flagging ``|tanh| >= 1 - 1e-6``), 3.6% of real charged
  constituents are saturated in d0 and 2.7% in dz (50,000-jet sample from
  JetClass_0.h5). Recovering ``d0 = arctanh(tanh_d0)`` therefore has an
  effective ceiling of ``arctanh(1 - 1e-6) ~= 7.25`` (in whatever units the
  network's preprocessing defines ``d0`` in); every downstream claim that uses
  the *magnitude* of a near-ceiling ``d0`` must say so, because the true value
  could be arbitrarily larger and is indistinguishable from the ceiling here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# -- column layout -------------------------------------------------------

DELTA_ETA_COLUMN = 0
DELTA_PHI_COLUMN = 1
LOG_PT_COLUMN = 2
LOG_ENERGY_COLUMN = 3
PID_COLUMN = 4
TANH_D0_COLUMN = 5
SIGMA_D0_COLUMN = 6
TANH_DZ_COLUMN = 7
SIGMA_DZ_COLUMN = 8

#: pid_int values observed (see module docstring) to carry a non-trivial
#: impact-parameter measurement -- probably charged pions and the four lepton
#: categories.
CHARGED_PID_VALUES: frozenset[int] = frozenset({0, 1, 2, 4, 5, 8})

#: pid_int values observed to carry no impact-parameter measurement at all --
#: probably photons and neutral kaons.
NEUTRAL_PID_VALUES: frozenset[int] = frozenset({6, 7})

#: Default clip for recovering d0/dz via arctanh(tanh_x). See module docstring
#: for the measured saturation fraction at this value.
DEFAULT_CLIP_EPS = 1e-6

#: Default |d0_significance| threshold defining a "displaced" track.
DEFAULT_SIG_THRESHOLD = 3.0

#: Frozen angular scale of the coherent-displacement compactness term (see
#: coherent_displacement_score). Chosen as a fraction of a typical jet cone,
#: not fit to any label; treat changing it as changing the definition of the
#: proxy, not as a bug fix.
DEFAULT_COMPACTNESS_SCALE = 0.15

#: Bump this if the coherent_displacement_score formula ever changes, so that
#: cached scores from different versions are never silently compared.
DISPLACEMENT_SCORE_VERSION = "v1"


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into [-pi, pi].

    delta_phi is already jet-relative and small in practice, but centroids
    computed from a subset of tracks can push a difference just outside that
    range; wrapping costs nothing and removes a rare source of a spuriously
    large angular spread.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _real_pt(data: np.ndarray) -> np.ndarray:
    """exp(log_pt), forced to exactly 0 at padding.

    Padding's log_pt is 0 by definition of constituent_mask, so exp(log_pt)
    would silently evaluate to 1 there -- a fake unit of momentum injected
    into every pT-weighted sum. Every pT weighting in this module goes through
    this helper instead of calling np.exp on the raw column directly.
    """
    mask = constituent_mask(data)
    return np.where(mask, np.exp(data[..., LOG_PT_COLUMN]), 0.0)


def constituent_mask(data: np.ndarray) -> np.ndarray:
    """(N, P) bool, True where a slot is a real constituent (non-zero log_pt)."""
    return data[..., LOG_PT_COLUMN] != 0


def is_charged(data: np.ndarray) -> np.ndarray:
    """(N, P) bool, True where a real constituent carries impact-parameter info.

    Always False at padding, regardless of whatever filler pid_int a given
    preprocessing version happens to write there (see module docstring) --
    the mask is applied explicitly rather than inferred from pid_int alone.
    """
    mask = constituent_mask(data)
    pid_int = np.rint(data[..., PID_COLUMN]).astype(np.int64)
    charged = np.isin(pid_int, np.fromiter(CHARGED_PID_VALUES, dtype=np.int64))
    return mask & charged


def tanh_saturation_fraction(
    data: np.ndarray, clip_eps: float = DEFAULT_CLIP_EPS
) -> dict[str, float]:
    """Fraction of constituents whose tanh_d0/tanh_dz sit at or beyond the clip.

    Reported both over all real constituents (as literally requested: neutral
    constituents have tanh == 0 exactly and dilute this number) and over real
    *charged* constituents only (the physically meaningful denominator, since
    only charged tracks carry a d0/dz measurement at all). Call this whenever
    d0/dz magnitudes from impact_parameters() are used for anything beyond a
    coarse threshold, to know whether the ceiling documented in the module
    docstring is being hit.
    """
    mask = constituent_mask(data)
    charged = is_charged(data)
    threshold = 1.0 - clip_eps

    def _frac(column: int, denom_mask: np.ndarray) -> float:
        denom = denom_mask.sum()
        if denom == 0:
            return 0.0
        saturated = denom_mask & (np.abs(data[..., column]) >= threshold)
        return float(saturated.sum()) / float(denom)

    result = {
        "d0_saturated_frac_real": _frac(TANH_D0_COLUMN, mask),
        "dz_saturated_frac_real": _frac(TANH_DZ_COLUMN, mask),
        "d0_saturated_frac_charged": _frac(TANH_D0_COLUMN, charged),
        "dz_saturated_frac_charged": _frac(TANH_DZ_COLUMN, charged),
    }
    logger.info("tanh saturation at clip_eps=%s: %s", clip_eps, result)
    return result


def impact_parameters(
    data: np.ndarray, clip_eps: float = DEFAULT_CLIP_EPS
) -> dict[str, np.ndarray]:
    """Recover impact parameters and their significances from raw columns.

    d0/dz are recovered as arctanh(tanh_d0)/arctanh(tanh_dz), with the
    argument clipped to [-1+clip_eps, 1-clip_eps] first because arctanh(+-1)
    is infinite and tanh does genuinely saturate in this data (see module
    docstring and tanh_saturation_fraction). All four raw/derived quantities
    are forced to exactly 0 at padding, regardless of whatever the raw column
    happens to hold there.

    Significance is d0/sigma_d0 (resp. dz/sigma_dz), defined as 0 wherever
    sigma_d0 <= 0 or the slot is not a real constituent -- a non-positive
    uncertainty means "no measurement", not "infinitely precise measurement".
    """
    mask = constituent_mask(data)

    clipped_d0 = np.clip(data[..., TANH_D0_COLUMN], -1.0 + clip_eps, 1.0 - clip_eps)
    clipped_dz = np.clip(data[..., TANH_DZ_COLUMN], -1.0 + clip_eps, 1.0 - clip_eps)
    d0 = np.where(mask, np.arctanh(clipped_d0), 0.0)
    dz = np.where(mask, np.arctanh(clipped_dz), 0.0)
    sigma_d0 = np.where(mask, data[..., SIGMA_D0_COLUMN], 0.0)
    sigma_dz = np.where(mask, data[..., SIGMA_DZ_COLUMN], 0.0)

    d0_significance = np.zeros_like(d0)
    valid_d0 = mask & (sigma_d0 > 0)
    d0_significance = np.where(valid_d0, np.divide(d0, np.where(valid_d0, sigma_d0, 1.0)), 0.0)

    dz_significance = np.zeros_like(dz)
    valid_dz = mask & (sigma_dz > 0)
    dz_significance = np.where(valid_dz, np.divide(dz, np.where(valid_dz, sigma_dz, 1.0)), 0.0)

    logger.debug(
        "impact_parameters: saturation diagnostic %s",
        tanh_saturation_fraction(data, clip_eps=clip_eps),
    )

    # Saturation mask. Where tanh(d0) reached +-1 in the stored float32, the
    # recovered |d0| is not a measurement but the clip ceiling arctanh(1-eps),
    # so any statistic dominated by the largest values is reporting the ceiling
    # rather than physics. Callers need to be able to exclude these.
    d0_saturated = mask & (np.abs(data[..., TANH_D0_COLUMN]) >= 1.0 - clip_eps)
    dz_saturated = mask & (np.abs(data[..., TANH_DZ_COLUMN]) >= 1.0 - clip_eps)

    return {
        "d0": d0,
        "sigma_d0": sigma_d0,
        "dz": dz,
        "sigma_dz": sigma_dz,
        "d0_significance": d0_significance,
        "dz_significance": dz_significance,
        "d0_saturated": d0_saturated,
        "dz_saturated": dz_saturated,
    }


def jet_observables(data: np.ndarray, global_: np.ndarray) -> dict[str, np.ndarray]:
    """(N,) per-jet kinematics: what `global` already stores, plus derived ones.

    The `global` column convention is NOT the same across datasets, and reading
    the wrong one is silent rather than fatal:

      JetClass (3 columns)  [log(jet_pt), log(jet_mass), n_constituents/100]
      Aspen    (4 columns)  [log(jet_pt), jet_eta, log(jet_mass), n_constituents/100]

    Applying the JetClass layout to an Aspen file returns eta as the jet mass and
    100*log(mass) as the constituent count, which is how that bug was found: the
    count came out with a mean of 306, a maximum of 589 and a MINIMUM OF -1324,
    the last being exactly 100 times the most negative log mass in the file. The
    activations were never affected, since the token mask is derived from `data`,
    but every observable-conditioned or sample-matched analysis would have been.

    The layout is therefore selected by column count and the multiplicity is
    checked against the count of real constituents, which is the one column whose
    meaning can be verified independently. n_charged/n_neutral come from
    constituent_mask + is_charged and were always correct; the pT-weighted
    quantities use _real_pt so padding contributes exactly 0 to every sum.
    """
    mask = constituent_mask(data)
    charged = is_charged(data)

    n_global = global_.shape[1]
    if n_global == 3:
        mass_col, count_col, eta_col = 1, 2, None
    elif n_global == 4:
        mass_col, count_col, eta_col = 2, 3, 1
    else:
        raise ValueError(
            f"unrecognised `global` layout with {n_global} columns; the column "
            f"meanings differ between datasets and must not be guessed"
        )

    jet_pt = np.exp(global_[:, 0])
    jet_mass = np.exp(global_[:, mass_col])
    n_constituents = np.rint(global_[:, count_col] * 100).astype(np.int64)

    # The multiplicity column is the one we can verify against the data array, so
    # use it to confirm the layout rather than trusting the column count alone.
    observed = mask.sum(axis=1).astype(np.int64)
    if n_constituents.size and not np.allclose(n_constituents, observed, atol=1):
        bad = int((np.abs(n_constituents - observed) > 1).sum())
        raise ValueError(
            f"`global` layout check failed: column {count_col} disagrees with the "
            f"number of unpadded constituents for {bad} of {observed.size} jets "
            f"(stored range {n_constituents.min()}..{n_constituents.max()}, "
            f"observed {observed.min()}..{observed.max()}). The column convention "
            f"is wrong for this file."
        )

    n_charged = (mask & charged).sum(axis=1).astype(np.int64)
    n_neutral = (mask & ~charged).sum(axis=1).astype(np.int64)

    pt = _real_pt(data)
    total_pt = pt.sum(axis=1)
    safe_total_pt = np.where(total_pt > 0, total_pt, 1.0)

    deta = data[..., DELTA_ETA_COLUMN]
    dphi = data[..., DELTA_PHI_COLUMN]
    pt_weighted_eta = (pt * deta).sum(axis=1) / safe_total_pt
    pt_weighted_phi = (pt * dphi).sum(axis=1) / safe_total_pt

    deta_c = deta - pt_weighted_eta[:, None]
    dphi_c = _wrap_angle(dphi - pt_weighted_phi[:, None])
    dR2 = deta_c**2 + dphi_c**2
    jet_width = np.sqrt(np.maximum((pt * dR2).sum(axis=1) / safe_total_pt, 0.0))
    jet_width = np.where(total_pt > 0, jet_width, 0.0)
    pt_weighted_eta = np.where(total_pt > 0, pt_weighted_eta, 0.0)
    pt_weighted_phi = np.where(total_pt > 0, pt_weighted_phi, 0.0)

    return {
        "jet_pt": jet_pt,
        "jet_mass": jet_mass,
        "n_constituents": n_constituents,
        "n_charged": n_charged,
        "n_neutral": n_neutral,
        "pt_weighted_eta": pt_weighted_eta,
        "pt_weighted_phi": pt_weighted_phi,
        "jet_width": jet_width,
        **({"jet_eta": global_[:, eta_col]} if eta_col is not None else {}),
    }


def displacement_features(
    data: np.ndarray,
    threshold: float = DEFAULT_SIG_THRESHOLD,
    clip_eps: float = DEFAULT_CLIP_EPS,
    dR0: float = DEFAULT_COMPACTNESS_SCALE,
) -> dict[str, np.ndarray]:
    """(N,) concept vocabulary for heavy-flavour / displaced-vertex probes.

    ``n_displaced_tracks`` counts real, charged constituents with
    ``|d0_significance| > threshold``. ``max_abs_d0_significance`` and
    ``second_max_abs_d0_significance`` rank ALL real charged constituents by
    |d0_significance| regardless of the threshold (0 if fewer than 1 or 2
    exist respectively), so they remain informative for jets that have no
    track above threshold at all. ``displaced_pt_fraction`` is the pT share
    (via _real_pt) carried by the displaced set relative to the whole jet, 0
    if the jet has no real pT (should not happen, guarded anyway).
    ``coherent_displacement`` is coherent_displacement_score with the same
    threshold/clip_eps/dR0.
    """
    mask = constituent_mask(data)
    charged = is_charged(data)
    ip = impact_parameters(data, clip_eps=clip_eps)
    d0_sig = ip["d0_significance"]

    charged_real = mask & charged
    displaced = charged_real & (np.abs(d0_sig) > threshold)
    n_displaced_tracks = displaced.sum(axis=1).astype(np.int64)

    ranked = np.where(charged_real, np.abs(d0_sig), -np.inf)
    ranked_sorted = -np.sort(-ranked, axis=1)
    n_cols = ranked_sorted.shape[1]
    max1 = ranked_sorted[:, 0] if n_cols >= 1 else np.zeros(data.shape[0])
    max2 = ranked_sorted[:, 1] if n_cols >= 2 else np.zeros(data.shape[0])
    max1 = np.where(np.isfinite(max1), max1, 0.0)
    max2 = np.where(np.isfinite(max2), max2, 0.0)

    pt = _real_pt(data)
    total_pt = pt.sum(axis=1)
    displaced_pt = np.where(displaced, pt, 0.0).sum(axis=1)
    displaced_pt_fraction = np.where(total_pt > 0, displaced_pt / np.where(total_pt > 0, total_pt, 1.0), 0.0)

    coherent = coherent_displacement_score(data, threshold=threshold, clip_eps=clip_eps, dR0=dR0)

    return {
        "n_displaced_tracks": n_displaced_tracks,
        "max_abs_d0_significance": max1,
        "second_max_abs_d0_significance": max2,
        "displaced_pt_fraction": displaced_pt_fraction,
        "coherent_displacement": coherent,
    }


def coherent_displacement_score(
    data: np.ndarray,
    threshold: float = DEFAULT_SIG_THRESHOLD,
    clip_eps: float = DEFAULT_CLIP_EPS,
    dR0: float = DEFAULT_COMPACTNESS_SCALE,
) -> np.ndarray:
    """(N,) proxy for a coherent multi-track displacement, FROZEN as DISPLACEMENT_SCORE_VERSION.

    This is a PROXY, not a secondary-vertex reconstruction. (d0, sigma_d0, dz,
    sigma_dz) alone cannot reconstruct a true secondary vertex: there is no
    track direction covariance and no z0 measured relative to the primary
    vertex, so two tracks with large, similar IP significance could originate from
    unrelated mismeasurements, from two independent heavy-flavour decays, or
    from one genuine displaced vertex, and this formula cannot tell those
    apart. It only captures a specific, named notion of "coherent displacement
    candidate": several tracks that are simultaneously significantly displaced,
    angularly close to each other, and displaced to the same side of the jet
    axis in sign.

    Definition (frozen -- change DISPLACEMENT_SCORE_VERSION if this changes):

    1. The "displaced set" is real, charged constituents with
       |d0_significance| > threshold. If fewer than 2 such tracks exist, the
       score is exactly 0 -- a single large-IP track, however extreme, never
       scores above 0 here, by construction (see the module tests for why
       this distinction is the point of the proxy).
    2. compactness = exp(-dR_rms / dR0), where dR_rms is the pT-weighted RMS
       angular separation (in the jet's local delta_eta/delta_phi frame) of
       the displaced tracks from their own pT-weighted centroid. compactness
       is 1.0 when the displaced tracks coincide and decays towards 0 as they
       spread out; dR0 (default 0.15) sets the angular scale and is frozen,
       not fit.
    3. sign_consistency = |sum(sign(d0_i))| / n_displaced over the displaced
       set: 1.0 if every displaced track's recovered d0 has the same sign,
       0.0 if they split evenly.
    4. score = compactness * sign_consistency, in [0, 1].
    """
    mask = constituent_mask(data)
    charged = is_charged(data)
    ip = impact_parameters(data, clip_eps=clip_eps)
    d0 = ip["d0"]
    d0_sig = ip["d0_significance"]

    displaced = mask & charged & (np.abs(d0_sig) > threshold)
    n_disp = displaced.sum(axis=1)

    pt = _real_pt(data)
    w = np.where(displaced, pt, 0.0)
    w_sum = w.sum(axis=1)
    safe_w_sum = np.where(w_sum > 0, w_sum, 1.0)

    deta = data[..., DELTA_ETA_COLUMN]
    dphi = data[..., DELTA_PHI_COLUMN]
    eta_c = (w * deta).sum(axis=1) / safe_w_sum
    phi_c = (w * dphi).sum(axis=1) / safe_w_sum

    deta_diff = deta - eta_c[:, None]
    dphi_diff = _wrap_angle(dphi - phi_c[:, None])
    dR2 = deta_diff**2 + dphi_diff**2
    weighted_dR2 = (w * dR2).sum(axis=1) / safe_w_sum
    dR_rms = np.sqrt(np.maximum(weighted_dR2, 0.0))
    compactness = np.exp(-dR_rms / dR0)

    sign_sum = np.where(displaced, np.sign(d0), 0.0).sum(axis=1)
    safe_n_disp = np.where(n_disp > 0, n_disp, 1)
    sign_consistency = np.where(n_disp > 0, np.abs(sign_sum) / safe_n_disp, 0.0)

    score = compactness * sign_consistency
    return np.where(n_disp >= 2, score, 0.0)




def unsaturated_displacement_features(
    data: np.ndarray,
    threshold: float = DEFAULT_SIG_THRESHOLD,
    clip_eps: float = DEFAULT_CLIP_EPS,
) -> dict[str, np.ndarray]:
    """Order statistics of impact-parameter significance, excluding saturated tracks.

    Measured on JetClass, ``max_abs_d0_significance`` computed over all charged
    constituents is **not a physics observable**: about 4% of charged tracks have
    ``tanh(d0)`` saturated at +-1 in the stored float32, so their recovered |d0| is
    the clip ceiling ``arctanh(1 - eps) = 7.248`` rather than a measurement. The
    maximum is then ``7.248 / min(sigma_d0)``, which came out as exactly 717.6 for
    both H->bb and H->gg -- identical, and ordered *backwards* with respect to
    heavy-flavour content (H->gg scored 153 versus 112 for H->bb at the jet level,
    because gluon jets simply have more charged tracks for the extreme-value
    statistic to draw from).

    The quantities here restrict to non-saturated charged constituents, so they
    measure displacement rather than clipping. ``n_displaced_tracks`` and the
    coherence proxy are threshold-based counts and were never affected.
    """
    mask = constituent_mask(data)
    charged = is_charged(data)
    ip = impact_parameters(data, clip_eps=clip_eps)
    usable = mask & charged & ~ip["d0_saturated"]

    significance = np.where(usable, np.abs(ip["d0_significance"]), 0.0)
    ordered = np.sort(significance, axis=1)[:, ::-1]
    n_usable = usable.sum(axis=1)

    first = np.where(n_usable >= 1, ordered[:, 0], 0.0)
    second = np.where(n_usable >= 2, ordered[:, 1], 0.0)
    third = np.where(n_usable >= 3, ordered[:, 2], 0.0)

    total = significance.sum(axis=1)
    median = np.zeros(data.shape[0], dtype=np.float64)
    for i in range(data.shape[0]):
        values = significance[i][usable[i]]
        if values.size:
            median[i] = float(np.median(values))

    return {
        "max_abs_d0_significance_unsat": first.astype(np.float64),
        "second_max_abs_d0_significance_unsat": second.astype(np.float64),
        "third_max_abs_d0_significance_unsat": third.astype(np.float64),
        "median_abs_d0_significance_unsat": median,
        "sum_abs_d0_significance_unsat": total.astype(np.float64),
        "n_unsaturated_charged": n_usable.astype(np.int64),
        "n_saturated_d0": (mask & charged & ip["d0_saturated"]).sum(axis=1).astype(np.int64),
    }
