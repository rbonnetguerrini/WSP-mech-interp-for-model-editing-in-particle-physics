"""The rule: drop every neutral particle that JetClass's detector simulation never reconstructs.

reco(raw) for raw jets of shape (n, 150, 9) (columns: deta, dphi, log_pt, log_E, pid, d0, sigma_d0, dz, sigma_dz).
A photon (pid 6) with pT < 0.80 GeV and E < 1.80 GeV, or a neutral hadron (pid 7) with E < 2.20 GeV, becomes a
padding row (all columns 0, pid 3). Everything else is unchanged. Identical to the reference used for scoring.
"""
import numpy as np

PHOTON_PT, PHOTON_E, NEUTRAL_HADRON_E = 0.80, 1.80, 2.20


def drop_mask(raw: np.ndarray) -> np.ndarray:
    """(n, 150) bool: the rows the rule drops."""
    real = raw[:, :, 2] != 0
    pt, E = np.exp(raw[:, :, 2].astype(np.float64)), np.exp(raw[:, :, 3].astype(np.float64))
    code = np.rint(raw[:, :, 4]).astype(int)
    return real & (((code == 6) & (pt < PHOTON_PT) & (E < PHOTON_E)) | ((code == 7) & (E < NEUTRAL_HADRON_E)))


def reco(raw: np.ndarray) -> np.ndarray:
    out = raw.copy()
    drop = drop_mask(out)
    out[drop] = 0.0
    out[drop, 4] = 3.0
    return out
