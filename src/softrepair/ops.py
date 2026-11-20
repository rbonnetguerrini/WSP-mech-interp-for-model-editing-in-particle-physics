"""Known input operations on JetClass jets, parameterised by a dose theta.

Raw columns: 0 delta_eta, 1 delta_phi, 2 log_pt, 3 log_energy, 4 pid, 5-8 impact parameters.
Only real constituents (log_pt != 0) are touched; padding stays bit-identical.

Energy is kept consistent with the edited kinematics multiplicatively,
    E' = E * (pt'/pt) * cosh(eta_j + deta') / cosh(eta_j + deta),
so every operation is EXACTLY the identity at theta = 0 (a massless recomputation would
not be, because constituents carry small masses). eta_j, the jet pseudorapidity, is not
stored; it is estimated once per jet from the clean constituents (``jet_eta``).

Deterministic operations are one-parameter groups: op(a) o op(b) = op(a + b), and
op(-a) inverts op(a) exactly (up to float32 rounding). The four angular ones generate the
similarity group of the (eta, phi) plane, which is non-abelian:
    D dilation, R rotation, Te / Tp translation in eta / phi.
D and R commute with each other; neither commutes with a translation.
J (energy scale) and P (fragmentation power, pt_i -> pt_i^k at fixed jet pt) act on
momenta only; they commute with each other and with every angular operation.

Stochastic operations (noise) form semigroups in the variance: Ne (log-pt smearing) and
Na (angular smearing), theta >= 0, noise variance = theta * unit^2.
"""

from __future__ import annotations

import numpy as np
import torch

# dose unit per operation: theta = 1 is a moderate, in-support change.
UNITS = {
    "D": 0.10,   # log scale factor of (deta, dphi)
    "R": 0.20,   # rotation angle [rad]
    "Te": 0.05,  # eta shift
    "Tp": 0.05,  # phi shift
    "J": 0.05,   # log energy-scale factor (5% JES)
    "P": 0.10,   # log of the fragmentation exponent k
    "Ne": 0.10,  # sigma of log-pt noise at theta = 1
    "Na": 0.02,  # sigma of angular noise at theta = 1
}
DETERMINISTIC = ("D", "R", "Te", "Tp", "J", "P")
STOCHASTIC = ("Ne", "Na")
ALL_OPS = DETERMINISTIC + STOCHASTIC
SIM2 = ("D", "R", "Te", "Tp")


def jet_eta(raw: np.ndarray) -> np.ndarray:
    """Per-jet pseudorapidity from massless constituents: E/pt = cosh(eta_j + deta)."""
    out = np.zeros(raw.shape[0], np.float32)
    for j in range(raw.shape[0]):
        r = raw[j, :, 2] != 0
        pt, E, de = np.exp(raw[j, r, 2]), np.exp(raw[j, r, 3]), raw[j, r, 0]
        a = np.arccosh(np.maximum(E / pt, 1.0))
        cands = [np.median(s * a - de) for s in (1.0, -1.0)]
        spread = [np.std(np.abs(c + de) - a) for c in cands]
        out[j] = cands[int(np.argmin(spread))]
    return out


def _logcosh(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(x, -x) - np.log(2.0)


def apply(raw: torch.Tensor, eta: torch.Tensor, op: str, theta, gen: torch.Generator | None = None) -> torch.Tensor:
    """Return op(theta) applied to raw (B, P, 9). theta: float or (B,) tensor."""
    out = raw.clone()
    real = raw[..., 2] != 0
    th = torch.as_tensor(theta, dtype=raw.dtype, device=raw.device)
    th = th.reshape(-1, 1) if th.ndim == 1 else th
    u = UNITS[op]
    de, dp, lpt = raw[..., 0], raw[..., 1], raw[..., 2]
    nde, ndp, nlpt = de, dp, lpt
    if op == "D":
        s = torch.exp(u * th)
        nde, ndp = de * s, dp * s
    elif op == "R":
        c, s = torch.cos(u * th), torch.sin(u * th)
        nde, ndp = c * de - s * dp, s * de + c * dp
    elif op == "Te":
        nde = de + u * th
    elif op == "Tp":
        ndp = dp + u * th
    elif op == "J":
        nlpt = lpt + u * th
    elif op == "P":
        k = torch.exp(u * th)
        neg = torch.finfo(raw.dtype).min / 4
        lse0 = torch.logsumexp(torch.where(real, lpt, neg), 1, keepdim=True)
        lse1 = torch.logsumexp(torch.where(real, k * lpt, neg), 1, keepdim=True)
        nlpt = k * lpt + (lse0 - lse1)
    elif op in ("Ne", "Na"):
        if torch.any(th < 0):
            raise ValueError(f"{op} is a noise semigroup: theta must be >= 0")
        sig = u * torch.sqrt(th)
        if op == "Ne":
            nlpt = lpt + sig * torch.randn(lpt.shape, generator=gen, device=raw.device, dtype=raw.dtype)
        else:
            nde = de + sig * torch.randn(de.shape, generator=gen, device=raw.device, dtype=raw.dtype)
            ndp = dp + sig * torch.randn(dp.shape, generator=gen, device=raw.device, dtype=raw.dtype)
    else:
        raise KeyError(op)
    # a real constituent must stay real for the model's padding criterion (log_pt != 0)
    nlpt = torch.where(nlpt.abs() < 1e-6, torch.full_like(nlpt, 1e-6), nlpt)
    e = eta.reshape(-1, 1)
    # delta first, so that a zero change is bit-exact
    nle = raw[..., 3] + ((nlpt - lpt) + (_logcosh(e + nde) - _logcosh(e + de)))
    out[..., 0] = torch.where(real, nde, raw[..., 0])
    out[..., 1] = torch.where(real, ndp, raw[..., 1])
    out[..., 2] = torch.where(real, nlpt, raw[..., 2])
    out[..., 3] = torch.where(real, nle, raw[..., 3])
    return out


def apply_seq(raw, eta, seq, gen=None):
    """seq: list of (op, theta), applied left to right (first element acts first)."""
    for op, th in seq:
        raw = apply(raw, eta, op, th, gen)
    return raw
