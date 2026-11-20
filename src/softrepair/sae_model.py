"""TopK sparse dictionaries (sparse autoencoder and crosscoder).

TopKSAE: a = TopK(ReLU(W_e (x - b_d) + b_e)), x_hat = W_d a + b_d, decoder columns unit norm.
Crosscoder: one latent code for two models' activations at the same site, a = TopK(ReLU(W_eA xA + W_eB xB + b_e)),
xA_hat = W_dA a + b_A, xB_hat = W_dB a + b_B. A latent whose decoder norm lives only in A is a feature of model A
that model B no longer has (and vice versa); shared latents have both.
Both are trained with Adam on the summed normalised MSE; dead latents are counted, not resampled.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def topk_relu(z: torch.Tensor, k: int) -> torch.Tensor:
    v, i = torch.topk(torch.relu(z), k, dim=-1)
    return torch.zeros_like(z).scatter_(-1, i, v)


class TopKSAE(nn.Module):
    def __init__(self, d: int, m: int, k: int, frozen_random: bool = False, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        Wd = torch.randn(d, m, generator=g)
        Wd /= Wd.norm(dim=0, keepdim=True)
        self.W_d = nn.Parameter(Wd, requires_grad=not frozen_random)
        self.W_e = nn.Parameter(Wd.T.clone(), requires_grad=not frozen_random)
        self.b_e = nn.Parameter(torch.zeros(m), requires_grad=not frozen_random)
        self.b_d = nn.Parameter(torch.zeros(d))
        self.k = k

    def encode(self, x):
        return topk_relu((x - self.b_d) @ self.W_e.T + self.b_e, self.k)

    def forward(self, x):
        a = self.encode(x)
        return a @ self.W_d.T + self.b_d, a

    @torch.no_grad()
    def renorm(self):
        if self.W_d.requires_grad:
            self.W_d /= self.W_d.norm(dim=0, keepdim=True).clamp_min(1e-8)


class Crosscoder(nn.Module):
    def __init__(self, d: int, m: int, k: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_dA = nn.Parameter(torch.randn(d, m, generator=g) / np.sqrt(d) * 0.5)
        self.W_dB = nn.Parameter(self.W_dA.detach().clone())
        self.W_eA = nn.Parameter(self.W_dA.detach().T.clone())
        self.W_eB = nn.Parameter(self.W_dB.detach().T.clone())
        self.b_e = nn.Parameter(torch.zeros(m))
        self.b_A, self.b_B = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.k = k

    def encode(self, xA, xB):
        return topk_relu(xA @ self.W_eA.T + xB @ self.W_eB.T + self.b_e, self.k)

    def forward(self, xA, xB):
        a = self.encode(xA, xB)
        return a @ self.W_dA.T + self.b_A, a @ self.W_dB.T + self.b_B, a


def train(model, data, steps: int, batch: int, lr: float = 3e-4, device: str = "cuda", log_every: int = 2000,
          seed: int = 1):
    """data: one (N, d) tensor (SAE) or a pair (crosscoder), already normalised, on device."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    pair = isinstance(data, tuple)
    N = (data[0] if pair else data).shape[0]
    var = [d.var(0).sum() for d in data] if pair else [data.var(0).sum()]
    log = []
    g = torch.Generator(device=device).manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), device=device, generator=g)
        if pair:
            xA, xB = data[0][idx], data[1][idx]
            hA, hB, _ = model(xA, xB)
            loss = ((hA - xA) ** 2).sum(1).mean() / var[0] + ((hB - xB) ** 2).sum(1).mean() / var[1]
        else:
            x = data[idx]
            h, _ = model(x)
            loss = ((h - x) ** 2).sum(1).mean() / var[0]
        if any(p.requires_grad for p in model.parameters()):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if hasattr(model, "renorm"):
                model.renorm()
        if step % log_every == 0 or step == steps - 1:
            log.append({"step": step, "nmse": float(loss)})
            print(f"[dict] step {step}: normalised MSE {float(loss):.4f}", flush=True)
    return log


@torch.no_grad()
def encode_all(model, data, batch: int = 65536):
    pair = isinstance(data, tuple)
    N = (data[0] if pair else data).shape[0]
    out = []
    for s in range(0, N, batch):
        out.append((model.encode(data[0][s:s + batch], data[1][s:s + batch]) if pair
                    else model.encode(data[s:s + batch])).to_sparse())
    return out
