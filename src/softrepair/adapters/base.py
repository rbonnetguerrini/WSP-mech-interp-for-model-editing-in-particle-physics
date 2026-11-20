"""The model-adapter interface.

Everything downstream -- activation capture, probes, erasure, attribution,
dictionaries, interventions -- talks to a model only through this interface, so
that a second architecture (Particle Transformer being the obvious candidate)
can be added without restructuring the project.

Design notes driven by the first adapter:

* Activation sites are **named**, not indexed by a bare integer. A sibling
  project addressed sites as "layer 3", which silently conflated the embedding
  output with block outputs and made configs unreadable.
* ``TokenLayout`` is explicit. A jet transformer's sequence is not homogeneous:
  it interleaves learned summary tokens, conditioning tokens and particle
  tokens. Analyses must be able to ask "which positions are real particles?"
  without hard-coding an offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable

import torch

#: A hook receives the site activation and returns a replacement (or ``None``
#: to leave it unchanged). It must preserve shape, dtype and device.
Intervention = Callable[[torch.Tensor], torch.Tensor | None]


@dataclass(frozen=True)
class SiteSpec:
    """One place inside the model where activations can be read or written."""

    name: str
    description: str
    #: "sequence" -> (batch, tokens, dim); "pooled" -> (batch, dim)
    kind: str = "sequence"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


@dataclass(frozen=True)
class TokenLayout:
    """Which sequence positions mean what, at a sequence-valued site.

    Positions are ``[summary tokens][auxiliary tokens][particle tokens]``.
    ``n_auxiliary`` counts conditioning-style tokens such as the diffusion-time
    token, which are real inputs to attention but are not particles and must be
    excluded from any per-particle analysis.
    """

    n_summary: int
    n_auxiliary: int

    @property
    def particle_start(self) -> int:
        return self.n_summary + self.n_auxiliary

    def summary_slice(self) -> slice:
        return slice(0, self.n_summary)

    def particle_slice(self) -> slice:
        return slice(self.particle_start, None)


@dataclass
class CheckpointIdentity:
    """Content-addressed identity of the loaded weights."""

    path: str
    sha256: str
    config: dict[str, Any]
    assumed_arguments: dict[str, Any]


@runtime_checkable
class ModelAdapter(Protocol):
    """The contract every model wrapper must satisfy."""

    def checkpoint_identity(self) -> CheckpointIdentity: ...

    def sites(self) -> list[SiteSpec]:
        """Every activation site this adapter can read or write."""

    def token_layout(self) -> TokenLayout: ...

    def preprocess(self, raw: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Turn a raw dataset record into the exact tensors the model expects."""

    def token_mask(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """(batch, tokens) boolean: True where the position is a real input."""

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        capture: list[str] | None = None,
        interventions: Mapping[str, Intervention] | None = None,
    ) -> dict[str, Any]:
        """Run inference, optionally capturing sites and/or editing them."""

    def head_scores(self, outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Named read-outs (logits, not probabilities) for causal measurement."""

    def capturing(self, names: list[str]) -> Iterator[dict[str, torch.Tensor]]:
        """Context manager yielding a dict that fills with captured activations."""
