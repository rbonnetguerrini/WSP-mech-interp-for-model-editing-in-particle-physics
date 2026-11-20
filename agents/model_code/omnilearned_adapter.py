"""Adapter for OmniLearned (PET2), pretrained checkpoints.

The pretrained checkpoint carries *two* heads over one frozen body: a 210-way
classifier (200 jet classes + 10 dataset-provenance classes) and a flow-matching
generative head. Both are exposed here, because comparing them is the central
experiment: a feature's causal role is a property of the feature-head pair, not
of the feature alone.

Sequence layout in ``pretrain`` mode is
``[4 summary tokens][time token][particle tokens]``. ``PET_body.forward``
prepends the conditioning and time tokens, then the summary tokens, and derives
its padding mask *once* from channel 2 of the embedded representation before
the transformer blocks run. Two consequences matter for interventions:

* editing a block output cannot change the mask used by later blocks, and
* an edit may write non-zero values into padded slots, which the final
  ``norm(x) * mask`` will zero but which would still corrupt any cached
  activation. This adapter therefore re-applies the mask after every edit.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from hep_ggb.adapters.base import (
    CheckpointIdentity,
    Intervention,
    SiteSpec,
    TokenLayout,
)
from hep_ggb.adapters.checkpoint import Pet2Config, infer_pet2_config
from hep_ggb.provenance import sha256_file

#: Column layout of the OmniLearned JetClass HDF5 ``data`` array.
KINEMATIC_COLUMNS = ("delta_eta", "delta_phi", "log_pt", "log_energy")
PID_COLUMN = 4
ADD_COLUMNS = ("tanh_d0", "sigma_d0", "tanh_dz", "sigma_dz")
PREPROCESS_VERSION = "omnilearned-jetclass-v1"


class OmniLearnedAdapter:
    """Frozen, evaluation-mode wrapper around a pretrained PET2."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        weights: str = "trained",
        compute_checksum: bool = True,
    ) -> None:
        """``weights='ema'`` loads the EMA body/generator instead of the live ones.

        The EMA copies are what a generative evaluation would normally use; the
        classifier head has no EMA counterpart in the checkpoint.
        """
        from omnilearned.network import PET2

        if weights not in ("trained", "ema"):
            raise ValueError(f"weights must be 'trained' or 'ema', got {weights!r}")

        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.weights = weights

        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.config: Pet2Config = infer_pet2_config(checkpoint)
        self.model = PET2(**self.config.to_kwargs())

        body_key = "ema_body" if weights == "ema" and "ema_body" in checkpoint else "body"
        gen_key = (
            "ema_generator"
            if weights == "ema" and "ema_generator" in checkpoint
            else "generator_head"
        )
        self.model.body.load_state_dict(checkpoint[body_key], strict=True)
        self.model.classifier.load_state_dict(checkpoint["classifier_head"], strict=True)
        if self.model.generator is not None:
            self.model.generator.load_state_dict(checkpoint[gen_key], strict=True)
        self._loaded_keys = {"body": body_key, "generator": gen_key}

        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._sha256 = sha256_file(self.checkpoint_path) if compute_checksum else ""
        self._handles: list[Any] = []

    # -- identity -------------------------------------------------------------

    def checkpoint_identity(self) -> CheckpointIdentity:
        return CheckpointIdentity(
            path=str(self.checkpoint_path),
            sha256=self._sha256,
            config={
                **self.config.to_kwargs(),
                "weights": self.weights,
                "loaded_keys": self._loaded_keys,
                "preprocess_version": PREPROCESS_VERSION,
            },
            assumed_arguments={
                key: getattr(self.config, key) for key in self.config.uninferable
            },
        )

    # -- structure ------------------------------------------------------------

    def token_layout(self) -> TokenLayout:
        # PET_body.num_add counts the conditioning and diffusion-time tokens.
        return TokenLayout(
            n_summary=self.config.num_tokens, n_auxiliary=int(self.model.body.num_add)
        )

    def _site_modules(self) -> dict[str, torch.nn.Module]:
        body = self.model.body
        modules: dict[str, torch.nn.Module] = {"embed": body.embed}
        for prefix, blocks in (
            ("body.block", body.in_blocks),
            ("body.out_block", getattr(body, "out_blocks", []) or []),
        ):
            for index, block in enumerate(blocks):
                modules[f"{prefix}{index}"] = block
                # MLP hidden units: fc1 -> GELU -> DynamicTanh -> fc2. Every step
                # between fc1 and fc2 is elementwise, so these channels are a
                # privileged basis, unlike the residual stream.
                modules[f"{prefix}{index}.mlp_pre"] = block.mlp.fc1
                modules[f"{prefix}{index}.mlp_post"] = block.mlp.norm
        modules["body.norm"] = body.norm
        if self.model.classifier is not None:
            for index, block in enumerate(self.model.classifier.in_blocks):
                modules[f"classifier.block{index}"] = block
            modules["classifier.fc"] = self.model.classifier.fc
        return modules

    def sites(self) -> list[SiteSpec]:
        specs = [
            SiteSpec("embed", "constituent embedding, before the transformer blocks")
        ]
        for index in range(self.config.num_transformers):
            specs.append(
                SiteSpec(
                    f"body.block{index}",
                    f"residual stream after body transformer block {index}",
                )
            )
            specs.append(
                SiteSpec(
                    f"body.block{index}.mlp_pre",
                    f"MLP hidden pre-activation (fc1 output) in body block {index}",
                )
            )
            specs.append(
                SiteSpec(
                    f"body.block{index}.mlp_post",
                    f"MLP hidden activation read by fc2 (w * tanh(alpha * GELU(pre))) "
                    f"in body block {index}",
                )
            )
        specs.append(
            SiteSpec("body.norm", "final body representation (x_body), both heads read this")
        )
        for index in range(self.config.num_transformers_head):
            specs.append(
                SiteSpec(
                    f"classifier.block{index}",
                    f"classifier head token-attention block {index}",
                )
            )
        specs.append(
            SiteSpec("classifier.fc", "pooled summary-token vector before the output layer",
                     kind="pooled")
        )
        return specs

    # -- data -----------------------------------------------------------------

    @staticmethod
    def preprocess(raw: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Split an OmniLearned ``data`` array into the model's input tensors.

        Mirrors ``omnilearned.dataloader.HEPDataset`` with ``use_pid=True,
        pid_idx=4, use_add=True, num_add=4`` and ``clip_inputs=False`` -- the
        upstream clipping path (R < 0.8, log pT > -0.7) is deliberately not
        reproduced, because the pretrained evaluation commands do not enable it.
        Kept as a separate, testable function because a preprocessing mismatch is
        silent and fatal.
        """
        data = raw["data"]
        if not torch.is_tensor(data):
            data = torch.as_tensor(data)
        data = data.to(torch.float32)
        if data.ndim != 3 or data.shape[-1] != 9:
            raise ValueError(
                f"expected (batch, particles, 9) JetClass array, got {tuple(data.shape)}"
            )

        pid = data[:, :, PID_COLUMN].to(torch.int32)
        rest = torch.cat([data[:, :, :PID_COLUMN], data[:, :, PID_COLUMN + 1 :]], dim=2)
        add_info = rest[:, :, -len(ADD_COLUMNS) :].contiguous()
        kinematics = rest[:, :, : -len(ADD_COLUMNS)].contiguous()

        batch: dict[str, torch.Tensor] = {
            "X": kinematics,
            "pid": pid,
            "add_info": add_info,
        }
        # Labels: accept the upstream HDF5 key ("pid") as well as an explicit
        # override. Upstream HEPDataset subtracts `label_shift`, which converts
        # the stored *global* index into a dataset-local one; that is correct for
        # fine-tuning on JetClass alone and wrong for the 200-way pretrained head,
        # so no shift is applied here.
        label = raw.get("pid_label", raw.get("labels", raw.get("pid_label_global")))
        if label is None and "pid" in raw and np.ndim(raw["pid"]) == 1:
            label = raw["pid"]
        if label is not None:
            batch["y"] = torch.as_tensor(np.asarray(label)).to(torch.float32)
        else:
            # y only conditions the generative head; classification ignores it.
            batch["y"] = torch.zeros(kinematics.shape[0], dtype=torch.float32)
        return batch

    @staticmethod
    def particle_mask(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """(batch, particles) True for real constituents.

        Uses the model's own criterion -- a non-zero log-pT channel -- so that
        our notion of "real" cannot drift from the network's.
        """
        return batch["X"][:, :, 2] != 0

    def token_mask(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """(batch, tokens) True for real positions, in full sequence coordinates."""
        particles = self.particle_mask(batch)
        layout = self.token_layout()
        prefix = torch.ones(
            particles.shape[0],
            layout.particle_start,
            dtype=torch.bool,
            device=particles.device,
        )
        return torch.cat([prefix, particles], dim=1)

    # -- inference ------------------------------------------------------------

    @contextlib.contextmanager
    def _hooks(
        self,
        capture: list[str] | None,
        interventions: Mapping[str, Intervention] | None,
        token_mask: torch.Tensor | None,
        particle_mask: torch.Tensor | None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Register capture/edit hooks, re-masking every edit.

        Two site-specific subtleties, both of which silently corrupt results if
        ignored:

        * ``body.embed`` is an ``InputBlock`` whose ``forward`` returns the tuple
          ``(x_embed, x_raw)``. Only the first element is the representation.
        * sites before the tokens are prepended are particle-length, while sites
          inside the blocks are full-sequence-length. The correct mask differs.
        """
        modules = self._site_modules()
        store: dict[str, torch.Tensor] = {}
        handles: list[Any] = []
        wanted = set(capture or [])
        edits = dict(interventions or {})
        unknown = (wanted | set(edits)) - set(modules)
        if unknown:
            raise KeyError(
                f"unknown activation site(s) {sorted(unknown)}; "
                f"available: {sorted(modules)}"
            )

        def mask_for(name: str, tensor: torch.Tensor) -> torch.Tensor | None:
            if tensor.ndim != 3:
                return None
            if particle_mask is not None and tensor.shape[1] == particle_mask.shape[1]:
                return particle_mask
            if token_mask is not None and tensor.shape[1] == token_mask.shape[1]:
                return token_mask
            return None

        def make_hook(name: str):
            def hook(module, inputs, output):  # noqa: ANN001
                is_tuple = isinstance(output, tuple)
                tensor = output[0] if is_tuple else output
                replaced = False
                if name in edits:
                    replacement = edits[name](tensor)
                    if replacement is not None:
                        if replacement.shape != tensor.shape:
                            raise ValueError(
                                f"intervention at '{name}' changed shape "
                                f"{tuple(tensor.shape)} -> {tuple(replacement.shape)}"
                            )
                        mask = mask_for(name, replacement)
                        if mask is not None:
                            replacement = replacement * mask.unsqueeze(-1).to(
                                replacement.dtype
                            )
                        tensor = replacement
                        replaced = True
                if name in wanted:
                    store[name] = tensor.detach()
                if not replaced:
                    return None
                return (tensor, *output[1:]) if is_tuple else tensor

            return hook

        try:
            for name in wanted | set(edits):
                handles.append(modules[name].register_forward_hook(make_hook(name)))
            yield store
        finally:
            for handle in handles:
                handle.remove()

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        capture: list[str] | None = None,
        interventions: Mapping[str, Intervention] | None = None,
    ) -> dict[str, Any]:
        """Deterministic classification pass.

        Deliberately does **not** call ``PET2.forward``. In ``pretrain`` mode that
        method samples ``time = torch.rand(...)``, runs the whole generative
        branch, and then evaluates the classifier a *second* time on the noised
        representation (``y_perturb = classifier(z_body)``). A forward hook on a
        classifier submodule therefore fires twice and the second, stochastic
        call overwrites the activation that actually produced ``y_pred``. Running
        the body and the classifier directly avoids that, keeps the pass
        deterministic, leaves the global RNG untouched, and skips the unused
        generative work.
        """
        device = self.device
        X = batch["X"].to(device)
        pid = batch["pid"].to(device) if batch.get("pid") is not None else None
        add_info = (
            batch["add_info"].to(device) if batch.get("add_info") is not None else None
        )

        token_mask = particle_mask = None
        if interventions:
            token_mask = self.token_mask(batch).to(device)
            particle_mask = self.particle_mask(batch).to(device)

        time = torch.zeros(X.shape[0], device=device)
        with self._hooks(capture, interventions, token_mask, particle_mask) as store:
            with torch.no_grad():
                x_body = self.model.body(X, None, pid, add_info, time)
                y_pred = self.model.classifier(x_body)

        return {
            "y_pred": y_pred,
            "x_body": x_body,
            "activations": store,
            "pass": "classify",
        }

    def head_scores(self, outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Logits, deliberately not probabilities.

        A softmax over 200 classes redistributes mass mechanically, so an
        intervention that raises one class appears to suppress every other. All
        causal measurement therefore uses logits and logit differences.
        """
        logits = outputs["y_pred"]
        return {
            "jet_logits": logits[:, :200],
            "dataset_logits": logits[:, 200:],
            "all_logits": logits,
        }
