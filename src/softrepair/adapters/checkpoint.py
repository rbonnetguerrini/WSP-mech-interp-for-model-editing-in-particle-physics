"""Recover a pretrained model's architecture from its weights alone.

OmniLearned checkpoints store only ``state_dict``s -- no hyper-parameters. The
published documentation is ambiguous about the pretrained configuration: the
repository README's evaluation example passes ``--interaction
--local-interaction`` while ``shell/eval.sh``'s own ``pretrain_s`` command does
not. Guessing wrong is not a loud failure, because ``restore_checkpoint`` loads
the body with ``strict=False``: mismatched modules are silently left at their
random initialisation.

So we infer the configuration from the parameter shapes and then demand a
``strict=True`` load. That is a strong check but *not* a proof of uniqueness:
constructor choices that carry no parameters -- the activation function, dropout
rates, the normalisation class -- can differ while every tensor name and shape
still loads, and the model would then compute something different without any
visible symptom. Those remain assumptions, recorded as such, and are tested
indirectly by reproducing task accuracy rather than by the load succeeding.

Two constructor arguments leave *no* trace in the weights and therefore cannot
be inferred: ``K`` (number of neighbours in the local-embedding kNN) and
``num_coord`` (how many leading features define the distance metric). Both
change behaviour but not parameter shapes. We take the values the training CLI
would have used and validate them empirically through task accuracy -- a wrong
neighbourhood definition degrades the classifier badly, so the reference-
reproduction check is what actually tests them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

# Values that the weights cannot pin down. `omnilearned.cli.train` defaults.
UNINFERABLE_DEFAULTS: dict[str, Any] = {
    "K": 10,
    "num_coord": 2,
    "int_type": "lhc",
}



class CheckpointInferenceError(RuntimeError):
    """Raised when a checkpoint's architecture cannot be determined."""


def _count_blocks(state: Mapping[str, Any], prefix: str) -> int:
    """Number of distinct ``<prefix>.<i>.`` module indices in a state dict.

    ``prefix`` carries no trailing dot and is matched from the start of the key,
    so ``"in_blocks"`` counts the body's own blocks without also picking up
    ``local_physics.in_blocks.*``.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = {int(m.group(1)) for key in state if (m := pattern.match(key))}
    return len(indices)


def _has(state: Mapping[str, Any], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state)


@dataclass
class Pet2Config:
    """A fully determined ``PET2`` constructor argument set."""

    # inferred from parameter shapes
    input_dim: int
    base_dim: int
    num_transformers: int
    num_transformers_head: int
    num_tokens: int
    num_heads: int
    mlp_ratio: int
    use_int: bool
    local_int: bool
    use_local: bool
    skip: bool
    conditional: bool
    cond_dim: int
    pid: bool
    pid_dim: int
    add_info: bool
    add_dim: int
    num_classes: int
    num_gen_classes: int
    mode: str
    # not inferable from weights; see module docstring
    K: int = UNINFERABLE_DEFAULTS["K"]
    num_coord: int = UNINFERABLE_DEFAULTS["num_coord"]
    int_type: str = UNINFERABLE_DEFAULTS["int_type"]
    # bookkeeping
    uninferable: tuple[str, ...] = ("K", "num_coord", "int_type")
    notes: list[str] = field(default_factory=list)

    def to_kwargs(self) -> dict[str, Any]:
        """Arguments for ``omnilearned.network.PET2``."""
        return {
            "input_dim": self.input_dim,
            "use_int": self.use_int,
            "local_int": self.local_int,
            "int_type": self.int_type,
            "conditional": self.conditional,
            "cond_dim": self.cond_dim,
            "pid": self.pid,
            "pid_dim": self.pid_dim,
            "add_info": self.add_info,
            "add_dim": self.add_dim,
            "mode": self.mode,
            "num_classes": self.num_classes,
            "num_gen_classes": self.num_gen_classes,
            "base_dim": self.base_dim,
            "num_transformers": self.num_transformers,
            "num_transformers_head": self.num_transformers_head,
            "num_tokens": self.num_tokens,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "K": self.K,
            "skip": self.skip,
            "num_coord": self.num_coord,
            "use_local": self.use_local,
        }


def infer_pet2_config(checkpoint: Mapping[str, Any]) -> Pet2Config:
    """Infer ``PET2`` arguments from an OmniLearned checkpoint dictionary."""
    if "body" not in checkpoint:
        raise CheckpointInferenceError(
            f"checkpoint has no 'body' entry; keys are {sorted(checkpoint)}"
        )
    body = checkpoint["body"]
    classifier = checkpoint.get("classifier_head")
    generator = checkpoint.get("generator_head")
    notes: list[str] = []

    # --- core dimensions -----------------------------------------------------
    try:
        num_tokens, base_dim = body["token"].shape[1], body["token"].shape[2]
        input_dim = body["embed.mlp.fc1.weight"].shape[1]
        hidden = body["embed.mlp.fc1.weight"].shape[0]
    except KeyError as exc:  # pragma: no cover - malformed checkpoint
        raise CheckpointInferenceError(f"missing expected body key: {exc}") from exc

    if hidden % base_dim:
        raise CheckpointInferenceError(
            f"embed hidden width {hidden} is not a multiple of base_dim {base_dim}"
        )
    mlp_ratio = hidden // base_dim

    num_transformers = _count_blocks(body, "in_blocks")
    skip = _has(body, "out_blocks")
    if skip:
        # PET_body splits the budget in half when `skip` is on.
        num_transformers += _count_blocks(body, "out_blocks")

    # --- optional feature blocks --------------------------------------------
    use_int = _has(body, "interaction.")
    if use_int:
        num_heads = body["interaction.mlp.fc2.weight"].shape[0]
    else:
        raise CheckpointInferenceError(
            "no interaction block: num_heads cannot be inferred from the weights "
            "(attention packs all heads into one projection). Supply it explicitly."
        )

    use_local = _has(body, "local_physics.")
    local_int = False
    if use_local:
        k_features = body["local_physics.mlp.fc1.weight"].shape[1]
        # LocalEmbeddingBlock: k_features = in_features (+3 when local_int).
        if k_features == input_dim + 3:
            local_int = True
        elif k_features != input_dim:
            raise CheckpointInferenceError(
                f"local_physics input width {k_features} matches neither "
                f"input_dim={input_dim} nor input_dim+3"
            )

    conditional = _has(body, "cond_embed.")
    cond_dim = body["cond_embed.0.fc1.weight"].shape[1] if conditional else 3

    pid = _has(body, "pid_embed.")
    pid_dim = body["pid_embed.0.weight"].shape[0] if pid else 9

    add_info = _has(body, "add_embed.")
    add_dim = body["add_embed.0.fc1.weight"].shape[1] if add_info else 4

    use_time = _has(body, "time_embed.")

    # --- heads ---------------------------------------------------------------
    if classifier is None:
        raise CheckpointInferenceError("checkpoint has no 'classifier_head'")
    num_classes = classifier["out.weight"].shape[0]
    num_transformers_head = _count_blocks(classifier, "in_blocks")

    fc_in = classifier["fc.fc1.weight"].shape[1]
    if fc_in != num_tokens * base_dim:
        raise CheckpointInferenceError(
            f"classifier fc expects {fc_in} inputs but num_tokens*base_dim = "
            f"{num_tokens * base_dim}"
        )

    num_gen_classes = 1
    if generator is not None:
        gen_out = generator["out.weight"].shape[0]
        # PET_generator output_size == input_dim when num_gen_classes == 1
        # (diffusion), otherwise it equals num_gen_classes (segmentation/ftag).
        num_gen_classes = 1 if gen_out == input_dim else gen_out
        gen_blocks = _count_blocks(generator, "in_blocks")
        if gen_blocks != num_transformers_head:
            notes.append(
                f"generator has {gen_blocks} blocks but classifier has "
                f"{num_transformers_head}; PET2 builds both from "
                "num_transformers_head"
            )

    # --- mode ----------------------------------------------------------------
    # `use_time` is not a PET2 argument: it is derived as
    # `mode in ("generator", "pretrain")`. A body carrying a time embedding must
    # therefore be rebuilt in one of those modes, or both the parameter set and
    # the token layout silently differ from the trained model.
    if classifier is not None and generator is not None:
        # Both `pretrain` and `ftag` build two heads; only the generative modes
        # ("generator", "pretrain") add the diffusion-time embedding, so the time
        # embedding is what separates them.
        if use_time:
            mode = "pretrain"
        else:
            mode = "ftag"
            notes.append(
                "both heads present but no time embedding: inferring mode='ftag'. "
                "num_gen_classes then denotes track-origin classes, not diffusion."
            )
    elif generator is not None:
        mode = "generator"
    else:
        mode = "classifier"
        if use_time:
            raise CheckpointInferenceError(
                "body carries a time embedding but the checkpoint has no "
                "generator head; cannot determine mode"
            )

    notes.append(
        "K, num_coord and int_type leave no trace in the weights; using training "
        f"CLI defaults {UNINFERABLE_DEFAULTS}. Validate via task accuracy."
    )
    notes.append(
        "Parameter-free constructor choices (act_layer=GELU, norm_layer="
        "DynamicTanh, dropout rates) are assumed at their upstream defaults; a "
        "strict load cannot distinguish them."
    )

    return Pet2Config(
        input_dim=input_dim,
        base_dim=base_dim,
        num_transformers=num_transformers,
        num_transformers_head=num_transformers_head,
        num_tokens=num_tokens,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_int=use_int,
        local_int=local_int,
        use_local=use_local,
        skip=skip,
        conditional=conditional,
        cond_dim=cond_dim,
        pid=pid,
        pid_dim=pid_dim,
        add_info=add_info,
        add_dim=add_dim,
        num_classes=num_classes,
        num_gen_classes=num_gen_classes,
        mode=mode,
        notes=notes,
    )
