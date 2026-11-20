"""Settings shared by the fine-tuning scripts (paper Sec. 3.1 and App. B.1).

VARIANTS maps the paper's names of the fine-tuning variants (also the file names of the published edits,
`edits/finetuning/<size>/<variant>.npz`) to the options of finetune_lora.py. Every variant uses the loss
L_hold + lambda L_IC + mu L_phys with lambda = 30, mu = 10 (unless stated), 3000 Adam steps of 128 jets,
learning rate 1e-3 with a 100-step warm-up and cosine decay, on the 10,240 practice jets.

SIZE_DEFAULTS holds the settings the published medium and large edits were trained with: gradient accumulation in
chunks of 32 (medium) or 8 (large) jets, which gives the same update as the whole batch with less GPU memory, the
math attention kernel, and a cap on the inference batch used for the reference logits. The small edits were trained
with the whole batch at once and the default (fused) attention kernels.
"""

from __future__ import annotations

SCOPES = ("attention", "whole_body_and_classifier")

VARIANTS = {
    # Tab. ft_variants
    "attention_only_both_neutral_types": dict(scope="attention", particles="both"),  # selected
    "attention_only_photons_only": dict(scope="attention", particles="photons"),
    "whole_body_and_classifier_both_neutral_types": dict(scope="whole_body_and_classifier", particles="both"),
    # App. B.1, further tests
    "attention_only_photon_and_neutral_hadron_per_jet": dict(scope="attention", particles="both", both_per_jet=True),
    "whole_body_and_classifier_photons_only": dict(scope="whole_body_and_classifier", particles="photons"),
    "whole_body_and_classifier_photons_only_without_Lphys": dict(scope="whole_body_and_classifier",
                                                                  particles="photons", mu=0.0),
    "whole_body_and_classifier_photons_only_rank1": dict(scope="whole_body_and_classifier", particles="photons",
                                                         rank=1),
}
SELECTED = "attention_only_both_neutral_types"

# particle codes of the added soft neutral: 6 = photon, 7 = neutral hadron
PARTICLES = {"photons": (6,), "both": (6, 7)}

SIZE_DEFAULTS = {
    "s": dict(micro_batch=0, math_attention=False, max_batch=None),
    "m": dict(micro_batch=32, math_attention=True, max_batch=128),
    "l": dict(micro_batch=8, math_attention=True, max_batch=32),
}
SIZE_NAMES = {"s": "small", "m": "medium", "l": "large"}
