"""All file locations, read from environment variables (see config.example.sh).

  OMNILEARNED_CHECKPOINTS  directory with best_model_pretrain_{s,m,l}.pt (the pretrained OmniLearned models)
  JETCLASS_DIR             directory with the JetClass validation files JetClass_<i>.h5
  JETCLASS2_DIR            directory with the JetClass-II files JetClass2_<i>.h5
  DATA_DIR                 prepared jet sets: <DATA_DIR>/<set>/{clean,labels,<variant>,<variant>.ok}.npy + meta.json
  EDITS_DIR                published edits and SAE checkpoints
  RESULTS_DIR              outputs (scores, caches, figures)

Each function reads its variable at call time, so a script may set them after import.
"""

from __future__ import annotations

import os
from pathlib import Path

VARIABLES = ("OMNILEARNED_CHECKPOINTS", "JETCLASS_DIR", "JETCLASS2_DIR", "DATA_DIR", "EDITS_DIR", "RESULTS_DIR")


def _get(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set (see config.example.sh)")
    return Path(value).expanduser()


def checkpoints_dir() -> Path:
    return _get("OMNILEARNED_CHECKPOINTS")


def checkpoint(size: str) -> Path:
    """Pretrained OmniLearned checkpoint of one size: s, m or l."""
    return checkpoints_dir() / f"best_model_pretrain_{size}.pt"


def jetclass_dir() -> Path:
    return _get("JETCLASS_DIR")


def jetclass2_dir() -> Path:
    return _get("JETCLASS2_DIR")


def data_dir() -> Path:
    return _get("DATA_DIR")


def edits_dir() -> Path:
    return _get("EDITS_DIR")


def results_dir() -> Path:
    return _get("RESULTS_DIR")
