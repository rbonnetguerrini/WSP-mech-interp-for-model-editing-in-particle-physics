#!/usr/bin/env python
"""Export a pretrained model's state dict as numpy (the weights file the agents received) plus a parameter manifest.

  python scripts/data/export_weights.py --size s --out DIR   -> DIR/weights_<size>.npz, DIR/param_manifest.json
(default DIR: RESULTS_DIR/agent_inputs_<size>). Keys are the adapter's state_dict keys (body.*, classifier.*,
generator.*), float32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from softrepair import paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="s", choices=("s", "m", "l"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ck = torch.load(paths.checkpoint(args.size), map_location="cpu", weights_only=False)
    state = {f"body.{k}": v for k, v in ck["body"].items()}
    state.update({f"classifier.{k}": v for k, v in ck["classifier_head"].items()})
    state.update({f"generator.{k}": v for k, v in ck["generator_head"].items()})
    out = args.out or paths.results_dir() / f"agent_inputs_{args.size}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"weights_{args.size}.npz", **{k: v.float().numpy() for k, v in state.items()})
    (out / "param_manifest.json").write_text(json.dumps(
        {k: {"shape": list(v.shape), "numel": int(v.numel())} for k, v in state.items()}, indent=1))
    print(len(state), "tensors,", sum(v.numel() for v in state.values()), "weights ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
