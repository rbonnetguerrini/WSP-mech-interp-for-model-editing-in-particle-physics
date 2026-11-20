"""Readers, SAE settings and edit recipes of the SAE-guided repair (paper Sec. 3.2, App. B.2, App. E).

A reader is written `b<block>-<head>` for a body attention head and `c<block>-<head>` for a classifier attention
head (paper notation b7/6, c1/2). The SAE of a reader reads the particle tokens at the input of the reader's
attention block: the norm1 output of a body block, or the body output for a classifier block.

Data-record layout (also used for the outputs of these scripts, under $RESULTS_DIR):
  sae/<size name>/<reader>/latents<N>_top<k>_seed<s>.pt (+ .json: latent identification)
  edits/sae/<size name>/<edit name>.npz
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

SIZE_NAMES = {"s": "small", "m": "medium", "l": "large"}

# reader -> (queries whose attention is measured, active latents k). queries "all" = summary and real-particle
# queries (used for b4-0, whose attention to soft neutrals comes from the particle queries), else summary only.
# Dictionary size: 32 x the token width (small 4096, medium 16384, large 32768 latents).
READERS = {
    "s": {"b7-6": ("summary", 32), "b4-0": ("all", 32), "c1-2": ("summary", 32)},
    "m": {"b7-0": ("summary", 32), "b11-7": ("summary", 64), "b11-0": ("summary", 64)},
    "l": {"b19-11": ("summary", 32), "b18-17": ("summary", 32), "c0-2": ("summary", 64), "c3-24": ("summary", 64)},
}
EXPANSION = 32
CAPTURE_BATCH = {"s": 256, "m": 256, "l": 32}  # jets per forward pass when capturing activations

# published SAE edits: name -> readers, edited one after the other (rank-one key edit at each, alpha = 1)
EDITS = {
    "s": {"one_reader_b7-6": ("b7-6",), "two_readers_b7-6_b4-0": ("b7-6", "b4-0"),
          "three_readers_b7-6_b4-0_c1-2": ("b7-6", "b4-0", "c1-2")},
    "m": {"one_reader_b7-0": ("b7-0",), "three_readers_b7-0_b11-7_b11-0": ("b7-0", "b11-7", "b11-0")},
    "l": {"one_reader_b19-11": ("b19-11",), "two_readers_b19-11_b18-17": ("b19-11", "b18-17"),
          "four_readers_b19-11_b18-17_c0-2_c3-24": ("b19-11", "b18-17", "c0-2", "c3-24")},
}
SELECTED = {"s": "three_readers_b7-6_b4-0_c1-2", "m": "three_readers_b7-0_b11-7_b11-0",
            "l": "four_readers_b19-11_b18-17_c0-2_c3-24"}


def parse_reader(reader: str) -> tuple[str, int]:
    """'b7-6' -> ('body.7', 6); 'c1-2' -> ('classifier.1', 2)."""
    m = re.fullmatch(r"([bc])(\d+)-(\d+)", reader)
    if not m:
        raise ValueError(f"reader {reader!r}: expected b<block>-<head> or c<block>-<head>")
    return ("body" if m.group(1) == "b" else "classifier") + "." + m.group(2), int(m.group(3))


def key_tensor(reader: str) -> str:
    """State-dict key of the fused query/key/value weights of the reader's attention block."""
    where, _ = parse_reader(reader)
    part, block = where.split(".")
    return f"{part}.in_blocks.{block}.attn.in_proj_weight"


def reader_settings(size: str, reader: str) -> tuple[str, int]:
    """(queries, k) of a reader of the paper, or ('summary', 32) for any other head."""
    return READERS[size].get(reader, ("summary", 32))


def sae_name(latents: int, k: int, seed: int) -> str:
    return f"latents{latents}_top{k}_seed{seed}"


def sae_root(path: Path | None = None) -> Path:
    """Folder with the SAE checkpoints: --sae-dir, else the data record's sae/ folder next to $EDITS_DIR."""
    if path is not None:
        return Path(path)
    from softrepair import paths
    return paths.edits_dir().parent / "sae"


def sae_path(root: Path, size: str, reader: str, seed: int = 0, latents: int | None = None,
             k: int | None = None) -> Path:
    """Checkpoint of the paper's SAE of one reader (default dictionary size and k of that reader)."""
    if latents is None or k is None:
        d = root / SIZE_NAMES[size] / reader
        hits = sorted(d.glob(f"latents*_top*_seed{seed}.pt"))
        if latents is not None:
            hits = [h for h in hits if h.name.startswith(f"latents{latents}_")]
        if k is not None:
            hits = [h for h in hits if f"_top{k}_" in h.name]
        if len(hits) != 1:
            # several dictionary sizes (the small b7-6 sweep): the paper's default is 32 x width
            default = [h for h in hits if h.name.startswith(("latents4096_", "latents16384_", "latents32768_"))]
            if len(default) != 1:
                raise FileNotFoundError(f"no unique SAE checkpoint for {size} {reader} seed {seed} in {d}: {hits}")
            hits = default
        return hits[0]
    return root / SIZE_NAMES[size] / reader / (sae_name(latents, k, seed) + ".pt")


def identification(sae_pt: Path) -> dict:
    """The latent-identification record written next to an SAE checkpoint."""
    return json.loads(Path(str(sae_pt)).with_suffix(".json").read_text())


def best_latents(rec: dict, top_k: int = 1) -> list[int]:
    """Latent(s) used by the edit: the identified latent (most selective, ties within 0.02 broken by key alignment;
    `best_latent_rule`), or for top_k > 1 the top_k most selective latents that pass the identification criteria.
    Records without `best_latent_rule` (no tie in the top 15) use the most selective identified latent."""
    found = set(rec["found"])
    ranked = [int(x["latent"]) for x in rec["top_latents"] if int(x["latent"]) in found]
    latents = ranked[:top_k] if ranked else [int(rec["top_latents"][0]["latent"])]
    if top_k == 1 and "best_latent_rule" in rec:
        latents = [int(rec["best_latent_rule"]["best"])]
    return latents


def set_size_env(size: str) -> None:
    os.environ["MODEL_SIZE"] = size
    if size == "l" and not os.environ.get("SOFTREPAIR_MAX_BATCH"):
        os.environ["SOFTREPAIR_MAX_BATCH"] = "32"
