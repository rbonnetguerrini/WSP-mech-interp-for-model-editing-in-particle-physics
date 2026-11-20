#!/usr/bin/env python
"""Recreate the links of the scoring sets after downloading the jet sets of the data record.

practice_ir/ and test_ir/ hold only the modified copies that are not already in practice/ and test/; the code reads
every variant of a scoring set from its own folder (core.load_set). This script links each missing file (clean jets,
labels and every variant listed in the set's meta.json) to the copy in practice/ or test/, with relative links, as
scoring.build does when it builds the sets from JetClass.

  python scripts/data/link_zenodo_sets.py [DATA_DIR]     (default: $DATA_DIR)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PAIRS = (("practice_ir", "practice"), ("test_ir", "test"))


def link(data_dir: Path) -> int:
    made = 0
    for name, src in PAIRS:
        d, s = data_dir / name, data_dir / src
        meta = json.loads((d / "meta.json").read_text())
        files = ["clean.npy", "labels.npy"] + [f"{v}{ext}" for v in meta["variants"] for ext in (".npy", ".ok.npy")]
        for f in files:
            if (d / f).exists():
                continue
            if not (s / f).exists():
                sys.exit(f"{name}/{f} is missing and {src}/{f} does not exist")
            if (d / f).is_symlink():  # dangling link from an earlier copy
                (d / f).unlink()
            os.symlink(Path("..") / src / f, d / f)
            made += 1
        missing = [f for f in files if not (d / f).exists()]
        print(f"{name}: {len(files) - len(missing)}/{len(files)} files present", flush=True)
        if missing:
            sys.exit(f"{name}: still missing {missing}")
    return made


def main() -> int:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("DATA_DIR", ""))
    if not str(data_dir) or not (data_dir / "practice").is_dir():
        sys.exit("usage: link_zenodo_sets.py DATA_DIR (the folder with practice/, test/, practice_ir/, test_ir/)")
    print(f"created {link(data_dir)} links", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
