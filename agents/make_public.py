#!/usr/bin/env python3
"""Public copies of the frozen task descriptions: the exact text as sent, with machine-specific paths replaced
by documented placeholders. The originals are the record of what was sent and are never edited.

The substitution table holds the machine-specific strings, so it is not part of this repository. It is a JSON
file {"rules": [[kind, pattern, placeholder], ...], "forbidden": [string, ...]}, kind "literal" or "regex",
applied in order (longest literal first, so that a prefix never pre-empts a longer match). "forbidden" lists
strings that must not remain in a public copy.

    python agents/make_public.py --table TABLE.json --map MAP.json [--out agents/task_descriptions]
MAP.json: {"<public file name>": "<path of the frozen original>", ...}. Writes the public copies and
SHA256SUMS (sha256 of the original as sent and of the public copy) and exits 1 if a forbidden string is left.

Placeholders (the same in the task descriptions and in the run records, see agents/scrub_records.py):
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLACEHOLDERS = {
    "{RUN}": "the run folder (the agent may write there)",
    "{INPUTS}": "the run's read-only inputs folder",
    "{RUN_ROOT}": "the parent folder of {INPUTS} and {RUN}",
    "{PYTHON}": "the agent's Python interpreter (with the audit hook)",
    "{AGENT_ENV}": "the virtual environment of {PYTHON}",
    "{ENV}": "the analysis Python environment ({ENV_PYTHON} its interpreter)",
    "{BASE_ENV}": "the base Python installation under {ENV}",
    "{OMNILEARNED_SRC}": "the installed OmniLearned package",
    "{REPO}": "the analysis code repository",
    "{RESULTS}": "the analysis output folder",
    "{RECORDS}": "the experimenter's records folder of one variant (outside every run folder)",
    "{RUNS_DIR}": "the root of the isolated run folders",
    "{HARNESS_TMP}": "the agent harness's own temporary folder of the session (background-task output)",
    "{PROJECT_SLUG}": "the harness's project name derived from the run folder path",
    "{LOCAL_TMP}": "the node-local temporary file system",
    "{HOME}": "the user's home folder",
    "{WORK}": "the project storage area",
    "{SCRATCH}": "the user's scratch storage area",
    "{SITE}": "the computing centre's name or file system",
    "{USER}": "the user name",
    "{EMAIL}": "an e-mail address",
    "{ACCOUNT}": "the compute-allocation account",
    "{NODE}": "a compute node host name",
    "{LOGIN_NODE}": "a login node host name",
    "{IP}": "an IP address",
    "{JOB}": "a Slurm job id",
    "{OTHER_PROJECT}": "an unrelated project of the user",
}
__doc__ += "".join(f"  {k:18s} {v}\n" for k, v in PLACEHOLDERS.items())


def load_table(path: Path) -> tuple[list[tuple[str, object, str]], list[str]]:
    t = json.loads(Path(path).read_text())
    rules = []
    for kind, pat, rep in t["rules"]:
        if kind not in ("literal", "regex"):
            raise ValueError(f"rule kind {kind!r}")
        rules.append((kind, re.compile(pat) if kind == "regex" else pat, rep))
    return rules, list(t.get("forbidden", []))


def apply(text: str, rules) -> str:
    for kind, pat, rep in rules:
        text = pat.sub(rep.replace("\\", "\\\\"), text) if kind == "regex" else text.replace(pat, rep)
    return text


def leftovers(text: str, forbidden) -> list[str]:
    low = text.lower()
    return [f for f in forbidden if f.lower() in low]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "task_descriptions")
    a = ap.parse_args()
    rules, forbidden = load_table(a.table)
    mapping = json.loads(a.map.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    lines, bad = [], []
    for name, src in mapping.items():
        raw = Path(src).read_bytes()
        text = apply(raw.decode(), rules)
        (a.out / name).write_text(text)
        lines.append(f"# original {sha(raw)}  {name}\n{sha(text.encode())}  {name}\n")
        bad += [f"{name}: {f}" for f in leftovers(text, forbidden)]
    head = ("# '# original' lines: sha256 of the frozen text as sent (as recorded at launch); the next line: sha256 of\n"
            "# the public copy in this folder (sha256sum -c SHA256SUMS). They are equal when no path was replaced.\n")
    (a.out / "SHA256SUMS").write_text(head + "".join(lines))
    print("public copies:", ", ".join(mapping))
    print("leftover forbidden strings:", bad or "none")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
