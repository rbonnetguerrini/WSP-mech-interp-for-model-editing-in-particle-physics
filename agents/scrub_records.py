#!/usr/bin/env python3
"""Deterministic scrubber of the agent-run records (transcripts, notes, requests, responses, ledgers, scores,
audit logs): replaces machine identifiers with the placeholders of agents/make_public.py. Content is never
rewritten; only paths, user, host and account names, e-mail addresses, job ids and internal run ids are replaced.

The table holds the machine-specific strings and is therefore not part of this repository. JSON:
  {"runs_root": "<root of the isolated run folders>",
   "runs": {"<internal run id>": "<public label>", ...},       e.g. "flaw_informed-medium-run1"
   "jobs": ["<slurm job id>", ...],
   "rules": [[kind, pattern, placeholder], ...],               kind "literal" | "regex", applied in order
   "forbidden": ["<string that must not remain>", ...]}

Order of the substitutions for one run:
  1. the run's own folders: <runs_root>/<id>/run -> {RUN}, <runs_root>/<id>/inputs -> {INPUTS}, <runs_root>/<id>
     -> {RUN_ROOT}; the folders of other runs: <runs_root>/<id> -> {RUNS_DIR}/<label>
  2. the table rules (site paths, host names, accounts, user name)
  3. the remaining internal run ids -> public labels (the own id -> this run's label)
  0. (first) e-mail addresses -> {EMAIL} (the harness's own noreply address is kept)
  4. job ids -> {JOB}

    python agents/scrub_records.py --table TABLE.json --own RUN_ID IN OUT     (text, .json, .jsonl, .gz)
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from pathlib import Path

from make_public import apply, leftovers, load_table

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|net|edu|eu|it|ch|de|fr|dk|uk|io|ai|gov)\b")
KEEP_EMAIL = ("noreply@anthropic.com",)
B64 = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


class Scrubber:
    def __init__(self, table: Path, own: str | None):
        t = json.loads(Path(table).read_text())
        self.rules, self.forbidden = load_table(table)
        root = t["runs_root"].rstrip("/")
        runs = t.get("runs", {})
        first = []
        if own:
            first += [("literal", f"{root}/{own}/run", "{RUN}"), ("literal", f"{root}/{own}/inputs", "{INPUTS}"),
                      ("literal", f"{root}/{own}", "{RUN_ROOT}")]
        first += [("literal", f"{root}/{rid}", "{RUNS_DIR}/" + lab) for rid, lab in runs.items() if rid != own]
        self.first = first
        ids = sorted(runs, key=len, reverse=True)
        self.ids = re.compile(r"(?<![0-9A-Za-z])(" + "|".join(map(re.escape, ids)) + r")(?![0-9A-Za-z])") if ids else None
        self.runs = runs
        jobs = sorted(set(map(str, t.get("jobs", []))), key=len, reverse=True)
        self.jobs = re.compile(r"(?<!\d)(" + "|".join(jobs) + r")(?!\d)") if jobs else None

    def text(self, s: str) -> str:
        s = EMAIL.sub(lambda m: m.group(0) if m.group(0) in KEEP_EMAIL else "{EMAIL}", s)
        s = apply(s, self.first)
        s = apply(s, self.rules)
        if self.ids is not None:
            s = self.ids.sub(lambda m: self.runs[m.group(1)], s)
        if self.jobs is not None:
            s = self.jobs.sub("{JOB}", s)
        return s

    def left(self, s: str) -> list[str]:
        """Forbidden strings left (case-insensitive), ignoring chance matches inside base64 data (images)."""
        out = []
        for f in leftovers(s, self.forbidden):
            for m in re.finditer(re.escape(f), s, re.I):
                a, b = m.start(), m.end()
                while a > 0 and s[a - 1] in B64:
                    a -= 1
                while b < len(s) and s[b] in B64:
                    b += 1
                if b - a < 64:
                    out.append(f)
                    break
        return out

    def file(self, src: Path, dst: Path) -> list[str]:
        """Scrubs one file (gzip in, gzip out with mtime 0); returns the forbidden strings left."""
        raw = src.read_bytes()
        gz = src.suffix == ".gz"
        text = (gzip.decompress(raw) if gz else raw).decode("utf-8", errors="surrogateescape")
        out = self.text(text)
        b = out.encode("utf-8", errors="surrogateescape")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if gz or dst.suffix == ".gz":
            buf = io.BytesIO()
            with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as g:
                g.write(b)
            b = buf.getvalue()
        dst.write_bytes(b)
        return self.left(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--own", default=None, help="internal id of the run the file belongs to")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    a = ap.parse_args()
    bad = Scrubber(a.table, a.own).file(a.src, a.dst)
    print("leftover forbidden strings:", bad or "none")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
