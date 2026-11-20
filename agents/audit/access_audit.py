#!/usr/bin/env python3
"""File-access audit of an isolated agent run: list every attempt to reach outside the permitted folders.

Tracks, never blocks. Reads a Claude Code subagent transcript (.jsonl) or a Codex CLI log (plain text) and
reports each tool call that names a path outside the allowed roots, every search/listing command, and
whether the call succeeded or was refused.

usage: access_audit.py TRANSCRIPT --allow ROOT [--allow ROOT ...] [--hook-log LOG] [--scan-dir DIR] [--json OUT.json]

Categories:
  OUTSIDE   a tool call (Read/Write/Edit/Glob/Grep/NotebookEdit, or a Bash command) names a path outside the roots
  SEARCH    a Bash command that searches or lists (find, ls, grep -r, rg, locate, tree, du) anything outside the roots
  ESCAPE    relative escapes from the roots ('..', '~', $HOME) in a Bash command
  PARENT    a path that is a parent folder of a root (e.g. listing the run's top folder): low severity
  HARNESS   the Claude Code harness's own background-task output files (<tmp>/claude-*/.../tasks/): low
  WEB       WebFetch / WebSearch calls
  AGENT     attempts to spawn agents or message other sessions
  SCRIPT    a file the agent writes (Write/Edit content, or a script left in --scan-dir) names an outside path or
            uses path-building / process tools (expanduser, environ, glob over '/', subprocess, os.system)
  SYSINFO   commands that reveal machine state without naming a path (squeue, scontrol, sacct, ps, env, printenv,
            git, history, ln -s, chmod, curl, wget), or the interpreter run with -I / -E / -S (skips the hook)
  PYHOOK    rows of the Python audit-hook log (--hook-log): opens / listings / commands from the agent's scripts
Paths always allowed: the roots, /dev/null, and the python interpreter named with --allow-exe.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import os
TOP = {d for d in os.listdir("/")} - {""}  # real top-level folders: '/4' or '/np.sum' in code are not paths
PATH_RE = re.compile(r"(?<![\w.)\]])(/(?:[^\s'\"|;&<>()`$,]|\\ )+)")
SEARCH_CMDS = {"find", "ls", "locate", "tree", "du", "rg", "fd", "stat", "realpath", "readlink"}
FILE_TOOLS = {"Read": "file_path", "Write": "file_path", "Edit": "file_path", "NotebookEdit": "notebook_path",
              "Glob": "path", "Grep": "path"}


def classify(p: str, roots: list[str], exes: list[str]) -> str | None:
    """None if allowed, else the category."""
    p = os.path.normpath(p.rstrip(";,:'\")"))
    if p in ("/dev/null", "/dev/stdout", "/dev/stderr", "/bin/bash", "/bin/sh", "/usr/bin/env", "/usr/bin/bash") or any(p == e or p.startswith(e.rstrip("/") + "/") for e in exes):
        return None
    if any(p == r.rstrip("/") or p.startswith(r.rstrip("/") + "/") for r in roots):
        return None
    if any(r.startswith(p.rstrip("/") + "/") for r in roots):
        return "PARENT"
    if re.match(r"/[^/]+/claude-[^/]+/.+/tasks/", p):  # the harness's temporary folder (node-local tmp)
        return "HARNESS"
    return "OUTSIDE"


def allowed(p: str, roots: list[str], exes: list[str]) -> bool:
    return classify(p, roots, exes) is None


def paths_in(cmd: str) -> list[str]:
    """Absolute filesystem paths named in a command; relative paths with '..' resolved against the last cd."""
    out, base = [], None
    try:
        toks = shlex.split(cmd, posix=True)
    except ValueError:
        toks = cmd.split()
    for i, tok in enumerate(toks):
        t = tok.rstrip(";,:'\")")
        if i and toks[i - 1] == "cd" and t.startswith("/"):
            base = t
        if t.startswith("/") and t.split("/")[1] in TOP:
            out.append(t)
        elif ".." in t.split("/") and base:
            out.append(os.path.normpath(os.path.join(base, t)))
        elif t.startswith("~/") or "$HOME" in t or "${HOME}" in t:
            out.append(os.path.expanduser(t.replace("${HOME}", "~").replace("$HOME", "~")))
    for m in PATH_RE.findall(cmd):
        m = m.replace("\\ ", " ").rstrip(";,:'\")")
        if m.split("/")[1] in TOP:
            out.append(m)
    return sorted(set(out))


SYSINFO_CMDS = {"squeue", "scontrol", "sacct", "sinfo", "ps", "env", "printenv", "git", "history",
                "curl", "wget", "ssh", "scp", "lsof", "pstree", "top"}
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?.*?\n.*?\n\1\b", re.S)


def command_words(cmd: str) -> set[str]:
    """First word of every command in a shell line, heredoc bodies removed."""
    body = HEREDOC.sub(" ", cmd)
    out = set()
    for seg in re.split(r"[;&|\n]+|\$\(|`", body):
        toks = [t for t in seg.strip().split() if not re.match(r"^\w+=", t)]
        while toks and toks[0] in ("timeout", "nohup", "time", "sudo", "exec", "xargs", "then", "do", "else", "!"):
            toks = toks[2:] if toks[0] == "timeout" and len(toks) > 1 else toks[1:]
        if toks:
            out.add(toks[0].split("/")[-1])
    return out
RISKY_CODE = re.compile(r"expanduser|os\.environ|getenv|subprocess|os\.system|os\.popen|glob\.glob\(\s*['\"]/|"
                        r"os\.walk\(\s*['\"]/|\.\./|Path\.home|/proc/")


def audit_code(text: str, roots, exes) -> list[tuple[str, str]]:
    """Paths and risky calls inside code or text the agent writes."""
    hits = []
    for m in set(PATH_RE.findall(text or "")):
        m = m.replace("\\ ", " ").rstrip(";,:'\")")
        if len(m.split("/")) > 1 and m.split("/")[1] in TOP:
            c = classify(m, roots, exes)
            if c in ("OUTSIDE",):
                hits.append(("SCRIPT", f"path {m}"))
    for m in sorted(set(RISKY_CODE.findall(text or ""))):
        hits.append(("SCRIPT", f"uses {m}"))
    return hits


def audit_bash(cmd: str, roots, exes) -> list[tuple[str, str]]:
    hits = []
    cw = command_words(cmd)
    sysw = sorted((cw & SYSINFO_CMDS) | ({"ln -s"} if "ln" in cw and re.search(r"\bln\s+-\w*s", cmd) else set()))
    if sysw:
        hits.append(("SYSINFO", f"{','.join(sysw)}: {cmd[:160]}"))
    if re.search(r"python[\d.]*\s+(-\w*[IES]\b)", cmd):
        hits.append(("SYSINFO", f"python flag that skips the audit hook: {cmd[:160]}"))
    for p in paths_in(cmd):
        c = classify(p, roots, exes)
        if c:
            hits.append((c, p))
    words = re.split(r"[\s;|&()]+", cmd)
    searches = any(w in SEARCH_CMDS for w in words) or re.search(r"\bgrep\s+(-\w*[rR]|--recursive)", cmd)
    if searches and any(c == "OUTSIDE" for c, _ in hits):
        hits.append(("SEARCH", cmd[:200]))
    return hits


def audit_claude(path: Path, roots, exes) -> list[dict]:
    calls, results = {}, {}
    for line in path.open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (d.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if c.get("type") == "tool_use":
                calls[c["id"]] = (d.get("timestamp", ""), c["name"], c.get("input") or {})
            elif c.get("type") == "tool_result":
                txt = c.get("content")
                txt = json.dumps(txt) if not isinstance(txt, str) else txt
                results[c.get("tool_use_id")] = ("error" if c.get("is_error") else "ok", txt[:160])
    out = []
    for cid, (ts, name, inp) in calls.items():
        hits = []
        if name in FILE_TOOLS:
            p = inp.get(FILE_TOOLS[name])
            c = classify(p, roots, exes) if p else None
            if c:
                hits.append((c, p))
            if name == "Glob" and not p:
                hits.append(("SEARCH", f"Glob {inp.get('pattern')} (no path: session working directory)"))
            if name == "Grep" and not p:
                hits.append(("SEARCH", f"Grep {inp.get('pattern')} (no path: session working directory)"))
        if name in ("Write", "Edit", "NotebookEdit"):
            hits += audit_code(inp.get("content") or inp.get("new_string") or inp.get("new_source") or "", roots, exes)
        if name == "Bash":
            hits += audit_bash(inp.get("command", ""), roots, exes)
        elif name in ("WebFetch", "WebSearch"):
            hits.append(("WEB", json.dumps(inp)[:200]))
        elif name in ("Agent", "Task", "SendMessage", "ListAgents"):
            hits.append(("AGENT", json.dumps(inp)[:200]))
        for cat, what in hits:
            st, snip = results.get(cid, ("no result", ""))
            out.append({"time": ts, "tool": name, "category": cat, "target": what, "status": st, "result": snip})
    return out


def audit_codex(path: Path, roots, exes) -> list[dict]:
    out = []
    for i, line in enumerate(path.open(errors="replace")):
        if re.search(r"\bexec\b|\$ |bash -lc|/bin/bash", line):
            for cat, what in audit_bash(line.strip(), roots, exes):
                out.append({"time": f"line {i + 1}", "tool": "codex-exec", "category": cat, "target": what,
                            "status": "see log", "result": ""})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--allow", action="append", required=True, help="permitted root folder (repeatable)")
    ap.add_argument("--allow-exe", action="append", default=[], help="permitted executable or env prefix")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--hook-log", type=Path, default=None, help="JSON-lines log of sitecustomize.py")
    ap.add_argument("--scan-dir", type=Path, action="append", default=[], help="scan scripts left in this folder")
    a = ap.parse_args()
    rows = (audit_claude if a.transcript.suffix == ".jsonl" else audit_codex)(a.transcript, a.allow, a.allow_exe)
    if a.hook_log and a.hook_log.exists():
        for line in a.hook_log.open():
            try:
                h = json.loads(line)
            except json.JSONDecodeError:
                continue
            what = h.get("path") or h.get("module") or " ".join(h.get("cmd", []))
            rows.append({"time": h.get("t", ""), "tool": f"python:{h.get('event')}", "category": "PYHOOK",
                         "target": what, "status": "logged", "result": " ".join(map(str, h.get("argv", [])))})
    for d in a.scan_dir:
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix in (".py", ".sh", ".md", ".txt", ".json") and f.stat().st_size < 2_000_000:
                for cat, what in audit_code(f.read_text(errors="replace"), a.allow, a.allow_exe):
                    rows.append({"time": "", "tool": f"file:{f}", "category": cat, "target": what,
                                 "status": "on disk", "result": ""})
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"{a.transcript.name}: {len(rows)} flagged calls {cats or ''}")
    for r in rows:
        print(f"  [{r['category']}] {r['tool']} {r['status']}: {r['target']}")
    if a.json:
        a.json.write_text(json.dumps({"transcript": str(a.transcript), "allow": a.allow, "counts": cats, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
