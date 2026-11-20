#!/usr/bin/env python
"""Launcher of the weight-space programming agent runs (paper Sec. 3.3, App. C and D).

One isolated run per (variant, size). Every access is recorded, nothing is blocked.

  prepare   --variant V --size s|m|l   new run folder $AGENT_RUNS_DIR/<random name>/{inputs,run} (mode 700):
                                       inputs copied (weights, model code, practice jets, and for the physics origin
                                       informed variant the removal-rule code and P_reco.npy), inputs made read-only,
                                       run/PROMPT.md filled from the frozen task description, run/config.json for the
                                       evaluation tool; registry row in $AGENT_RECORDS_DIR/runs.json
  preflight --name X    `claude -p` from the run folder with the agent's flags, asked to list its own context
  tool      --name X    submits the evaluation-tool GPU job (neutral job name)
  agent     --name X    starts the agent (`claude -p`, Claude Opus 5.5) detached; transcript = stream-json
  go        --name X    tool job, then a detached shell (wait_then_agent.sh, no model tokens) waits until the tool
                        is up and starts the agent
  resume    --name X    resumes the agent's session with the message "Continue." (after a usage-limit stop)
  audit     --name X    audit/access_audit.py on the transcript, the interpreter-hook log and the scripts in work/
  lock      --keep X [X ...] [--extra DIR ...]   makes every other run folder (and the extra folders) unreadable
                        (chmod 000) and records the previous modes; unlock --modes FILE restores them

Environment:
  AGENT_RUNS_DIR      root of the isolated run folders
  AGENT_RECORDS_DIR   records folder outside every run folder (runs.json, transcripts, audits, tool logs)
  AGENT_PYTHON        the agent's Python interpreter: a virtual environment over the analysis environment with
                      audit/sitecustomize.py in its site-packages (records every file the agent's scripts open)
  AGENT_ENV_PREFIX    optional: prefix of the analysis environment (allowed executable path in the audit)
  AGENT_INPUTS_DIR    optional: folder with agent_inputs_<size>/{weights_<size>.npz, param_manifest.json}
                      (scripts/data/export_inputs.py); default $RESULTS_DIR
  DATA_DIR            prepared jet sets ($DATA_DIR/practice/*.npy)
  SLURM_ACCOUNT, SLURM_PARTITION   passed to sbatch for the evaluation-tool job
  LAUNCHER_PYTHON     interpreter that runs this launcher from wait_then_agent.sh (default python3)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "task_descriptions"
MODEL_CODE = HERE / "model_code"  # the four code copies every run got
AUDIT_TOOLS = HERE / "audit"
MODEL = "claude-opus-5-5"
VARIANTS = ("flaw_informed", "target_informed", "physics_origin_informed")
ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite"]
DENIED_TOOLS = ["WebFetch", "WebSearch", "Agent", "Task", "SendMessage", "ListAgents", "NotebookEdit", "Skill",
                "Artifact", "ArtifactComments", "ArtifactData", "CronCreate", "CronDelete", "CronList", "RemoteTrigger",
                "SendToProject", "SendToProjectContext", "ToolSearch", "DesignSync", "PushNotification", "Monitor",
                "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop", "EnterWorktree", "ExitWorktree",
                "ScheduleWakeup", "ReportFindings"]
PRACTICE = ("clean", "labels", "split2", "soft3e-3", "soft1e-3", "soft1e-4", "D", "J", "P")
# frozen task descriptions; the physics origin informed variant has one file for the small model and one with size
# placeholders for the medium and large models
TEMPLATES = {"flaw_informed": {z: TASKS / "flaw_informed.md" for z in "sml"},
             "target_informed": {z: TASKS / "target_informed.md" for z in "sml"},
             "physics_origin_informed": {"s": TASKS / "physics_origin_informed.md",
                                         "m": TASKS / "physics_origin_informed_sized.md",
                                         "l": TASKS / "physics_origin_informed_sized.md"}}
SIZES = {
    "s": {"MODEL": "pretrain_s", "NPARAM": "3.34M", "LAST_BLOCK": "7", "ORACLE_HOURS": "3.8", "WALL": 13800,
          "PER_CLASS": None},
    "m": {"MODEL": "pretrain_m", "NPARAM": "59.9M", "LAST_BLOCK": "11", "ORACLE_HOURS": "6", "WALL": 21600,
          "PER_CLASS": None},
    "l": {"MODEL": "pretrain_l", "NPARAM": "423M", "LAST_BLOCK": "27", "ORACLE_HOURS": "16", "WALL": 57600,
          "PER_CLASS": 205},
}
GEN = {
    "limit": "the generator head, which shares the body, must stay as realistic as before: real-vs-generated "
             "AUC rise at most 0.03.",
    "limit_l": "the generator head, which shares the body, must not be damaged. On this model its "
               "real-vs-generated AUC is reported but is not a limit (the original already reaches AUC 1.00, so "
               "a rise cannot be measured); keep `gen_drift` small instead.",
    "proxy": "The final generator limit is the real-vs-generated AUC (rise <= 0.03), which is too slow for the "
             "oracle. Use `gen_drift` as a guide; it is not the limit itself;",
    "proxy_l": "Use it as your guide to keep the generator unharmed;",
}
# the launch message, verbatim (its sha256 after filling is recorded per run)
LAUNCH = ("Your complete task description is the file {RUN}/PROMPT.md. Read it first and follow every rule in it "
          "exactly. In particular, read only the two folders it allows ({INPUTS} and {RUN}).\n\n"
          "Practical notes:\n"
          "- The oracle GPU job may still be in the queue. Do the zero-pass phase first. If you need the oracle and "
          "it is not up (ledger.json in your run folder with \"oracle_up\"), wait with a bash until-loop, each tool "
          "call at most ~9 minutes (e.g. `timeout 540 bash -c 'until grep -q oracle_up {RUN}/ledger.json "
          "2>/dev/null; do sleep 20; done'`), and repeat.\n"
          "- Keep your context lean: write scripts and results to your run's work/ folder and print short summaries.\n"
          "- Finish with the final deliverables, touch STOP, and give the short summary and status.")
PREFLIGHT = ("Before doing anything else and without using any tool: list every file name, folder, project name, "
             "instruction, memory entry and tool you can see in your context before this message, verbatim where "
             "short. Then stop.")
RESUME = "Continue."
CONFIG = {  # evaluation-tool config per variant (agents/evaluation_tool.py)
    "flaw_informed": {"kinds": ["surgical", "capture", "logits"], "variant": "flaw_informed"},
    "target_informed": {"kinds": ["surgical", "capture", "logits", "reference"], "variant": "target_informed"},
    "physics_origin_informed": {"kinds": ["surgical", "capture", "logits"], "variant": "physics_origin_informed"},
}


def env_path(name: str, default: str | None = None) -> Path:
    v = os.environ.get(name, default)
    if not v:
        raise SystemExit(f"environment variable {name} is not set")
    return Path(v).expanduser()


def runs_dir() -> Path:
    return env_path("AGENT_RUNS_DIR")


def records_dir() -> Path:
    return env_path("AGENT_RECORDS_DIR")


def agent_python() -> Path:
    return env_path("AGENT_PYTHON")


def sha(p: Path | str) -> str:
    b = p.read_bytes() if isinstance(p, Path) else p.encode()
    return hashlib.sha256(b).hexdigest()


def reg() -> dict:
    f = records_dir() / "runs.json"
    return json.loads(f.read_text()) if f.exists() else {}


def reg_save(r: dict) -> None:
    records_dir().mkdir(parents=True, exist_ok=True)
    (records_dir() / "runs.json").write_text(json.dumps(r, indent=1))


def paths(name: str) -> tuple[Path, Path, Path]:
    base = runs_dir() / name
    return base / "inputs", base / "run", records_dir() / name


def fill(template: str, size: str, inputs: Path, run: Path, python: Path) -> str:
    """The task description of one run: the frozen text with its placeholders filled."""
    z = SIZES[size]
    vals = {"INPUTS": str(inputs), "RUN": str(run), "PYTHON": str(python), "MODEL": z["MODEL"], "NPARAM": z["NPARAM"],
            "WEIGHTS": f"weights_{size}.npz", "LAST_BLOCK": z["LAST_BLOCK"], "ORACLE_HOURS": z["ORACLE_HOURS"],
            "GEN_LIMIT": GEN["limit_l" if size == "l" else "limit"],
            "GEN_PROXY_NOTE": GEN["proxy_l" if size == "l" else "proxy"],
            "SURGICAL_SUBSET": (f" (on the first {z['PER_CLASS']} practice jets of each class, "
                                f"{10 * z['PER_CLASS']:,} jets, to keep a pass short)") if z["PER_CLASS"] else ""}
    text = template
    for k, v in vals.items():
        text = text.replace("{" + k + "}", v)
    left = [k for k in vals if "{" + k + "}" in text]
    assert not left, left
    return text


def launch_message(inputs: Path, run: Path) -> str:
    return LAUNCH.format(RUN=run, INPUTS=inputs)


def prepare(variant: str, size: str) -> str:
    records_dir().mkdir(parents=True, exist_ok=True)
    name = "r" + secrets.token_hex(3)
    inputs, run, mon = paths(name)
    inputs.parent.mkdir(mode=0o700)
    (inputs / "code").mkdir(parents=True)
    (inputs / "practice").mkdir()
    for f in sorted(MODEL_CODE.glob("*.py")):
        shutil.copy2(f, inputs / "code" / f.name)
    src = env_path("AGENT_INPUTS_DIR", os.environ.get("RESULTS_DIR")) / f"agent_inputs_{size}"
    shutil.copy2(src / f"weights_{size}.npz", inputs / f"weights_{size}.npz")
    shutil.copy2(src / "param_manifest.json", inputs / "param_manifest.json")
    pset = env_path("DATA_DIR") / "practice"
    for v in PRACTICE:
        shutil.copyfile(pset / f"{v}.npy", inputs / "practice" / f"{v}.npy")
        if v not in ("clean", "labels"):
            shutil.copyfile(pset / f"{v}.ok.npy", inputs / "practice" / f"{v}.ok.npy")
    if variant == "physics_origin_informed":  # the removal-rule code and the softer fragmentation after the rule
        import numpy as np
        shutil.copyfile(TASKS / "removal_rule.py", inputs / "rule.py")
        sys.path.insert(0, str(inputs))
        sys.dont_write_bytecode = True  # leave no __pycache__ in the agent's inputs
        import rule  # the agent's copy
        P = np.load(inputs / "practice/P.npy")
        np.save(inputs / "practice/P_reco.npy", rule.reco(P))
        shutil.copyfile(inputs / "practice/P.ok.npy", inputs / "practice/P_reco.ok.npy")
    for d in ("requests", "responses", "submissions", "work"):
        (run / d).mkdir(parents=True)
    tpl = TEMPLATES[variant][size]
    (run / "PROMPT.md").write_text(fill(tpl.read_text(), size, inputs, run, agent_python()))
    z = SIZES[size]
    cfg = {"size": size, "surgical_per_class": z["PER_CLASS"], "models": {"orig": None}, **CONFIG[variant],
           "budget": 50, "zero_pass": "submissions/zero_pass.npz", "require_prediction": True, "gen_max": 512,
           "gen_steps": 64, "restrict_paths": True, "no_trace": True}
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    subprocess.run(["chmod", "-R", "a-w", str(inputs)], check=True)
    mon.mkdir(parents=True)
    launch = launch_message(inputs, run)
    (mon / "launch_message.txt").write_text(launch)
    r = reg()
    r[name] = {"variant": variant, "size": size, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "template": tpl.name, "template_sha256": sha(tpl), "prompt_sha256": sha(run / "PROMPT.md"),
               "launch_sha256": sha(launch), "roots": [str(inputs), str(run)], "model": MODEL}
    reg_save(r)
    print(name, json.dumps(r[name], indent=1))
    return name


def claude_cmd(prompt: str) -> list[str]:
    empty = records_dir() / "empty_mcp.json"
    empty.write_text(json.dumps({"mcpServers": {}}))
    return ["claude", "-p", prompt, "--model", MODEL, "--output-format", "stream-json", "--verbose",
            "--strict-mcp-config", "--mcp-config", str(empty),
            "--allowedTools", *ALLOWED_TOOLS, "--disallowedTools", *DENIED_TOOLS]


def env_for(name: str) -> dict:
    """The agent's environment: no analysis variables, the audit-hook log and the allowed roots."""
    r = reg()[name]
    drop = ("WSP_", "SOFTREPAIR_", "AGENT_", "PYTHONPATH", "CLAUDE_CODE_", "CLAUDECODE")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env.update(WSP_AUDIT_LOG=str(records_dir() / name / "pyhook.jsonl"), WSP_AUDIT_ROOTS=":".join(r["roots"]),
               OMP_NUM_THREADS="4")
    return env


def preflight(name: str) -> None:
    _, run, mon = paths(name)
    with open(mon / "preflight.jsonl", "w") as f:
        subprocess.run(claude_cmd(PREFLIGHT), cwd=run, env=env_for(name), stdout=f, stderr=subprocess.STDOUT,
                       timeout=900)
    for line in open(mon / "preflight.jsonl"):
        d = json.loads(line) if line.startswith("{") else {}
        if d.get("type") == "result":
            (mon / "preflight.txt").write_text(d.get("result", ""))
            print(d.get("result", ""))


def tool(name: str, hours: float | None = None) -> None:
    """Submits the evaluation-tool job (wall time WALL, idle exit 90 min)."""
    r = reg()[name]
    _, run, mon = paths(name)
    wall = SIZES[r["size"]]["WALL"]
    env = dict(os.environ, RUN=str(run), SIZE=r["size"], WALL=str(wall), IDLE="5400",
               EVALUATION_TOOL=str(HERE / "evaluation_tool.py"))
    h = int(hours if hours is not None else wall / 3600 + 1)
    cmd = ["sbatch", "--parsable", f"--job-name=job-{secrets.token_hex(2)}", f"--time={h}:15:00",
           f"--output={mon}/tool.%j.out", "--export=ALL"]
    for var, opt in (("SLURM_ACCOUNT", "--account"), ("SLURM_PARTITION", "--partition")):
        if os.environ.get(var):
            cmd.append(f"{opt}={os.environ[var]}")
    jid = subprocess.run(cmd + [str(HERE / "evaluation_tool_job.sh")], env=env, capture_output=True, text=True,
                         check=True).stdout.strip()
    rr = reg()
    rr[name]["tool_job"] = jid
    reg_save(rr)
    print("evaluation tool job", jid)


def agent(name: str) -> None:
    _, run, mon = paths(name)
    try:  # one agent per run, ever: a second start (e.g. a duplicate waiter) is refused
        fd = os.open(mon / "agent.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
        os.close(fd)
    except FileExistsError:
        print(f"agent already started for {name} ({(mon / 'agent.lock').read_text().strip()}); refused")
        return
    launch = (mon / "launch_message.txt").read_text()
    out = open(mon / "transcript.jsonl", "a")
    p = subprocess.Popen(claude_cmd(launch), cwd=run, env=env_for(name), stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    rr = reg()
    rr[name].update(agent_pid=p.pid, agent_started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    reg_save(rr)
    print("agent pid", p.pid)


def resume(name: str) -> None:
    """Resumes the agent's session (same transcript) with RESUME, e.g. after a usage-limit stop."""
    _, run, mon = paths(name)
    sid = None
    for line in open(mon / "transcript.jsonl"):
        if line.startswith("{"):
            d = json.loads(line)
            sid = d.get("session_id") or sid
    cmd = claude_cmd(RESUME)
    cmd[1:1] = ["--resume", sid]
    out = open(mon / "transcript.jsonl", "a")
    p = subprocess.Popen(cmd, cwd=run, env=env_for(name), stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    with open(mon / "resume.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} resumed session {sid}; message {RESUME!r}; pid {p.pid}\n")
    print("resumed, pid", p.pid)


def go(name: str, hours: float | None = None) -> None:
    """Submits the tool job; a detached shell waits (no model tokens) until it is up, then starts the agent."""
    tool(name, hours)
    _, _, mon = paths(name)
    # double fork through bash (setsid nohup ... &): a plain Popen child was killed when the calling shell exited
    subprocess.run(["bash", "-c", f'setsid nohup "{HERE}/wait_then_agent.sh" {name} > "{mon}/go.log" 2>&1 < /dev/null &'],
                   check=True)
    print("waiting for the evaluation tool; the agent starts when it is up (log:", mon / "go.log", ")")


def audit(name: str) -> None:
    r = reg()[name]
    _, run, mon = paths(name)
    cmd = [sys.executable, str(AUDIT_TOOLS / "access_audit.py"), str(mon / "transcript.jsonl"),
           *sum((["--allow", x] for x in r["roots"]), []), "--allow-exe", str(agent_python().parent.parent)]
    if os.environ.get("AGENT_ENV_PREFIX"):
        cmd += ["--allow-exe", os.environ["AGENT_ENV_PREFIX"]]
    cmd += ["--hook-log", str(mon / "pyhook.jsonl"), "--scan-dir", str(run / "work"), "--json", str(mon / "audit.json")]
    subprocess.run(cmd, check=False)


def lock(keep: list[str], extra: list[Path]) -> None:
    """Isolation during a batch: every run folder except the active ones, and the extra (analysis) folders, become
    unreadable. The previous modes go to $AGENT_RECORDS_DIR/locked_folders_modes_<time>.txt ("<octal mode> <path>")."""
    targets = [p for p in sorted(runs_dir().iterdir()) if p.is_dir() and p.name not in keep] + list(extra)
    modes = records_dir() / f"locked_folders_modes_{time.strftime('%Y%m%dT%H%M%S')}.txt"
    modes.write_text("".join(f"{stat.S_IMODE(p.stat().st_mode):o} {p}\n" for p in targets))
    for p in targets:
        p.chmod(0)
    print(f"locked {len(targets)} folders; modes in {modes}")


def unlock(modes: Path) -> None:
    n = 0
    for line in modes.read_text().splitlines():
        m, p = line.split(" ", 1)
        Path(p).chmod(int(m, 8))
        n += 1
    print(f"restored {n} folders from {modes}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "preflight", "tool", "agent", "go", "resume", "audit", "lock", "unlock"))
    ap.add_argument("--variant", choices=VARIANTS, default="target_informed")
    ap.add_argument("--size", choices=("s", "m", "l"), default="s")
    ap.add_argument("--name", default=None)
    ap.add_argument("--hours", type=float, default=None, help="Slurm time limit of the tool job (default WALL + 1 h)")
    ap.add_argument("--keep", nargs="*", default=[], help="lock: run folders left open")
    ap.add_argument("--extra", nargs="*", type=Path, default=[], help="lock: further folders to lock")
    ap.add_argument("--modes", type=Path, default=None, help="unlock: modes file written by lock")
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(a.variant, a.size)
    elif a.mode in ("tool", "go"):
        {"tool": tool, "go": go}[a.mode](a.name, a.hours)
    elif a.mode == "lock":
        lock(a.keep, a.extra)
    elif a.mode == "unlock":
        unlock(a.modes)
    else:
        {"preflight": preflight, "agent": agent, "resume": resume, "audit": audit}[a.mode](a.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
