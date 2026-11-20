"""Access monitor for isolated agent runs: log (never block) what the agent's own Python processes touch.

Loaded automatically when this folder is on PYTHONPATH. Logs, as JSON lines to $WSP_AUDIT_LOG:
  open / os.listdir / os.scandir / glob.glob / os.chdir / shutil.copy* events whose path lies outside
  $WSP_AUDIT_ROOTS (colon-separated) and outside the interpreter's own install;
  every subprocess / os.system / os.exec* command;
  every import of the forbidden packages ($WSP_AUDIT_FORBIDDEN, comma-separated; default omnilearned and the
  analysis package softrepair; the runs of the paper used the analysis package's development name).
Limits: opens done in C (h5py, torch.load's own reader) do not raise these events; python -I / -E / -S
or an unset PYTHONPATH skips this file (access_audit.py flags those command lines).
"""

import json
import os
import sys
import time

_LOG = os.environ.get("WSP_AUDIT_LOG")
_ROOTS = tuple(p.rstrip("/") + "/" for p in os.environ.get("WSP_AUDIT_ROOTS", "").split(":") if p)
_QUIET = tuple(p.rstrip("/") + "/" for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, "/usr", "/lib",
                                               "/lib64", "/etc/ld.so.cache", "/proc/self", "/dev",
                                               "/sys/devices/system/cpu", "/sys/fs/cgroup"})
_FORBIDDEN = tuple(p for p in os.environ.get("WSP_AUDIT_FORBIDDEN", "omnilearned,softrepair").split(",") if p)
_busy = False


def _outside(p) -> bool:
    try:
        p = os.path.abspath(os.fsdecode(p)) if not isinstance(p, int) else None
    except Exception:  # noqa: BLE001
        return False
    if p is None:
        return False
    q = p.rstrip("/") + "/"
    return not (q.startswith(_ROOTS) or q.startswith(_QUIET))


def _write(rec: dict) -> None:
    global _busy
    if _busy or not _LOG:
        return
    _busy = True
    try:
        rec.update(t=time.strftime("%Y-%m-%dT%H:%M:%S"), pid=os.getpid(), argv=sys.argv[:3])
        with open(_LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass
    finally:
        _busy = False


def _hook(event: str, args) -> None:
    if _busy:
        return
    try:
        if event == "open" and args and _outside(args[0]):
            _write({"event": event, "path": os.fsdecode(args[0]), "mode": str(args[1]) if len(args) > 1 else ""})
        elif event in ("os.listdir", "os.scandir", "glob.glob", "os.chdir", "shutil.copyfile", "shutil.copytree",
                       "os.symlink", "os.rename") and args and _outside(args[0]):
            _write({"event": event, "path": os.fsdecode(args[0])})
        elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn"):
            _write({"event": event, "cmd": [str(a) for a in args][:3]})
        elif event == "import" and args and str(args[0]).split(".")[0] in _FORBIDDEN:
            _write({"event": "forbidden_import", "module": str(args[0])})
    except Exception:  # noqa: BLE001
        pass


if _LOG:
    sys.addaudithook(_hook)
