#!/usr/bin/env python
"""Evaluation tool for the weight-space programming agent runs (runs inside a GPU job).

The agent never runs the model. It drops request files into RUN/requests/ and reads the replies from
RUN/responses/. Every served request costs one query from the budget (50 in all runs of the paper).

Variants (paper Sec. 3.3, App. C). They share the tool and differ only in what a reply contains:
  physics_origin_informed  surgical replies carry the remaining error against the target predictions under its
                           own names (rule, E_rule, P_reco, ...); the agent also has P_reco.npy and the removal-rule
                           code in its inputs.
  target_informed          the same replies with neutral names (BLIND: ref, E_ref, P_ref, ...) and the extra kind
                           "reference", which returns the target predictions f_orig(removal_rule(x)) on any jet.
  flaw_informed            as target_informed, without the kind "reference" and with the "ref" block (every
                           remaining-error number) removed from surgical replies.

RUN/config.json (written by agents/launch.py) sets the run:
  {"size": "s"|"m"|"l", "variant": <one of the three>, "kinds": [...], "budget": 50,
   "zero_pass": "submissions/zero_pass.npz", "require_prediction": true, "gen_max": 512, "gen_steps": 64,
   "surgical_per_class": null | k, "restrict_paths": true, "no_trace": true}
The configs of the published runs use the older keys "rule" (physics_origin_informed), "ref_task"
(target_informed) and "flaw_task" (flaw_informed); they are read as aliases. --variant overrides the config.

Request RUN/requests/<id>.json (write <id>.json.tmp, then rename). Every request needs a "hypothesis" (string)
and a "prediction" (dict of numbers) when require_prediction is set.
  {"kind": "surgical", "weights": "<abs .npz>" | null}   practice-set limits, responses, remaining error, size
  {"kind": "logits", "weights": ..., "input": SRC, "indices": [...]}
  {"kind": "capture", "weights": ..., "input": SRC, "indices": [...], "sites": [...]}   (at most 4 sites)
  {"kind": "reference", "input": SRC, "indices": [...]}   target predictions (target_informed only)
      SRC: "clean" | a practice file name | abs path of an .npz with key 'data' (n <= 2048, 150, 9) inside RUN.
      Reply npz: logits (n, 210), labels, act:<site> (float16).
An edit ("weights") is an .npz of full replacement tensors (state_dict keys) applied on top of the original model.
Reply RUN/responses/<id>.json: {"ok", "error"?, "pass", "remaining", ...}. Malformed requests are rejected without
cost. Nothing is served before the zero-pass file exists. The job stops when RUN/STOP exists, the wall time is
reached, or no request arrives for --idle seconds.

    python agents/evaluation_tool.py --run RUN [--variant V] [--wall s] [--idle s]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import softrepair.core as W

MAX_JETS, MAX_SITES = 2048, 4
VARIANTS = ("flaw_informed", "target_informed", "physics_origin_informed")
# config keys of the published runs -> variant
LEGACY_KEYS = {"flaw_task": "flaw_informed", "ref_task": "target_informed", "rule": "physics_origin_informed"}
KINDS = {"flaw_informed": ["surgical", "capture", "logits"],
         "target_informed": ["surgical", "capture", "logits", "reference"],
         "physics_origin_informed": ["surgical", "capture", "logits"]}
DEFAULTS = {"size": "s", "models": {"orig": None}, "kinds": ["surgical", "capture", "logits"], "budget": 50,
            "zero_pass": "submissions/zero_pass.npz", "gen_max": 512, "gen_steps": 64,
            "require_prediction": True, "restrict_paths": False, "no_trace": False, "surgical_per_class": None,
            "variant": None, "practice_set": "practice"}
# batch caps used by the GPU jobs of the paper (the original job scripts), per model size
BATCH_ENV = {"s": {"SOFTREPAIR_EVAL_BATCH": "256"},
             "m": {"SOFTREPAIR_MAX_BATCH": "128", "SOFTREPAIR_EVAL_BATCH": "128"},
             "l": {"SOFTREPAIR_MAX_BATCH": "32", "SOFTREPAIR_EVAL_BATCH": "32"}}
# Reply keys as the agents saw them (the task descriptions refer to them): softrepair key -> reply key. A no-op for
# keys the library does not rename.
AGENT_KEYS = {"response_probes": "R_vis", "response_in_range": "R_in", "response_sub_threshold": "R_sub",
              "response_held_out": "R_held", "passes_limits_no_generator": "gate_no_g2",
              "P_removal": "P_reco", "passes_limits_removal_no_generator": "gate_rule_no_g2",
              "max_log_rho_removal": "max_log_rho_rule", "remaining_error": "E_rule",
              "remaining_error_held_out": "E_rule_held", "changed_parameters": "dof"}
# Neutral names for the target_informed and flaw_informed replies: the removal rule is never named.
BLIND = {"rule": "ref", "E_rule": "E_ref", "E_rule_held": "E_ref_held", "gate_rule_no_g2": "gate_ref_no_g2",
         "max_log_rho_rule": "max_log_rho_ref", "P_reco": "P_ref"}


def agent_keys(d):
    if isinstance(d, dict):
        return {AGENT_KEYS.get(k, k): agent_keys(v) for k, v in d.items()}
    return d


def blind(d):
    if isinstance(d, dict):
        return {BLIND.get(k, k): blind(v) for k, v in d.items()}
    return d


def resolve_config(cfg: dict, variant: str | None) -> dict:
    """Variant from --variant, else from "variant", else from the legacy keys; sets the internal flags."""
    cfg = dict(cfg)
    if variant is None:
        variant = cfg.get("variant") or next((v for k, v in LEGACY_KEYS.items() if cfg.get(k)), None)
    cfg["variant"] = variant
    if variant is not None:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant!r}; have {VARIANTS}")
        cfg["rule"] = True  # surgical replies with the remaining error against the target predictions
        cfg["blind"] = variant != "physics_origin_informed"
        cfg["flaw_task"] = variant == "flaw_informed"
    else:
        raise SystemExit(f"no variant: pass --variant or set it in config.json ({VARIANTS})")
    if cfg.get("flaw_task") and "reference" in cfg["kinds"]:
        raise SystemExit("flaw_informed runs must not offer the kind 'reference'")
    return cfg


def ledger_write(run: Path, led: dict) -> None:
    tmp = run / "ledger.json.tmp"
    tmp.write_text(json.dumps(led, indent=1))
    tmp.replace(run / "ledger.json")


def reply(run: Path, rid: str, body: dict) -> None:
    tmp = run / "responses" / f"{rid}.json.tmp"
    tmp.write_text(json.dumps(body, indent=1))
    tmp.replace(run / "responses" / f"{rid}.json")


def size_reply(state: dict, orig: dict) -> dict:
    import softrepair.repair_eval as B
    sz = B.edit_size(state, orig)
    out = {AGENT_KEYS.get(k, k): v for k, v in sz.items() if k != "tensors"}
    out["tensors"] = {k: {"rank": v["rank"], "changed": v["changed"]} for k, v in sz["tensors"].items()}
    return out


class EvaluationTool:
    def __init__(self, run: Path, cfg: dict):
        self.run, self.cfg = run, cfg
        self.ad = W.load_adapter("cuda", cfg["size"])
        self.orig = W.original_state(self.ad)
        self.bases = {k: (W.load_candidate(v, self.orig) if v else dict(self.orig)) for k, v in cfg["models"].items()}
        self.practice = W.load_set(cfg["practice_set"], None, mmap=False)
        if cfg.get("rule") and not cfg.get("blind"):  # physics_origin_informed: P_reco is a practice input too
            self.practice["P_reco"] = W.reco_fix(np.asarray(self.practice["P"]))
            self.practice["P_reco:ok"] = self.practice["P:ok"]
        self.Y0 = None

    def gen_drift(self, state: dict) -> dict:
        """Relative change of the generator's velocity on 512 fixed noised practice W jets (t = 0.2/0.5/0.85)."""
        ad = self.ad
        if not hasattr(self, "_gd"):
            idx = np.flatnonzero(self.practice["labels"] == 9)[:512]
            raw = np.asarray(self.practice["clean"][idx])
            eps, _ = W.gen_drift_inputs(raw, seed=11)
            W.set_state(ad, self.orig)
            ref = {t: W.gen_velocity(ad, raw, self.practice["labels"][idx], eps, t) for t in (0.2, 0.5, 0.85)}
            self._gd = (raw, self.practice["labels"][idx], eps, ref)
        raw, lab, eps, ref = self._gd
        W.set_state(ad, state)
        out = {f"t{t}": float(np.linalg.norm(W.gen_velocity(ad, raw, lab, eps, t) - ref[t]) / np.linalg.norm(ref[t]))
               for t in ref}
        out["max"] = max(out.values())
        return out

    def rule_practice(self) -> dict:
        """Practice set of the surgical kind; with surgical_per_class = k, the first k jets of each class."""
        k = self.cfg.get("surgical_per_class")
        if not k:
            return self.practice
        if not hasattr(self, "_rp"):
            lab = np.asarray(self.practice["labels"])
            idx = np.sort(np.concatenate([np.flatnonzero(lab == c)[:k] for c in np.unique(lab)]))
            n = len(lab)
            self._rp = {key: (np.asarray(v)[idx] if hasattr(v, "__len__") and not isinstance(v, (str, dict)) and len(v) == n else v)
                        for key, v in self.practice.items()}
        return self._rp

    def state(self, req: dict) -> dict:
        mid = req.get("model", "orig")
        if mid not in self.bases:
            raise KeyError(f"unknown model {mid!r}; have {sorted(self.bases)}")
        base = self.bases[mid]
        return W.load_candidate(self.inside(req["weights"]), base) if req.get("weights") else base

    def inside(self, p):
        """With restrict_paths, a request may only name files inside RUN."""
        q = Path(p).resolve()
        if self.cfg.get("restrict_paths") and not q.is_relative_to(self.run.resolve()):
            raise ValueError(f"files must lie inside {self.run}")
        return q

    def inputs(self, req: dict, max_n: int):
        src = req.get("input", "clean")
        if src == "clean" or (src in self.practice and src != "labels" and not src.endswith(":ok")
                              and src != "meta"):
            idx = np.asarray(req.get("indices", list(range(min(512, max_n)))), int)
            if idx.size == 0 or idx.size > max_n or idx.min() < 0 or idx.max() >= len(self.practice["labels"]):
                raise ValueError(f"indices must be 1..{max_n} practice rows")
            return np.asarray(self.practice[src][idx]), self.practice["labels"][idx]
        with np.load(self.inside(src)) as z:
            raw = np.asarray(z["data"], np.float32)
        if raw.ndim != 3 or raw.shape[1:] != (W.N_SLOTS, 9) or not 0 < len(raw) <= max_n:
            raise ValueError(f"custom input must be (1..{max_n}, 150, 9), got {raw.shape}")
        if not np.isfinite(raw).all():
            raise ValueError("custom input has non-finite values")
        return raw, np.full(len(raw), -1)

    def chunk(self) -> int:
        return {"s": 256, "m": 128}.get(self.cfg["size"], 32)  # l: small chunks fit the GPU

    def handle(self, req: dict, rid: str) -> dict:
        kind = req.get("kind")
        if kind not in self.cfg["kinds"]:
            raise ValueError(f"kind {kind!r} not available here; have {self.cfg['kinds']}")
        npz = self.run / "responses" / f"{rid}.npz"
        if kind == "reference":  # target predictions f_orig(removal_rule(x)); the rule stays inside the tool
            if req.get("weights") or req.get("model", "orig") != "orig":
                raise ValueError("reference takes no 'weights' or 'model'")
            raw, labels = self.inputs(req, MAX_JETS)
            W.set_state(self.ad, self.orig)
            x = W.reco_fix(np.asarray(raw, np.float32))
            ys, cb = [], self.chunk()
            with torch.no_grad():
                for s0 in range(0, len(x), cb):
                    o = self.ad.forward(self.ad.preprocess({"data": torch.as_tensor(np.ascontiguousarray(x[s0:s0 + cb]))}))
                    ys.append(o["y_pred"].float().cpu().numpy())
            arrays = {"logits": np.concatenate(ys), "labels": labels}
            np.savez(npz, **arrays)
            return {"npz": str(npz), "shapes": {k: list(v.shape) for k, v in arrays.items()}}
        state = self.state(req)
        W.set_state(self.ad, state)
        ad = self.ad
        if kind == "surgical":  # the three variants of the paper
            import softrepair.removal_rule as R
            res = R.surgical(ad, state, self.orig, self.rule_practice(), self.__dict__.setdefault("_rule", {}))
            res["gen_drift"] = self.gen_drift(state)
            res["size"] = size_reply(state, self.orig)
            res = agent_keys(res)
            res = blind(res) if self.cfg.get("blind") else res
            if self.cfg.get("flaw_task"):  # flaw_informed: no remaining error of any kind in the reply
                res.pop("ref", None)
            return res
        if kind in ("logits", "capture"):
            sites = list(req.get("sites", [])) if kind == "capture" else []
            if len(sites) > MAX_SITES:
                raise ValueError(f"at most {MAX_SITES} sites")
            raw, labels = self.inputs(req, MAX_JETS)
            ys, acts = [], {s: [] for s in sites}
            with torch.no_grad():
                cb = self.chunk()
                for s0 in range(0, len(raw), cb):
                    o = ad.forward(ad.preprocess({"data": torch.as_tensor(raw[s0:s0 + cb])}), capture=sites)
                    ys.append(o["y_pred"].float().cpu().numpy())
                    for s in sites:
                        acts[s].append(o["activations"][s].half().cpu().numpy())
            arrays = {"logits": np.concatenate(ys), "labels": labels}
            arrays.update({f"act:{s}": np.concatenate(v) for s, v in acts.items()})
        else:
            raise ValueError(f"unknown kind {kind!r}")
        np.savez(npz, **arrays)
        return {"npz": str(npz), "shapes": {k: list(v.shape) for k, v in arrays.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--variant", choices=VARIANTS, default=None, help="overrides the variant of RUN/config.json")
    ap.add_argument("--budget", type=int, default=None, help="overrides config budget")
    ap.add_argument("--wall", type=float, default=3.8 * 3600, help="seconds before self-exit")
    ap.add_argument("--idle", type=float, default=1800, help="exit after this many seconds without a request")
    args = ap.parse_args()
    run = args.run
    for d in ("requests", "responses", "submissions", "work"):
        (run / d).mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS)
    if (run / "config.json").exists():
        cfg.update(json.loads((run / "config.json").read_text()))
    if args.variant is not None and not (run / "config.json").exists():
        cfg["kinds"] = KINDS[args.variant]
    if args.budget is not None:
        cfg["budget"] = args.budget
    cfg = resolve_config(cfg, args.variant)
    os.environ["MODEL_SIZE"] = cfg["size"]
    for k, v in BATCH_ENV.get(cfg["size"], {}).items():
        os.environ.setdefault(k, v)
    W.use_math_sdp()  # every run of the paper used the math attention kernel
    t0 = time.time()
    lock = run / "oracle.lock"  # file names oracle.lock / "oracle_up" as seen by the agents
    me = f"{os.environ.get('SLURM_JOB_ID', 'local')}@{os.uname().nodename}"
    while lock.exists() and time.time() - lock.stat().st_mtime < 90 and lock.read_text().strip() != me:
        if time.time() - t0 > args.wall:  # another instance served this run for our whole wall time
            print("[tool] another instance held the run for the whole wall time, exit", flush=True)
            return 0
        time.sleep(15)  # another instance is serving this run: wait until its heartbeat stops
    lock.write_text(me)
    tool = EvaluationTool(run, cfg)
    led_path = run / "ledger.json"
    led = json.loads(led_path.read_text()) if led_path.exists() else {"budget": cfg["budget"], "used": 0, "log": []}
    led["oracle_up"] = time.strftime("%Y-%m-%dT%H:%M:%S")  # the agents wait for this key
    ledger_write(run, led)
    print(f"[tool] up in {time.time() - t0:.0f}s, size {cfg['size']}, variant {cfg['variant']}, "
          f"budget {led['budget']}, used {led['used']}", flush=True)
    last = time.time()
    while time.time() - t0 < args.wall:
        lock.write_text(me)  # heartbeat
        if time.time() - last > args.idle:
            print(f"[tool] idle for {args.idle:.0f}s, exit", flush=True)
            break
        if (run / "STOP").exists():
            print("[tool] STOP", flush=True)
            break
        reqs = sorted(p for p in (run / "requests").glob("*.json") if not (run / "responses" / p.name).exists())
        if not reqs:
            time.sleep(3)
            continue
        p = reqs[0]
        rid = p.stem
        last = time.time()
        remaining = led["budget"] - led["used"]
        try:
            req = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            reply(run, rid, {"ok": False, "error": f"unreadable request: {e}", "remaining": remaining})
            continue
        if cfg.get("require_prediction") and (not str(req.get("hypothesis", "")).strip()
                                              or not isinstance(req.get("prediction"), dict) or not req["prediction"]):
            reply(run, rid, {"ok": False, "error": "every request needs a non-empty 'hypothesis' (string) and "
                             "'prediction' (dict of numbers) written before it is served", "remaining": remaining})
            continue
        if cfg.get("zero_pass") and not (run / cfg["zero_pass"]).exists():
            reply(run, rid, {"ok": False, "error": f"submit {cfg['zero_pass']} first", "remaining": remaining})
            continue
        if remaining <= 0:
            reply(run, rid, {"ok": False, "error": "budget exhausted", "remaining": 0})
            continue
        t = time.time()
        try:
            body = tool.handle(req, rid)
        except Exception as e:  # noqa: BLE001  (malformed: no charge)
            reply(run, rid, {"ok": False, "error": f"{type(e).__name__}: {e}",
                             "remaining": remaining, **({} if cfg.get("no_trace") else {"trace": traceback.format_exc(limit=3)})})
            print(f"[tool] {rid} rejected: {e}", flush=True)
            continue
        led["used"] += 1
        led["log"].append({"id": rid, "kind": req.get("kind"), "model": req.get("model", "orig"),
                           "hypothesis": req.get("hypothesis"), "prediction": req.get("prediction"),
                           "weights": req.get("weights"), "input": req.get("input"),
                           "time": time.strftime("%H:%M:%S"), "sec": round(time.time() - t, 1)})
        ledger_write(run, led)
        reply(run, rid, {"ok": True, "pass": led["used"], "remaining": led["budget"] - led["used"], **body})
        print(f"[tool] {rid} {req.get('kind')} {req.get('model', 'orig')} pass {led['used']} "
              f"({time.time() - t:.1f}s)", flush=True)
        last = time.time()  # idle time counts from the end of the last pass, not its start
    if lock.exists() and lock.read_text().strip() == me:
        lock.unlink()
    print("[tool] exit", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
