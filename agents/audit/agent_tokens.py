#!/usr/bin/env python3
"""Token use and cost of agent runs, from their Claude Code transcripts (stream-json, plain or .gz).

usage: agent_tokens.py TRANSCRIPT [TRANSCRIPT ...]
Token totals: the per-model counts of the last result line (Claude Code's running totals of the session); without a
result line, the usage of each API message counted once (by message id; stream-json messages may carry partial
output counts). 'input total' = fresh + cache-read + cache-write tokens.
The cost is the one reported by Claude Code in the transcript's result lines: total_cost_usd of the last result
line (a session resumed after a stop reports the running total of the whole session), with the cost of each
result line since the previous one listed separately.
"""
import gzip
import json
import os
import sys


def lines(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        yield from f


def tally(path):
    t = {"calls": 0, "out": 0, "fresh": 0, "cread": 0, "cwrite": 0, "model": set(), "results": [], "last_usage": None}
    seen = set()
    for line in lines(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") == "result":
            since = d.get("model_usage_since_last_result") or d.get("modelUsage") or {}
            t["last_usage"] = d.get("modelUsage") or t["last_usage"]
            t["results"].append({"total_cost_usd": d.get("total_cost_usd"),
                                 "cost_since_last_usd": sum(u.get("costUSD", 0) for u in since.values()),
                                 "num_turns": d.get("num_turns"), "duration_ms": d.get("duration_ms"),
                                 "is_error": d.get("is_error"), "api_error": d.get("api_error")})
            continue
        m = d.get("message") or {}
        u, mid = m.get("usage"), m.get("id")
        if not u or mid in seen:
            continue
        seen.add(mid)
        t["calls"] += 1
        t["out"] += u.get("output_tokens", 0) or 0
        t["fresh"] += u.get("input_tokens", 0) or 0
        t["cread"] += u.get("cache_read_input_tokens", 0) or 0
        t["cwrite"] += u.get("cache_creation_input_tokens", 0) or 0
        if m.get("model"):
            t["model"].add(m["model"])
    t["cost_usd"] = t["results"][-1]["total_cost_usd"] if t["results"] else None
    if t["last_usage"]:
        u = t["last_usage"]
        t["out"] = sum(x.get("outputTokens", 0) for x in u.values())
        t["fresh"] = sum(x.get("inputTokens", 0) for x in u.values())
        t["cread"] = sum(x.get("cacheReadInputTokens", 0) for x in u.values())
        t["cwrite"] = sum(x.get("cacheCreationInputTokens", 0) for x in u.values())
        t["model"] |= set(u)
    return t


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    print(f"{'transcript':40s} {'calls':>6s} {'output':>10s} {'cache read':>13s} {'cache write':>12s} "
          f"{'input total':>13s} {'cost USD':>9s}")
    for p in paths:
        if not os.path.exists(p):
            print(f"{p}: no transcript")
            continue
        t = tally(p)
        cost = f"{t['cost_usd']:.2f}" if t["cost_usd"] is not None else "-"
        print(f"{p[-40:]:40s} {t['calls']:6d} {t['out']:10,d} {t['cread']:13,d} {t['cwrite']:12,d} "
              f"{t['fresh'] + t['cread'] + t['cwrite']:13,d} {cost:>9s}  {','.join(sorted(t['model']))}")
        if len(t["results"]) > 1:
            for r in t["results"]:
                print(f"{'':40s} result: total {r['total_cost_usd']:.2f}, since previous {r['cost_since_last_usd']:.2f}, "
                      f"{r['num_turns']} turns, error {r['api_error'] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
