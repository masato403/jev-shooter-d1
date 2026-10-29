"""Jev に、d1-3B と同じ「短い問題文」で同じ120問を解かせて比べる（有料。費用は logs/usage.jsonl に記録）。

  python bench_jev_short.py No.11
"""
import json
import statistics
import sys

from bench_fast import short_option, short_state
from brain import JevBrain
from budget import Ledger
from shooter import SHORT_INSTRUCTIONS

LOGS = {
    "弾幕": "logs/shoot_danmaku_20261009_140421_jev.jsonl",
    "上下左右": "logs/shoot_free_20261009_140204_jev.jsonl",
}


def main():
    ledger = Ledger(sys.argv[1], "bench_jev_short")
    brain = JevBrain(SHORT_INSTRUCTIONS, ledger)
    res = {"brain": "jev", "prompt": "short"}
    for name, path in LOGS.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if "criteria" in r and r.get("source") == "jev"][:120]
        ms, agree, hit, good, picks, toks = [], 0, 0, 0, [], []
        for r in rows:
            d = brain.decide(short_state(r["state"]) or "-", {k: short_option(v) for k, v in r["criteria"].items()}, r["rule"])
            if d.source == "jev":
                ms.append(d.latency_ms)
                toks.append(d.input_tokens)
            a = d.action
            picks.append(a)
            agree += a == r["action"]
            hit += "gets hit" in r["criteria"][a]
            good += any(w in r["criteria"][a] for w in ("best for", "good for"))
        n = len(rows)
        q = sorted(ms)
        res[name] = {"avg_ms": round(statistics.mean(ms), 1), "median_ms": round(statistics.median(ms), 1),
                     "p95_ms": round(q[int(len(q) * .95) - 1], 1), "tokens": round(statistics.mean(toks)),
                     "calls": len(ms), "jev_agree": round(100 * agree / n), "good": round(100 * good / n),
                     "hit": hit, "n": n, "picks": picks}
        print(name, {k: v for k, v in res[name].items() if k != "picks"}, flush=True)
    res["usage"] = ledger.report()
    with open("logs/jev_short_bench.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print("費用:", res["usage"])


if __name__ == "__main__":
    main()
