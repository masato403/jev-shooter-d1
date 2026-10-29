"""LiquidAI d1-3B（ローカルの判断モデル）に、Jev に送ったのと同じ問題を答えさせて、速さと一致率を測る。

  python bench_d1.py [--compile] [--n 150]

d1-3B は Jev と同じ形式（state + questions{type, instructions, criteria}）で答える。
"""
import argparse
import json
import statistics
import time

import torch
from transformers import AutoModel

import shooter

LOGS = {
    "弾幕（No.10）": "logs/shoot_danmaku_20261009_140421_jev.jsonl",
    "上下左右（No.9）": "logs/shoot_free_20261009_140204_jev.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help='model.compile(mode="reduce-overhead") で CUDA グラフを使う')
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    model = AutoModel.from_pretrained("models/d1-3B", trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    if args.compile:
        model.compile(mode="reduce-overhead")
    print("GPU:", torch.cuda.get_device_name(0), "／compile:", args.compile)
    for name, path in LOGS.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if "criteria" in r and r.get("source") == "jev"][:args.n]

        def ask(r):
            q = {"next_action": {"type": "choice", "instructions": shooter.INSTRUCTIONS, "criteria": r["criteria"]}}
            return model.system_one(r["state"], q)

        t = time.perf_counter()
        for r in rows[:8]:                         # 準備運動（最初の形の準備・コンパイルの時間は入れない）
            ask(r)
        warm = time.perf_counter() - t
        ms, toks, agree, hit, good = [], [], 0, 0, 0
        for r in rows:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = ask(r)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
            a = out["answers"]["next_action"]["choice"]
            toks.append(out["usage"]["input_tokens"])
            agree += a == r["action"]
            hit += "gets hit" in r["criteria"][a]
            good += any(w in r["criteria"][a] for w in ("best for", "good for"))
        q = sorted(ms)
        n = len(rows)
        print(f"{name} | 平均 {statistics.mean(ms):.1f}ms 中央 {statistics.median(ms):.1f}ms 95% {q[int(n * .95) - 1]:.1f}ms "
              f"最大 {max(ms):.1f}ms | 入力 平均{statistics.mean(toks):.0f}トークン | Jevと一致 {100 * agree / n:.0f}% | "
              f"攻撃に良い/最適 {100 * good / n:.0f}% | 当たる {hit}回/{n}問 | 準備運動 {warm:.1f}秒", flush=True)
    print("例:", json.dumps(ask(rows[0])["answers"], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
