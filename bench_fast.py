"""高速版（CUDA グラフ＋指示文の使い回し）の速さと、Jev と同じ行動を選ぶ割合を測る。

  python bench_fast.py models/Qwen2.5-0.5B-Instruct

Jev が実際に答えた記録の問題文を、短い書き方に直して使う。
"""
import json
import statistics
import sys

import torch

from fast_brain import FastBrain

SHORT_INSTRUCTIONS = (
    "You play a shooting game. Pick the best option. Never pick 'hit' (it loses the game). "
    "Among safe options, prefer 'on target', then 'closer to target'."
)
LOGS = {
    "弾幕（No.10）": "logs/shoot_danmaku_20261009_140421_jev.jsonl",
    "上下左右（No.9）": "logs/shoot_free_20261009_140204_jev.jsonl",
}


def short_option(text):
    """Jev に送った長い選択肢の説明を、同じ意味の短い言葉に直す。"""
    if "gets hit" in text:
        return "hit"
    head = "risky" if "very close" in text else ("very safe" if "lots of room" in text else "safe")
    if "lined up so" in text:
        tail = "on target"
    elif "closer to" in text:
        tail = "closer to target"
    elif "away from" in text:
        tail = "away from target"
    else:
        tail = "no change"
    return f"{head}, {tail}"


def short_state(state):
    """状況の文章から、選択肢の判断に関係の薄い行を省いて短くする。"""
    keep = [l for l in state.split("\n") if l.startswith(("Aim point", "Side bullet", "Ship is invincible"))]
    return "\n".join(l.replace(" (boss, where the shots will meet it)", "") for l in keep)


VARIANTS = {
    "選択肢を短く＋状況そのまま": lambda r: (r["state"], {k: f"{k}: {short_option(v)}" for k, v in r["criteria"].items()}),
    "選択肢を短く＋状況も短く": lambda r: (short_state(r["state"]), {k: f"{k}: {short_option(v)}" for k, v in r["criteria"].items()}),
    "選択肢だけ": lambda r: ("", {k: f"{k}: {short_option(v)}" for k, v in r["criteria"].items()}),
}


def main():
    model_dir = sys.argv[1]
    brain = FastBrain(model_dir, SHORT_INSTRUCTIONS)
    print("モデル:", model_dir, "／指示文", brain.P, "トークン（先に計算済み）")
    for vname, conv in VARIANTS.items():
        for name, path in LOGS.items():
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            rows = [r for r in rows if "criteria" in r and r.get("source") == "jev"]
            for r in rows[:5]:
                brain.decide(*conv(r))
            ms, tok, agree, hit, best = [], [], 0, 0, 0
            for r in rows:
                d = brain.decide(*conv(r))
                ms.append(d.latency_ms)
                tok.append(d.input_tokens)
                agree += d.action == r["action"]
                hit += "gets hit" in r["criteria"][d.action]
                best += any(w in r["criteria"][d.action] for w in ("best for", "good for"))
            q = sorted(ms)
            print(f"{vname} | {name} | 平均 {statistics.mean(ms):.1f}ms 95% {q[int(len(q) * .95) - 1]:.1f}ms "
                  f"最大 {max(ms):.1f}ms | 変わる部分 平均{statistics.mean(tok):.0f}トークン | "
                  f"Jevと一致 {100 * agree / len(rows):.0f}% | 攻撃に良い/最適 {100 * best / len(rows):.0f}% | 当たる {hit}回 / {len(rows)}問")


if __name__ == "__main__":
    main()
