"""AIの「判断」部分。ゲームはここを通してだけ行動を決める。

RuleBrain : ゲーム側が計算したルールの行動をそのまま返す（APIキー不要）
JevBrain  : TypeSafe AI の Jev に状態文（State）を渡し、Choice で行動を選ばせる
D1Brain   : ローカルの GPU で LiquidAI d1-3B に、Jev と同じ形式で選ばせる（無料）
ShortPrompt : 上のどれかに、問題文を短く書き直してから渡す
"""
import json
import os
import time
from dataclasses import dataclass


@dataclass
class Decision:
    action: str
    source: str                      # rule / jev / cache / fallback
    confidence: float | None = None
    probabilities: dict | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0


class RuleBrain:
    name = "rule"

    def decide(self, state_text, actions, rule_action):
        return Decision(rule_action, "rule")

    def stats(self):
        return {}

    def close(self):
        pass


class JevBrain:
    """Jev で行動を選ぶ。同じ状態文は結果を使い回す（API呼び出し回数の節約）。

    gate > 0 のとき、confidence が gate 未満ならルールの行動に切り替える
    （記事の Confidence-Gated Routing）。既定の 0 は常に Jev の判断を使う。
    """
    name = "jev"

    def __init__(self, instructions, ledger, gate=0.0, model="jev-latest"):
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise SystemExit("環境変数 TYPESAFE_API_KEY が設定されていません")
        from typesafe_sdk import TypeSafeClient
        self.client = TypeSafeClient(model=model)
        self.instructions = instructions
        self.ledger = ledger
        ledger.check()
        self.gate = gate
        self.cache = {}
        self.calls = self.cache_hits = self.fallbacks = self.tokens = 0
        self.total_ms = 0.0

    def decide(self, state_text, actions, rule_action):
        from typesafe_sdk import Choice
        key = state_text + "\n" + json.dumps(actions, sort_keys=True, ensure_ascii=False)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return Decision(hit.action, "cache", hit.confidence, hit.probabilities)

        self.ledger.check()
        t0 = time.perf_counter()
        resp = self.client.system_one(
            state=state_text,
            questions={"next_action": Choice(instructions=self.instructions, criteria=actions)},
        )
        ms = (time.perf_counter() - t0) * 1000
        ans = resp.choices["next_action"]
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "input_tokens", 0) or 0
        out_tokens = getattr(usage, "output_tokens", 0) or 0
        self.calls += 1
        self.total_ms += ms
        self.tokens += tokens
        self.ledger.record(tokens, out_tokens, "platformer")  # 基準を超えたら BudgetExceeded

        if ans.confidence < self.gate:
            self.fallbacks += 1
            d = Decision(rule_action, "fallback", ans.confidence, dict(ans.probabilities), ms, tokens)
        else:
            d = Decision(ans.choice, "jev", ans.confidence, dict(ans.probabilities), ms, tokens)
        self.cache[key] = d
        return d

    def stats(self):
        return {
            "jev_calls": self.calls,
            "cache_hits": self.cache_hits,
            "fallbacks": self.fallbacks,
            "avg_latency_ms": round(self.total_ms / self.calls, 1) if self.calls else None,
            "input_tokens": self.tokens,
            "cost_usd": round(self.tokens / 1e6 * 0.042, 6),
        }

    def close(self):
        close = getattr(self.client, "close", None)
        if close:
            close()


class D1Brain:
    """d1-3B（d1_fast.D1Fast）で行動を選ぶ。お金はかからないので記録は時間だけ。"""
    name = "d1"

    def __init__(self, instructions, quant="bf16", graph="cuda"):
        from d1_fast import D1Fast
        self.m = D1Fast(quant=quant, graph=graph)
        if graph != "none":                      # 始める前に、使いそうな長さを全部一度通して準備を済ませる（リアルタイムで不利にならないように）
            from d1_fast import BUCKETS
            for L in BUCKETS:
                if L <= 640:
                    for _ in range(2):
                        self.m.logz([self.m.pad_id] * (L - 1))
        self.instructions = instructions
        self.calls, self.total_ms, self.tokens, self.ms = 0, 0.0, 0, []

    def decide(self, state_text, actions, rule_action):
        o = self.m.system_one(state_text or None, {"q": {"type": "choice", "instructions": self.instructions,
                                                         "criteria": actions}})
        a = o["answers"]["q"]
        self.calls += 1
        self.total_ms += o["latency_ms"]
        self.ms.append(o["latency_ms"])
        self.tokens += o["usage"]["input_tokens"]
        return Decision(a["choice"], "d1", a["confidence"], a["probabilities"], o["latency_ms"], o["usage"]["input_tokens"])

    def stats(self):
        q = sorted(self.ms)
        return {"d1_calls": self.calls,
                "avg_latency_ms": round(self.total_ms / self.calls, 1) if self.calls else None,
                "median_latency_ms": round(q[len(q) // 2], 1) if q else None,   # 初回の準備（グラフ記録）の分を除いた目安
                "input_tokens": self.tokens}

    def close(self):
        pass


class ShortPrompt:
    """問題文を短く書き直してから中の判断役（Jev / d1）に渡す。選ぶ行動の名前は同じ。"""

    def __init__(self, brain, short_state, short_option):
        self.brain, self.short_state, self.short_option = brain, short_state, short_option
        self.name = brain.name + "_short"

    def decide(self, state_text, actions, rule_action):
        return self.brain.decide(self.short_state(state_text), {k: self.short_option(v) for k, v in actions.items()},
                                 rule_action)

    def stats(self):
        return self.brain.stats()

    def close(self):
        self.brain.close()
