"""Jev の費用管理（使用量の記録と、中断の基準による自動停止）。

- Jev を1回呼ぶたびに logs/usage.jsonl へ1行記録する（実測の入力トークンと費用）。
- 中断の基準は budget.json に置く。budget.json の doc_txt_url に共有した資料の
  テキスト出力URL（Google ドキュメントの .../export?format=txt）があれば、実行のたびに
  資料の「中断の基準」を読み直し、資料の値を優先する。読めなかったときは budget.json の値を使う。
"""
import json
import os
import re
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "logs", "usage.jsonl")
CONFIG = os.path.join(HERE, "budget.json")
PRICE_PER_MTOK = 0.042          # jev-1.13.0、入力トークン100万あたりのドル（出力は無料）

DEFAULTS = {
    "per_instruction_usd": 0.05,   # Claude への指示1回
    "per_day_usd": 0.5,            # 1日の合計
    "per_task_usd": 2.0,           # 課題全体
    "max_tokens_per_call": 1500,
    "max_calls_per_run": 1500,
    "doc_txt_url": "",
}
# 資料の表から読む項目: (探す見出し, キー, 値の正規表現)
DOC_RULES = [
    ("Claude への指示1回", "per_instruction_usd", r"([0-9]+(?:\.[0-9]+)?)\s*ドル"),
    ("1日の合計", "per_day_usd", r"([0-9]+(?:\.[0-9]+)?)\s*ドル"),
    ("課題全体", "per_task_usd", r"([0-9]+(?:\.[0-9]+)?)\s*ドル"),
    ("入力トークンが", "max_tokens_per_call", r"^([0-9,]+)"),
    ("呼び出しが", "max_calls_per_run", r"^([0-9,]+)"),
]


class BudgetExceeded(Exception):
    pass


def cost_of(tokens):
    return tokens / 1e6 * PRICE_PER_MTOK


def load_limits():
    """(基準の dict, 値の出どころの説明) を返す。"""
    limits = dict(DEFAULTS)
    if os.path.exists(CONFIG):
        with open(CONFIG, encoding="utf-8") as f:
            limits.update(json.load(f))
    else:
        with open(CONFIG, "w", encoding="utf-8") as f:
            json.dump(DEFAULTS, f, ensure_ascii=False, indent=2)
    source = "budget.json"
    url = limits.get("doc_txt_url")
    if url:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                text = r.read().decode("utf-8-sig")
            sec = text[text.index("中断の基準"):]
            found = {}
            for label, key, pat in DOC_RULES:
                i = sec.find(label)
                if i < 0:
                    continue
                m = re.search(pat, sec[i + len(label):i + len(label) + 200].lstrip(), re.M)
                if m:
                    found[key] = float(m.group(1).replace(",", ""))
            if found:
                limits.update(found)
                source = "資料（" + ", ".join(f"{k}={v:g}" for k, v in found.items()) + "）"
        except Exception as e:  # 読めなければ budget.json の値で続ける
            source = f"budget.json（資料を読めなかったため: {type(e).__name__}）"
    return limits, source


class Ledger:
    """使用量の記録と基準のチェック。instruction は資料の「実行記録」の No. に合わせる。"""

    def __init__(self, instruction, program):
        self.instruction = instruction
        self.program = program
        self.limits, self.source = load_limits()
        self.run_calls = 0
        self.run_tokens = 0
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)

    def _rows(self):
        if not os.path.exists(LEDGER):
            return []
        with open(LEDGER, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def spent(self):
        today = datetime.now().strftime("%Y-%m-%d")
        rows = self._rows()
        return {
            "instruction": sum(r["cost_usd"] for r in rows if r.get("instruction") == self.instruction),
            "day": sum(r["cost_usd"] for r in rows if r["time"].startswith(today)),
            "task": sum(r["cost_usd"] for r in rows),
        }

    def check(self):
        """基準を超えていたら BudgetExceeded を投げる。Jev を呼ぶ前と後に使う。"""
        s, L = self.spent(), self.limits
        if s["instruction"] >= L["per_instruction_usd"]:
            raise BudgetExceeded(f"指示1回の上限 {L['per_instruction_usd']}ドル に達した（{s['instruction']:.6f}ドル）")
        if s["day"] >= L["per_day_usd"]:
            raise BudgetExceeded(f"1日の上限 {L['per_day_usd']}ドル に達した（{s['day']:.6f}ドル）。担当者に報告する")
        if s["task"] >= L["per_task_usd"]:
            raise BudgetExceeded(f"課題全体の上限 {L['per_task_usd']}ドル に達した（{s['task']:.6f}ドル）。担当者に相談する")
        if self.run_calls >= L["max_calls_per_run"]:
            raise BudgetExceeded(f"1回の実行での呼び出しが {L['max_calls_per_run']:g}回 に達した")

    def record(self, input_tokens, output_tokens=0, note=""):
        self.run_calls += 1
        self.run_tokens += input_tokens
        row = {"time": datetime.now().isoformat(timespec="seconds"), "instruction": self.instruction,
               "program": self.program, "input_tokens": input_tokens, "output_tokens": output_tokens,
               "cost_usd": round(cost_of(input_tokens), 9), "note": note}
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if input_tokens > self.limits["max_tokens_per_call"]:
            raise BudgetExceeded(f"1回の入力トークンが {input_tokens} で基準 {self.limits['max_tokens_per_call']:g} を超えた")
        self.check()

    def report(self):
        s = self.spent()
        return {"instruction": self.instruction, "run_calls": self.run_calls, "run_input_tokens": self.run_tokens,
                "run_cost_usd": round(cost_of(self.run_tokens), 6),
                "instruction_total_usd": round(s["instruction"], 6), "today_total_usd": round(s["day"], 6),
                "task_total_usd": round(s["task"], 6), "limits_from": self.source}
