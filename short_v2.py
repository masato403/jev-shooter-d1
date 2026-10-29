"""短い問題文の改良版。元の短い版（bench_fast.short_option）は、弾に当たる行動を "hit" の一語で書いていたが、
シューティングでは「敵に当てる（良いこと）」とも読めるため、d1-3B も Jev もそれを選ぶことがあった。

  A : "hit" → "shot down (game over)" に言い換えるだけ
  B : A に加えて、攻撃の良さを (best) / (good) / (bad) と、Jev に送った長い文と同じ評価の言葉で付ける
  C : B と同じ選択肢で、指示文を「安全が第一」の順にはっきり書く
"""
from bench_fast import short_state

INSTR = {
    "A": "Pick the best action. Never pick one that is shot down; prefer on target, then closer to target.",
    "B": "Pick the best action. Never pick 'shot down'. Prefer 'best', then 'good'. Avoid 'risky' if possible.",
    "C": ("Safety first: never pick 'shot down' (it ends the game), and avoid 'risky' when a safe option exists. "
          "Among safe options, prefer 'best', then 'good'."),
}


def short_option_v2(text, variant):
    """Jev に送った長い選択肢の説明を、取り違えのない短い言葉に直す。"""
    if "gets hit" in text:
        return "shot down (game over)" if variant == "A" else "shot down (never choose)"
    head = "risky" if "very close" in text else ("very safe" if "lots of room" in text else "safe")
    if variant == "A":
        tail = ("on target" if "lined up so" in text else "closer to target" if "closer to" in text
                else "away from target" if "away from" in text else "no change")
    else:
        tail = ("on target (best)" if "lined up so" in text else "closer to target (good)" if "closer to" in text
                else "away from target (bad)" if "away from" in text else "no change")
    return f"{head}, {tail}"


__all__ = ["INSTR", "short_option_v2", "short_state"]
