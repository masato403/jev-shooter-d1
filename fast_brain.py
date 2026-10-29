"""ローカルの言語モデル（GPU）で、Jev と同じ方式（状況＋選択肢を読んで1つ選ぶ）を高速に行う。

速くするための工夫:
  1. 毎回同じ指示文は、先に計算して StaticCache に置いておく（プレフィックスキャッシュ）
  2. 毎回変わる部分（状況と選択肢）は長さを固定（右を埋める）し、その1回分の計算を
     CUDA グラフに記録して再生する（GPU への細かい命令の待ち時間がなくなる）
  3. 文章は生成せず、「次に書く1文字」（選択肢の記号）の確率だけを見る
"""
import string
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from brain import Decision

LETTERS = string.ascii_uppercase


class FastBrain:
    name = "local"

    def __init__(self, model_dir, instructions, buckets=(64, 96, 128, 160, 192, 256)):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16,
                                                          attn_implementation="sdpa").cuda().eval()
        self.letter_ids = torch.tensor([self.tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS],
                                       device="cuda")
        self.pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        prefix_ids = self.tok(instructions.strip() + "\n", return_tensors="pt").input_ids.cuda()
        self.P = prefix_ids.shape[1]
        self.buckets = buckets
        self.max_len = max_len = self.P + max(buckets)
        self.cache = StaticCache(config=self.model.config, max_cache_len=max_len)
        with torch.inference_mode():
            self.model(input_ids=prefix_ids, past_key_values=self.cache, use_cache=True,
                       cache_position=torch.arange(self.P, device="cuda"))
        self.graphs = {}
        for L in buckets:
            self._capture(L)
        self.calls, self.total_ms = 0, 0.0

    @torch.inference_mode()
    def _capture(self, L):
        ids = torch.full((1, L), self.pad_id, dtype=torch.long, device="cuda")
        cpos = torch.arange(self.P, self.P + L, device="cuda")
        # マスクは長さごとに決まった形なので先に作る（毎回作る処理は CUDA グラフに記録できないため）
        keys = torch.arange(self.max_len, device="cuda")
        allow = keys[None, :] <= cpos[:, None]
        mask = torch.zeros((1, 1, L, self.max_len), dtype=torch.float16, device="cuda")
        mask.masked_fill_(~allow[None, None], torch.finfo(torch.float16).min)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model(input_ids=ids, attention_mask=mask, past_key_values=self.cache, use_cache=True,
                           cache_position=cpos)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            logits = self.model(input_ids=ids, attention_mask=mask, past_key_values=self.cache, use_cache=True,
                                cache_position=cpos).logits
        self.graphs[L] = (g, ids, logits, mask)

    def build(self, state_text, actions):
        keys = list(actions)
        opts = "\n".join(f"{LETTERS[i]}. {actions[k]}" for i, k in enumerate(keys))
        return keys, f"{state_text}\nOptions:\n{opts}\nAnswer:"

    @torch.inference_mode()
    def decide(self, state_text, actions, rule_action=None):
        keys, body = self.build(state_text, actions)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        new = self.tok(body, add_special_tokens=False).input_ids
        n = len(new)
        L = next((b for b in self.buckets if b >= n), None)
        if L is None:
            raise ValueError(f"問題文が長すぎます（{n}トークン、上限{max(self.buckets)}）")
        g, ids, logits, _ = self.graphs[L]
        ids.fill_(self.pad_id)
        ids[0, :n] = torch.tensor(new, device="cuda")
        g.replay()
        probs = torch.softmax(logits[0, n - 1, self.letter_ids[:len(keys)]].float(), dim=0)
        p = probs.tolist()                      # ここで GPU の計算が終わるのを待つ
        ms = (time.perf_counter() - t0) * 1000
        self.calls += 1
        self.total_ms += ms
        best = max(range(len(keys)), key=lambda i: p[i])
        srt = sorted(p, reverse=True)
        return Decision(keys[best], "local", round(srt[0] - (srt[1] if len(srt) > 1 else 0), 3),
                        {k: round(v, 4) for k, v in zip(keys, p)}, ms, n)

    def stats(self):
        return {"local_calls": self.calls,
                "avg_latency_ms": round(self.total_ms / self.calls, 2) if self.calls else None}

    def close(self):
        pass
