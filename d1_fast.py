"""LiquidAI d1-3B を速く動かす（Jev と同じ形式の判断を、ローカルの GPU で）。

  quant : "bf16"（そのまま） / "int8"（重みだけ8ビット） / "int4"（重みだけ4ビット）
          "w8a8"（重みも計算も8ビット整数） / "fp8"（重みも計算も8ビット小数）
  graph : "none"（d1-3B の標準の動かし方）
          "cuda"（入力を決まった長さに右詰めでそろえ、計算を CUDA グラフに記録して再生）
          "compile"（同じく長さをそろえ、torch.compile の reduce-overhead で動かす）

答えは問題文の最後の位置で読むので、右に詰め物を足しても答えは変わらない
（前の位置の計算は後ろの文字の影響を受けない）。
"""
import importlib
import time

import torch
from transformers import AutoModel

BUCKETS = (64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768)


class D1Fast:
    name = "d1"

    def __init__(self, model_dir="models/d1-3B", quant="bf16", graph="none"):
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
        if quant != "bf16":
            from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
            from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig,
                                              Int8DynamicActivationInt8WeightConfig)
            cfg = {"int8": Int8WeightOnlyConfig(),                      # 重みだけ8ビット（計算は16ビット）
                   "int4": Int4WeightOnlyConfig(group_size=128),         # 重みだけ4ビット
                   "w8a8": Int8DynamicActivationInt8WeightConfig(),      # 重みも計算も8ビット整数
                   "fp8": Float8DynamicActivationFloat8WeightConfig(),   # 重みも計算も8ビット小数（FP8）
                   }[quant]
            # 文章を読む部分（言語モデル）だけを軽くする。答えを出す最後の層（lm_head）はそのまま
            quantize_(self.model.model.language_model, cfg)
        self.engine = self.model.engine
        pkg = type(self.engine).__module__.rsplit(".", 1)[0]
        self.as_question = importlib.import_module(pkg + ".prompt").as_question
        self.answer = importlib.import_module(pkg + ".api").answer
        self.tok = self.engine.tokenizer
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.graph = graph
        self.graphs = {}
        self.fwd = torch.compile(self.model.forward, mode="reduce-overhead") if graph == "compile" else self.model.forward
        self.calls, self.total_ms = 0, 0.0

    def _bucket(self, n):
        for b in BUCKETS:
            if b >= n:
                return b
        raise ValueError(f"問題文が長すぎます（{n}トークン）")

    @torch.inference_mode()
    def _get_graph(self, L):
        if L in self.graphs:
            return self.graphs[L]
        ids = torch.full((1, L), self.pad_id, dtype=torch.long, device="cuda")
        idx = torch.zeros(1, dtype=torch.long, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model(input_ids=ids, logits_to_keep=idx)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.model(input_ids=ids, logits_to_keep=idx).logits
        self.graphs[L] = (g, ids, idx, out)
        return self.graphs[L]

    @torch.inference_mode()
    def logz(self, row):
        n = len(row)
        if self.graph == "none":
            r = self.model(input_ids=torch.tensor([row], device="cuda"), logits_to_keep=1).logits[0, -1].float()
        else:
            L = self._bucket(n)
            if self.graph == "cuda":
                g, ids, idx, out = self._get_graph(L)
                ids.fill_(self.pad_id)
                ids[0, :n] = torch.tensor(row, device="cuda")
                idx.fill_(n - 1)
                g.replay()
                r = out[0, -1].float()
            else:
                ids = torch.full((1, L), self.pad_id, dtype=torch.long, device="cuda")
                ids[0, :n] = torch.tensor(row, device="cuda")
                idx = torch.tensor([n - 1], device="cuda")
                r = self.fwd(input_ids=ids, logits_to_keep=idx).logits[0, -1].float()
        return r - torch.logsumexp(r, dim=-1)

    def system_one(self, state, questions):
        """Jev と同じ形の答え（1問だけ）を返す。"""
        (name, qd), = questions.items()
        q = self.as_question(qd)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        row = self.tok.encode(self.engine.render(state, q), add_special_tokens=False)
        probs = self.engine._readout(q, self.logz(row))
        ms = (time.perf_counter() - t0) * 1000
        self.calls += 1
        self.total_ms += ms
        return {"answers": {name: self.answer(q, probs)}, "usage": {"input_tokens": len(row)}, "latency_ms": ms}
