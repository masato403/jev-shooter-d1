"""d1-3B に毎フレームのカメラ画像を読ませて、速く判断させる（ロボットのカメラを想定）。

カメラの画像サイズと質問を固定すると、入力の形が毎回同じになるので、
  - 画像の前処理（縮小・正規化・16×16 のマス目に切る）を CPU ではなく GPU で行う
  - 判断の計算全体を CUDA グラフに記録して、毎フレームは再生するだけにする
ことができる。文章を読む部分は d1_fast と同じく fp8 などで軽くできる。

  m = D1ImageFast(state, question, size=(256, 256), quant="fp8", graph="cuda")
  choice, probs, ms = m.decide(frame)      # frame：PIL 画像、または H×W×3 の uint8（numpy / torch）
"""
import importlib
import time

import numpy as np
import torch

from d1_fast import D1Fast

PATCH = 16


class D1ImageFast:
    def __init__(self, state, question, size=(256, 256), quant="bf16", graph="cuda", gpu_prep=True, vquant="bf16", d1=None):
        # d1：読み込み済みの D1Fast を渡すと、モデルを読み直さずに別の文（状態・質問）用のグラフを作れる
        self.m = d1 or D1Fast(quant=quant)
        self.model = self.m.model
        if vquant != "bf16" and d1 is None:          # 画像を読む部分（SigLIP2）も軽くする
            from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig,
                                              Int8DynamicActivationInt8WeightConfig, quantize_)
            cfg = {"fp8": Float8DynamicActivationFloat8WeightConfig(),
                   "w8a8": Int8DynamicActivationInt8WeightConfig()}[vquant]
            quantize_(self.model.model.vision_tower, cfg)
        eng = self.model.engine
        pkg = type(eng).__module__.rsplit(".", 1)[0]
        P = importlib.import_module(pkg + ".prompt")
        self.q = P.as_question(question)
        text = (P.prefix_text(eng.tokenizer, state, eng.bos, eng.state_style, eng.system, eng._image_markup(1))
                + P.suffix_text(eng.tokenizer, self.q, eng.lead, eng.option_style))
        self.eng, self.text, self.size = eng, text, size
        self.gpu_prep, self.graph = gpu_prep, graph
        from PIL import Image
        blank = Image.new("RGB", size)
        self.inp = eng._image_inputs(text, [blank])          # 固定の部分（文字のトークンなど）はここで一度だけ作る
        # 画像のマス目の数（縦・横）は、画像を読む部分が途中で CPU に取り出す。
        # GPU に置くと CUDA グラフに記録できないので、最初から CPU に置いておく（大きさは固定なので値は変わらない）
        self.inp["spatial_shapes"] = self.inp["spatial_shapes"].cpu()
        self.n_patch = self.inp["pixel_values"].shape[1]
        gh, gw = (int(x) for x in self.inp["spatial_shapes"][0])
        assert gh * gw == self.n_patch and (gh * PATCH, gw * PATCH) == (size[1], size[0]), \
            f"画像は {gw * PATCH}x{gh * PATCH} の1枚（分割なし）になる大きさにする"
        self.gh, self.gw = gh, gw
        with torch.inference_mode():
            ids = self.inp["input_ids"]
            self.base = self.model.model.get_input_embeddings()(ids)      # 文字の部分は毎回同じなので先に作る
            self.img_idx = (ids[0] == self.model.config.image_token_id).nonzero().squeeze(1)
        self.g = None

    # ------------------------------------------------------------ 前処理
    def _prep_cpu(self, frame):
        from PIL import Image
        img = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame))
        return self.eng._image_inputs(self.text, [img])["pixel_values"]

    def _prep_gpu(self, frame):
        """縮小 → 0〜1 → (x-0.5)/0.5 → 16×16 のマス目ごとに (縦, 横, 色) の順に並べる（元の前処理と同じ並び）。"""
        if not torch.is_tensor(frame):
            frame = torch.from_numpy(np.asarray(frame))
        x = frame.to("cuda", non_blocking=True).permute(2, 0, 1).unsqueeze(0).float()
        W, H = self.size
        if x.shape[-2:] != (H, W):
            x = torch.nn.functional.interpolate(x, size=(H, W), mode="bicubic", align_corners=False)
        x = (x / 255.0 - 0.5) / 0.5
        gh, gw = H // PATCH, W // PATCH
        x = x.reshape(1, 3, gh, PATCH, gw, PATCH).permute(0, 2, 4, 3, 5, 1)   # (1, gh, gw, 16, 16, 3)
        return x.reshape(1, gh * gw, PATCH * PATCH * 3)

    # ------------------------------------------------------------ 計算
    def _forward_ref(self, pixel_values):
        """元のモデルの計算（答え合わせ用）。途中で GPU の値を CPU に取り出すので CUDA グラフには記録できない。"""
        i = self.inp
        return self.model(input_ids=i["input_ids"], attention_mask=i["attention_mask"], pixel_values=pixel_values,
                          pixel_attention_mask=i["pixel_attention_mask"], spatial_shapes=i["spatial_shapes"],
                          logits_to_keep=1).logits[0, -1].float()

    def _forward(self, pixel_values):
        """元の計算と同じ中身を、形が毎回同じになるように書き直したもの（CUDA グラフに記録できる）。
        画像の特徴の長さ・画像トークンの位置は、大きさが固定なので最初に一度だけ決めておく。"""
        mm, i = self.model.model, self.inp
        vt = mm.vision_tower(pixel_values=pixel_values, spatial_shapes=i["spatial_shapes"],
                             pixel_attention_mask=i["pixel_attention_mask"], return_dict=True).last_hidden_state
        f = vt[0, :self.n_patch].reshape(1, self.gh, self.gw, -1)
        emb = mm.multi_modal_projector(f).reshape(-1, self.base.shape[-1])
        x = self.base.clone()
        x[0, self.img_idx] = emb.to(x.dtype)          # 画像トークンの場所に画像の特徴を入れる
        h = mm.language_model(inputs_embeds=x, attention_mask=i["attention_mask"]).last_hidden_state[:, -1]
        return self.model.lm_head(h)[0].float()

    @torch.inference_mode()
    def _capture(self):
        self.pix = self.inp["pixel_values"].clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward(self.pix)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = self._forward(self.pix)

    @torch.inference_mode()
    def decide(self, frame):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pv = self._prep_gpu(frame) if self.gpu_prep else self._prep_cpu(frame)
        if self.graph == "cuda":
            if self.g is None:
                self._capture()
            self.pix.copy_(pv)
            self.g.replay()
            row = self.out
        else:
            row = self._forward(pv)
        probs = self.eng._readout(self.q, row - torch.logsumexp(row, dim=-1))
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        best = max(range(len(probs)), key=probs.__getitem__)
        return list(self.q.criteria)[best], probs, ms
