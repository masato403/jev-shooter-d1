# JEV SHOOTER：判断 AI「Jev」と手元の GPU の「d1-3B」にシューティングを遊ばせて比べる

Python（pygame）で作った縦シューティングを、判断 AI に操作させるコードです。
AI には「今の状況（文章）」と「行動ごとの予測結果を書いた選択肢」を渡し、次の行動を 1 つ選ばせます。

- **Jev**：TypeSafe 社の判断 AI（クラウドの API）
- **d1-3B**：Liquid AI の判断専用モデル。ノートパソコンの GPU（RTX 5050 Laptop、8GB）で動かす

どちらも同じ形（状態＋説明つきの選択肢 → 選択肢ごとの確率）で使い、追加学習はしていません。

## ゲームのモード

| モード | 中身 |
|---|---|
| `h` | 横移動だけ |
| `free` | 上下左右に動ける。横からの弾・狙い撃ち・扇形の弾あり |
| `danmaku` | 弾幕。3 段階のボス弾幕（最大約 270 発）、斜め移動と低速移動あり |

どのモードもライフ 1、5 秒でボスが出て、ボスを倒すとクリアです。

## 使い方

```
python shooter.py                          # 横移動だけを、ルールで動く仮の AI で
python shooter.py --mode free --human      # 自分で遊ぶ（矢印キー、Shift で低速）
python shooter.py --mode danmaku --brain jev                     # Jev（有料。TYPESAFE_API_KEY が必要）
python shooter.py --mode danmaku --brain d1 --prompt full        # d1-3B、長い問題文
python shooter.py --mode free --brain d1 --prompt B --quant fp8 --graph cuda --realtime --lead auto
                                           # d1-3B、短い問題文 B、考えている間もゲームが進む
python shooter.py --replay last            # 最新の Jev の記録を再生（無料）
```

- `--realtime`：AI が考えている間もゲームが進む（ふつうは答えが出るまでゲームを止めて待つ）
- `--lead auto`：答えにかかる時間の分だけ先の位置で結果を予測する（遅れの見越し）
- `--prompt`：`full`（長い文）、`short`（短い文）、`A`・`B`・`C`（短い文の改良版、`short_v2.py`）

## ファイル

| ファイル | 中身 |
|---|---|
| `shooter.py` | ゲーム本体。状況の文章と、行動ごとの予測結果（当たるか・狙えるか）を作る |
| `brain.py` | 判断する側（ルール・Jev・d1-3B） |
| `budget.py`・`budget.json` | Jev の費用を記録し、上限を超えたら止める |
| `d1_fast.py` | d1-3B を速く動かす（FP8 化、入力の長さをそろえて CUDA グラフを使い回す） |
| `short_v2.py` | 短い問題文の改良版 |
| `bench_d1.py`・`bench_fast.py`・`bench_jev_short.py` | Jev が実際に答えた 120 問を、同じ問題でほかのモデルに解かせて比べる |
| `logs/shoot_*_jev.jsonl` | その 120 問（Jev がプレイした時の状況と答え） |
| `run_*.sh` | 比べた時の実行スクリプト（シードを変えて 5 回ずつ） |
| `cam_screen.py`・`cam_player.py` | Web カメラで画面を読み、d1-3B が操作する版 |
| `make_compare.py` | 比較動画を並べて 1 本にする |
| `docs/比較.md` | Jev と d1-3B の比較結果 |

## 入っていないもの

- d1-3B のモデルの重み（Liquid AI の配布元から取得し、`models/d1-3B` に置く）
- Jev の API キー（環境変数 `TYPESAFE_API_KEY`）
- 録画と、比較の時のプレイの記録

## 動作環境

- Windows 11、RTX 5050 Laptop GPU（8GB）
- Python、pygame、PyTorch、transformers、torchao、typesafe-sdk、OpenCV（カメラ版のみ）、imageio-ffmpeg（録画のみ）

## 付記

株式会社スリーアップ・テクノロジー様でのインターンシップにおける実習として取り組んだ内容です。
