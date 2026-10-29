#!/bin/sh
# リアルタイム＋反応時間を見越した予測（--lead auto）で Jev（長い問題文）
cd "$(dirname "$0")"
for m in danmaku free; do
  for s in 0 1 2 3 4; do
    .venv/Scripts/python.exe shooter.py --mode $m --seed $s --realtime --lead auto --headless --brain jev --prompt full --instruction No.11 > logs/rtlead_jev_${m}_s${s}.txt 2>&1
    echo "lead jev $m $s done"
  done
done
