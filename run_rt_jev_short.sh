#!/bin/sh
# リアルタイムモードの Jev（短い問題文）。GPU は使わない
cd "$(dirname "$0")"
for m in danmaku free; do
  for s in 0 1 2 3 4; do
    .venv/Scripts/python.exe shooter.py --mode $m --seed $s --realtime --headless --brain jev --prompt short --instruction No.11 > logs/rt_jev_short_${m}_s${s}.txt 2>&1
    echo "jev_short $m $s done"
  done
done
