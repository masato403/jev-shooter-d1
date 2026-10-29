#!/bin/sh
# Jev（短い問題文）に弾幕・上下左右を5つの出方（種）でプレイさせる（有料、No.11 として記録）
cd "$(dirname "$0")"
for m in danmaku free; do
  for s in 0 1 2 3 4; do
    rec=""
    [ $s = 0 ] && rec="--record videos/shoot_${m}_jev_short.mp4"
    .venv/Scripts/python.exe shooter.py --mode $m --brain jev --prompt short --seed $s --instruction No.11 --headless $rec > logs/play_jev_short_${m}_s${s}.txt 2>&1
    echo "jev short $m $s done"
  done
done
