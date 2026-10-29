#!/bin/sh
# d1-3B に弾幕・上下左右を5つの出方（種）で実際にプレイさせる（Jev と同じ長さの問題文／短い問題文）。1つずつ順番に動かす
cd "$(dirname "$0")"
for p in full short; do
  for m in danmaku free; do
    for s in 0 1 2 3 4; do
      rec=""
      [ $s = 0 ] && rec="--record videos/shoot_${m}_d1_${p}.mp4"
      .venv/Scripts/python.exe shooter.py --mode $m --brain d1 --prompt $p --seed $s --headless $rec > logs/play_d1_${p}_${m}_s${s}.txt 2>&1
      echo "$p $m $s done"
    done
  done
done
