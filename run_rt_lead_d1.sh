#!/bin/sh
# リアルタイムで d1-3B：短い問題文の改良版 B（FP8）を見越しなし／見越しありで、長い問題文（16ビット）を見越しありで
cd "$(dirname "$0")"
run() {
  tag=$1; shift
  for m in danmaku free; do
    for s in 0 1 2 3 4; do
      .venv/Scripts/python.exe shooter.py --mode $m --seed $s --realtime --headless "$@" > logs/rtlead_${tag}_${m}_s${s}.txt 2>&1
      echo "$tag $m $s done"
    done
  done
}
run d1B_nolead --brain d1 --prompt B --quant fp8 --graph cuda --lead 0
run d1B_lead --brain d1 --prompt B --quant fp8 --graph cuda --lead auto
run d1full_lead --brain d1 --prompt full --quant bf16 --graph cuda --lead auto
