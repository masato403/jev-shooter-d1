"""比較用の動画を videos/比較_20261013/ に作る（3つの AI を横に並べた動画と、名前を分かりやすくした個別の動画）。

  python make_compare.py

止めて待つやり方の動画は、待っている間の画面も録画されていて長いので、記録から作り直してから並べる。
"""
import glob
import os
import subprocess
import sys

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT = "videos/比較_20261013"
FONT = "C\\:/Windows/Fonts/meiryob.ttc"
PY = sys.executable


def latest(pattern):
    fs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return fs[-1] if fs else None


def replay(log, mode, out):
    if not os.path.exists(out):
        subprocess.run([PY, "shooter.py", "--mode", mode, "--replay", log, "--headless", "--record", out],
                       check=True, capture_output=True)
    return out


def dur_of(path):
    t = subprocess.run([FF, "-i", path], capture_output=True, text=True, errors="replace").stderr
    h, m, sec = t.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def side_by_side(items, title, out):
    """items: [(動画, 見出し)]。縦 360 に縮めて横に並べ、上に見出しの帯を付ける。短い動画は最後の画面で止めて長さをそろえる。"""
    args, parts = [FF, "-y", "-loglevel", "error"], []
    for i, (path, label) in enumerate(items):
        args += ["-i", path]
        parts.append(f"[{i}:v]scale=624:360,tpad=stop_mode=clone:stop_duration=120,"
                     f"pad=624:410:0:50:color=0x101018,"
                     f"drawtext=fontfile='{FONT}':text='{label}':fontcolor=white:fontsize=22:x=(w-tw)/2:y=14[v{i}]")
    n = len(items)
    dur = max(dur_of(p) for p, _ in items) + 1.5
    if n == 4:                                   # 2×2（上の段 d1-3B、下の段 Jev）
        stack = "[v0][v1]hstack=inputs=2[r0];[v2][v3]hstack=inputs=2[r1];[r0][r1]vstack=inputs=2[h];"
    else:
        stack = "".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[h];"
    graph = ";".join(parts) + ";" + stack +         f"[h]pad=iw:ih+56:0:56:color=0x000000,drawtext=fontfile='{FONT}':text='{title}':fontcolor=0xffd060:"         f"fontsize=26:x=(w-tw)/2:y=14[o]"
    args += ["-filter_complex", graph, "-map", "[o]", "-t", f"{dur:.1f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "24", out]
    subprocess.run(args, check=True)


AIS = (("d1", "d1-3B 長い指示文"), ("d1_short", "d1-3B 短い指示文"), ("jev", "Jev 長い指示文"), ("jev_short", "Jev 短い指示文"))


def log_for(mode, name, realtime):
    """種0の記録。止めて待つ Jev（長い指示文）は 10/9 にクリアしたときの記録を使う。"""
    if realtime:
        return latest(f"logs/shoot_{mode}_2026101[3]_*_{name}_rt.jsonl")
    if name == "jev":
        return {"danmaku": "logs/shoot_danmaku_20261009_140421_jev.jsonl", "free": "logs/shoot_free_20261009_140204_jev.jsonl"}[mode]
    return latest(f"logs/shoot_{mode}_2026101[3]_*_{name}.jsonl")


def main():
    tmp = f"{OUT}/個別"
    os.makedirs(tmp, exist_ok=True)
    for f in glob.glob(f"{OUT}/*.mp4") + glob.glob(f"{tmp}/*.mp4"):   # 前に作った比較動画は作り直す
        os.remove(f)
    for realtime, how, title in ((True, "1_リアルタイム", "リアルタイム（考えている間もゲームが進む）"),
                                 (False, "2_止めて待つ", "止めて待つ（答えが出るまでゲームを止める）")):
        for mode, jp in (("free", "上下左右"), ("danmaku", "弾幕")):
            items = []
            for name, label in AIS:
                log = log_for(mode, name, realtime)
                if not log:
                    print("記録なし:", how, jp, label)
                    continue
                items.append((replay(log, mode, f"{tmp}/{how[2:]}_{jp}_{label.replace(' ', '_')}.mp4"), label))
            side_by_side(items, f"{title}／{jp}", f"{OUT}/{how}_{jp}_{len(items)}つ並べて.mp4")
    print("ok")


if __name__ == "__main__":
    main()
