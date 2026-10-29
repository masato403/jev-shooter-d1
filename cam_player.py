"""Web カメラでゲームの画面を見て、AI（d1-3B）がゲームを操作する。

  1) shooter.py --mode free --human --markers --udp 9200   （ゲーム：画面の四隅に目印。キー入力を UDP でも受ける）
  2) python cam_player.py --brain d1                          （この AI）

毎フレーム（カメラ 30FPS）：
  カメラ画像 → 目印で斜めの画面をまっすぐに直す（480x600）→ 色で 自機・弾・敵・ボス・横からの弾の予告 を見つける
  → 前のフレームとの位置の差で、弾の速さと向きを求める
  → これまでと同じ「状況の文章」と「行動ごとの予測結果」を作る（shooter.Game.observe をそのまま使う）
  → d1-3B が行動を選ぶ → キー入力（L/R/U/D）を UDP でゲームへ送る

カメラで見てから動くまでの遅れ（画面の表示・カメラ・判断）は、最初に測って予測で見越す（--lead auto）。
"""
import argparse
import json
import math
import os
import socket
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from cam_screen import Camera, ScreenReader
from shooter import FH, FPS, FW, MODES, Game, Obj, instr_for

KEYS = {"move_left": "L", "move_right": "R", "move_up": "U", "move_down": "D", "stay": "",
        "move_up_left": "UL", "move_up_right": "UR", "move_down_left": "DL", "move_down_right": "DR",
        "slow_left": "SL", "slow_right": "SR", "slow_up": "SU", "slow_down": "SD"}


# ------------------------------------------------------------ 画像 → 見えている物
class Perception:
    """色と大きさで物を見分ける。数値は、カメラの露出 -7 で、ゲームの本当の位置と比べて決めた（robot/tune_perception.py）。"""
    DEFAULT = dict(tri_h=(60, 112), tri_v=90, bul_s=35, bul_v=120, bul_a=25, open_k=7, boss_v=100)

    def __init__(self, **kw):
        self.p = {**self.DEFAULT, **kw}
        self.ship = None             # 前のフレームの自機の位置（自機らしい物が複数あるときは近いほうを選ぶ）

    def find(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
        out = {"ship": None, "bullets": [], "enemies": [], "boss": None, "warnings": []}
        # 自機と敵：緑〜水色の明るい三角。色はカメラの明るさで変わるので、三角の向きで見分ける（自機は上向き、敵は下向き）
        P = self.p
        m = (((H >= P["tri_h"][0]) & (H <= P["tri_h"][1]) & (V >= P["tri_v"])) & (np.arange(FH)[:, None] > 28)).astype(np.uint8)
        m[FH - 8:] = 0                  # 画面のふち（白い余白がにじむ）は使わない。自機がふちに触れてもくっつかないように
        m[:, :3] = 0
        m[:, FW - 3:] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["open_k"], P["open_k"])))   # 自機の弾（細い線）を消す
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        cands = []
        for k in range(1, n):
            x, y, w, h, a = st[k]
            if not (100 <= a <= 2500 and 12 <= w <= 60 and 14 <= h <= 46 and 0.25 <= a / (w * h) <= 0.85):
                continue
            rows = (lab[y:y + h, x:x + w] == k).sum(1)
            t3 = max(1, h // 3)
            up = rows[:t3].mean() < rows[-t3:].mean()      # 上が細い ＝ 上向き ＝ 自機
            # 位置は形の重心から求める（囲む四角の中心は、動いたときの残像で伸び縮みしてぶれる）
            # 自機の三角の重心は、自機の位置より 2.7 下。敵の三角の重心は、敵の位置より 3 上
            if up:
                cands.append((cen[k][0], cen[k][1] - 2.7, a))
            else:
                out["enemies"].append((cen[k][0], cen[k][1] + 3))
        if cands:
            if self.ship is not None:
                c = min(cands, key=lambda c: math.hypot(c[0] - self.ship[0], c[1] - self.ship[1]))
            else:
                c = max(cands, key=lambda c: c[2])
            out["ship"] = (c[0], c[1])
            self.ship = out["ship"]
        # 弾・横からの弾の予告線：ピンク〜赤
        m = ((((H >= 138) | (H <= 8)) & (S >= P["bul_s"]) & (V >= P["bul_v"])) & (np.arange(FH)[:, None] > 28)).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        for k in range(1, n):
            x, y, w, h, a = st[k]
            if w > 100 and h < 30:                       # 横いっぱいの細い線 ＝ 横からの弾の予告
                cols = (lab[y:y + h, x:x + w] == k).sum(0)
                left = cols[:20].max() > cols[-20:].max()     # 三角の印がある側から飛んでくる
                out["warnings"].append((y + h / 2, left))
            elif a >= P["bul_a"] and w <= 70 and h <= 70:
                if a > 260:                               # 弾が重なって見えるときは、大きい弾として扱う
                    out["bullets"].append((cen[k][0], cen[k][1], max(w, h) / 2))
                else:
                    out["bullets"].append((cen[k][0], cen[k][1], 5))
        # ボス：紫の大きな楕円
        m = ((H >= 120) & (H <= 145) & (S >= 60) & (V >= P["boss_v"])).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        big = [k for k in range(1, n) if st[k][4] > 1500]
        if big:
            k = max(big, key=lambda k: st[k][4])
            x, y, w, h, a = st[k]
            out["boss"] = (x + w / 2, y + h / 2)
            bx, by = out["boss"]                          # ボスの中の明るい部分を弾と見まちがえないように
            out["bullets"] = [b for b in out["bullets"] if not (abs(b[0] - bx) < 52 and abs(b[1] - by) < 30)]
        return out


class Tracker:
    """前のフレームで見えた物と近いものを同じ物とみなして、速さ（1 ゲームフレームあたりの px）を求める。"""

    def __init__(self, default_v=(0.0, 3.0), radius=24.0, side=False):
        self.prev, self.default_v, self.radius, self.side = [], default_v, radius, side

    def update(self, pts, dt_frames):
        out, used = [], set()
        for p in pts:
            best, bd = None, self.radius
            for j, q in enumerate(self.prev):
                if j in used:
                    continue
                # 前の速さで進めた位置との差で比べる
                px, py = q[0] + q[2] * dt_frames, q[1] + q[3] * dt_frames
                d = math.hypot(p[0] - px, p[1] - py)
                if d < bd:
                    best, bd = j, d
            if best is None:
                vx, vy = self.default_v
                if self.side and (p[0] < 30 or p[0] > FW - 30):      # 画面の端に急に出てきた弾 ＝ 横からの弾
                    vx, vy = (4.5 if p[0] < FW / 2 else -4.5), 0.0
            else:
                used.add(best)
                q = self.prev[best]
                vx = 0.5 * q[2] + 0.5 * (p[0] - q[0]) / dt_frames
                vy = 0.5 * q[3] + 0.5 * (p[1] - q[1]) / dt_frames
            out.append((p[0], p[1], vx, vy) + tuple(p[2:]))
        self.prev = out
        return out


# ------------------------------------------------------------ 見えている物 → ゲームの状況（これまでと同じ形）
class StateBuilder:
    def __init__(self, mode, margin=4.0):
        self.mode = mode
        self.margin = margin            # カメラの読み取りのぶれと遅れの揺れの分、弾を大きめに見積もる（px）
        self.bullets = Tracker((0.0, 3.0), side=True)
        self.warn_mem = []            # 横からの弾の予告線は点滅するので、少しの間覚えておく（y, 左から？, 見えた時刻）
        self.enemies = Tracker((0.0, 1.6))
        self.boss = Tracker((2.0, 0.0), radius=60)
        self.last_ship = (FW / 2, MODES[mode]["home"])

    def build(self, seen, dt_frames, lead, cur):
        g = Game(self.mode)
        g.frame = 1000
        g.wi = len(g.waves)                       # 出現の予定は使わない（見えている物だけで判断する）
        g.lead, g.cur = lead, cur
        if seen["ship"]:
            self.last_ship = seen["ship"]
        g.player.x = min(max(self.last_ship[0], 15), FW - 15)
        g.player.y = min(max(self.last_ship[1], g.cfg["top"]), g.cfg["bottom"])
        g.bullets = [Obj(x, y, vx, vy, r + self.margin) for x, y, vx, vy, r in self.bullets.update(seen["bullets"], dt_frames)]
        g.enemies = [Obj(x, y, 0, max(vy, 0.5), 15, kind="basic", hp=1)
                     for x, y, vx, vy in self.enemies.update(seen["enemies"], dt_frames)]
        bs = self.boss.update([seen["boss"]] if seen["boss"] else [], dt_frames)
        if bs:
            x, y, vx, vy = bs[0]
            g.boss = Obj(x, y, 2.0 if vx >= 0 else -2.0, 0.0, 38, hp=1, cool=60)
        now = time.perf_counter()
        for y, left in seen["warnings"]:
            self.warn_mem = [w for w in self.warn_mem if abs(w[0] - y) > 10]
            self.warn_mem.append((y, left, now))
        self.warn_mem = [w for w in self.warn_mem if now - w[2] < 0.35]
        g.warnings = [Obj(-12 if left else FW + 12, y, 4.5 if left else -4.5, 0, 6, fire=g.frame + 15)
                      for y, left, _ in self.warn_mem]
        return g


# ------------------------------------------------------------ B：画像をそのまま d1-3B に見せる
IMG_STATE = ("Screenshot of a vertical shooter game seen from the camera. Your ship is the cyan triangle pointing up near the bottom. "
             "Small pink dots are enemy bullets; touching any bullet or enemy loses the game. Green triangles pointing down are enemies, "
             "the purple oval at the top is the boss. Your ship fires upward automatically, so being under the boss or an enemy hits it.")
IMG_Q = {"type": "choice",
         "instructions": "Which move keeps the ship away from bullets and, if safe, lines it up under the boss?",
         "criteria": {"move_left": "move left", "move_right": "move right", "stay": "stay still",
                      "move_up": "move up", "move_down": "move down"}}


def trail(cur, prev):
    """今の画像に、少し前の画像を半分の明るさで重ねる（残像で動く向きが分かる）。"""
    return np.maximum(cur, (prev.astype(np.uint16) // 2).astype(np.uint8))


class ImageBrain:
    def __init__(self, mode, size=256):
        from d1_image_fast import D1ImageFast
        self.m = D1ImageFast(IMG_STATE, IMG_Q, size=(size, size), quant="fp8", graph="cuda")
        self.ms = []

    def decide_image(self, rgb):
        choice, probs, ms = self.m.decide(rgb)
        self.ms.append(ms)
        return choice


# ------------------------------------------------------------ キー入力を送る
class Keys:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ("127.0.0.1", port)
        self.keys = ""
        self.running = True
        threading.Thread(target=self._beat, daemon=True).start()

    def send(self, keys):
        self.keys = keys
        self.sock.sendto(keys.encode(), self.addr)

    def _beat(self):                     # 同じキーを押し続けていることを伝え続ける（届かないとゲームは離したとみなす）
        while self.running:
            self.sock.sendto(self.keys.encode(), self.addr)
            time.sleep(0.05)


def measure_lag(cam, rd, per, keys, tries=3):
    """キーを送ってから、カメラで自機が動いて見えるまでの時間（＝見てから動くまでの遅れ）を測る。"""
    lags = []
    last = -1
    for i in range(tries):
        key = "R" if i % 2 == 0 else "L"
        keys.send("")
        t_end = time.perf_counter() + 0.5
        x0 = None
        while time.perf_counter() < t_end:
            f, t, last = cam.read(last)
            rd.update(f)
            s = per.find(rd.warp(f))["ship"] if rd.ready() else None
            if s:
                x0 = s[0]
        if x0 is None:
            continue
        t0 = time.perf_counter()
        keys.send(key)
        while time.perf_counter() - t0 < 1.0:
            f, t, last = cam.read(last)
            rd.update(f)
            s = per.find(rd.warp(f))["ship"] if rd.ready() else None
            if s and abs(s[0] - x0) > 4:
                lags.append(t - t0)
                break
        keys.send("")
    keys.send("")
    return sorted(lags)[len(lags) // 2] if lags else 0.1


# ------------------------------------------------------------ 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="d1", choices=["d1", "rule", "d1img"],
                    help="d1=見つけた物を文章にして d1-3B が判断（A）／d1img=画像をそのまま d1-3B に見せて判断（B）／rule=ルール")
    ap.add_argument("--size", type=int, default=256, help="d1img：d1-3B に見せる画像の一辺")
    ap.add_argument("--mode", default="free")
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--exposure", type=float, default=-7, help="カメラの露出（自動だと画面の明るさで色が変わる）")
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--lead", default="auto", help="見越す遅れ（ゲームのフレーム数）。auto＝最初に測る")
    ap.add_argument("--show", action="store_true", help="AI が見ている画像を窓に出す")
    ap.add_argument("--margin", type=float, default=4.0, help="弾を何 px 大きく見積もるか")
    ap.add_argument("--hold", type=float, default=0.1, help="一度決めた行動を続ける時間（秒）。元のゲームの 6 フレームに合わせる")
    ap.add_argument("--seconds", type=float, default=200)
    ap.add_argument("--record", default="", help="B の追加学習用：256x256 の画像（今と 2 フレーム前）と、先生（ルール）の答えを保存するフォルダ")
    ap.add_argument("--explore", type=float, default=0.0, help="記録用：この割合で、ときどきでたらめな行動を 0.3 秒混ぜる（いろいろな場面を集める）")
    ap.add_argument("--save", default="", help="AI が見ている画像（印つき）を保存するフォルダ")
    a = ap.parse_args()

    from brain import D1Brain, RuleBrain, ShortPrompt
    if a.brain == "d1":
        from short_v2 import short_option_v2, short_state
        brain = ShortPrompt(D1Brain(instr_for("B"), "fp8", "cuda"), short_state, lambda t: short_option_v2(t, "B"))
    elif a.brain == "d1img":
        brain = ImageBrain(a.mode, a.size)
    else:
        brain = RuleBrain()

    cam = Camera(a.cam, exposure=a.exposure)
    rd, per = ScreenReader(), Perception()
    keys = Keys(a.port)
    sb = StateBuilder(a.mode, a.margin)
    keys.send("")                                          # ゲームを始める
    lag_s = measure_lag(cam, rd, per, keys)
    print(f"見てから動くまでの遅れ：{lag_s * 1000:.0f} ms", flush=True)

    os.makedirs("logs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = open(f"logs/cam_{a.mode}_{stamp}_{a.brain}.jsonl", "w", encoding="utf-8")
    if a.show:
        cv2.namedWindow("AI view", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI view", 360, 450)
        cv2.moveWindow("AI view", 760, 0)
    action, last, t_prev = "stay", -1, None
    hold_until, decided = 0.0, 0
    import random
    rnd, explore_until, explore_act = random.Random(), 0.0, "stay"
    prev256 = []
    rec = None
    if a.record:
        os.makedirs(a.record, exist_ok=True)
        rec = open(os.path.join(a.record, f"labels_{stamp}.jsonl"), "w", encoding="utf-8")
    n, no_marker, ms_think, ms_all = 0, 0.0, [], []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < a.seconds:
        f, t_cap, last = cam.read(last)
        t0 = time.perf_counter()
        rd.update(f)
        if len(rd.seen) < 2:
            no_marker = no_marker or time.perf_counter()
            if time.perf_counter() - no_marker > 3:         # ゲームの窓が閉じた（終わった）
                break
            continue
        no_marker = 0.0
        img = rd.warp(f)
        seen = per.find(img)
        dt_frames = (t_cap - t_prev) * FPS if t_prev else 2.0
        t_prev = t_cap
        # 見越す遅れ：カメラで見てから動くまで ＋ 判断にかかる時間
        think = (np.median(ms_think[-20:]) / 1000) if ms_think else 0.03
        lead = int(a.lead) if a.lead != "auto" else max(1, round((lag_s + think) * FPS))
        from shooter import move_vec
        g = sb.build(seen, dt_frames, lead, move_vec(a.mode, action))
        state, rule, crit = g.observe()
        t1 = time.perf_counter()
        # 一度決めた行動は hold 秒続ける（毎フレーム決め直すと、遅れのせいで左右に揺れて逃げられない）。
        # ただし、続けると当たると予測されたら、すぐに決め直す
        if a.brain == "d1img":                   # B：画像だけで判断する（予測の文章は使わない）
            if time.perf_counter() < hold_until:
                new = action
            else:
                new = brain.decide_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                hold_until = time.perf_counter() + a.hold
                decided += 1
        elif time.perf_counter() < hold_until and "gets hit" not in crit.get(action, "gets hit"):
            new = action
        else:
            d = brain.decide(state, crit, rule)
            new = d.action if d.action in KEYS else "stay"
            hold_until = time.perf_counter() + a.hold
            decided += 1
        if a.explore and a.brain == "rule":       # 記録用：ときどきでたらめに動いて、いろいろな場面を作る
            now = time.perf_counter()
            if now >= explore_until and rnd.random() < a.explore / 9:
                explore_until, explore_act = now + 0.3, rnd.choice(list(crit))
            if now < explore_until:
                new = explore_act
        if rec is not None:                       # 今の画像・2 フレーム前の画像・先生（ルール）の答えと、安全な行動の一覧
            small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            prev256.append(small)
            if len(prev256) > 3:
                prev256.pop(0)
            name = f"{stamp}_{n:05d}"
            cv2.imwrite(os.path.join(a.record, name + ".png"), small)
            rec.write(json.dumps({"img": name + ".png", "prev": f"{stamp}_{max(0, n - 2):05d}.png", "rule": rule,
                                  "safe": [k for k, v in crit.items() if "gets hit" not in v],
                                  "best": [k for k, v in crit.items() if "lined up so" in v and "gets hit" not in v],
                                  "acted": new}) + "\n")
        t2 = time.perf_counter()
        action = new
        keys.send(KEYS[action])
        ms_think.append((t2 - t1) * 1000)
        ms_all.append((t2 - t_cap) * 1000)
        log.write(json.dumps({"t": round(t_cap - t_start, 3), "wall": time.time() - (time.perf_counter() - t_cap),
                              "bullets": [[round(b.x, 1), round(b.y, 1), round(b.vx, 2), round(b.vy, 2)] for b in g.bullets], "ship": seen["ship"], "n_bullets": len(seen["bullets"]),
                              "boss": seen["boss"] is not None, "lead": lead, "rule": rule, "action": action,
                              "think_ms": round((t2 - t1) * 1000, 1), "see_ms": round((t1 - t0) * 1000, 1),
                              "state": state, "criteria": crit}, ensure_ascii=False) + "\n")
        n += 1
        if a.show or a.save:
            show = img.copy()
            if seen["ship"]:
                cv2.circle(show, (int(seen["ship"][0]), int(seen["ship"][1])), 14, (255, 255, 0), 2)
            for b in g.bullets:
                cv2.circle(show, (int(b.x), int(b.y)), int(b.r) + 3, (0, 0, 255), 1)
                cv2.line(show, (int(b.x), int(b.y)), (int(b.x + b.vx * 8), int(b.y + b.vy * 8)), (0, 255, 255), 1)
            for e in g.enemies:
                cv2.circle(show, (int(e.x), int(e.y)), 16, (0, 255, 0), 1)
            if g.boss:
                cv2.circle(show, (int(g.boss.x), int(g.boss.y)), 40, (255, 0, 255), 1)
            cv2.putText(show, f"{action}  think {ms_think[-1]:.0f}ms  lead {lead}f", (6, 590),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if a.save:
                os.makedirs(a.save, exist_ok=True)
                cv2.imwrite(f"{a.save}/{n:05d}_{t_cap - t_start:.2f}.jpg", show)
                cv2.imwrite(f"{a.save}/{n:05d}_{t_cap - t_start:.2f}_raw.png", img)      # 印なし（見分け方の確認用）
            if a.show:
                cv2.imshow("AI view", show)
                if n == 1:
                    cv2.setWindowProperty("AI view", cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)
    keys.send("")
    keys.running = False
    wall = time.perf_counter() - t_start
    summary = {"brain": a.brain, "mode": a.mode, "margin": a.margin, "frames": n, "decisions": decided, "decisions_per_s": round(decided / wall, 1), "hold": a.hold,
               "lag_ms": round(lag_s * 1000), "d1img_ms_median": round(float(np.median(brain.ms)), 1) if getattr(brain, "ms", None) else None, "think_ms_median": round(float(np.median(ms_think)), 1) if ms_think else None,
               "camera_to_key_ms_median": round(float(np.median(ms_all)), 1) if ms_all else None, "seconds": round(wall, 1)}
    log.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    log.close()
    if rec is not None:
        rec.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    cam.close()


if __name__ == "__main__":
    main()
