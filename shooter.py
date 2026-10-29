"""縦シューティング「JEV SHOOTER」 — AI が自機を操作して、最後のボスを倒す。

  python shooter.py                        横移動だけのモードを、ルールで動く仮のAIで実行
  python shooter.py --mode free            上下左右に動けるモード（横からの弾・狙い撃ち・扇形の弾あり）
  python shooter.py --mode danmaku         弾幕モード（3段階のボス弾幕・ランダムな横からの弾・低速移動）
  python shooter.py --brain jev            Jev が判断して実行（--instruction No.N で費用報告の番号）
  python shooter.py --human                自分で操作（矢印キー、Shift で低速）
  python shooter.py --replay last          最新の Jev の記録を再生（無料）
  python shooter.py --headless --record videos/x.mp4   画面なしで実行して動画に保存

弾は自動で上に撃つ。AI は一定フレームごとに、状況と「各行動をとったらどうなるか」の
予測を英語の文章（State）にして、次の行動を Choice で選ぶ。
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime

from brain import D1Brain, Decision, JevBrain, RuleBrain, ShortPrompt
from budget import BudgetExceeded, Ledger

FW, FH = 480, 600          # 戦う場所の大きさ
PANEL_W = 560              # 右側の説明パネル
FPS = 60
U = 40                     # AI への説明で使う距離の単位（px）
SHOT_SPEED = 10.0
LIVES = 1
BOSS_AT = 5 * FPS          # ボスが出るまでの時間（5秒）
TIME_LIMIT = 180 * FPS
BASE_Y = 545

MODES = {
    "h": dict(label="横移動だけ", decide=6, speed=5.0, slow=None, r=7, top=BASE_Y, bottom=BASE_Y,
              boss_hp=45, shot_every=10, invincible=90, home=BASE_Y),
    "free": dict(label="上下左右", decide=6, speed=5.0, slow=None, r=7, top=300, bottom=575,
                 boss_hp=35, shot_every=10, invincible=90, home=BASE_Y),
    "danmaku": dict(label="弾幕", decide=4, speed=4.5, slow=1.8, r=3, top=230, bottom=580,
                    boss_hp=150, shot_every=5, invincible=120, home=520),
}
D = 1 / math.sqrt(2)
ALL_MOVES = {
    "move_left": (-1, 0, "Move left."),
    "move_right": (1, 0, "Move right."),
    "stay": (0, 0, "Stay."),
    "move_up": (0, -1, "Move up."),
    "move_down": (0, 1, "Move down."),
    "move_up_left": (-D, -D, "Move up-left."),
    "move_up_right": (D, -D, "Move up-right."),
    "move_down_left": (-D, D, "Move down-left."),
    "move_down_right": (D, D, "Move down-right."),
    "slow_left": (-1, 0, "Move slowly left."),
    "slow_right": (1, 0, "Move slowly right."),
    "slow_up": (0, -1, "Move slowly up."),
    "slow_down": (0, 1, "Move slowly down."),
}
MODE_ACTIONS = {
    "h": ["move_left", "move_right", "stay"],
    "free": ["move_left", "move_right", "stay", "move_up", "move_down"],
    "danmaku": ["move_left", "move_right", "stay", "move_up", "move_down", "move_up_left", "move_up_right",
                "move_down_left", "move_down_right", "slow_left", "slow_right", "slow_up", "slow_down"],
}
INSTRUCTIONS = (
    "You control a ship in a vertical shooter; it fires upward automatically. Getting hit loses the game; "
    "defeating the boss wins, so attacking matters. Each action's description gives its predicted result. "
    "Never pick an action that gets hit. Among safe actions, pick the one best for hitting the target. "
    "Which action now?"
)


def actions_for(mode):
    return {k: ALL_MOVES[k][2] for k in MODE_ACTIONS[mode]}


def move_vec(mode, action):
    dx, dy, _ = ALL_MOVES[action]
    cfg = MODES[mode]
    sp = cfg["slow"] if action.startswith("slow_") else cfg["speed"]
    return dx * sp, dy * sp


class Obj:
    def __init__(self, x, y, vx=0.0, vy=0.0, r=5, **kw):
        self.x, self.y, self.vx, self.vy, self.r = x, y, vx, vy, r
        self.__dict__.update(kw)


def half(v):
    """AI に渡す距離は 0.5 単位に丸める（同じ文章が増え、答えを使い回せる）"""
    return f"{round(v * 2) / 2:.1f}"


def hit(a, b, margin=0.0):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 <= (a.r + b.r + margin) ** 2


SHORT_INSTRUCTIONS = "Pick the best action. Never pick one that gets hit; prefer on target, then closer to target."


def instr_for(prompt):
    if prompt in ("A", "B", "C"):
        from short_v2 import INSTR
        return INSTR[prompt]
    return SHORT_INSTRUCTIONS if prompt == "short" else INSTRUCTIONS


def make_waves(mode, seed=0):
    """(出てくるフレーム, 種類, x) の一覧。乱数の種を固定しているので毎回同じ（seed を変えると別の並び）。ボスは BOSS_AT に出る。"""
    rnd = random.Random(7 + 100 * seed)
    w = []
    if mode == "danmaku":
        for i in range(5):                           # 妖精がリング弾を撃ってくる
            w.append((30 + i * 50, "fairy", rnd.choice([70, 150, 240, 330, 410])))
    else:
        for i in range(6):                           # 弱い敵がまっすぐ降りてくる
            w.append((30 + i * 40, "basic", rnd.choice([80, 160, 240, 320, 400])))
        if mode == "free":                           # 横から飛んでくる弾（予告のあと発射）
            for i in range(24):
                w.append((BOSS_AT + 120 + i * 100, "side", rnd.choice([0, 1])))
    w.append((BOSS_AT, "boss", FW / 2))
    return sorted(w, key=lambda t: t[0])


PHASE_NAMES = {1: "第1段階：回転弾幕", 2: "第2段階：リング＋狙い撃ち", 3: "第3段階：弾の雨"}


class Game:
    def __init__(self, mode, seed=0):
        self.mode = mode
        self.cfg = MODES[mode]
        self.rng = random.Random(11 + 100 * seed)
        self.waves = make_waves(mode, seed)
        self.wi = 0
        self.player = Obj(FW / 2, self.cfg["home"], r=self.cfg["r"])
        self.enemies, self.shots, self.bullets, self.warnings = [], [], [], []
        self.boss = None
        self.phase = 0
        self.lives, self.score, self.frame = LIVES, 0, 0
        self.invincible = 0
        self.result = None
        self.hits = []
        self.boss_max = self.cfg["boss_hp"]
        self.lead, self.cur = 0, (0.0, 0.0)     # 予測で見越す遅れ（フレーム）と、その間続く今の動き
        self.next_side = 420

    # --- 出現
    def spawn(self):
        while self.wi < len(self.waves) and self.waves[self.wi][0] <= self.frame:
            _, kind, x = self.waves[self.wi]
            self.wi += 1
            if kind == "basic":
                self.enemies.append(Obj(x, -20, 0, 1.6, 15, kind=kind, hp=1, cool=0, t=0, x0=x))
            elif kind == "shooter":
                self.enemies.append(Obj(x, -20, 0, 1.2, 16, kind=kind, hp=2, cool=40, t=0, x0=x, shots=0))
            elif kind == "zigzag":
                self.enemies.append(Obj(x, -20, 0, 1.3, 14, kind=kind, hp=1, cool=0, t=0, x0=x))
            elif kind == "fairy":
                self.enemies.append(Obj(x, -20, 0, 1.5, 13, kind=kind, hp=3, cool=30, t=0, x0=x, shots=0,
                                        stop=self.rng.uniform(90, 170)))
            elif kind == "boss":
                self.boss = Obj(FW / 2, -60, 2.0, 1.0, 38, hp=self.boss_max, cool=60, ang=0.0, t=0)
            elif kind == "side":
                y = min(max(self.player.y + random.Random(self.frame).choice([-30, 0, 30]), self.cfg["top"] + 10),
                        self.cfg["bottom"])
                self.add_warning(x == 0, y, 4.5)
        if self.mode == "danmaku" and self.frame >= self.next_side:   # ランダムな横からの弾
            for _ in range(self.rng.choice([1, 1, 2])):
                self.add_warning(self.rng.random() < 0.5, self.rng.uniform(self.cfg["top"], self.cfg["bottom"]),
                                 self.rng.uniform(3.8, 5.5))
            self.next_side = self.frame + self.rng.randint(80, 170)

    def add_warning(self, from_left, y, speed):
        self.warnings.append(Obj(-12 if from_left else FW + 12, y, speed if from_left else -speed, 0, 6,
                                 fire=self.frame + 45))

    def aimed(self, src, speed, spread=0.0, r=5):
        if self.mode == "h":
            return Obj(src.x, src.y + 18, 0, speed, r)
        a = math.atan2(self.player.y - src.y, self.player.x - src.x) + spread
        return Obj(src.x, src.y + 10, math.cos(a) * speed, math.sin(a) * speed, r)

    def ring(self, src, n, speed, offset=0.0, r=5, color=0):
        for k in range(n):
            a = offset + k * 2 * math.pi / n
            self.bullets.append(Obj(src.x, src.y, math.cos(a) * speed, math.sin(a) * speed, r, color=color))

    # --- ボスの攻撃
    def boss_attack(self, b):
        p = self.player
        if self.mode != "danmaku":
            b.cool -= 1
            if b.cool <= 0:
                rage = b.hp < self.boss_max / 2
                if self.mode == "h":
                    for ox in (-45, 0, 45):
                        self.bullets.append(Obj(b.x + ox, b.y + 30, 0, 3.4, 5))
                else:
                    base = math.atan2(p.y - b.y, p.x - b.x)
                    for k in (-2, -1, 0, 1, 2):
                        a = base + k * 0.32
                        self.bullets.append(Obj(b.x, b.y + 30, math.cos(a) * 3.2, math.sin(a) * 3.2, 5))
                b.cool = 38 if rage else 52
            return
        ratio = b.hp / self.boss_max
        phase = 1 if ratio > 2 / 3 else (2 if ratio > 1 / 3 else 3)
        if phase != self.phase:                       # 段階が変わったら弾を消す（スペルカードの切り替え）
            self.phase = phase
            self.bullets = [o for o in self.bullets if getattr(o, "side", False)]
            b.t = 0
        b.t += 1
        if phase == 1:
            if b.t % 5 == 0:
                for k in range(4):
                    a = b.ang + k * 2 * math.pi / 4
                    self.bullets.append(Obj(b.x, b.y, math.cos(a) * 2.3, math.sin(a) * 2.3, 5, color=1))
                    self.bullets.append(Obj(b.x, b.y, math.cos(-a) * 1.7, math.sin(-a) * 1.7, 4, color=2))
                b.ang += 0.21
        elif phase == 2:
            if b.t % 40 == 0:
                self.ring(b, 28, 2.0, offset=(b.t // 40) * 0.11, color=3)
            if b.t % 60 == 30:
                for s in (-0.2, -0.1, 0, 0.1, 0.2):
                    for sp in (2.8, 3.4, 4.0):
                        self.bullets.append(self.aimed(b, sp, s, r=4))
        else:
            if b.t % 4 == 0:
                self.bullets.append(Obj(self.rng.uniform(10, FW - 10), -5, self.rng.uniform(-0.4, 0.4),
                                        self.rng.uniform(2.0, 2.9), 4, color=4))
            if b.t % 50 == 0:
                self.ring(b, 18, 1.8, offset=self.rng.uniform(0, 1), r=6, color=5)

    # --- 1フレーム進める
    def step(self, dx, dy):
        if self.result:
            return
        self.frame += 1
        if self.frame >= TIME_LIMIT:
            self.result = "TIME UP"
            return
        self.spawn()
        p, cfg = self.player, self.cfg
        p.x = min(max(p.x + dx, 15), FW - 15)
        p.y = min(max(p.y + dy, cfg["top"]), cfg["bottom"])
        if self.frame % cfg["shot_every"] == 0:
            self.shots.append(Obj(p.x, p.y - 16, 0, -SHOT_SPEED, 4))

        for e in self.enemies:
            e.t += 1
            if e.kind == "zigzag":
                e.x = e.x0 + 70 * math.sin(e.t / 28)
                e.vx = 70 / 28 * math.cos(e.t / 28)
                e.y += e.vy
            elif e.kind == "shooter":
                if e.y < 130 and e.shots < 5:
                    e.y += e.vy
                elif e.shots < 5:
                    e.vy = 0
                    e.cool -= 1
                    if e.cool <= 0:
                        self.bullets.append(self.aimed(e, 3.4))
                        e.shots += 1
                        e.cool = 75
                else:
                    e.vy = 2.5
                    e.y += e.vy
            elif e.kind == "fairy":
                if e.y < e.stop and e.shots < 3:
                    e.y += e.vy
                elif e.shots < 3:
                    e.vy = 0
                    e.cool -= 1
                    if e.cool <= 0:
                        self.ring(e, 10, 1.9, offset=self.rng.uniform(0, 1), color=3)
                        e.shots += 1
                        e.cool = 70
                else:
                    e.vy = -1.8
                    e.y += e.vy
            else:
                e.y += e.vy
        b = self.boss
        if b:
            if b.y < 90 + (20 if self.mode == "danmaku" else 0):
                b.y += b.vy
            else:
                if self.mode == "danmaku":
                    b.vy = 0
                    nx = FW / 2 + 110 * math.sin(self.frame / 140)
                    b.vx, b.x = nx - b.x, nx
                else:
                    b.vy = 0
                    b.x += b.vx
                    if b.x < 60 or b.x > FW - 60:
                        b.vx = -b.vx
                self.boss_attack(b)
        for w in list(self.warnings):
            if self.frame >= w.fire:
                self.warnings.remove(w)
                self.bullets.append(Obj(w.x, w.y, w.vx, 0, 6, side=True))

        for o in self.shots + self.bullets:
            o.x += o.vx
            o.y += o.vy
        self.shots = [s for s in self.shots if s.y > -20]
        self.bullets = [o for o in self.bullets if -30 < o.x < FW + 30 and -30 < o.y < FH + 30]
        self.enemies = [e for e in self.enemies if -40 < e.y < FH + 30]

        for s in list(self.shots):
            for e in self.enemies:
                if hit(s, e):
                    e.hp -= 1
                    if s in self.shots:
                        self.shots.remove(s)
                    break
            else:
                if b and b.y >= 60 and hit(s, b):
                    b.hp -= 1
                    self.shots.remove(s)
        for e in [e for e in self.enemies if e.hp <= 0]:
            self.enemies.remove(e)
            self.score += 100
        if b and b.hp <= 0:
            self.score += 5000
            self.boss = None
            self.result = "CLEAR"
            return

        if self.invincible:
            self.invincible -= 1
            return
        for o in self.bullets + self.enemies + ([b] if b else []):
            if hit(p, o):
                if getattr(self, "god", False):     # 無敵：当たった回数と場面だけ数えて、そのまま続ける
                    self.hits.append({"frame": self.frame, "by": "boss" if o is b else
                                      ("side bullet" if getattr(o, "side", False) else
                                       ("bullet" if o in self.bullets else "enemy")), "x": round(p.x), "y": round(p.y)})
                    self.invincible = 30            # 同じ弾で何度も数えないよう、0.5 秒だけ当たらない
                    break
                self.lives -= 1
                self.hits.append({"frame": self.frame, "by": "boss" if o is b else
                                  ("side bullet" if getattr(o, "side", False) else
                                   ("bullet" if o in self.bullets else "enemy"))})
                # 当たったら周りの弾を消して、しばらく無敵
                self.bullets = [q for q in self.bullets if math.hypot(q.x - p.x, q.y - p.y) > 90]
                self.invincible = cfg["invincible"]
                if self.lives <= 0:
                    self.result = "GAME OVER"
                break

    # --- 先読み：その行動を少し続けて、そのあと止まったら当たるか（弾はまっすぐ進むとして計算）
    def threats(self, horizon):
        p = self.player
        out = []
        reach = 40 + horizon * (self.cfg["speed"] + 6)
        for o in self.bullets + self.enemies + ([self.boss] if self.boss else []):
            if abs(o.x - p.x) < reach and abs(o.y - p.y) < reach:
                out.append((o.x, o.y, o.vx, o.vy, o.r, 0))
        for w in self.warnings:
            t0 = w.fire - self.frame
            if t0 < horizon:
                out.append((w.x, w.y, w.vx, 0.0, w.r, t0))
        return out

    def simulate(self, action, things):
        cfg = self.cfg
        mx, my = move_vec(self.mode, action)
        n = cfg["decide"]
        L = self.lead
        px, py = self.player.x, self.player.y
        pr = cfg["r"]
        mind = 999.0
        for f in range(1, L + 3 * n + 1):
            if f <= L:                                   # 答えが出るまでは今の動きのまま
                px = min(max(px + self.cur[0], 15), FW - 15)
                py = min(max(py + self.cur[1], cfg["top"]), cfg["bottom"])
            elif f <= L + n:
                px = min(max(px + mx, 15), FW - 15)
                py = min(max(py + my, cfg["top"]), cfg["bottom"])
            for (x, y, vx, vy, r, t0) in things:
                if f < t0:
                    continue
                k = f - t0
                d = math.hypot(x + vx * k - px, y + vy * k - py) - r - pr
                if d < mind:
                    mind = d
                if d <= 1.5 and f > self.invincible:
                    return False, f, px, py, d
        return True, None, px, py, mind

    def aim_x(self, tgt, from_y, delay):
        """今から delay フレーム後に from_y から撃った弾が、狙いの高さに届くときの狙いの x（動く先を読む）"""
        k = delay + max(0.0, (from_y - 16 - tgt.y) / SHOT_SPEED)
        if tgt is self.boss and tgt.y >= 90:
            if self.mode == "danmaku":
                return FW / 2 + 110 * math.sin((self.frame + k) / 140)
            lo, hi = 60, FW - 60                      # 端で跳ね返る動き
            span = hi - lo
            t = (tgt.x + tgt.vx * k - lo) % (2 * span)
            return lo + (t if t <= span else 2 * span - t)
        return tgt.x + tgt.vx * k

    def target(self):
        if self.boss:
            return self.boss, "the boss"
        above = [e for e in self.enemies if e.y < self.player.y - 40 and e.y > 0]
        if not above:
            return None, None
        e = min(above, key=lambda e: abs(e.x - self.player.x) + 0.3 * (self.player.y - e.y))
        return e, "an enemy"

    def observe(self):
        p, cfg = self.player, self.cfg
        acts = MODE_ACTIONS[self.mode]
        lines = []
        pos = f"Ship: {half(p.x / U)} from left wall, {half((FW - p.x) / U)} from right wall"
        if cfg["top"] != cfg["bottom"]:
            pos += f", {half((p.y - cfg['top']) / U)} below top, {half((cfg['bottom'] - p.y) / U)} above bottom of its area"
        lines.append(pos + " (units).")
        if self.invincible:
            lines.append("Ship is invincible for a moment.")
        tgt, tname = self.target()
        if tgt is not None:
            # 動く相手は、弾が届くときの位置（狙う点）を伝える
            dxu = (self.aim_x(tgt, p.y, 0) - p.x) / U
            side = "directly above" if abs(dxu) < 0.25 else f"{half(abs(dxu))} {'right' if dxu > 0 else 'left'}"
            lines.append(f"Aim point ({'boss' if tname == 'the boss' else 'enemy'}, where the shots will meet it): "
                         f"{side}, {half((p.y - tgt.y) / U)} above.")
        else:
            lines.append("Target: none.")
        near = sorted(self.bullets, key=lambda o: math.hypot(o.x - p.x, o.y - p.y))
        close = [o for o in near if math.hypot(o.x - p.x, o.y - p.y) < 3 * U]
        if self.mode == "danmaku":
            lines.append(f"Bullets within 3 units: {len(close)}.")
        for o in near[:3]:
            if math.hypot(o.x - p.x, o.y - p.y) / U > 6:
                break
            dxu, dyu = (o.x - p.x) / U, (p.y - o.y) / U
            kind = "Side bullet" if getattr(o, "side", False) else "Bullet"
            lines.append(f"{kind}: {half(abs(dxu))} {'right' if dxu > 0 else 'left'}, "
                         f"{half(abs(dyu))} {'above' if dyu >= 0 else 'below'}.")
        if not near or math.hypot(near[0].x - p.x, near[0].y - p.y) / U > 6:
            lines.append("No bullet nearby.")
        for w in self.warnings:
            lines.append(f"Side bullet coming from the {'left' if w.vx > 0 else 'right'} in "
                         f"{(w.fire - self.frame) / FPS:.1f}s, {half(abs(w.y - p.y) / U)} "
                         f"{'above' if w.y < p.y else 'below'} the ship.")

        things = self.threats(3 * cfg["decide"] + self.lead)
        results, crit = {}, {}
        for a in acts:
            safe, when, qx, qy, mind = self.simulate(a, things)
            aim = self.aim_x(tgt, qy, cfg["decide"]) if tgt is not None else None
            lined = tgt is not None and abs(aim - qx) < max(14, tgt.r * 0.6)
            results[a] = (safe, when, qx, qy, mind, lined, aim)
            move = ALL_MOVES[a][2]
            if not safe:
                crit[a] = move + " Result: the ship gets hit and the game is lost (never choose)."
                continue
            room = "" if self.mode != "danmaku" else (
                " with lots of room" if mind > 1.2 * U else (" with some room" if mind > 0.4 * U
                                                            else ", but a bullet passes very close (risky)"))
            if tgt is None:
                effect = "."
            elif lined:
                effect = ", and the ship is lined up so its shots hit the target (best for attacking)."
            elif abs(qx - p.x) > 1:                   # 近づく・離れるは左右に動いたときだけ判定する
                before, after = abs(self.aim_x(tgt, qy, cfg["decide"]) - p.x), abs(aim - qx)
                effect = (", and gets closer to lining up with the target (good for attacking)." if after < before - 3 else
                          (", but moves away from the target (bad for attacking)." if after > before + 3 else "."))
            else:
                effect = ", but does not help line up with the target."
            crit[a] = move + " Result: safe" + room + effect

        # ルールAI：安全な行動のうち、弾から離れていて狙いに近づくものを選ぶ
        def score(a):
            safe, when, qx, qy, mind, lined, aim = results[a]
            if not safe:
                return 10000 - when
            s = abs(aim - qx) * (0.25 if self.mode == "danmaku" else 1.0) if tgt is not None else abs(qx - FW / 2) * 0.2
            if cfg["top"] != cfg["bottom"]:
                s += abs(qy - cfg["home"]) * (0.15 if self.mode == "danmaku" else 0.3)
            if self.mode == "danmaku":
                s -= min(mind, 50) * 3
            return s
        rule = min(acts, key=score)
        return "\n".join(lines), rule, crit


# ---------------------------------------------------------------- 描画
BULLET_COLORS = {0: (255, 90, 160), 1: (110, 170, 255), 2: (180, 120, 255), 3: (255, 210, 90),
                 4: (120, 230, 255), 5: (255, 120, 90)}


MARK = 96                  # 目印（ArUco）の一辺（px）
MARGIN = MARK + 16         # 目印を置く白い余白


def marker_layout():
    """--markers のときの目印の位置。{番号: 左上の点（戦う場所の左上を (0,0) とする座標）}。カメラ側（cam_player.py）も使う。"""
    g = 8
    return {0: (-MARK - g, -MARK - g), 1: (FW + g, -MARK - g), 2: (FW + g, FH + g), 3: (-MARK - g, FH + g),
            # 各辺の真ん中（カメラのレンズのゆがみを直すため）
            4: (FW / 2 - MARK / 2, -MARK - g), 5: (FW + g, FH / 2 - MARK / 2),
            6: (FW / 2 - MARK / 2, FH + g), 7: (-MARK - g, FH / 2 - MARK / 2)}


class View:
    def __init__(self, pygame, markers=False):
        self.pg = pygame
        self.markers = markers
        if markers:                # Web カメラで読むための目印を、戦う場所の四隅の外に置く
            import cv2
            # 説明パネルは出さず、戦う場所と目印だけにする（カメラの映像の窓を横に並べられるように）
            import os as _os
            _os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "8,32")
            self.display = pygame.display.set_mode((FW + 2 * MARGIN, FH + 2 * MARGIN))
            try:                       # いつも一番手前に表示する（ほかの窓に隠れるとカメラから見えない）
                import ctypes
                hwnd = pygame.display.get_wm_info()["window"]
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
            except Exception:
                pass
            self.screen = pygame.Surface((FW + PANEL_W, FH))
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.marks = {}
            for k, xy in marker_layout().items():
                img = cv2.aruco.generateImageMarker(d, k, MARK)
                surf = pygame.image.frombuffer(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).tobytes(), (MARK, MARK), "RGB")
                self.marks[k] = (surf.copy(), (MARGIN + int(xy[0]), MARGIN + int(xy[1])))
        else:
            self.screen = pygame.display.set_mode((FW + PANEL_W, FH))
        pygame.display.set_caption("JEV SHOOTER")
        self.font = pygame.font.SysFont("meiryo,yugothicui,msgothic", 16)
        self.small = pygame.font.SysFont("consolas,meiryo", 12)
        self.big = pygame.font.SysFont("meiryo,yugothicui,msgothic", 40, bold=True)
        # 背景：濃い紺のグラデーションに、暗くて小さい星（弾と見間違えないように白く光らせない）
        self.bg = pygame.Surface((FW, FH))
        for y in range(FH):
            k = y / FH
            self.bg.fill((int(6 + 10 * k), int(8 + 12 * k), int(26 + 22 * k)), (0, y, FW, 1))
        rnd = random.Random(3)
        self.stars = [(rnd.randrange(FW), rnd.randrange(FH)) for _ in range(60)]

    def draw(self, g, brain_name, acts, last, thinking, stats):
        pg, s = self.pg, self.screen
        s.blit(self.bg, (0, 0))
        for (x, y) in self.stars:
            s.set_at((x, int(y + g.frame * 0.3) % FH), (45, 55, 90))
        if g.cfg["top"] != g.cfg["bottom"]:
            pg.draw.rect(s, (25, 35, 70), (0, g.cfg["top"] - 8, FW, g.cfg["bottom"] - g.cfg["top"] + 16), 1)
        for w in g.warnings:
            if (g.frame // 6) % 2:
                pg.draw.line(s, (255, 90, 90), (0, w.y), (FW, w.y), 1)
                tx = 6 if w.vx > 0 else FW - 22
                pg.draw.polygon(s, (255, 90, 90), [(tx, w.y - 7), (tx + 14, w.y), (tx, w.y + 7)] if w.vx > 0
                                else [(tx + 14, w.y - 7), (tx, w.y), (tx + 14, w.y + 7)])
        for e in g.enemies:
            col = {"basic": (90, 200, 120), "shooter": (240, 150, 60), "zigzag": (120, 160, 255),
                   "fairy": (255, 170, 220)}[e.kind]
            if e.kind == "fairy":
                pg.draw.circle(s, col, (int(e.x), int(e.y)), e.r)
                pg.draw.ellipse(s, (200, 230, 255), (e.x - e.r - 10, e.y - 8, 12, 16), 2)
                pg.draw.ellipse(s, (200, 230, 255), (e.x + e.r - 2, e.y - 8, 12, 16), 2)
            else:
                pg.draw.polygon(s, col, [(e.x, e.y + e.r), (e.x - e.r, e.y - e.r * 0.6), (e.x + e.r, e.y - e.r * 0.6)])
            pg.draw.circle(s, (255, 255, 255), (int(e.x), int(e.y - 2)), 3)
        b = g.boss
        if b:
            pg.draw.ellipse(s, (170, 70, 210), (b.x - 50, b.y - 28, 100, 56))
            pg.draw.ellipse(s, (230, 150, 255), (b.x - 22, b.y - 14, 44, 28))
            pg.draw.rect(s, (60, 20, 70), (FW / 2 - 150, 10, 300, 10))
            pg.draw.rect(s, (230, 80, 160), (FW / 2 - 150, 10, 300 * b.hp / g.boss_max, 10))
            if g.mode == "danmaku" and g.phase:
                s.blit(self.font.render(PHASE_NAMES[g.phase], True, (255, 200, 240)), (10, 24))
        for o in g.shots:
            pg.draw.line(s, (255, 240, 120), (o.x, o.y), (o.x, o.y + 10), 3)
        for o in g.bullets:
            col = (255, 70, 70) if getattr(o, "side", False) else BULLET_COLORS[getattr(o, "color", 0)]
            pg.draw.circle(s, col, (int(o.x), int(o.y)), o.r)
            pg.draw.circle(s, (255, 245, 250), (int(o.x), int(o.y)), max(1, o.r - 3))
        p = g.player
        if not g.invincible or (g.invincible // 5) % 2:
            pg.draw.polygon(s, (80, 230, 255), [(p.x, p.y - 16), (p.x - 13, p.y + 12), (p.x + 13, p.y + 12)])
            pg.draw.circle(s, (255, 255, 255), (int(p.x), int(p.y)), p.r)
            if g.mode == "danmaku":
                pg.draw.circle(s, (255, 60, 60), (int(p.x), int(p.y)), p.r, 1)
        if g.result:
            msg = {"CLEAR": "STAGE CLEAR!", "GAME OVER": "GAME OVER", "TIME UP": "TIME UP",
                   "BUDGET STOP": "費用の上限で中断", "END": "記録はここまで"}[g.result]
            t = self.big.render(msg, True, (255, 255, 255))
            pg.draw.rect(s, (15, 36, 71), (FW / 2 - t.get_width() / 2 - 24, 230, t.get_width() + 48, 80), border_radius=12)
            s.blit(t, (FW / 2 - t.get_width() / 2, 245))

        # 右のパネル
        x0 = FW
        pg.draw.rect(s, (15, 36, 71), (x0, 0, PANEL_W, FH))
        s.blit(self.font.render(f"AI: {brain_name.upper()}   モード: {g.cfg['label']}   弾 {len(g.bullets)}",
                                True, (230, 235, 245)), (x0 + 14, 8))
        s.blit(self.font.render(f"LIFE {g.lives}   SCORE {g.score}   TIME {g.frame / FPS:5.1f}s", True, (230, 235, 245)), (x0 + 14, 30))
        if thinking:
            who = "d1-3B" if brain_name.startswith("d1") else "Jev" if brain_name.startswith("jev") else "AI"
            s.blit(self.font.render(f"{who} が考え中" +"." * (int(time.time() * 4) % 4), True, (255, 120, 200)), (x0 + 330, 30))
        s.blit(self.font.render("AI に渡している状況（State）", True, (150, 180, 230)), (x0 + 14, 56))
        if last:
            y = 80
            for line in last["state"].split("\n"):
                for part in textwrap.wrap(line, 76):
                    if y > FH - 150:
                        break
                    s.blit(self.small.render(part, True, (230, 235, 245)), (x0 + 14, y))
                    y += 13
            d = last["decision"]
            y0 = FH - 140
            s.blit(self.font.render("判断（Choice）上位5つ", True, (150, 180, 230)), (x0 + 14, y0))
            probs = d.probabilities or {d.action: 1.0}
            top = sorted(acts, key=lambda a: -probs.get(a, 0.0))[:5]
            for i, a in enumerate(top):
                pv = probs.get(a, 0.0)
                yy = y0 + 24 + i * 18
                col = (255, 120, 200) if a == d.action else (120, 140, 170)
                s.blit(self.small.render(a, True, col), (x0 + 14, yy))
                pg.draw.rect(s, (40, 60, 100), (x0 + 140, yy + 2, 220, 11))
                pg.draw.rect(s, col, (x0 + 140, yy + 2, int(220 * pv), 11))
                s.blit(self.small.render(f"{pv:.2f}", True, (230, 235, 245)), (x0 + 368, yy))
            info = f"source={d.source}" + (f"  conf={d.confidence:.2f}" if d.confidence is not None else "")
            if stats.get("jev_calls"):
                info += f"  calls={stats['jev_calls']}"
            s.blit(self.small.render(info, True, (180, 190, 210)), (x0 + 14, FH - 18))
        if self.markers:
            self.display.fill((255, 255, 255))
            self.display.blit(s, (MARGIN, MARGIN), (0, 0, FW, FH))
            for surf, pos in self.marks.values():
                self.display.blit(surf, pos)
        pg.display.flip()


class ReplayBrain(RuleBrain):
    name = "replay"

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        self.rows = [r for r in rows if "action" in r]
        summ = next((r["summary"] for r in rows if "summary" in r), {})
        self.src = summ.get("brain", "jev")
        self.seed, self.realtime = summ.get("seed", 0), summ.get("realtime", False)
        self.i = 0

    def decide(self, state_text, actions, rule_action):
        if self.i >= len(self.rows):
            self.ended = True                       # 記録の終わり。ここで再生をやめる（ルールAIで続けない）
            return Decision("stay" if "stay" in actions else "wait", "end")
        r = self.rows[self.i]
        self.i += 1
        return Decision(r["action"], r.get("source", "?"), r.get("confidence"), r.get("probabilities"))


# ---------------------------------------------------------------- 実行
def main():
    ap = argparse.ArgumentParser(description="JEV SHOOTER")
    ap.add_argument("--mode", choices=list(MODES), default="h", help="h=横移動だけ / free=上下左右 / danmaku=弾幕")
    ap.add_argument("--brain", choices=["rule", "jev", "d1"], default="rule")
    ap.add_argument("--prompt", choices=["full", "short", "A", "B", "C"], default="full",
                    help="short=問題文を短く書き直して渡す / A・B・C=短い問題文の改良版（short_v2.py）")
    ap.add_argument("--lead", default="0", help="予測で見越す遅れ（フレーム数）。auto=最近の答えにかかった時間から決める")
    ap.add_argument("--decide", type=int, default=0, help="何フレームごとに AI に聞くか（0=モードの決まった値、1=答えが出たらすぐ次を聞く）")
    ap.add_argument("--live", action="store_true",
                    help="画面ありのリアルタイムで、答えを待つ間もゲームを止めずに 60FPS で進める（実際の時計で遅れが決まる）")
    ap.add_argument("--seed", type=int, default=0, help="敵や弾の出方を変える（0 がこれまでと同じ）")
    ap.add_argument("--realtime", action="store_true",
                    help="AI が考えている間もゲームを止めない（答えにかかった時間ぶん、前の動きのまま進む）")
    ap.add_argument("--quant", default="bf16", help="d1 の軽くし方（d1_fast.py）")
    ap.add_argument("--graph", default="cuda", help="d1 の動かし方（d1_fast.py）")
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--replay", help="記録を再生する。last で最新の Jev の記録")
    ap.add_argument("--instruction", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--dump", help="毎フレームの本当の位置（自機・弾・敵・ボス）を時刻つきで保存する（カメラの読み取りの答え合わせ用）")
    ap.add_argument("--god", action="store_true", help="当たっても終わらない。当たった回数と場面を記録して、クリアまで続ける（AI の弱点を 1 回で調べる）")
    ap.add_argument("--markers", action="store_true", help="Web カメラで画面を読むための目印（ArUco）を四隅に出す")
    ap.add_argument("--udp", type=int, default=0, help="--human のとき、このポートで受けたキー入力（cam_player.py から）も使う")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--record", help="画面を動画（mp4）に保存する")
    args = ap.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.init()
    view = View(pygame, markers=args.markers) if (not args.headless or args.record) else None
    udp_keys, udp_t, sock = "", 0.0, None
    dump = open(args.dump, "w", encoding="utf-8") if args.dump else None
    if args.udp:                     # 別のプログラム（cam_player.py）からのキー入力。"LRUDS" の押されている文字を送ってくる
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", args.udp))
        sock.setblocking(False)
    clock = pygame.time.Clock()
    rec = None
    if args.record:
        import imageio_ffmpeg
        rec = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                "-pix_fmt", "rgb24", "-s", f"{FW + PANEL_W}x{FH}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", args.record],
                               stdin=subprocess.PIPE)

    ledger = None
    if args.replay:
        path = args.replay
        if path == "last":
            path = max(glob.glob(f"logs/shoot_{args.mode}_*_jev.jsonl"), key=os.path.getmtime)
        brain = ReplayBrain(path)
        args.seed, args.realtime = brain.seed, brain.realtime       # 記録と同じ出方・同じ遅れで再生する
        print("再生:", path)
    elif args.brain == "jev":
        ledger = Ledger(args.instruction, f"shooter_{args.mode}")
        print("中断の基準:", ledger.limits, "／出どころ:", ledger.source)
        try:
            brain = JevBrain(instr_for(args.prompt), ledger, gate=args.gate)
        except BudgetExceeded as e:
            sys.exit(f"開始前に中断: {e}")
    elif args.brain == "d1":
        brain = D1Brain(instr_for(args.prompt), args.quant, args.graph)
    else:
        brain = RuleBrain()
    if args.prompt in ("A", "B", "C") and args.brain != "rule":
        from short_v2 import short_option_v2, short_state
        v = args.prompt
        brain = ShortPrompt(brain, (lambda s: short_state(s) or "-") if args.brain == "jev" else short_state,
                            lambda t: short_option_v2(t, v))
        brain.name = brain.brain.name + "_" + v
    if args.prompt == "short" and args.brain != "rule":
        from bench_fast import short_option, short_state
        # Jev は空の状況文を受け付けない可能性があるので "-" にする（d1 は空のまま）
        brain = ShortPrompt(brain, (lambda s: short_state(s) or "-") if args.brain == "jev" else short_state, short_option)
    name = "human" if args.human else (f"{brain.src} 再生" if args.replay else brain.name)
    acts = actions_for(args.mode)
    decide_every = args.decide or MODES[args.mode]["decide"]
    live = args.live and args.realtime and not args.headless and not args.replay
    live_frames = 0                   # live：今の質問を出してから進んだフレーム数
    pool = ThreadPoolExecutor(1)
    game = Game(args.mode, args.seed)
    game.god = args.god
    os.makedirs("logs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = open(f"logs/shoot_{args.mode}_{stamp}_{name.replace(' ', '_')}{f'_s{args.seed}' if args.seed else ''}"
               f"{'_rt' if args.realtime else ''}{'_live' if live else ''}.jsonl",
               "w", encoding="utf-8")

    action, last, pending = "stay", None, None
    lagging, lags = None, []          # リアルタイム：答えが出たあと、考えていた時間ぶん前の動きで進めてから使う
    decisions, stop_reason, end_wait = 0, None, None
    running = True

    def shown(state, crit):
        return state + "\n" + "\n".join(f"[{a}] {t.split('Result: ', 1)[-1]}" for a, t in crit.items())

    def write(state, rule, d, crit=None):
        log.write(json.dumps({"frame": game.frame, "state": state, "criteria": crit, "rule": rule, "action": d.action,
                              "source": d.source, "confidence": d.confidence, "probabilities": d.probabilities,
                              "latency_ms": round(d.latency_ms, 1)}, ensure_ascii=False) + "\n")

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
        if game.result:
            if rec:
                for _ in range(2 * FPS):
                    view.draw(game, name, acts, last, False, brain.stats())
                    rec.stdin.write(pygame.image.tobytes(view.screen, "RGB"))
                break
            if args.headless:
                break
            end_wait = end_wait or time.time()
            if time.time() - end_wait > 4:
                break
        elif args.replay and args.realtime:
            while brain.i < len(brain.rows) and brain.rows[brain.i]["frame"] <= game.frame:
                r = brain.rows[brain.i]
                brain.i += 1
                action = r["action"]
                last = {"state": shown(r["state"], r["criteria"]),
                        "decision": Decision(r["action"], r.get("source", "?"), r.get("confidence"), r.get("probabilities"),
                                             r.get("latency_ms", 0.0))}
                decisions += 1
            game.step(*move_vec(args.mode, action))
        elif args.human:
            k = pygame.key.get_pressed()
            if sock:
                try:
                    while True:
                        udp_keys, udp_t = sock.recv(64).decode(), time.time()
                except (BlockingIOError, OSError):
                    pass
            if sock and udp_t == 0.0:                           # 操作するプログラムがつながるまでは始めない
                view.draw(game, "camera AI 待ち", acts, last, False, brain.stats())
                clock.tick(FPS)
                continue
            u = udp_keys if time.time() - udp_t < 0.5 else ""      # 0.5 秒届かなければ離したことにする
            if dump:
                r1 = lambda v: round(v, 1)
                dump.write(json.dumps({"t": time.time(), "frame": game.frame, "keys": u, "ship": [r1(game.player.x), r1(game.player.y)],
                                       "bullets": [[r1(o.x), r1(o.y), r1(o.vx), r1(o.vy), o.r] for o in game.bullets],
                                       "enemies": [[r1(o.x), r1(o.y)] for o in game.enemies],
                                       "boss": [r1(game.boss.x), r1(game.boss.y), r1(game.boss.vx)] if game.boss else None,
                                       "warnings": [[r1(w.y), w.vx > 0, w.fire - game.frame] for w in game.warnings]}) + "\n")
            left, right = k[pygame.K_LEFT] or "L" in u, k[pygame.K_RIGHT] or "R" in u
            up, down = k[pygame.K_UP] or "U" in u, k[pygame.K_DOWN] or "D" in u
            slow = k[pygame.K_LSHIFT] or "S" in u
            sp = MODES[args.mode]["slow"] if (slow and MODES[args.mode]["slow"]) else MODES[args.mode]["speed"]
            game.step((right - left) * sp, (down - up) * sp)
        elif lagging is not None:                    # リアルタイム：考えていた間も弾は動き、自機は前の動きのまま
            game.step(*move_vec(args.mode, action))
            lagging[4] -= 1
            if lagging[4] <= 0:
                d, st, ru, cr, _ = lagging
                action = d.action
                last = {"state": shown(st, cr), "decision": d}
                decisions += 1
                write(st, ru, d, cr)
                lagging = None
        elif pending is not None and live:
            if pending[0].done():                   # 答えが届いたらすぐ使う（遅れは実際に進んだフレーム数）
                try:
                    d = pending[0].result()
                except BudgetExceeded as e:
                    stop_reason = str(e)
                    game.result = "BUDGET STOP"
                    pending = None
                    continue
                lags.append(max(1, live_frames))
                action = d.action
                last = {"state": shown(pending[1], pending[3]), "decision": d}
                decisions += 1
                write(pending[1], pending[2], d, pending[3])
                pending = None
            game.step(*move_vec(args.mode, action))   # 待っている間も止めずに、今の動きのまま進める
            live_frames += 1
        elif pending is not None:
            if not pending[0].done():               # 答えを待つ間は休む（休まず確かめ続けると AI の計算が遅くなる）
                wait([pending[0]], timeout=1 / FPS)
            if pending[0].done() and args.realtime:
                try:
                    d = pending[0].result()
                except BudgetExceeded as e:
                    stop_reason = str(e)
                    game.result = "BUDGET STOP"
                    pending = None
                    continue
                lag = max(1, math.ceil(d.latency_ms * FPS / 1000))      # 考えていた時間を、ゲームのフレーム数に直す
                lags.append(lag)
                lagging = [d, pending[1], pending[2], pending[3], lag]
                pending = None
                continue
            if pending[0].done():
                try:
                    d = pending[0].result()
                except BudgetExceeded as e:
                    stop_reason = str(e)
                    game.result = "BUDGET STOP"
                    pending = None
                    continue
                action = d.action
                last = {"state": shown(pending[1], pending[3]), "decision": d}
                decisions += 1
                write(pending[1], pending[2], d, pending[3])
                pending = None
                game.step(*move_vec(args.mode, action))
        else:
            if game.frame % decide_every == 0:
                game.cur = move_vec(args.mode, action)
                if args.lead == "auto":
                    recent = sorted(lags[-10:])
                    game.lead = recent[len(recent) // 2] if recent else 0
                else:
                    game.lead = int(args.lead)
                state, rule, crit = game.observe()
                if isinstance(brain, RuleBrain):
                    d = brain.decide(state, crit, rule)
                    if getattr(brain, "ended", False):
                        game.result = "END"
                        continue
                    action, last = d.action, {"state": shown(state, crit), "decision": d}
                    decisions += 1
                    write(state, rule, d, crit)
                else:
                    pending = (pool.submit(brain.decide, state, crit, rule), state, rule, crit)
                    if not live:
                        if view:
                            view.draw(game, name, acts, last, True, brain.stats())
                        continue
                    live_frames = 0
            game.step(*move_vec(args.mode, action))
            live_frames += 1

        if view and (rec or not args.headless):
            view.draw(game, name, acts, last, pending is not None, brain.stats())
            if rec:
                rec.stdin.write(pygame.image.tobytes(view.screen, "RGB"))
            if not args.headless:
                clock.tick(FPS * args.speed)

    summary = {"brain": name, "mode": args.mode, "seed": args.seed, "realtime": args.realtime, "lead": args.lead, "live": live, "decide_every": decide_every,
               "result": game.result or "QUIT", "time_s": round(game.frame / FPS, 1),
               "lives_left": game.lives, "hits": game.hits, "score": game.score, "decisions": decisions,
               **brain.stats()}
    if lags:
        summary["lag_frames_avg"] = round(sum(lags) / len(lags), 1)
        summary["lag_frames_max"] = max(lags)
    if stop_reason:
        summary["stop_reason"] = stop_reason
    if ledger:
        summary["usage"] = ledger.report()
    log.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    log.close()
    if rec:
        rec.stdin.close()
        rec.wait()
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    brain.close()
    pool.shutdown(wait=False)
    pygame.quit()


if __name__ == "__main__":
    main()
