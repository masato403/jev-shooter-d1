"""Web カメラでゲームの画面を読む部品。

  Camera      : カメラから最新の 1 枚をいつでも取り出せるようにする（別スレッドで読み続ける）
  ScreenReader: 画面の四隅の目印（ArUco）から、斜めに映った画面をまっすぐな「戦う場所」の画像（480x600）に直す

  python cam_screen.py          # 調整用：カメラの映像と、まっすぐに直した画像を表示する
     キー：+ / - 明るさ（露出）、a 自動露出、q 終了
"""
import os
import threading
import time

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # これがないとカメラの起動に30秒かかる
import cv2  # noqa: E402
import numpy as np

from shooter import FH, FW, MARK, marker_layout


class Camera:
    def __init__(self, index=1, w=1280, h=720, fps=30, exposure=None):
        self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)      # DirectShow だと 1280x720 で 5FPS しか出なかった
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.exposure = exposure
        self.frame, self.t, self.n = None, 0.0, 0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def set_exposure(self, e):
        self.exposure = e
        if e is None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, e)

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok and self.n == 5 and self.exposure is not None:
                self.set_exposure(self.exposure)        # 開いた直後に設定しても効かないので、映り始めてから設定する
            if ok:
                with self.lock:
                    self.frame, self.t, self.n = f, time.perf_counter(), self.n + 1

    def read(self, after=-1):
        """after より新しい 1 枚が来るまで待って返す（(画像, 撮れた時刻, 通し番号)）。"""
        while True:
            with self.lock:
                if self.frame is not None and self.n > after:
                    return self.frame, self.t, self.n
            time.sleep(0.001)

    def close(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


class ScreenReader:
    """目印（ArUco）の角と、ゲームの中での位置の対応から、カメラの画像 → まっすぐな「戦う場所」の画像 への変換を求める。

    目印が 6 個以上見えるとき：ゲームの位置 → カメラの位置 を 3 次の多項式で表す（レンズの丸いゆがみも直せる）
    それより少ないとき      ：射影変換（ホモグラフィー）。見えないときは前の値を使う。
    同じ番号が複数見えたとき（カメラの映像の窓に映った目印など）は、いちばん大きいものだけを使う。"""

    def __init__(self, scale=1.0, refit_every=15):
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        self.scale = scale
        self.size = (int(FW * scale), int(FH * scale))
        self.H, self.maps = None, None
        self.seen, self.n, self.refit_every, self.err = [], 0, refit_every, None
        lay = marker_layout()
        sq = np.array([[0, 0], [MARK, 0], [MARK, MARK], [0, MARK]], np.float32)
        self.ref = {k: (sq + np.array(xy, np.float32)) * scale for k, xy in lay.items()}
        gx, gy = np.meshgrid(np.arange(self.size[0], dtype=np.float32), np.arange(self.size[1], dtype=np.float32))
        self.grid = (gx, gy)

    @staticmethod
    def _terms(x, y):
        x, y = x / 300.0, y / 300.0
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3], -1)

    def update(self, frame):
        corners, ids, _ = self.det.detectMarkers(frame)
        best = {}
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                i = int(i)
                if i not in self.ref:
                    continue
                area = cv2.contourArea(c.reshape(4, 2))
                if i not in best or area > best[i][0]:
                    best[i] = (area, c.reshape(4, 2))
        self.seen = sorted(best)
        self.n += 1
        if len(best) >= 2:
            src = np.concatenate([best[i][1] for i in self.seen]).astype(np.float32)
            dst = np.concatenate([self.ref[i] for i in self.seen]).astype(np.float32)
            if len(best) >= 6:
                if self.maps is None or self.n % self.refit_every == 0:
                    A = self._terms(dst[:, 0], dst[:, 1])
                    cx, *_ = np.linalg.lstsq(A, src[:, 0], rcond=None)
                    cy, *_ = np.linalg.lstsq(A, src[:, 1], rcond=None)
                    self.err = float(np.sqrt(np.mean((A @ cx - src[:, 0]) ** 2 + (A @ cy - src[:, 1]) ** 2)))
                    T = self._terms(*self.grid)
                    self.maps = ((T @ cx).astype(np.float32), (T @ cy).astype(np.float32))
                self.H = None
            else:
                H, _ = cv2.findHomography(src, dst, 0)
                if H is not None:
                    self.H, self.maps = H, None
        return [best[i][1].reshape(1, 4, 2) for i in self.seen], (np.array(self.seen).reshape(-1, 1) if best else None)

    def ready(self):
        return self.maps is not None or self.H is not None

    def warp(self, frame):
        if self.maps is not None:
            return cv2.remap(frame, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
        if self.H is None:
            return None
        return cv2.warpPerspective(frame, self.H, self.size, flags=cv2.INTER_LINEAR)


def main():
    import sys
    cam = Camera(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    rd = ScreenReader()
    exp = None
    n0, t0, fps, last = 0, time.perf_counter(), 0.0, -1
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera", 640, 360)
    cv2.moveWindow("camera", 760, 0)
    cv2.namedWindow("game area (from camera)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("game area (from camera)", 288, 360)
    cv2.moveWindow("game area (from camera)", 760, 420)
    first = True
    while True:
        f, t, last = cam.read(last)
        corners, ids = rd.update(f)
        if time.perf_counter() - t0 > 1:
            fps, n0, t0 = (last - n0) / (time.perf_counter() - t0), last, time.perf_counter()
        show = f.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(show, corners, ids)
        ok = len(rd.seen) == 8
        msg = f"markers {len(rd.seen)}/8  {'OK' if ok else 'need 8'}  {fps:.0f}fps  exposure {'auto' if exp is None else exp}"
        cv2.rectangle(show, (0, 0), (1280, 44), (0, 0, 0), -1)
        cv2.putText(show, msg, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if ok else (0, 128, 255), 2)
        cv2.imshow("camera", show)
        w = rd.warp(f)
        if w is not None:
            cv2.imshow("game area (from camera)", w)
        if first:                    # ほかの窓に隠れないよう、いつも一番手前に表示する
            for name in ("camera", "game area (from camera)"):
                try:
                    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
                except cv2.error:
                    pass
            print("表示を開始しました", flush=True)
            first = False
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in (ord("+"), ord(";")):
            exp = (exp if exp is not None else -6) + 1
            cam.set_exposure(exp)
        if k == ord("-"):
            exp = (exp if exp is not None else -6) - 1
            cam.set_exposure(exp)
        if k == ord("a"):
            exp = None
            cam.set_exposure(None)
    cam.close()


if __name__ == "__main__":
    main()
