"""
Orbiting-gem detector for The Tower autoclicker.

Finds the magenta-bordered gem box that orbits the tower. The box is
tidally locked (one face always points toward the tower), so it can
appear at any rotation — this detector is rotation-invariant via
cv2.minAreaRect + circularity filter + contour-to-edge alignment.

Also provides:
- find_tower_center(): grayscale Hough circle on the orbit ring to locate
  the tower center (needed because the bottom menu shifts the tower when open).
- screencap_raw():     ~2x faster than `screencap -p` since it skips
  on-device PNG encoding.
- predict_tap(): extrapolate the gem's orbital position forward to
  compensate for the screencap+detect+tap pipeline latency.
"""
import math
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np


# ---------- gem detector ----------

# Magenta outline HSV range (OpenCV H is 0-180). S>=120 drops the soft halo
# glow so the clean bright-magenta outline dominates the mask — this is
# what lets contour fitting recover a tight rotated square.
MAGENTA_LO = np.array([135, 120, 100])
MAGENTA_HI = np.array([160, 255, 255])

# Expected gem side length (pixels) on native 1080x2400 frames.
MIN_SIDE = 50
MAX_SIDE = 120

# Side-length uniformity: max(side)/min(side) <= this.
MAX_SIDE_RATIO = 1.35

# Max deviation from 90° at each corner.
MAX_ANGLE_DEV_DEG = 22

# Max contour circularity (4πA/P²). The floating gem is a rotated square
# outline (~0.785); healing enemies (circle + 4 nubs) are ~0.79+. Raised to
# 0.85 to let the gem through; interior density filter catches hollow enemies.
MAX_CIRCULARITY = 0.85

# Fraction of contour points that must lie within CONTOUR_MATCH_DIST of a
# fitted-rectangle edge. Rotation-invariant (no rasterization artifacts).
CONTOUR_MATCH_DIST = 3.0
MIN_EDGE_ALIGN = 0.65

# Interior magenta density bounds (eroded polygon). Gem icon reliably puts
# 0.13-0.55; circle-with-nubs enemies have 0.05-0.15.
MIN_INTERIOR = 0.13
MAX_INTERIOR = 0.60


def magenta_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, MAGENTA_LO, MAGENTA_HI)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)


def _quad_geometry(pts):
    p = pts.reshape(-1, 2).astype(np.float64)
    cx, cy = p.mean(axis=0)
    order = np.argsort(np.arctan2(p[:, 1] - cy, p[:, 0] - cx))
    p = p[order]
    sides = [float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4)]
    corner_angles = []
    for i in range(4):
        v1 = p[(i - 1) % 4] - p[i]
        v2 = p[(i + 1) % 4] - p[i]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        corner_angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    return sides, corner_angles, p


def _score_quad(sides, angles):
    mn, mx = min(sides), max(sides)
    side_ratio = mx / max(mn, 1e-9)
    max_angle_dev = max(abs(a - 90.0) for a in angles)
    if side_ratio > MAX_SIDE_RATIO or max_angle_dev > MAX_ANGLE_DEV_DEG:
        return 0.0
    side_score = 1.0 - (side_ratio - 1.0) / (MAX_SIDE_RATIO - 1.0)
    angle_score = 1.0 - max_angle_dev / MAX_ANGLE_DEV_DEG
    return 0.6 * side_score + 0.4 * angle_score


def _edge_alignment(contour, pts):
    c = contour.reshape(-1, 2).astype(np.float64)
    edges = []
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        v = b - a
        L = np.linalg.norm(v)
        if L < 1e-6:
            continue
        t = v / L
        n = np.array([-t[1], t[0]])
        edges.append((a, b, t, n, L))
    if not edges:
        return 0.0
    dists = []
    for p in c:
        best = 1e9
        for (a, b, t, n, L) in edges:
            ap = p - a
            proj = np.dot(ap, t)
            if proj < 0:
                d = np.linalg.norm(p - a)
            elif proj > L:
                d = np.linalg.norm(p - b)
            else:
                d = abs(np.dot(ap, n))
            if d < best:
                best = d
        dists.append(best)
    return float((np.array(dists) <= CONTOUR_MATCH_DIST).mean())


def _interior_magenta_density(mask, pts):
    H, W = mask.shape
    filled = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(filled, [pts.astype(np.int32)], 255)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    interior = cv2.erode(filled, k, iterations=2)
    ip = int((interior > 0).sum())
    if ip == 0:
        return 0.0
    inter = cv2.bitwise_and(interior, mask)
    return int((inter > 0).sum()) / ip


def detect_gem_near(img_bgr, predicted_xy, search_r=130):
    """Lightweight gem locator for tracking: find the magenta centroid
    within search_r pixels of predicted_xy. Returns (cx, cy, score) or None.
    Uses blob centroid instead of contour fitting so it works on H.264
    compressed frames where the gem outline is too fragmented for the
    full contour-based detector."""
    px, py = int(predicted_xy[0]), int(predicted_xy[1])
    H, W = img_bgr.shape[:2]
    x0, y0 = max(0, px - search_r), max(0, py - search_r)
    x1, y1 = min(W, px + search_r), min(H, py + search_r)
    roi = img_bgr[y0:y1, x0:x1]
    mask = magenta_mask(roi)
    n = int(np.count_nonzero(mask))
    if n < 80:
        return None
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"]) + x0
    cy = int(M["m01"] / M["m00"]) + y0
    score = min(1.0, n / 600.0)
    return cx, cy, score


def detect_gem(img_bgr):
    """
    Return (score, cx, cy, side) for the best gem candidate, or None.
    """
    mask = magenta_mask(img_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri < 4 * MIN_SIDE * 0.7 or peri > 4 * MAX_SIDE * 1.8:
            continue
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        circularity = 4 * math.pi * area / (peri * peri)
        if circularity > MAX_CIRCULARITY:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        rw, rh = max(rw, rh), min(rw, rh)
        if rh < 1 or not (MIN_SIDE <= rh <= MAX_SIDE and MIN_SIDE <= rw <= MAX_SIDE):
            continue
        if rw / rh > MAX_SIDE_RATIO:
            continue

        pts = cv2.boxPoints(rect).astype(np.float64)
        sides, angles, pts = _quad_geometry(pts)
        quad_score = _score_quad(sides, angles)
        if quad_score <= 0:
            continue

        align = _edge_alignment(c, pts)
        if align < MIN_EDGE_ALIGN:
            continue

        interior = _interior_magenta_density(mask, pts)
        if interior < MIN_INTERIOR or interior > MAX_INTERIOR:
            continue

        score = 0.35 * quad_score + 0.35 * align + 0.30 * min(1.0, interior / 0.30)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        side = sum(sides) / 4.0
        if best is None or score > best[0]:
            best = (score, cx, cy, side)

    return best


# ---------- tower center ----------


def _ring_mean_brightness(gray, cx, cy, r, n_samples=32):
    """Sample n_samples pixels on the circumference, return mean grayscale value."""
    h, w = gray.shape
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    xs = np.clip(np.round(cx + r * np.cos(angles)).astype(int), 0, w - 1)
    ys = np.clip(np.round(cy + r * np.sin(angles)).astype(int), 0, h - 1)
    return float(gray[ys, xs].mean())


def find_tower_center(img_bgr, min_ring_brightness=60):
    """Orbit-ring Hough circle → tower center, or None if not found.

    Detects the orbit ring the gem travels along. Color-agnostic (grayscale).
    Filters to circles whose center x is within 5% of the screen's horizontal
    midpoint and whose ring fits entirely within the image. Among candidates,
    picks the largest ring (orbit ring dominates). Radius range: 16%–50% of
    screen width.

    The brightness filter rejects decorative background arcs (dark, ~<20 mean)
    that appear on the game-stats/round-end screen. The real orbit ring glows
    white/blue and easily clears the default threshold of 60.
    """
    h, w = img_bgr.shape[:2]
    cx_screen = w / 2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=2, minDist=400,
        param1=60, param2=80, minRadius=w * 16 // 100, maxRadius=w // 2,
    )
    if circles is None:
        return None
    tolerance = w * 0.05
    candidates = [
        c for c in circles[0]
        if abs(c[0] - cx_screen) <= tolerance  # horizontally centered
        and c[2] <= c[0] <= w - c[2]           # fits within image width
        and c[2] <= c[1] <= h - c[2]           # fits within image height
        and _ring_mean_brightness(gray, c[0], c[1], c[2]) >= min_ring_brightness
    ]
    if not candidates:
        return None
    c = max(candidates, key=lambda c: c[2])  # largest on-screen ring = orbit ring
    return int(round(c[0])), int(round(c[1])), int(round(c[2]))


# ---------- prediction ----------

def predict_tap(gem_xy, tower_xy, latency_s, period_s=12.0, omega=None, clockwise=True):
    """
    Extrapolate the gem's position forward by `latency_s` seconds, assuming
    it moves at constant angular velocity along a circle centered on the
    tower. In image coordinates with y pointing down, clockwise motion on
    screen means atan2 angle INCREASES.

    Pass `omega` (rad/s) directly to override the period_s default — used
    when the caller has measured angular velocity from consecutive frames.
    """
    dx = gem_xy[0] - tower_xy[0]
    dy = gem_xy[1] - tower_xy[1]
    r = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    w = omega if omega is not None else (2 * math.pi / period_s)
    if not clockwise:
        w = -w
    theta_future = theta + w * latency_s
    return (
        int(round(tower_xy[0] + r * math.cos(theta_future))),
        int(round(tower_xy[1] + r * math.sin(theta_future))),
    )


# ---------- screencap ----------

def screencap_raw(adb_cmd=None):
    """
    Raw-format screencap over ADB (RGBA). ~2x faster than `screencap -p`
    because the device doesn't PNG-encode. Returns a BGR numpy array.

    `adb_cmd` lets callers supply a custom adb prefix (e.g. running adb
    in a different container); defaults to plain `adb` in $PATH.
    """
    cmd = (adb_cmd or ["adb"]) + ["exec-out", "screencap"]
    r = subprocess.run(cmd, capture_output=True, timeout=15)
    buf = r.stdout
    if len(buf) < 16:
        raise RuntimeError(f"screencap returned only {len(buf)} bytes")
    w = int.from_bytes(buf[0:4], "little")
    h = int.from_bytes(buf[4:8], "little")
    # Modern Android uses a 16-byte header (width, height, format, color-space).
    header = 16
    expected = w * h * 4
    if len(buf) - header != expected:
        # Fallback for older 12-byte header.
        header = 12
        if len(buf) - header != expected:
            raise RuntimeError(
                f"unexpected screencap size: {len(buf)} bytes, want {expected}+12 or +16"
            )
    arr = np.frombuffer(buf[header:header + expected], dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


# ---------- live stream reader ----------

class FrameStream:
    """
    Connects to the droidctrl WebSocket stream, decodes H.264 via ffmpeg,
    and keeps only the latest frame in a deque(maxlen=1). Intended for
    short-lived use during the gem tracking loop.

    Usage:
        with FrameStream("ws://droidctrl:6080/ws", width, height) as fs:
            ts, bgr = fs.latest()   # None if no frame yet
    """

    def __init__(self, ws_url, width, height):
        self._url = ws_url
        self._w = width
        self._h = height
        self._buf = deque(maxlen=1)
        self._stop = threading.Event()
        self._ffmpeg = None
        self._threads = []

    def __enter__(self):
        frame_size = self._w * self._h * 3
        sps_event = threading.Event()
        # Protected by sps_event: written before set(), read after wait().
        sps_initial: list[bytes] = []

        def _ws_reader():
            import websocket
            # Buffer until SPS NAL (00 00 00 01 67 or 00 00 01 67) —
            # the start of a clean GOP. Writing mid-stream slices to ffmpeg
            # before it has SPS/PPS causes "non-existing PPS" decode errors.
            pre_buf = bytearray()
            synced = False

            def _has_sps(data):
                for i in range(len(data) - 4):
                    # 4-byte start code
                    if data[i:i+4] == b'\x00\x00\x00\x01' and (data[i+4] & 0x1f) == 7:
                        return i
                    # 3-byte start code
                    if i + 3 < len(data) and data[i:i+3] == b'\x00\x00\x01' and (data[i+3] & 0x1f) == 7:
                        return i
                return -1

            def _on_message(ws, msg):
                nonlocal pre_buf, synced
                if self._stop.is_set():
                    ws.close()
                    return
                if not isinstance(msg, (bytes, bytearray)):
                    return  # skip JSON control messages
                if synced:
                    if self._ffmpeg is not None:
                        try:
                            self._ffmpeg.stdin.write(msg)
                            self._ffmpeg.stdin.flush()
                        except (BrokenPipeError, OSError):
                            ws.close()
                else:
                    pre_buf += msg
                    idx = _has_sps(pre_buf)
                    if idx >= 0:
                        synced = True
                        chunk = bytes(pre_buf[idx:])
                        print(f"[~] FrameStream: SPS at offset {idx} ({len(chunk)} bytes), starting ffmpeg", flush=True)
                        sps_initial.append(chunk)
                        sps_event.set()
                        pre_buf = bytearray()

            def _on_error(ws, err):
                if not self._stop.is_set():
                    print(f"[!] FrameStream WS error: {err}", flush=True)

            ws_app = websocket.WebSocketApp(self._url, on_message=_on_message, on_error=_on_error)
            ws_app.run_forever()
            # Signal that no more data is coming (WS closed before SPS found).
            sps_event.set()
            try:
                if self._ffmpeg is not None:
                    self._ffmpeg.stdin.close()
            except OSError:
                pass

        def _frame_reader():
            while not self._stop.is_set():
                data = b""
                while len(data) < frame_size:
                    try:
                        chunk = self._ffmpeg.stdout.read(frame_size - len(data))
                    except OSError:
                        return
                    if not chunk:
                        return
                    data += chunk
                arr = np.frombuffer(data, dtype=np.uint8).reshape(self._h, self._w, 3)
                self._buf.append((time.monotonic(), arr.copy()))

        # Start WS reader first; it buffers until SPS then signals sps_event.
        t_ws = threading.Thread(target=_ws_reader, daemon=True)
        t_ws.start()
        self._threads.append(t_ws)

        # Wait for first SPS before starting ffmpeg so it gets clean input
        # and doesn't error on "unspecified size" from an empty/mid-stream pipe.
        if not sps_event.wait(timeout=15):
            print("[!] FrameStream: timeout waiting for SPS", flush=True)
            self._stop.set()
            return self

        if not sps_initial:
            # WS closed without SPS
            print("[!] FrameStream: WS closed before SPS received", flush=True)
            self._stop.set()
            return self

        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "warning",
                "-f", "h264",
                "-i", "pipe:0",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self._w}x{self._h}",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        # Feed the buffered SPS immediately so ffmpeg can parse codec params
        self._ffmpeg.stdin.write(sps_initial[0])
        self._ffmpeg.stdin.flush()

        t_fr = threading.Thread(target=_frame_reader, daemon=True)
        t_fr.start()
        self._threads.append(t_fr)

        return self

    def latest(self):
        """Return (timestamp, bgr_array) or None if no frame received yet."""
        return self._buf[-1] if self._buf else None

    def __exit__(self, *_):
        self._stop.set()
        ffmpeg = self._ffmpeg
        if ffmpeg:
            ffmpeg.terminate()
            try:
                ffmpeg.wait(timeout=3)
            except subprocess.TimeoutExpired:
                ffmpeg.kill()
