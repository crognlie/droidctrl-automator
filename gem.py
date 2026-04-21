"""
Orbiting-gem detector for The Tower autoclicker.

Finds the magenta-bordered gem box that orbits the tower. The box is
tidally locked (one face always points toward the tower), so it can
appear at any rotation — this detector is rotation-invariant via
cv2.minAreaRect + circularity filter + contour-to-edge alignment.

Also provides:
- find_tower_center(): cyan-ring Hough circle for the constant tower
  landmark (needed because the bottom menu shifts the tower when open).
- screencap_raw():     ~2x faster than `screencap -p` since it skips
  on-device PNG encoding.
- predict_tap(): extrapolate the gem's orbital position forward to
  compensate for the screencap+detect+tap pipeline latency.
"""
import math
import subprocess

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

# Max contour circularity (4πA/P²). A square outline is ~0.78; the common
# "circle + 4 cardinal nubs" healing enemy is 0.79+, so cut at 0.77.
MAX_CIRCULARITY = 0.77

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

# Cyan / teal range for the tower's concentric rings.
CYAN_LO = np.array([80, 100, 150])
CYAN_HI = np.array([100, 255, 255])


def find_tower_center(img_bgr):
    """Top-ranked cyan ring → tower center, or None if not found."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CYAN_LO, CYAN_HI)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=200,
        param1=80, param2=30, minRadius=80, maxRadius=400,
    )
    if circles is None:
        return None
    c = circles[0][0]
    return int(round(c[0])), int(round(c[1]))


# ---------- prediction ----------

def predict_tap(gem_xy, tower_xy, latency_s, period_s=12.0, clockwise=True):
    """
    Extrapolate the gem's position forward by `latency_s` seconds, assuming
    it moves at constant angular velocity along a circle centered on the
    tower. In image coordinates with y pointing down, clockwise motion on
    screen means atan2 angle INCREASES.
    """
    dx = gem_xy[0] - tower_xy[0]
    dy = gem_xy[1] - tower_xy[1]
    r = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    omega = 2 * math.pi / period_s
    if not clockwise:
        omega = -omega
    theta_future = theta + omega * latency_s
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
