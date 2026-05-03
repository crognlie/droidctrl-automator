"""
Automates clicking the orbiting gem, CLAIM, and RETRY in The Tower (Tech
Tree Games) on Android. Each tick takes a raw screenshot via ADB and:

  1. Runs the CV gem detector (rotation-invariant contour match, see
     gem.py). If a gem is found within the orbit-radius tolerance, taps
     the predicted *future* position that compensates for the screenshot
     + detect + tap pipeline latency.
  2. Runs Tesseract OCR for CLAIM and taps if found.
  3. On a longer cadence (RETRY_INTERVAL, default 5 minutes) also OCRs
     for RETRY.

Priority order: gem > claim > retry. At most one tap per tick so that
we don't accidentally double-input on the phone.

Why OCR for CLAIM/RETRY but CV for gem: The Tower is a Unity game that
renders its UI inside a single SurfaceView, so Android's accessibility
tree sees nothing useful. Text labels OCR cleanly; the gem is a pure
sprite with no text so it needs shape-based detection.
"""
import io
import os
import subprocess
import time

import requests

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFilter

import gem

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
RETRY_INTERVAL = int(os.environ.get("RETRY_INTERVAL", "300"))
RETRY_WAIT = int(os.environ.get("RETRY_WAIT", "180"))
RETRY_WEBHOOK = os.environ.get("RETRY_WEBHOOK", "")
RETRY_WEBHOOK_MESSAGE = os.environ.get(
    "RETRY_WEBHOOK_MESSAGE",
    "Run ended — retry pending (waiting {wait}s).",
)
RETRY_WEBHOOK_AVATAR = os.environ.get(
    "RETRY_WEBHOOK_AVATAR",
    "https://raw.githubusercontent.com/crognlie/droidctrl/main/favicons/novnc-64x64.png",
)
MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "80"))
SCALE = int(os.environ.get("SCALE", "3"))

GEM_MIN_SCORE = float(os.environ.get("GEM_MIN_SCORE", "0.70"))
GEM_RADIUS_MIN = int(os.environ.get("GEM_RADIUS_MIN", "280"))
GEM_RADIUS_MAX = int(os.environ.get("GEM_RADIUS_MAX", "360"))
GEM_PERIOD = float(os.environ.get("GEM_PERIOD", "12.0"))
# Image-capture-to-tap latency (seconds). The gem moves at 2π·radius /
# period ≈ 160 px/s on a 1080x2400 frame, so a ~900 ms latency requires
# predicting ~27° of orbital rotation forward.
PIPELINE_LATENCY = float(os.environ.get("PIPELINE_LATENCY", "0.9"))


def bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def preprocess_for_ocr(img_pil):
    img = img_pil.resize((img_pil.width * SCALE, img_pil.height * SCALE), Image.LANCZOS)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.convert("L")
    img = img.point(lambda x: 255 if x > 140 else 0)
    # Button outlines (long horizontal/vertical white runs) confuse Tesseract's
    # layout analysis and cause it to skip the text inside — notably RETRY/HOME
    # on the post-death screen. Morphologically extract those long lines and
    # subtract them, leaving the glyph strokes intact.
    arr = np.array(img)
    line_len = 40 * SCALE
    h_lines = cv2.morphologyEx(arr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1)))
    v_lines = cv2.morphologyEx(arr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len)))
    arr = cv2.subtract(arr, cv2.bitwise_or(h_lines, v_lines))
    return Image.fromarray(arr)


def ocr_find(img_pil, word):
    """Return (cx, cy) of the first high-confidence OCR match of `word`, or None."""
    processed = preprocess_for_ocr(img_pil)
    data = pytesseract.image_to_data(
        processed,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )
    for i, text in enumerate(data["text"]):
        if text.lower().strip() == word and int(data["conf"][i]) >= MIN_CONFIDENCE:
            cx = (data["left"][i] + data["width"][i] // 2) // SCALE
            cy = (data["top"][i] + data["height"][i] // 2) // SCALE
            return (cx, cy)
    return None


def tap(x, y):
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)], timeout=5)


def wait_for_device(timeout=60):
    print("[*] Waiting for ADB device...", flush=True)
    for _ in range(timeout):
        r = subprocess.run(["adb", "devices"], capture_output=True, text=True)
        if "device\n" in r.stdout:
            print("[*] Device ready", flush=True)
            return
        time.sleep(1)
    raise RuntimeError("Timed out waiting for ADB device")


def try_gem(img_bgr, tower):
    """
    Return predicted tap position (x, y) if a gem is detected on-orbit,
    else None. Logs the detection. `tower` is the tower center from
    find_tower_center (required — callers must gate on tower-found).
    """
    g = gem.detect_gem(img_bgr)
    if g is None:
        return None
    score, cx, cy, side = g
    if score < GEM_MIN_SCORE:
        return None

    dx, dy = cx - tower[0], cy - tower[1]
    radius = (dx * dx + dy * dy) ** 0.5
    # Radius filter disabled 2026-04-22: the in-game orbit radius actually
    # varies (not ~320 ± 3 as previously assumed), so this band was rejecting
    # real gems. On the current screenshots/ test set, the detector score
    # alone (GEM_MIN_SCORE) catches all real gems with zero false positives.
    # Re-enable (or replace with a motion-based filter) if reward-box drops
    # start producing false taps again — capture a screenshot of that case
    # first so we have a regression test.
    # if not (GEM_RADIUS_MIN <= radius <= GEM_RADIUS_MAX):
    #     print(f"[-] gem-shape at ({cx},{cy}) rejected — r={radius:.0f} off-orbit", flush=True)
    #     return None

    pred = gem.predict_tap((cx, cy), tower, PIPELINE_LATENCY, period_s=GEM_PERIOD)
    print(f"[+] gem at ({cx},{cy}) score={score:.2f} r={radius:.0f} → tap ({pred[0]},{pred[1]})", flush=True)
    return pred


def send_retry_webhook(img_pil):
    if not RETRY_WEBHOOK:
        return
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    msg = RETRY_WEBHOOK_MESSAGE.format(wait=RETRY_WAIT)
    try:
        requests.post(
            RETRY_WEBHOOK,
            data={"content": msg, "avatar_url": RETRY_WEBHOOK_AVATAR},
            files={"file": ("screenshot.png", buf, "image/png")},
            timeout=10,
        )
    except Exception as e:
        print(f"[!] webhook failed: {e}", flush=True)


def run():
    wait_for_device()
    print(
        f"[*] Automator started — poll every {POLL_INTERVAL}s "
        f"(retry_wait={RETRY_WAIT}s, gem min_score={GEM_MIN_SCORE}, "
        f"latency={PIPELINE_LATENCY}s)",
        flush=True,
    )

    retry_time = 0.0

    while True:
        try:
            img_bgr = gem.screencap_raw()
            tower = gem.find_tower_center(img_bgr)
            img_pil = bgr_to_pil(img_bgr)

            if tower is not None:
                tap_pos = try_gem(img_bgr, tower)
                if tap_pos:
                    tap(*tap_pos)
                    time.sleep(POLL_INTERVAL)
                    continue

            claim_pos = ocr_find(img_pil, "claim")
            if claim_pos:
                print(f"[+] 'claim' at {claim_pos} — tapping", flush=True)
                tap(*claim_pos)
                time.sleep(POLL_INTERVAL)
                continue

            if tower is None:
                retry_pos = ocr_find(img_pil, "retry")
                if retry_pos:
                    if not RETRY_WEBHOOK:
                        print(f"[+] 'retry' at {retry_pos} — tapping", flush=True)
                        tap(*retry_pos)
                    else:
                        now = time.monotonic()
                        if retry_time == 0.0:
                            retry_time = now
                            print(
                                f"[+] 'retry' found at {retry_pos} — waiting {RETRY_WAIT}s before clicking",
                                flush=True,
                            )
                            send_retry_webhook(img_pil)
                        elif now - retry_time >= RETRY_WAIT:
                            print(f"[+] 'retry' at {retry_pos} — tapping after {now - retry_time:.0f}s", flush=True)
                            tap(*retry_pos)
                            retry_time = 0.0
                        else:
                            print(
                                f"[-] 'retry' found — waiting ({now - retry_time:.0f}/{RETRY_WAIT}s)",
                                flush=True,
                            )
                else:
                    print("[-] No tower — menu mode, no retry found", flush=True)

        except Exception as e:
            print(f"[!] {e}", flush=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
