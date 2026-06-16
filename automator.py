"""
Automates clicking the orbiting gem, CLAIM, and RETRY in The Tower (Tech
Tree Games) on Android. Each tick takes a raw screenshot via ADB and:

  1. Runs the CV gem detector (rotation-invariant contour match, see
     gem.py). If a gem is found within the orbit-radius tolerance, taps
     the predicted *future* position that compensates for the screenshot
     + detect + tap pipeline latency.
  2. Runs Tesseract OCR for CLAIM and taps if found.
  3. OCRs for RETRY on the death screen.

Priority order: gem > claim > retry. At most one tap per tick so that
we don't accidentally double-input on the phone.

Why OCR for CLAIM/RETRY but CV for gem: The Tower is a Unity game that
renders its UI inside a single SurfaceView, so Android's accessibility
tree sees nothing useful. Text labels OCR cleanly; the gem is a pure
sprite with no text so it needs shape-based detection.
"""
import io
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFilter

import gem

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
RETRY_WAIT = int(os.environ.get("RETRY_WAIT", "180"))

RETRY_WEBHOOK = os.environ.get("RETRY_WEBHOOK", "")
RETRY_WEBHOOK_MESSAGE = os.environ.get(
    "RETRY_WEBHOOK_MESSAGE",
    "Run ended — retry pending (waiting {wait}s).",
)
RETRY_WEBHOOK_AVATAR = os.environ.get(
    "RETRY_WEBHOOK_AVATAR",
    "https://raw.githubusercontent.com/crognlie/droidctrl/main/favicons/droidctrl-64x64.png",
)
BACKUP_DIR = os.environ.get("BACKUP_DIR", "")
RETURN_WAIT = 180  # seconds before auto-tapping "return to game" overlay

PLAYERINFO_SRC = "/sdcard/Android/data/com.TechTreeGames.TheTower/files/playerInfo.dat"

TOWER_PACKAGE = "com.TechTreeGames.TheTower"
TOWER_ACTIVITY = f"{TOWER_PACKAGE}/com.unity3d.player.UnityPlayerActivity"

MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "80"))
SCALE = int(os.environ.get("SCALE", "3"))

GEM_MIN_SCORE = float(os.environ.get("GEM_MIN_SCORE", "0.70"))
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


def ocr_find_in_region(img_pil, word, x0_frac=0.0, y0_frac=0.0, x1_frac=1.0, y1_frac=1.0):
    """Like ocr_find but restricted to a fractional region of the image.
    Returns coordinates in full-image space."""
    w, h = img_pil.size
    x0, y0 = int(w * x0_frac), int(h * y0_frac)
    x1, y1 = int(w * x1_frac), int(h * y1_frac)
    crop = img_pil.crop((x0, y0, x1, y1))
    pos = ocr_find(crop, word)
    if pos is None:
        return None
    return (pos[0] + x0, pos[1] + y0)


def is_tower_focused():
    r = subprocess.run(
        ["adb", "shell", "dumpsys", "window"],
        capture_output=True, text=True, timeout=5,
    )
    return any(
        TOWER_PACKAGE in line
        for line in r.stdout.splitlines()
        if "mCurrentFocus" in line
    )


def tap(x, y):
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)], timeout=5)


def exit_tower():
    """Press Home and wait for playerInfo.dat to update (game saved)."""
    def get_mtime():
        r = subprocess.run(["adb", "shell", "stat", "-c", "%Y", PLAYERINFO_SRC],
                           capture_output=True, text=True, timeout=5)
        try:
            return int(r.stdout.strip())
        except ValueError:
            return 0

    mtime_before = get_mtime()
    print("[*] exit_tower: pressing Home — waiting for game to save", flush=True)
    subprocess.run(["adb", "shell", "input", "keyevent", "KEYCODE_HOME"],
                   capture_output=True, timeout=5)
    for _ in range(10):
        time.sleep(1)
        if get_mtime() > mtime_before:
            print("[*] exit_tower: save detected", flush=True)
            return
    print("[!] exit_tower: no save detected after 10s", flush=True)


def kill_tower():
    """Kill the Tower process and wait for it to exit."""
    print("[*] kill_tower: killing process", flush=True)
    subprocess.run(["adb", "shell", "am", "kill", TOWER_PACKAGE],
                   capture_output=True, timeout=5)
    for _ in range(10):
        r = subprocess.run(["adb", "shell", "pidof", TOWER_PACKAGE],
                           capture_output=True, text=True, timeout=5)
        if not r.stdout.strip():
            print("[*] kill_tower: Tower exited", flush=True)
            return
        time.sleep(1)
    print("[!] kill_tower: Tower still running after 10s — continuing anyway", flush=True)


def start_tower():
    """Launch Tower and wait up to 15s for it to become focused."""
    print("[*] restart: launching Tower", flush=True)
    subprocess.run(["adb", "shell", "am", "start", "-n", TOWER_ACTIVITY],
                   capture_output=True, timeout=5)
    for _ in range(15):
        if is_tower_focused():
            print("[*] restart: Tower is focused — done", flush=True)
            return
        time.sleep(1)
    print("[!] restart: Tower didn't focus within 15s", flush=True)


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
    else None. `tower` is the (cx, cy, ring_r) tuple from find_tower_center.
    Radius bounds are computed from the orbit ring: min = 5% screen width,
    max = ring radius.
    """
    g = gem.detect_gem(img_bgr)
    if g is None:
        return None
    score, cx, cy, side = g
    if score < GEM_MIN_SCORE:
        return None

    dx, dy = cx - tower[0], cy - tower[1]
    radius = (dx * dx + dy * dy) ** 0.5
    r_min = img_bgr.shape[1] * 0.05
    r_max = tower[2]

    if not (r_min <= radius <= r_max):
        print(f"[-] gem-shape at ({cx},{cy}) rejected — r={radius:.0f} off-orbit (tower={tower[0]},{tower[1]} ring_r={tower[2]})", flush=True)
        return None

    pred = gem.predict_tap((cx, cy), tower, PIPELINE_LATENCY, period_s=GEM_PERIOD)
    print(f"[+] gem at ({cx},{cy}) score={score:.2f} r={radius:.0f} tower=({tower[0]},{tower[1]}) → tap ({pred[0]},{pred[1]})", flush=True)
    return pred, cx, cy, score, radius


def read_gem_count(img_pil):
    """Read the floating gem count (3rd stat row, upper-left). Returns int or None."""
    arr = np.array(img_pil)
    # The number is white text — isolate near-white pixels to cut through the background
    mask = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 180) & (arr[:, :, 2] > 180)
    white = np.zeros_like(arr)
    white[mask] = 255
    crop = Image.fromarray(white).crop((30, 330, 200, 370))
    crop = crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS)
    text = pytesseract.image_to_string(crop, config="--psm 7 digits").strip()
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def log_gem_attempt(img_pil, detected, pred, score, radius,
                    count_before, count_after, elapsed_s, success):
    """Log every gem tap attempt (success or miss) with timing to /backup/gem_taps.log.
    On miss, also saves a screenshot to /backup/false_positives/."""
    if not BACKUP_DIR:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = "hit" if success else "miss"
    elapsed_str = f"{elapsed_s:.1f}" if elapsed_s is not None else "?"
    count_str = f"{count_before}→{count_after}" if (count_before is not None and count_after is not None) else "?"
    line = (f"{ts} {result} elapsed={elapsed_str}s "
            f"detected=({detected[0]},{detected[1]}) score={score:.2f} r={radius:.0f} "
            f"tap=({pred[0]},{pred[1]}) count={count_str}\n")
    log_path = Path("/backup") / "gem_taps.log"
    with open(log_path, "a") as f:
        f.write(line)
    if success:
        print(f"[+] gem hit! +{(count_after or 0)-(count_before or 0)} gems in {elapsed_str}s", flush=True)
    else:
        fp_dir = Path("/backup") / "false_positives"
        fp_dir.mkdir(exist_ok=True)
        img_path = fp_dir / f"{ts}_fp.png"
        ann = img_pil.copy()
        draw = ImageDraw.Draw(ann)
        px, py, r = pred[0], pred[1], 60
        green = (0, 255, 0)
        draw.line([(px - r, py), (px + r, py)], fill=green, width=2)
        draw.line([(px, py - r), (px, py + r)], fill=green, width=2)
        draw.ellipse([(px - r, py - r), (px + r, py + r)], outline=green, width=2)
        ann.save(str(img_path))
        print(f"[!] gem miss logged → {img_path.name}", flush=True)


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


DROIDCTRL_URL = "http://droidctrl:6080"

_item_states = {
    "automator":      True,
    "gem_click":      True,
    "claim_click":    True,
    "tower_focus":    True,
    "return_to_game": True,
    "backup":         True,
    "resume_round":   True,
    "save_game":      False,
    "restart_game":   False,
    "retry_wait":     float(RETRY_WAIT),
    "return_wait":    float(RETURN_WAIT),
}
_item_lock = threading.Lock()
_wakeup = threading.Event()  # set by poll thread on any state change; clears at top of main loop

ITEMS = [
    {"id": "automator",      "type": "toggle",  "desc": "Automator on/off",              "state": "true",             "preserve_state": "true",  "order": "0"},
    {"id": "tower_focus",    "type": "toggle",  "desc": "Require Tower in foreground",   "state": "true",             "preserve_state": "true",  "order": "10"},
    {"id": "resume_round",   "type": "toggle",  "desc": "Start/Resume round",     "state": "true",             "preserve_state": "true",  "order": "20"},
    {"id": "retry_wait",     "type": "numeric", "desc": "Autostart delay (s)",          "state": str(RETRY_WAIT),    "preserve_state": "true",  "order": "25"},
    {"id": "return_to_game", "type": "toggle",  "desc": "Return to game (from menus)",   "state": "true",             "preserve_state": "true",  "order": "30"},
    {"id": "return_wait",    "type": "numeric", "desc": "Return delay (s)",      "state": str(RETURN_WAIT),   "preserve_state": "true",  "order": "35"},
    {"id": "claim_click",    "type": "toggle",  "desc": "Ad Gem claiming",            "state": "true",             "preserve_state": "true",  "order": "40"},
    {"id": "gem_click",      "type": "toggle",  "desc": "Floating gem claiming",         "state": "true",             "preserve_state": "true",  "order": "50"},
    {"id": "backup",         "type": "toggle",  "desc": "playerInfo.dat backup",         "state": "true",             "preserve_state": "true",  "order": "60"},
    {"id": "save_game",      "type": "button",  "desc": "Save game",         "state": "false",                                         "order": "65"},
    {"id": "restart_game",   "type": "button",  "desc": "Restart Tower (graceful)",      "state": "false",                                         "order": "70"},
]


def flag(name, default=True):
    with _item_lock:
        return bool(_item_states.get(name, default))


def val(name, default=None):
    with _item_lock:
        return _item_states.get(name, default)


def _set_item(item_id, state):
    try:
        requests.get(f"{DROIDCTRL_URL}/item/set",
                     params={"id": item_id, "state": state}, timeout=5)
    except Exception:
        pass
    with _item_lock:
        _item_states[item_id] = state
    print(f"[*] item {item_id} → {state}", flush=True)


def _register_items():
    for item in ITEMS:
        try:
            requests.get(f"{DROIDCTRL_URL}/item/register", params=item, timeout=5)
        except Exception:
            pass
    try:
        states = requests.get(f"{DROIDCTRL_URL}/item/states", timeout=5).json()
        with _item_lock:
            for key, val in states.items():
                if key in _item_states and _item_states[key] != val:
                    print(f"[*] item {key}: {_item_states[key]} → {val}", flush=True)
                _item_states[key] = val
        print(f"[*] Item states: {states}", flush=True)
    except Exception:
        pass


def _do_save():
    """Save Tower state and return to game: called from poll thread within ~1s of button press."""
    print("[*] save_game — saving and returning", flush=True)
    _set_item("save_game", "false")
    prev_focus = flag("tower_focus")
    if prev_focus:
        _set_item("tower_focus", "false")
    try:
        exit_tower()
        start_tower()
        subprocess.run(["python3", "/backup.py"], check=False)
    finally:
        if prev_focus:
            _set_item("tower_focus", "true")
    print("[*] save_game complete", flush=True)


def _do_restart():
    """Graceful Tower restart: called from the poll thread within ~1s of button press."""
    print("[*] restart_game — starting graceful restart", flush=True)
    _set_item("restart_game", "false")  # acknowledge before the long wait
    prev_focus = flag("tower_focus")
    if prev_focus:
        _set_item("tower_focus", "false")
    try:
        exit_tower()
        kill_tower()
        start_tower()
    finally:
        if prev_focus:
            _set_item("tower_focus", "true")
    print("[*] restart_game complete", flush=True)


def _poll_items():
    while True:
        try:
            states = requests.get(f"{DROIDCTRL_URL}/item/states", timeout=5).json()
            changed = False
            with _item_lock:
                for key, new_val in states.items():
                    if key in _item_states and _item_states[key] != new_val:
                        print(f"[*] item {key}: {_item_states[key]} → {new_val}", flush=True)
                        changed = True
                    _item_states[key] = new_val
            if changed:
                _wakeup.set()
            if _item_states.get("save_game"):
                _do_save()
            if _item_states.get("restart_game"):
                _do_restart()
        except Exception as e:
            print(f"[!] poll: {e}", flush=True)
        time.sleep(1)


def run():
    wait_for_device()
    _register_items()
    threading.Thread(target=_poll_items, daemon=True).start()

    print(
        f"[*] Automator started — poll every {POLL_INTERVAL}s "
        f"(retry_wait={val("retry_wait", RETRY_WAIT):.0f}s, gem min_score={GEM_MIN_SCORE}, "
        f"latency={PIPELINE_LATENCY}s)",
        flush=True,
    )

    retry_time = 0.0
    return_time = 0.0
    last_dat_mtime = 0
    gem_first_seen = None
    last_gem_tap_time = None
    status_time = time.monotonic()
    tower_ticks = 0
    total_ticks = 0

    while True:
        _wakeup.clear()
        try:
            if not flag("automator"):
                _wakeup.wait(timeout=POLL_INTERVAL)
                continue

            if flag("tower_focus") and not is_tower_focused():
                start_tower()
                continue

            img_bgr = gem.screencap_raw()
            tower = gem.find_tower_center(img_bgr)
            img_pil = bgr_to_pil(img_bgr)

            total_ticks += 1
            if tower is not None:
                tower_ticks += 1
            now = time.monotonic()
            if now - status_time >= 300:
                gem_ago = f"{(now - last_gem_tap_time) / 60:.1f}m ago" if last_gem_tap_time else "never"
                print(f"[~] tower found {tower_ticks}/{total_ticks} polls | last gem tap: {gem_ago}", flush=True)
                tower_ticks = 0
                total_ticks = 0
                status_time = now

            if tower is not None:
                result = try_gem(img_bgr, tower)
                if result:
                    tap_pos, det_x, det_y, det_score, det_r = result
                    if gem_first_seen is None:
                        gem_first_seen = time.monotonic()
                    elapsed = time.monotonic() - gem_first_seen
                    if flag("gem_click"):
                        tap(*tap_pos)
                        time.sleep(1)
                        img_after = bgr_to_pil(gem.screencap_raw())
                        count_before = read_gem_count(img_pil)
                        count_after = read_gem_count(img_after)
                        success = (count_before is not None and count_after is not None
                                   and count_after > count_before)
                        log_gem_attempt(img_pil, (det_x, det_y), tap_pos, det_score, det_r,
                                        count_before, count_after, elapsed, success)
                        if success:
                            gem_first_seen = None
                            last_gem_tap_time = time.monotonic()
                    else:
                        print(f"[.] gem_click off — would tap ({tap_pos[0]},{tap_pos[1]})", flush=True)
                    _wakeup.wait(timeout=max(0, POLL_INTERVAL - 1))
                    continue
                else:
                    gem_first_seen = None  # gem gone from orbit
                claim_pos = ocr_find_in_region(img_pil, "claim", y0_frac=0.5, x1_frac=0.5)
                if claim_pos:
                    if flag("claim_click"):
                        print(f"[+] 'claim' at {claim_pos} — tapping", flush=True)
                        tap(*claim_pos)
                    else:
                        print(f"[.] claim_click off — would tap {claim_pos}", flush=True)
                    _wakeup.wait(timeout=POLL_INTERVAL)
                    continue
            else:
                print("[-] No tower center found", flush=True)

                resume_pos = ocr_find_in_region(img_pil, "resume", x0_frac=0.5, y0_frac=0.5) if flag("resume_round") else None
                if resume_pos:
                    print(f"[+] 'Welcome Back' dialog — tapping Resume at {resume_pos}", flush=True)
                    tap(*resume_pos)
                    _wakeup.wait(timeout=POLL_INTERVAL)
                    continue

            # Return-to-game check runs regardless of tower detection (overlay
            # can appear even when a false-positive tower is detected)
            return_pos = ocr_find(img_pil, "return") if flag("return_to_game") else None
            if return_pos:
                now = time.monotonic()
                if return_time == 0.0:
                    return_time = now
                    print(f"[+] 'return to game' overlay — waiting {val('return_wait', RETURN_WAIT):.0f}s before tapping", flush=True)
                elif now - return_time >= val("return_wait", RETURN_WAIT):
                    print(f"[+] 'return to game' still present after {now - return_time:.0f}s — tapping", flush=True)
                    tap(*return_pos)
                    return_time = 0.0
                else:
                    print(f"[-] 'return to game' overlay ({now - return_time:.0f}/{val('return_wait', RETURN_WAIT):.0f}s)", flush=True)
                _wakeup.wait(timeout=POLL_INTERVAL)
                continue

            if return_time != 0.0:
                print("[-] 'return to game' gone — resuming normal play", flush=True)
                return_time = 0.0

            if tower is None:
                retry_pos = ocr_find(img_pil, "retry")
                if retry_pos:
                    auto_retry = flag("resume_round")
                    if not RETRY_WEBHOOK:
                        if auto_retry:
                            print(f"[+] 'retry' at {retry_pos} — tapping", flush=True)
                            tap(*retry_pos)
                        else:
                            print(f"[-] 'retry' found — auto-resume off, waiting for manual input", flush=True)
                    else:
                        now = time.monotonic()
                        if retry_time == 0.0:
                            send_retry_webhook(img_pil)
                            if not auto_retry:
                                print(f"[-] 'retry' found — auto-resume off, waiting for manual input", flush=True)
                            elif val("retry_wait", RETRY_WAIT) == 0:
                                print(f"[+] 'retry' at {retry_pos} — tapping immediately", flush=True)
                                tap(*retry_pos)
                            else:
                                retry_time = now
                                print(
                                    f"[+] 'retry' found at {retry_pos} — waiting {val("retry_wait", RETRY_WAIT):.0f}s before clicking",
                                    flush=True,
                                )
                        elif auto_retry and now - retry_time >= val("retry_wait", RETRY_WAIT):
                            print(f"[+] 'retry' at {retry_pos} — tapping after {now - retry_time:.0f}s", flush=True)
                            tap(*retry_pos)
                            retry_time = 0.0
                        else:
                            print(
                                f"[-] 'retry' found — {'waiting' if auto_retry else 'auto-resume off'} ({now - retry_time:.0f}/{val("retry_wait", RETRY_WAIT):.0f}s)",
                                flush=True,
                            )
                else:
                    if retry_time != 0.0:
                        print("[-] retry screen gone — resetting timer", flush=True)
                        retry_time = 0.0
                    else:
                        print("[-] No tower — menu mode, no retry found", flush=True)

            if BACKUP_DIR and flag("backup"):
                r = subprocess.run(
                    ["adb", "shell", "stat", "-c", "%Y", PLAYERINFO_SRC],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0:
                    mtime = int(r.stdout.strip())
                    if mtime > last_dat_mtime:
                        last_dat_mtime = mtime
                        subprocess.run(["python3", "/backup.py"], check=False)

        except Exception as e:
            print(f"[!] {e}", flush=True)

        _wakeup.wait(timeout=POLL_INTERVAL)


if __name__ == "__main__":
    run()
