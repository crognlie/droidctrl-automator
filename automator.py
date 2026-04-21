"""
Automates clicking CLAIM and RETRY buttons in The Tower (Tech Tree Games) on
Android. Takes a screenshot via ADB, runs Tesseract OCR over an upscaled and
thresholded version of it, and taps the first matching target's centroid.

Why OCR and not UIAutomator: The Tower is a Unity game that renders its UI
inside a single SurfaceView, so Android's accessibility tree sees nothing
useful. il2cpp also rules out Mono-heap walking. OCR over screenshots is the
simplest reliable option.
"""
import io
import os
import subprocess
import time
from PIL import Image, ImageFilter
import pytesseract

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
TARGETS = [t.strip().lower() for t in os.environ.get("TARGETS", "claim,retry").split(",") if t.strip()]
MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "80"))
SCALE = int(os.environ.get("SCALE", "3"))


def screenshot():
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        capture_output=True,
        timeout=15,
    )
    return Image.open(io.BytesIO(result.stdout))


def preprocess(img):
    img = img.resize((img.width * SCALE, img.height * SCALE), Image.LANCZOS)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.convert("L")
    img = img.point(lambda x: 255 if x > 140 else 0)
    return img


def find_targets(img):
    processed = preprocess(img)
    data = pytesseract.image_to_data(
        processed,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )
    found = {}
    for i, word in enumerate(data["text"]):
        key = word.lower().strip()
        if key in TARGETS and int(data["conf"][i]) >= MIN_CONFIDENCE:
            if key not in found:
                cx = (data["left"][i] + data["width"][i] // 2) // SCALE
                cy = (data["top"][i] + data["height"][i] // 2) // SCALE
                found[key] = (cx, cy)
    return found


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


def run():
    wait_for_device()
    print(f"[*] Automator started — polling every {POLL_INTERVAL}s for {TARGETS}", flush=True)
    while True:
        try:
            img = screenshot()
            found = find_targets(img)

            if found:
                for target in TARGETS:
                    if target in found:
                        cx, cy = found[target]
                        print(f"[+] '{target}' at ({cx}, {cy}) — tapping", flush=True)
                        tap(cx, cy)
                        break
            else:
                print("[-] No targets found", flush=True)

        except Exception as e:
            print(f"[!] {e}", flush=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
