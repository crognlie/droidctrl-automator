# droidctrl-automator

Autoclicker for **The Tower** by Tech Tree Games (Android). Sits next to a
[droidctrl](https://github.com/crognlie/droidctrl) container and:

- Taps the **orbiting gem** using CV-based detection with tap-ahead prediction.
- Taps **CLAIM** when a gem drop shows up via OCR.
- Waits a configurable delay after a run ends, then taps **RETRY** automatically.
- Posts a screenshot to a Discord webhook when a run ends (optional).

Each tick: screencap via ADB → CV gem detector → OCR for CLAIM (in-game) or RETRY (death screen).

## Why OCR for text but CV for the gem

The Tower is a Unity game, so its UI lives inside a single `SurfaceView` and
Android's accessibility tree sees nothing useful. Text labels (CLAIM, RETRY)
OCR cleanly. The gem is a pure sprite with no text — it needs shape-based
detection: HSV magenta mask → contour fitting → rotated-rectangle scoring.

## Requirements

- [droidctrl](https://github.com/crognlie/droidctrl) running against the phone
  you want to automate (provides the ADB server this container talks to).
- The Tower installed and running on the phone.

## Setup

### Option A — as a subfolder of droidctrl

```bash
cd path/to/droidctrl
git clone https://github.com/crognlie/droidctrl-automator automator
docker compose --profile automator up -d --build automator
```

Env vars come from droidctrl's `.env`. The parent `compose.yml` has the
automator service definition; this repo's own `compose.yml` is ignored.

### Option B — standalone

```bash
git clone https://github.com/crognlie/droidctrl-automator droidctrl-automator
cd droidctrl-automator

cp .env.example .env
$EDITOR .env      # set RETRY_WEBHOOK and anything else you want to override

docker compose up -d --build
```

Requires droidctrl to be running first (so the Docker network and ADB server
on `droidctrl:5037` exist).

## Configuration (`.env`)

| Variable | Meaning | Default |
|----------|---------|---------|
| `ADB_KEY_DIR` | Host path containing `adbkey` + `adbkey.pub` | `$HOME/.android` |
| `NETWORK` | Docker network to join (must match droidctrl) | `droidctrl_default` |
| `POLL_INTERVAL` | Seconds between screenshot polls | `10` |
| `MIN_CONFIDENCE` | Tesseract confidence threshold (0–100) for CLAIM/RETRY | `80` |
| `SCALE` | Screenshot upscale factor before OCR | `3` |
| `RETRY_WAIT` | Seconds to wait after RETRY is first detected before clicking | `180` |
| `RETRY_WEBHOOK` | Discord webhook URL; if set, posts a screenshot on run death | _(none)_ |
| `RETRY_WEBHOOK_MESSAGE` | Message template sent with the webhook (`{wait}` is replaced) | `"Run ended — retry pending (waiting {wait}s)."` |
| `RETRY_WEBHOOK_AVATAR` | Avatar URL for the webhook post | noVNC favicon from droidctrl repo |
| `GEM_MIN_SCORE` | CV detector confidence threshold (0–1) to tap a gem | `0.70` |
| `GEM_RADIUS_MIN` | Orbit-radius filter lower bound in pixels | `280` |
| `GEM_RADIUS_MAX` | Orbit-radius filter upper bound in pixels | `360` |
| `GEM_PERIOD` | Gem orbit period in seconds (used for tap-ahead prediction) | `12.0` |
| `PIPELINE_LATENCY` | Capture-to-tap latency in seconds (controls how far ahead to predict) | `0.9` |

### Retry webhook

Set `RETRY_WEBHOOK` to a Discord webhook URL. When the RETRY button is first
detected (run death), the automator posts a screenshot with the message from
`RETRY_WEBHOOK_MESSAGE`, then waits `RETRY_WAIT` seconds before clicking.
This gives you a window to inspect the run or intervene before it auto-restarts.

If `RETRY_WEBHOOK` is not set, RETRY is clicked immediately with no delay.

## Logs

```bash
docker compose logs -f                 # standalone mode
docker logs -f droidctrl-automator    # subfolder/profile mode
```

Sample output:
```
[*] Automator started — poll every 10s (retry_wait=180s, gem min_score=0.7, latency=0.9s)
[+] gem at (541,812) score=0.84 r=318 → tap (562,798)
[+] 'claim' at (540, 1200) — tapping
[+] 'retry' found at (540, 1400) — waiting 180s before clicking
[-] 'retry' found — waiting (23/180s)
[+] 'retry' at (540, 1400) — tapping after 181s
```

## Updates and restarts

`automator.py` and `gem.py` are bind-mounted into the container — Python edits
pick up on `docker compose restart` (~10s) without a rebuild. Use `--build`
only when changing the Dockerfile or pip dependencies.

```bash
# Standalone mode:
docker compose restart               # Python edits only
docker compose up -d --build         # Dockerfile / dep changes

# Subfolder/profile mode (from droidctrl/):
docker compose restart automator
docker compose --profile automator up -d --build automator
```

## Tuning

- **Misses gems**: lower `POLL_INTERVAL`; verify `PIPELINE_LATENCY` by watching
  whether taps land before or after the gem.
- **False gem taps**: raise `GEM_MIN_SCORE`; tighten `GEM_RADIUS_MIN`/`MAX` if
  stationary reward boxes are the culprit.
- **Misses CLAIM**: lower `POLL_INTERVAL` or `MIN_CONFIDENCE` (try 50–70).
- **False OCR taps**: raise `MIN_CONFIDENCE`.
- **Stylized font misreads**: raise `SCALE` to 4 (higher memory/CPU during OCR).

## Limitations

- Gem detection recall is ~22% per frame; at a 10s poll interval you get roughly
  one tap opportunity per 12s orbit, so very brief gem windows may be missed.
- Taps the centroid of detected text, not a button hitbox — unrelated text near
  a button can cause misclicks.
- Each tap preempts phone input for ~100ms, colliding with concurrent
  scrcpy/droidctrl interaction.
