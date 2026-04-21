# droidctrl-automator

Autoclicker for **The Tower** by Tech Tree Games (Android). Sits next to an
[droidctrl](https://github.com/crognlie/droidctrl) container and:

- Taps **CLAIM** when a gem drop shows up in the corner.
- Taps **RETRY** when a run ends, so the next run starts without you.

Screenshots the phone via ADB, OCRs the image with Tesseract, and taps the
centroid of the first matching label. Default poll interval is 10 seconds —
low enough to catch a CLAIM before it times out.

## Why OCR instead of accessibility / hooking

The Tower is a Unity game, so its UI lives inside a single `SurfaceView` and
Android's accessibility tree sees nothing useful. `dumpsys window` gives you
the activity and one opaque surface. il2cpp rules out walking the Mono heap.
Frida would work but is fragile across game updates. OCR over a screenshot
is version-proof, simple, and fast enough at a 5-minute cadence.

## Requirements

- [droidctrl](https://github.com/crognlie/droidctrl) running against the phone
  you want to automate (provides the ADB server this container talks to).
- The Tower installed and running on the phone (launched to the main game
  view; the first CLAIM/RETRY it sees will be tapped).

## Setup

### Option A — as a subfolder of droidctrl

```bash
cd path/to/droidctrl
git clone https://github.com/crognlie/droidctrl-automator automator
docker compose --profile automator up -d
```

Env vars come from droidctrl's `.env`. The parent `compose.yml` has
the automator service definition; this repo's own `compose.yml` is ignored
in this mode.

### Option B — standalone

```bash
git clone https://github.com/crognlie/droidctrl-automator droidctrl-automator
cd droidctrl-automator

cp .env.example .env
$EDITOR .env      # set ADB_KEY_DIR and (if non-default) NETWORK

docker compose up -d
```

Requires the droidctrl container to be running first (so the Docker
network and the ADB server on `droidctrl:5037` exist).

## Configuration (`.env`)

| Variable | Meaning | Default |
|----------|---------|---------|
| `ADB_KEY_DIR` | Host path containing `adbkey` + `adbkey.pub` | `$HOME/.android` |
| `POLL_INTERVAL` | Seconds between screenshot polls | `10` |
| `TARGETS` | Comma-separated button labels (priority order) | `claim,retry` |
| `MIN_CONFIDENCE` | Tesseract confidence threshold (0–100) | `80` |
| `SCALE` | Screenshot upscale factor before OCR | `3` |
| `NETWORK` | Docker network to join (must match droidctrl) | `droidctrl_default` |

`CLAIM` is checked first because gem drops time out and only appear briefly;
`RETRY` only appears after a run ends, which is stable for much longer.

## Logs

```bash
docker compose logs -f                   # standalone mode
docker logs -f droidctrl-automator         # subfolder/profile mode (named container)
```

Each poll prints either `[-] No targets found` or `[+] '<target>' at (x, y) — tapping`.

## Updates and restarts

This container rebuilds and restarts independently of droidctrl —
scrcpy keeps streaming while you iterate on automator code.

```bash
# Standalone mode (inside this repo's directory):
docker compose up -d --build     # rebuild + restart
docker compose restart           # just restart, no rebuild

# Subfolder/profile mode (from droidctrl/):
docker compose up -d --build automator
docker compose restart automator
```

Python-only edits to `automator.py` rebuild sub-second thanks to Docker
layer caching.

## Running multiple automators

You can have several of these running in parallel against the same phone —
different configs (different `TARGETS`, poll intervals) or different games.
Just clone this repo into separate sibling directories next to
droidctrl; each is its own compose project and gets a unique
container name derived from the directory.

```bash
~/code/
├── droidctrl/
├── tower-automator/       # this repo, claim+retry
└── events-automator/      # this repo cloned again with different TARGETS
```

```bash
cd ~/code/tower-automator && docker compose up -d
cd ~/code/events-automator && docker compose up -d
```

Note: if two automators try to tap the phone at overlapping moments they'll
collide. Space out their `POLL_INTERVAL` values to avoid simultaneous
screenshots/OCR passes.

## Tuning

- **Misses CLAIM**: drop `POLL_INTERVAL` to 5 (more screenshots = more
  CPU but catches faster-expiring drops). Or lower `MIN_CONFIDENCE` to
  50–70 if Tesseract's reading the stylized font below threshold.
- **False taps**: raise `MIN_CONFIDENCE` back toward 80–90. If a cosmetic
  element is matching, add a more specific string (e.g.
  `TARGETS=claim gem,retry`).
- **Unity font still unreadable**: raise `SCALE` to 4. Costs memory and CPU
  during OCR; rest of the interval it's idle.

## Adding more targets

`TARGETS` is just a list of lowercase strings Tesseract needs to find
verbatim in the screenshot. Common extensions:

- `continue` — continue an offline run summary.
- `collect` — collect a quest reward.

Order matters: earlier entries take priority when multiple match in the
same frame.

## Limitations

- Taps the center of the detected text, not a button specifically. If the
  game ever puts unrelated text right next to a button, this will misclick.
- Stops whatever you're doing on the phone for ~100ms when it taps — if
  you're interacting over droidctrl at the same time, expect the
  occasional collision.
- The Tower's UI evolves; if CLAIM or RETRY disappears/renames in a future
  update, edit `TARGETS` in `.env`.
