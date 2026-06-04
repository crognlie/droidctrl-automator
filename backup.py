#!/usr/bin/env python3
import subprocess
import sys

SRC = "/sdcard/Android/data/com.TechTreeGames.TheTower/files/playerInfo.dat"
DST_DAT = "/backup/playerInfo.dat"
DST_JSON = "/backup/playerInfo.json"

result = subprocess.run(
    ["adb", "pull", SRC, DST_DAT],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print(f"[!] playerInfo.dat backup failed: {result.stderr.strip()}", flush=True)
    sys.exit(1)
print("[*] playerInfo.dat backed up", flush=True)

result = subprocess.run(
    ["nrbfdump", "--format", "json", DST_DAT],
    capture_output=True, text=True,
)
if result.returncode == 0:
    with open(DST_JSON, "w") as f:
        f.write(result.stdout)
    print("[*] playerInfo.json written", flush=True)
else:
    print(f"[!] playerInfo.json decode failed: {result.stderr.strip()}", flush=True)
