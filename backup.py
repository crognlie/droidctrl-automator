#!/usr/bin/env python3
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import requests

SRC = "/sdcard/Android/data/com.TechTreeGames.TheTower/files/playerInfo.dat"
DST_DAT = "/backup/playerInfo.dat"

st = os.stat("/backup")
_uid, _gid = st.st_uid, st.st_gid


def chown(path):
    try:
        os.chown(path, _uid, _gid)
    except OSError:
        pass


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


with tempfile.NamedTemporaryFile(dir="/backup", delete=False) as tmp:
    tmp_path = tmp.name

try:
    result = subprocess.run(
        ["adb", "pull", SRC, tmp_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[!] playerInfo.dat backup failed: {result.stderr.strip()}", flush=True)
        sys.exit(1)

    if os.path.exists(DST_DAT) and md5(tmp_path) == md5(DST_DAT):
        print("[*] playerInfo.dat unchanged — skipping", flush=True)
        sys.exit(0)

    shutil.move(tmp_path, DST_DAT)
    chown(DST_DAT)
    datestamp = datetime.now().strftime("%Y%m%d")
    daily_dat = f"/backup/playerInfo.{datestamp}.dat"
    shutil.copy2(DST_DAT, daily_dat)
    chown(daily_dat)
    print(f"[*] playerInfo.dat changed — backed up → also {datestamp}", flush=True)

    webhook = os.environ.get("BACKUP_WEBHOOK", "")
    if webhook:
        print(f"[*] backup webhook: POSTing playerInfo.dat to {webhook}", flush=True)
        try:
            with open(DST_DAT, "rb") as f:
                resp = requests.post(
                    webhook,
                    files={"file": ("playerInfo.dat", f, "application/octet-stream")},
                    timeout=15,
                )
            if resp.ok:
                print(f"[*] backup webhook: OK {resp.status_code}", flush=True)
            else:
                print(f"[!] backup webhook: {resp.status_code} {resp.text[:200]}", flush=True)
        except Exception as e:
            print(f"[!] backup webhook: failed — {e}", flush=True)
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
