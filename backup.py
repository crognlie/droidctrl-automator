#!/usr/bin/env python3
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

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
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
