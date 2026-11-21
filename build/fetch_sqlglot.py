"""Download sqlglot at the pinned commit (GitHub archive), unpack to /app/sqlglot,
and check every file under sqlglot/ and tests/ against the committed git blob hashes,
so the maps in this repository always describe exactly the code that runs."""

import hashlib
import io
import os
import shutil
import sys
import tarfile
import time
import urllib.request

SHA = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
URLS = [
    f"https://github.com/tobymao/sqlglot/archive/{SHA}.tar.gz",
    f"https://codeload.github.com/tobymao/sqlglot/tar.gz/{SHA}",
]
DEST = sys.argv[1] if len(sys.argv) > 1 else "/app/sqlglot"
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.txt")


def fetch():
    last = None
    for attempt in range(6):
        for url in URLS:
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    return r.read()
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 + attempt * 3)
    raise SystemExit(f"download failed: {last!r}")


def blob_sha(data):
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def main():
    data = fetch()
    tmp = DEST + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(tmp, filter="data")
    tops = os.listdir(tmp)
    if len(tops) != 1:
        raise SystemExit(f"unexpected archive layout: {tops}")
    shutil.rmtree(DEST, ignore_errors=True)
    shutil.move(os.path.join(tmp, tops[0]), DEST)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(DEST, "sqlglot-integration-tests"), exist_ok=True)
    bad = []
    n = 0
    for line in open(MANIFEST):
        sha, rel = line.rstrip("\n").split(" ", 1)
        p = os.path.join(DEST, rel)
        try:
            got = blob_sha(open(p, "rb").read())
        except OSError:
            got = None
        n += 1
        if got != sha:
            bad.append(rel)
    if bad:
        raise SystemExit(f"{len(bad)} files differ from the pinned commit, e.g. {bad[:5]}")
    print(f"sqlglot {SHA[:7]} ok: {n} files verified", file=sys.stderr)


if __name__ == "__main__":
    main()
