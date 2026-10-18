"""Download the inputs for the CTCF experiment.

1. The CTCF ChIP-seq dataset from Tang et al. (2025), Zenodo record 8279715
   (files hosted under record 13836032). Only data/chip/CTCF_200.h5 is kept.
2. The JASPAR CTCF motifs MA0139.1 and MA0139.2.

Usage:
    python scripts/download_data.py            # download everything missing
    python scripts/download_data.py --keep-archive
"""

import argparse
import hashlib
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ZENODO_URL = "https://zenodo.org/api/records/13836032/files/data.tar.gz/content"
ZENODO_MD5 = "d638cc27d988f2fbdb94851262e22ccd"
ZENODO_SIZE = 486_392_602
CTCF_MEMBER = "./data/chip/CTCF_200.h5"

JASPAR_URL = "https://jaspar.elixir.no/api/v1/matrix/{}.jaspar"
JASPAR_IDS = ["MA0139.1", "MA0139.2"]


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url, dest, size=None, retries=20):
    """Stream url to dest, printing progress every 50 MB.

    Zenodo often drops long connections without an error, so when the expected
    size is known the download resumes from where it stopped with HTTP Range
    requests until the file is complete.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries + 1):
        done = tmp.stat().st_size if tmp.exists() and size else 0
        request = urllib.request.Request(url)
        if done:
            request.add_header("Range", f"bytes={done}-")
            print(f"  resuming at {done / 1e6:.0f} MB")
        next_report = (done // (50 << 20) + 1) * (50 << 20)
        try:
            with urllib.request.urlopen(request, timeout=60) as resp, \
                    open(tmp, "ab" if done else "wb") as out:
                while block := resp.read(1 << 20):
                    out.write(block)
                    done += len(block)
                    if done >= next_report:
                        print(f"  {done / 1e6:.0f} MB")
                        next_report += 50 << 20
        except OSError as err:
            print(f"  connection error: {err}")
        if size is None or done == size:
            tmp.replace(dest)
            return
        time.sleep(min(5 * (attempt + 1), 60))
    raise RuntimeError(f"download incomplete after {retries} retries: {done} of {size} bytes")


def get_ctcf(keep_archive):
    target = RAW / "CTCF_200.h5"
    if target.exists():
        print(f"{target.name} already present, skipping")
        return
    archive = RAW / "tang2025_data.tar.gz"
    if not archive.exists() or archive.stat().st_size != ZENODO_SIZE:
        print(f"Downloading Tang et al. data archive ({ZENODO_SIZE / 1e6:.0f} MB)...")
        download(ZENODO_URL, archive, size=ZENODO_SIZE)
    print("Verifying checksum...")
    digest = md5sum(archive)
    if digest != ZENODO_MD5:
        raise RuntimeError(f"MD5 mismatch: got {digest}, expected {ZENODO_MD5}")
    print("Extracting CTCF_200.h5...")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(CTCF_MEMBER)
        with tar.extractfile(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    if not keep_archive:
        archive.unlink()
    print(f"Saved {target}")


def get_jaspar():
    for motif_id in JASPAR_IDS:
        target = RAW / f"{motif_id}.jaspar"
        if target.exists():
            print(f"{target.name} already present, skipping")
            continue
        download(JASPAR_URL.format(motif_id), target)
        print(f"Saved {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep-archive", action="store_true",
                        help="keep the 486 MB Zenodo archive after extracting")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    # Two runs appending to the same partial file corrupt it, so allow only one.
    lock = RAW / "download.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"Another download is already running (delete {lock} if it is not).")
    try:
        get_jaspar()
        get_ctcf(args.keep_archive)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
