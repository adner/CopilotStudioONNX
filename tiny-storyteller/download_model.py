#!/usr/bin/env python3
"""
Download model.onnx from Hugging Face into this directory.

Run this once before packaging or testing locally:
    python download_model.py

The model file (model.onnx, ~2.6 MB) is NOT included in the git repo
because it has no explicit license. It is downloaded directly from
the upstream source:  onnx-community/TinyStories-656K-ONNX on Hugging Face.

Requires only the Python standard library (uses urllib).
"""

import sys
import shutil
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/onnx-community/TinyStories-656K-ONNX/resolve/main/onnx/model.onnx"
DEST = Path(__file__).resolve().parent / "model.onnx"


def download():
    if DEST.exists():
        print(f"model.onnx already present ({DEST.stat().st_size / 1024**2:.2f} MB) -- skipping.")
        print("Delete it and re-run to force a fresh download.")
        return

    print(f"Downloading TinyStories-656K model (~2.6 MB) ...")
    print(f"  from: {URL}")
    print(f"  to:   {DEST}")

    tmp = DEST.with_suffix(".onnx.tmp")
    try:
        with urllib.request.urlopen(URL) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1024**2
                        print(f"\r  {pct:.0f}%  {mb:.1f} MB", end="", flush=True)
        print()
        shutil.move(tmp, DEST)
        print(f"Done. {DEST.stat().st_size / 1024**2:.2f} MB saved to {DEST.name}")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"ERROR: download failed: {e}")


if __name__ == "__main__":
    download()
