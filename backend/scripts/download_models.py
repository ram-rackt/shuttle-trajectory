"""Downloads the model weights used by the CV pipeline into backend/data/models.

- YOLO11n (person detection), from the Ultralytics release assets (AGPL-3.0).
- TrackNetV3 (shuttle detection), from the authors' release (MIT).

Usage: python scripts/download_models.py
"""

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "data" / "models"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
TRACKNET_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA&export=download&confirm=t"
)


def fetch(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.rename(dest)


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    yolo = MODELS / "yolo11n.pt"
    if not yolo.exists():
        fetch(YOLO_URL, yolo)
    tracknet = MODELS / "ckpts" / "TrackNet_best.pt"
    if not tracknet.exists():
        archive = MODELS / "TrackNetV3_ckpts.zip"
        if not archive.exists():
            fetch(TRACKNET_URL, archive)
        if not zipfile.is_zipfile(archive):
            archive.unlink()
            print("TrackNet download did not return a zip (Google Drive may be rate-limiting).")
            print("Download it manually from https://github.com/qaz812345/TrackNetV3 and unzip")
            print(f"into {MODELS}/ckpts/.")
            return 1
        with zipfile.ZipFile(archive) as z:
            z.extractall(MODELS)
        archive.unlink()
    print(f"Models ready in {MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
