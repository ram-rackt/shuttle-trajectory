"""Downloads the model weights used by the CV pipeline into the models directory
(RALLYREVIEW_MODELS_DIR, default backend/data/models).

- RT-DETRv2 (person detection, RALLYREVIEW_PERSON_MODEL), cached under <models>/hf.
- TrackNetV3 (shuttle detection), from the authors' release (MIT).

Usage: python scripts/download_models.py
"""

import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("RALLYREVIEW_MODELS_DIR", BACKEND / "data" / "models"))
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
    sys.path.insert(0, str(BACKEND))
    from app.config import DEFAULT_PERSON_MODEL

    person_model = os.environ.get("RALLYREVIEW_PERSON_MODEL", DEFAULT_PERSON_MODEL)
    try:
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
    except ImportError:
        print("transformers is not installed; skipping RT-DETRv2 (pip install -e 'backend[cv]').")
    else:
        print(f"Caching {person_model} ...", flush=True)
        RTDetrImageProcessor.from_pretrained(person_model, cache_dir=MODELS / "hf")
        RTDetrV2ForObjectDetection.from_pretrained(person_model, cache_dir=MODELS / "hf")
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
