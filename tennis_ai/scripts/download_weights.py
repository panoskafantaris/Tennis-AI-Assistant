"""
Download TrackNet V2 weights. Run once:  python scripts/download_weights.py

Source: yastrebksv/TrackNet (MIT License)
Weights hosted on Google Drive.
"""
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEST = WEIGHTS_DIR / "tracknet_v2.pt"
GDRIVE_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"


def main():
    if DEST.exists():
        logger.info(f"Weights already present at {DEST}")
        return

    try:
        import gdown
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"],
        )
        import gdown

    DEST.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    logger.info(f"Downloading -> {DEST}")
    gdown.download(url, str(DEST), quiet=False)

    if DEST.exists():
        mb = DEST.stat().st_size / 1e6
        logger.info(f"Done ({mb:.1f} MB)")
    else:
        logger.error(f"Failed. Download manually: {url}")
        sys.exit(1)


if __name__ == "__main__":
    main()
