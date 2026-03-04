"""
Standalone script to download TrackNet weights before running the pipeline.
Run once: python download_weights.py

Weights source: yastrebksv/TrackNet (PyTorch implementation, MIT License)
https://github.com/yastrebksv/TrackNet
Weights hosted on Google Drive (file ID: 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl)

Requires: pip install gdown
"""
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR   = Path(__file__).parent / "weights"
DEST          = WEIGHTS_DIR / "tracknet_v2.pt"

# Google Drive file ID from yastrebksv/TrackNet (MIT License)
GDRIVE_FILE_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"


def ensure_gdown() -> None:
    """Install gdown if not already present."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        logger.info("Installing gdown (required for Google Drive download)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        logger.info("gdown installed.")


def download_from_gdrive(file_id: str, dest: Path) -> None:
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading TrackNet weights from Google Drive...")
    logger.info(f"Source : {url}")
    logger.info(f"Saving → {dest}")
    gdown.download(url, str(dest), quiet=False)


if __name__ == "__main__":
    if DEST.exists():
        logger.info(f"✅ Weights already present at {DEST}")
        logger.info("   Delete the file and re-run to download again.")
        sys.exit(0)

    ensure_gdown()
    download_from_gdrive(GDRIVE_FILE_ID, DEST)

    if DEST.exists():
        size_mb = DEST.stat().st_size / 1e6
        logger.info(f"\n✅ Done! Weights saved ({size_mb:.1f} MB)")
        logger.info("\nYou can now run:")
        logger.info("  python main.py --source <your_video_or_youtube_url>")
    else:
        logger.error(
            "\n❌ Download failed. Manual fallback:\n"
            "  1. Open: https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl\n"
            "  2. Download the file\n"
            f"  3. Rename it to 'tracknet_v2.pt' and place it in: {WEIGHTS_DIR}"
        )
        sys.exit(1)