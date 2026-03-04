"""
Setup TrackNetV3 — clone repo + download pretrained weights.

Run once:  python setup_tracknetv3.py

Source: https://github.com/qaz812345/TrackNetV3 (MIT License)
Paper:  TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations
        and Trajectory Rectification (ACM MM 2023)

Tested on tennis by: https://github.com/nickluo/TrackNetV3
"""
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
V3_DIR = ROOT / "tracknetv3_repo"
CKPT_DIR = ROOT / "weights" / "tracknetv3"

# Google Drive file from the official TrackNetV3 repo
CKPT_GDRIVE_ID = "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"


def clone_repo():
    """Clone TrackNetV3 repository (shallow clone)."""
    if V3_DIR.exists():
        logger.info(f"✅ TrackNetV3 repo already at {V3_DIR}")
        return
    logger.info("📥 Cloning TrackNetV3 repository (MIT License)...")
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/qaz812345/TrackNetV3.git",
        str(V3_DIR),
    ])
    logger.info("✅ Cloned successfully.")


def ensure_gdown():
    """Install gdown if needed."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        logger.info("Installing gdown...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "gdown", "-q"
        ])


def download_weights():
    """Download pretrained checkpoints from Google Drive."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tracknet_pt = CKPT_DIR / "TrackNet_best.pt"
    inpaint_pt = CKPT_DIR / "InpaintNet_best.pt"

    if tracknet_pt.exists() and inpaint_pt.exists():
        logger.info(f"✅ Weights already present:")
        logger.info(f"   {tracknet_pt}")
        logger.info(f"   {inpaint_pt}")
        return

    ensure_gdown()
    import gdown

    zip_path = CKPT_DIR / "TrackNetV3_ckpts.zip"
    url = f"https://drive.google.com/uc?id={CKPT_GDRIVE_ID}"

    logger.info("📥 Downloading TrackNetV3 checkpoints...")
    logger.info(f"   Source: {url}")
    gdown.download(url, str(zip_path), quiet=False)

    if not zip_path.exists():
        logger.error("❌ Download failed. Manual fallback:")
        logger.error(f"   1. Open: https://drive.google.com/file/d/{CKPT_GDRIVE_ID}")
        logger.error(f"   2. Download and extract to: {CKPT_DIR}")
        sys.exit(1)

    logger.info("📦 Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(CKPT_DIR)
    zip_path.unlink()

    # Move files from nested ckpts/ folder if needed
    nested = CKPT_DIR / "ckpts"
    if nested.exists():
        for f in nested.glob("*.pt"):
            f.rename(CKPT_DIR / f.name)
        nested.rmdir()

    # Verify
    for name in ["TrackNet_best.pt", "InpaintNet_best.pt"]:
        path = CKPT_DIR / name
        if path.exists():
            mb = path.stat().st_size / 1e6
            logger.info(f"   ✅ {name} ({mb:.1f} MB)")
        else:
            logger.warning(f"   ⚠️  {name} not found!")


if __name__ == "__main__":
    clone_repo()
    download_weights()

    logger.info("\n" + "=" * 55)
    logger.info("  TrackNetV3 setup complete!")
    logger.info("=" * 55)
    logger.info("\nNext steps:")
    logger.info("  python diagnose_v3.py --source tennis_match_2.mp4")
    logger.info("  python main.py --source video.mp4 --detector tracknetv3")