"""
Setup TrackNetV3 — clone repo + download weights. Run once:
    python scripts/setup_v3.py

Source: qaz812345/TrackNetV3 (MIT License)
"""
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REPO_DIR = ROOT / "tracknetv3_repo"
CKPT_DIR = ROOT / "weights" / "tracknetv3"
GDRIVE_ID = "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"


def clone_repo():
    if REPO_DIR.exists():
        logger.info(f"Repo already at {REPO_DIR}")
        return
    logger.info("Cloning TrackNetV3 (MIT License)...")
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/qaz812345/TrackNetV3.git",
        str(REPO_DIR),
    ])


def download_weights():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tracknet = CKPT_DIR / "TrackNet_best.pt"
    inpaint = CKPT_DIR / "InpaintNet_best.pt"

    if tracknet.exists() and inpaint.exists():
        logger.info("Weights already present.")
        return

    try:
        import gdown
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"],
        )
        import gdown

    zip_path = CKPT_DIR / "ckpts.zip"
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    logger.info("Downloading TrackNetV3 checkpoints...")
    gdown.download(url, str(zip_path), quiet=False)

    if not zip_path.exists():
        logger.error(f"Failed. Download manually: {url}")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(CKPT_DIR)
    zip_path.unlink()

    nested = CKPT_DIR / "ckpts"
    if nested.exists():
        for f in nested.glob("*.pt"):
            f.rename(CKPT_DIR / f.name)
        nested.rmdir()

    for name in ["TrackNet_best.pt", "InpaintNet_best.pt"]:
        p = CKPT_DIR / name
        if p.exists():
            logger.info(f"  {name} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            logger.warning(f"  {name} not found!")


if __name__ == "__main__":
    clone_repo()
    download_weights()
    logger.info("\nDone! Run:  python main.py --source video.mp4 --detector tracknetv3")
