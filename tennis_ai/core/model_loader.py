"""
Weight loading & model initialisation.

Handles:
  - Loading pre-trained weights with automatic key adaptation
  - Downloading weights from yastrebksv/TrackNet (MIT License)
  - Moving model to GPU + optional FP16

Weight source: yastrebksv/TrackNet (PyTorch, MIT License)
https://github.com/yastrebksv/TrackNet
"""
import logging
import subprocess
import sys
from pathlib import Path

import torch

from core.tracknet_model import TrackNetV2
from core.weight_adapter import adapt_state_dict
from config.settings import TRACKNET, TRACKNET_WEIGHTS, DEVICE, USE_FP16
from utils.device import get_device, to_fp16_if_available

logger = logging.getLogger(__name__)

# Google Drive file ID — yastrebksv/TrackNet (MIT License)
_GDRIVE_FILE_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"


def _ensure_gdown() -> None:
    """Install gdown if not already present."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        logger.info("Installing gdown for Google Drive download...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"]
        )


def _download_weights(dest: Path) -> None:
    """Download TrackNet weights from Google Drive via gdown."""
    _ensure_gdown()
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={_GDRIVE_FILE_ID}"

    logger.info(f"⬇️  Downloading TrackNet weights → {dest}")
    logger.info(f"   Source: yastrebksv/TrackNet (MIT License)")
    gdown.download(url, str(dest), quiet=False)

    if not dest.exists():
        raise RuntimeError(
            f"Download failed. Manual fallback:\n"
            f"  1. Open: https://drive.google.com/file/d/{_GDRIVE_FILE_ID}\n"
            f"  2. Download and save as: {dest}"
        )
    size_mb = dest.stat().st_size / 1e6
    logger.info(f"✅ Weights downloaded ({size_mb:.1f} MB)")


def _load_raw_state(path: Path) -> dict:
    """Load and unwrap checkpoint dict → raw state_dict."""
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in state:
                return state[key]
    return state


def load_model(weights_path: Path = TRACKNET_WEIGHTS) -> torch.nn.Module:
    """
    Build TrackNetV2, load adapted weights, move to GPU.
    Returns the model in eval mode — ready for inference.
    """
    device = get_device(DEVICE)

    model = TrackNetV2(input_frames=TRACKNET["input_frames"])

    if not weights_path.exists():
        logger.warning(f"Weights not found at {weights_path}.")
        try:
            _download_weights(weights_path)
        except Exception as exc:
            logger.error(f"Auto-download failed: {exc}")
            raise

    # Load raw state and adapt keys to our model
    raw_state = _load_raw_state(weights_path)
    adapted = adapt_state_dict(model, raw_state)

    # Load with strict=True — adapter already handled remapping
    missing, unexpected = model.load_state_dict(adapted, strict=False)

    loaded = len(adapted)
    total = len(model.state_dict())
    logger.info(f"✅ Loaded {loaded}/{total} weight tensors from {weights_path}")

    if missing:
        logger.warning(f"   {len(missing)} keys still missing (random init)")
    if unexpected:
        logger.warning(f"   {len(unexpected)} unexpected keys ignored")

    model = to_fp16_if_available(model.to(device), USE_FP16)
    model.eval()
    return model