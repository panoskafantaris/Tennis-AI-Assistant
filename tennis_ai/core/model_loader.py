"""
Weight loading & model initialisation.

Handles:
  - Loading pre-trained weights from disk
  - Downloading weights from the open-source repo if missing
  - Moving model to GPU + optional FP16
"""
import logging
import urllib.request
from pathlib import Path

import torch

from core.tracknet_model import TrackNetV2
from config.settings import TRACKNET, TRACKNET_WEIGHTS, DEVICE, USE_FP16
from utils.device import get_device, to_fp16_if_available

logger = logging.getLogger(__name__)

# Public weights trained on the TrackNet dataset (Chang-Chia-Chi, MIT License)
_WEIGHTS_URL = (
    "https://github.com/Chang-Chia-Chi/TrackNet/releases/download/v1.0/"
    "TrackNet_best.pt"
)


def _download_weights(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"⬇️  Downloading TrackNet weights → {dest}")
    logger.info("    Source: Chang-Chia-Chi/TrackNet (MIT License)")

    def _progress(block, block_size, total):
        if total > 0:
            pct = min(block * block_size / total * 100, 100)
            print(f"\r    {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(_WEIGHTS_URL, dest, reporthook=_progress)
    print()  # newline after progress
    logger.info("✅ Weights downloaded.")


def load_model(weights_path: Path = TRACKNET_WEIGHTS) -> torch.nn.Module:
    """
    Build TrackNetV2, load weights, move to GPU, optionally cast to FP16.
    Returns the model in eval mode — ready for inference.
    """
    device = get_device(DEVICE)

    model = TrackNetV2(input_frames=TRACKNET["input_frames"])

    if not weights_path.exists():
        logger.warning(f"Weights not found at {weights_path}.")
        try:
            _download_weights(weights_path)
        except Exception as exc:
            logger.error(
                f"Auto-download failed: {exc}\n"
                "Please download weights manually:\n"
                f"  {_WEIGHTS_URL}\n"
                f"  → save to {weights_path}"
            )
            raise

    state = torch.load(weights_path, map_location=device, weights_only=False)
    # Support both raw state_dict and checkpoint dicts
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    logger.info(f"✅ TrackNet weights loaded from {weights_path}")

    model = to_fp16_if_available(model.to(device), USE_FP16)
    model.eval()
    return model