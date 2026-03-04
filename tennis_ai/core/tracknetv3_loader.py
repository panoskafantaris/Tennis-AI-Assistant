"""
TrackNetV3 model loader.

Imports TrackNet(in_dim, out_dim) from the cloned
qaz812345/TrackNetV3 repository and loads pretrained weights.

Checkpoint contains param_dict with training config:
  seq_len=8, bg_mode='concat' → in_dim = (8+1)*3 = 27

Source: https://github.com/qaz812345/TrackNetV3 (MIT License)
"""
import logging
import sys
from pathlib import Path

import torch

from config.settings import ROOT_DIR, DEVICE
from utils.device import get_device

logger = logging.getLogger(__name__)

V3_REPO = ROOT_DIR / "tracknetv3_repo"
V3_WEIGHTS = ROOT_DIR / "weights" / "tracknetv3"


def _ensure_repo_on_path():
    repo_str = str(V3_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    if not V3_REPO.exists():
        raise FileNotFoundError(
            f"TrackNetV3 repo not found at {V3_REPO}.\n"
            "Run: python setup_tracknetv3.py"
        )


def load_tracknetv3(
    weights_path: Path = None,
) -> tuple:
    """
    Build TrackNetV3 model and load pretrained weights.

    Returns:
        (model, param_dict) — model in eval mode, plus the
        training config so the inference engine knows seq_len etc.
    """
    _ensure_repo_on_path()
    from model import TrackNet  # noqa: E402

    device = get_device(DEVICE)

    if weights_path is None:
        weights_path = V3_WEIGHTS / "TrackNet_best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Run: python setup_tracknetv3.py"
        )

    ckpt = torch.load(
        weights_path, map_location=device, weights_only=False
    )
    state_dict = ckpt["model"]
    param_dict = ckpt.get("param_dict", {})

    # Extract dimensions from checkpoint
    seq_len = param_dict.get("seq_len", 8)
    bg_mode = param_dict.get("bg_mode", "concat")

    # in_dim from first conv weight shape
    first_key = next(k for k in state_dict if "weight" in k)
    in_dim = state_dict[first_key].shape[1]

    # out_dim from last conv weight shape
    last_key = [k for k in state_dict if "weight" in k][-1]
    out_dim = state_dict[last_key].shape[0]

    logger.info(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
    logger.info(f"  seq_len={seq_len}, bg_mode={bg_mode}")
    logger.info(f"  TrackNet(in_dim={in_dim}, out_dim={out_dim})")

    model = TrackNet(in_dim=in_dim, out_dim=out_dim)
    result = model.load_state_dict(state_dict, strict=True)
    logger.info(f"✅ TrackNetV3 loaded — all keys matched")

    model = model.to(device).eval()
    return model, param_dict