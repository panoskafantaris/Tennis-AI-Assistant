"""GPU device selection and precision management."""
import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preferred: str = "cuda") -> torch.device:
    """Return best available device, preferring GPU."""
    if preferred == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {name} ({vram:.1f} GB VRAM)")
        return dev

    if preferred == "cuda":
        logger.warning("CUDA unavailable — falling back to CPU")
    return torch.device("cpu")


def to_fp16_if_available(
    model: torch.nn.Module, use_fp16: bool,
) -> torch.nn.Module:
    """Cast to FP16 on GPU. Halves VRAM on RTX 4050 (6 GB)."""
    if use_fp16 and torch.cuda.is_available():
        logger.info("Using FP16 inference")
        return model.half()
    return model
