"""
Device management.
Centralises all torch.device logic so the rest of the codebase
never needs to import torch directly for device selection.
"""
import torch
import logging

logger = logging.getLogger(__name__)


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Return the best available device.
    Always tries GPU first; warns and falls back to CPU if unavailable.
    """
    if preferred == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU selected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            return device
        else:
            logger.warning("⚠️  CUDA not available — falling back to CPU. "
                           "Install CUDA toolkit + torch-cuda for GPU acceleration.")
    return torch.device("cpu")


def log_gpu_memory():
    """Log current GPU memory usage — handy for debugging OOM issues."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e6
        reserved  = torch.cuda.memory_reserved(0)  / 1e6
        logger.debug(f"GPU memory — allocated: {allocated:.0f} MB | reserved: {reserved:.0f} MB")


def to_fp16_if_available(model: torch.nn.Module, use_fp16: bool) -> torch.nn.Module:
    """
    Cast model weights to float16 on GPU.
    On an RTX 4050 (6 GB), FP16 roughly halves VRAM use with
    no meaningful accuracy drop for inference.
    """
    if use_fp16 and torch.cuda.is_available():
        logger.info("🔧 Using FP16 (half precision) for inference.")
        return model.half()
    return model