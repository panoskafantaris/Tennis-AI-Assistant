"""
Weight adapter — remaps state_dict keys between TrackNet variants.

Strategies (tried in order):
  1. Direct match — keys already align
  2. Strip 'module.' prefix (DataParallel artifact)
  3. Shape-based positional mapping (same architecture, different names)
"""
import logging
from collections import OrderedDict
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _strip_prefix(state: dict) -> OrderedDict:
    """Remove 'module.' prefix from DataParallel checkpoints."""
    return OrderedDict(
        (k.replace("module.", "", 1) if k.startswith("module.") else k, v)
        for k, v in state.items()
    )


def _shape_mapping(
    model_state: OrderedDict, weight_state: OrderedDict,
) -> Optional[Dict[str, str]]:
    """Map weight_key -> model_key by matching tensor shapes in order."""
    def groups(state):
        g = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                g.setdefault(tuple(v.shape), []).append(k)
        return g

    m_grp, w_grp = groups(model_state), groups(weight_state)
    mapping = {}
    for shape, m_keys in m_grp.items():
        w_keys = w_grp.get(shape, [])
        if len(m_keys) != len(w_keys):
            return None
        for wk, mk in zip(w_keys, m_keys):
            mapping[wk] = mk
    return mapping


def adapt_state_dict(
    model: torch.nn.Module, raw_state: dict,
) -> OrderedDict:
    """Adapt raw state_dict to match model keys. Raises on failure."""
    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    # Unwrap checkpoint wrappers
    if isinstance(raw_state, dict):
        for wrapper in ("model_state_dict", "state_dict"):
            if wrapper in raw_state:
                raw_state = raw_state[wrapper]
                break

    # Strategy 1: direct match
    if model_keys <= set(raw_state.keys()):
        logger.info("Weight keys match directly.")
        return OrderedDict((k, raw_state[k]) for k in model_state)

    # Strategy 2: strip module prefix
    stripped = _strip_prefix(raw_state)
    if model_keys <= set(stripped.keys()):
        logger.info("Matched after stripping 'module.' prefix.")
        return OrderedDict((k, stripped[k]) for k in model_state)

    # Strategy 3: shape mapping
    mapping = _shape_mapping(model_state, raw_state)
    if mapping and len(mapping) >= len(model_keys):
        logger.info(f"Shape-mapped {len(mapping)} keys.")
        return OrderedDict(
            (mk, raw_state[wk])
            for wk, mk in mapping.items() if mk in model_keys
        )

    # Partial fallback
    result = OrderedDict()
    for k in model_state:
        if k in raw_state:
            result[k] = raw_state[k]
    if mapping:
        for wk, mk in mapping.items():
            if mk not in result and mk in model_keys:
                result[mk] = raw_state[wk]

    if len(result) < len(model_keys) * 0.5:
        raise RuntimeError(
            f"Only {len(result)}/{len(model_keys)} keys mapped. "
            "Incompatible weight file."
        )
    logger.warning(f"Partial load: {len(result)}/{len(model_keys)} keys.")
    return result
