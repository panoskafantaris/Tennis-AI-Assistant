"""
Weight adapter — remaps state_dict keys between different TrackNet variants.

TrackNet has several open-source implementations that share the same
architecture but use different naming conventions:

  yastrebksv/TrackNet   : conv1.block.0.weight  (our model format)
  Chang-Chia-Chi/TrackNet: may use different prefixes
  nttcom/TrackNetV2     : completely different names

This adapter handles all three strategies:
  1. Direct match  — keys already align, pass through
  2. Shape mapping — same shapes in same order, remap keys
  3. Manual rules  — known prefix/suffix transformations
"""
import logging
from collections import OrderedDict
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _strip_module_prefix(state: dict) -> OrderedDict:
    """Remove 'module.' prefix from DataParallel-wrapped checkpoints."""
    cleaned = OrderedDict()
    for k, v in state.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        cleaned[new_key] = v
    return cleaned


def _shape_order_mapping(
    model_state: OrderedDict,
    weight_state: OrderedDict,
) -> Optional[Dict[str, str]]:
    """
    Build {weight_key → model_key} by matching tensor shapes in order.

    Works when two models have identical architecture but different key names.
    Groups tensors by shape, then pairs them positionally.
    """
    def shape_groups(state):
        groups = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                s = tuple(v.shape)
                groups.setdefault(s, []).append(k)
        return groups

    m_groups = shape_groups(model_state)
    w_groups = shape_groups(weight_state)

    mapping = {}
    for shape, m_keys in m_groups.items():
        w_keys = w_groups.get(shape, [])
        if len(m_keys) != len(w_keys):
            return None  # ambiguous — can't map safely
        for wk, mk in zip(w_keys, m_keys):
            mapping[wk] = mk

    return mapping


def adapt_state_dict(
    model: torch.nn.Module,
    raw_state: dict,
) -> OrderedDict:
    """
    Adapt a raw state_dict to match the model's expected keys.

    Tries in order:
      1. Direct loading (keys already match)
      2. Strip 'module.' prefix (DataParallel artifact)
      3. Shape-based positional mapping

    Returns the remapped state_dict ready for model.load_state_dict().
    Raises RuntimeError if no strategy works.
    """
    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    # ── Strategy 1: direct match ──────────────────────────────────
    raw_keys = set(raw_state.keys())
    overlap = model_keys & raw_keys

    if len(overlap) == len(model_keys):
        logger.info("Weight keys match directly.")
        return OrderedDict(
            (k, raw_state[k]) for k in model_state if k in raw_state
        )

    # ── Strategy 2: strip 'module.' prefix ────────────────────────
    stripped = _strip_module_prefix(raw_state)
    stripped_keys = set(stripped.keys())
    overlap2 = model_keys & stripped_keys

    if len(overlap2) == len(model_keys):
        logger.info("Matched after stripping 'module.' prefix.")
        return OrderedDict(
            (k, stripped[k]) for k in model_state if k in stripped
        )

    # ── Strategy 3: shape-based mapping ───────────────────────────
    mapping = _shape_order_mapping(model_state, raw_state)

    if mapping and len(mapping) >= len(model_keys):
        logger.info(
            f"Shape-mapped {len(mapping)} keys "
            f"(model expects {len(model_keys)})."
        )
        adapted = OrderedDict()
        for w_key, m_key in mapping.items():
            if m_key in model_keys:
                adapted[m_key] = raw_state[w_key]
        return adapted

    # ── Strategy 4: partial best-effort ───────────────────────────
    # Use whatever direct matches exist + shape mapping for the rest
    logger.warning(
        f"Partial match only: {len(overlap)} direct, "
        f"{len(mapping or {})} shape-mapped out of {len(model_keys)}."
    )
    result = OrderedDict()
    for k in model_state:
        if k in raw_state:
            result[k] = raw_state[k]
    if mapping:
        for w_key, m_key in mapping.items():
            if m_key not in result and m_key in model_keys:
                result[m_key] = raw_state[w_key]

    loaded = len(result)
    total = len(model_keys)
    if loaded < total * 0.5:
        raise RuntimeError(
            f"Only {loaded}/{total} keys mapped. "
            "Weight file likely from incompatible architecture."
        )

    logger.warning(f"Loaded {loaded}/{total} keys. Some layers are random!")
    return result