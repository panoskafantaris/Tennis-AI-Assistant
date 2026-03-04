"""
Weight Inspector — deep diagnostic for TrackNet weight compatibility.

Run: python -m tools.weight_inspector
     (from the tennis_ai/ directory)

This tool answers three questions:
  1. Do the weight keys match the model keys?
  2. Do the tensor shapes match where keys align?
  3. Are mismatched keys fixable via automatic remapping?

Output: a clear report + suggested key mapping if fixable.
"""
import sys
import torch
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tracknet_model import TrackNetV2
from config.settings import TRACKNET_WEIGHTS


def load_raw_state(path: Path) -> OrderedDict:
    """Load state dict, unwrapping checkpoint wrappers."""
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for wrapper in ("model_state_dict", "state_dict"):
            if wrapper in state:
                state = state[wrapper]
                break
    return state


def shape_signature(state: dict) -> list:
    """List of (key, shape_tuple) sorted by key."""
    return sorted((k, tuple(v.shape)) for k, v in state.items()
                  if isinstance(v, torch.Tensor))


def build_shape_map(entries: list) -> dict:
    """shape_tuple → list of keys with that shape."""
    smap = {}
    for key, shape in entries:
        smap.setdefault(shape, []).append(key)
    return smap


def try_shape_mapping(model_entries, weight_entries):
    """
    Attempt to build a key mapping by matching shapes in order.

    Logic: group keys by shape. If model and weights have the same
    number of keys per shape, pair them in order.
    Returns: {weight_key: model_key} or None if ambiguous.
    """
    m_map = build_shape_map(model_entries)
    w_map = build_shape_map(weight_entries)

    mapping = {}
    unmapped_shapes = []

    for shape in sorted(set(list(m_map.keys()) + list(w_map.keys()))):
        m_keys = m_map.get(shape, [])
        w_keys = w_map.get(shape, [])

        if len(m_keys) == len(w_keys) and len(m_keys) > 0:
            for wk, mk in zip(w_keys, m_keys):
                mapping[wk] = mk
        else:
            unmapped_shapes.append((shape, len(m_keys), len(w_keys)))

    return mapping, unmapped_shapes


def inspect(weights_path: Path = TRACKNET_WEIGHTS):
    """Run full diagnostic and print report."""
    print("=" * 65)
    print("  TrackNet Weight Inspector")
    print("=" * 65)

    # ── Load both sides ───────────────────────────────────────────
    if not weights_path.exists():
        print(f"\n❌ Weight file not found: {weights_path}")
        print("   Run: python download_weights.py")
        return False

    model = TrackNetV2(input_frames=3)
    model_state = model.state_dict()
    weight_state = load_raw_state(weights_path)

    m_keys = set(model_state.keys())
    w_keys = set(weight_state.keys())

    print(f"\nModel keys  : {len(m_keys)}")
    print(f"Weight keys : {len(w_keys)}")

    # ── Direct match analysis ─────────────────────────────────────
    matched = m_keys & w_keys
    missing = m_keys - w_keys  # in model, not in file
    unexpected = w_keys - m_keys  # in file, not in model

    print(f"\nDirect matches   : {len(matched)}")
    print(f"Missing (model)  : {len(missing)}")
    print(f"Unexpected (file): {len(unexpected)}")

    if len(matched) == len(m_keys):
        # Perfect match — check shapes
        print("\n✅ All keys match by name!")
        bad = _check_shapes(model_state, weight_state, matched)
        return bad == 0

    # ── Show first few mismatches ─────────────────────────────────
    print("\n── Sample model keys (first 8) ──")
    for k in sorted(m_keys)[:8]:
        print(f"  {k:45s} {tuple(model_state[k].shape)}")

    print("\n── Sample weight keys (first 8) ──")
    for k in sorted(w_keys)[:8]:
        print(f"  {k:45s} {tuple(weight_state[k].shape)}")

    # ── Attempt shape-based mapping ───────────────────────────────
    print("\n── Attempting shape-based key mapping ──")
    m_entries = shape_signature(model_state)
    w_entries = shape_signature(weight_state)

    mapping, unmapped = try_shape_mapping(m_entries, w_entries)

    mapped_count = len(mapping)
    print(f"Mapped by shape  : {mapped_count} / {len(m_entries)}")

    if unmapped:
        print(f"Unmappable shapes: {len(unmapped)}")
        for shape, mc, wc in unmapped[:5]:
            print(f"  shape={shape}  model_has={mc}  weights_has={wc}")

    if mapped_count == len(m_entries):
        print("\n✅ Full mapping possible! All shapes align in order.")
        print("   The weight_adapter module can remap automatically.")
        _print_mapping_sample(mapping)
        _save_mapping(mapping, weights_path.parent)
        return True
    elif mapped_count > len(m_entries) * 0.8:
        print(f"\n⚠️  Partial mapping ({mapped_count}/{len(m_entries)}).")
        print("   Some layers will stay randomly initialised.")
        _print_mapping_sample(mapping)
        return False
    else:
        print("\n❌ Cannot map — architectures are likely different.")
        print("   The weight file may be for a different TrackNet variant.")
        return False


def _check_shapes(model_state, weight_state, keys):
    bad = 0
    for k in sorted(keys):
        ms = tuple(model_state[k].shape)
        ws = tuple(weight_state[k].shape)
        if ms != ws:
            print(f"  ❌ Shape mismatch: {k}  model={ms}  file={ws}")
            bad += 1
    if bad == 0:
        print("   All matched key shapes are correct.")
    return bad


def _print_mapping_sample(mapping, n=6):
    print(f"\n── Key mapping sample (first {n}) ──")
    for i, (wk, mk) in enumerate(sorted(mapping.items())):
        if i >= n:
            print(f"  ... ({len(mapping) - n} more)")
            break
        arrow = "→" if wk != mk else "="
        print(f"  {wk:45s} {arrow} {mk}")


def _save_mapping(mapping, directory):
    """Save mapping dict for use by weight_adapter."""
    out = directory / "key_mapping.txt"
    with open(out, "w") as f:
        for wk, mk in sorted(mapping.items()):
            f.write(f"{wk}\t{mk}\n")
    print(f"\nMapping saved → {out}")


if __name__ == "__main__":
    inspect()