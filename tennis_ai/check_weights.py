"""
Quick weight check — run after downloading weights.
Run: python check_weights.py

Reports:
  1. Raw key match (before adaptation)
  2. Adapted key match (after remapping)
  3. Shape verification
  4. Quick forward pass test on dummy data
"""
import sys
import torch

sys.path.insert(0, ".")
from core.tracknet_model import TrackNetV2
from core.weight_adapter import adapt_state_dict
from config.settings import TRACKNET_WEIGHTS, DEVICE, TRACKNET
from utils.device import get_device


def run():
    if not TRACKNET_WEIGHTS.exists():
        print(f"❌ Weight file not found: {TRACKNET_WEIGHTS}")
        print("   Run: python download_weights.py")
        return

    device = get_device(DEVICE)
    model = TrackNetV2(input_frames=3)
    model_keys = set(model.state_dict().keys())

    # Load raw weights
    state = torch.load(str(TRACKNET_WEIGHTS), map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    raw_keys = set(state.keys())

    # ── Step 1: Raw comparison ────────────────────────────────────
    direct = model_keys & raw_keys
    print("=" * 55)
    print("  Step 1: Raw Key Comparison")
    print("=" * 55)
    print(f"  Model keys  : {len(model_keys)}")
    print(f"  Weight keys : {len(raw_keys)}")
    print(f"  Direct match: {len(direct)}")
    print(f"  Missing     : {len(model_keys - raw_keys)}")
    print(f"  Unexpected  : {len(raw_keys - model_keys)}")

    if len(direct) < len(model_keys):
        print("\n  Sample model keys:")
        for k in sorted(model_keys)[:5]:
            print(f"    {k}")
        print("  Sample weight keys:")
        for k in sorted(raw_keys)[:5]:
            print(f"    {k}")

    # ── Step 2: Adapted loading ───────────────────────────────────
    print(f"\n{'=' * 55}")
    print("  Step 2: Adapted Key Mapping")
    print("=" * 55)
    try:
        adapted = adapt_state_dict(model, state)
        print(f"  Adapted keys: {len(adapted)}/{len(model_keys)}")

        missing, unexpected = model.load_state_dict(adapted, strict=False)
        print(f"  Still missing   : {len(missing)}")
        print(f"  Still unexpected: {len(unexpected)}")

        if len(missing) == 0:
            print("\n  ✅ All model layers have weights loaded!")
        else:
            print("\n  ⚠️  Some layers missing (will use random init):")
            for k in missing[:10]:
                print(f"    {k}")
    except RuntimeError as e:
        print(f"  ❌ Adaptation failed: {e}")
        return

    # ── Step 3: Forward pass test ─────────────────────────────────
    print(f"\n{'=' * 55}")
    print("  Step 3: Forward Pass Smoke Test")
    print("=" * 55)
    model = model.to(device).eval()
    H, W = TRACKNET["input_height"], TRACKNET["input_width"]
    dummy = torch.randn(1, 9, H, W, device=device)

    with torch.no_grad():
        out = model(dummy)

    print(f"  Input  : {tuple(dummy.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Expected: (1, 256, {H}, {W})")

    if out.shape == (1, 256, H, W):
        # Check output isn't collapsed
        vals = out[0].float()
        ch_var = vals.var(dim=0).mean().item()
        print(f"  Cross-channel variance: {ch_var:.4f}")
        if ch_var < 1e-6:
            print("  ⚠️  Output collapsed — all channels identical.")
            print("     Weights may be wrong or FP16 issue.")
        else:
            print("  ✅ Output looks healthy!")
    else:
        print("  ❌ Unexpected output shape!")


if __name__ == "__main__":
    run()