"""
Run: python check_weights.py
Prints which weight keys matched vs were skipped.
"""
import sys, torch
sys.path.insert(0, ".")
from core.tracknet_model import TrackNetV2
from config.settings import TRACKNET_WEIGHTS, DEVICE
from utils.device import get_device

device = get_device(DEVICE)
model  = TrackNetV2(input_frames=3)

state = torch.load(str(TRACKNET_WEIGHTS), map_location="cpu", weights_only=False)
if "model_state_dict" in state: state = state["model_state_dict"]
elif "state_dict"       in state: state = state["state_dict"]

result = model.load_state_dict(state, strict=False)

print("=== MISSING KEYS (in model, not in weights file) ===")
for k in result.missing_keys:
    print(f"  MISSING : {k}")

print("\n=== UNEXPECTED KEYS (in weights file, not in model) ===")
for k in result.unexpected_keys:
    print(f"  UNEXPECTED: {k}")

print(f"\nTotal weight keys  : {len(state)}")
print(f"Missing  keys      : {len(result.missing_keys)}")
print(f"Unexpected keys    : {len(result.unexpected_keys)}")

# Also print model's own key names so we can compare
print("\n=== MODEL KEY NAMES (first 10) ===")
for k in list(model.state_dict().keys())[:10]:
    print(f"  {k}")

print("\n=== WEIGHT FILE KEY NAMES (first 10) ===")
for k in list(state.keys())[:10]:
    print(f"  {k}")