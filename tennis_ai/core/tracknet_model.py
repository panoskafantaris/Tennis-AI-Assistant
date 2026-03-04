"""
TrackNet V2 architecture — matched exactly to yastrebksv/TrackNet weights.

Source: https://github.com/yastrebksv/TrackNet (MIT License)

Layer map (verified against weight keys):
  Encoder : conv1–conv10  with MaxPool2d after conv2, conv4, conv7, conv10
  Decoder : conv11–conv17 with bilinear Upsample before conv11, conv14, conv16
  Head    : conv18  [64 → 256 → softmax across 256 classes → heatmap]

Input  : [B, 9, H, W]  — 3 consecutive RGB frames stacked channel-wise
Output : [B, 256, H, W] raw logits  (caller picks argmax channel or applies softmax)

Note: this model outputs 256-class heatmaps (one hot-encoded per pixel),
not a single sigmoid map. The inference engine handles this correctly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Conv2d(0) → ReLU(1) → BatchNorm2d(2)
    Order matches the yastrebksv/TrackNet weight file exactly:
      block.0 = Conv2d  (has weight + bias)
      block.2 = BatchNorm2d (has weight, bias, running_mean, running_var)
    ReLU sits at index 1 but has no parameters so it doesn't appear in state_dict.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class ConvBlock(nn.Module):
    """Single named conv block matching the weight-file key pattern (conv1.block…)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = _conv_bn_relu(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TrackNetV2(nn.Module):
    """
    Flat encoder-decoder — NO skip connections, NO transposed convolutions.
    Uses MaxPool2d for downsampling and bilinear Upsample for upsampling.
    Architecture reverse-engineered from the yastrebksv/TrackNet weight file.

    Encoder channel progression:
      9 → 64 → 64 → pool → 128 → 128 → pool → 256 → 256 → 256
        → pool → 512 → 512 → 512 → pool

    Decoder channel progression (bilinear up before each stage):
      512 → up → 256 → 256 → 256 → up → 128 → 128 → up → 64 → 64
        → 256  (head / output)
    """

    def __init__(self, input_frames: int = 3):
        super().__init__()
        in_ch = input_frames * 3   # 9

        # ── Encoder ───────────────────────────────────────────────────────
        self.conv1  = ConvBlock(in_ch, 64)
        self.conv2  = ConvBlock(64,   64)
        # pool
        self.conv3  = ConvBlock(64,  128)
        self.conv4  = ConvBlock(128, 128)
        # pool
        self.conv5  = ConvBlock(128, 256)
        self.conv6  = ConvBlock(256, 256)
        self.conv7  = ConvBlock(256, 256)
        # pool
        self.conv8  = ConvBlock(256, 512)
        self.conv9  = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        # pool

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Decoder ───────────────────────────────────────────────────────
        # upsample ×2
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        # upsample ×2
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        # upsample ×2
        self.conv16 = ConvBlock(128,  64)
        self.conv17 = ConvBlock(64,   64)

        # ── Output head ───────────────────────────────────────────────────
        # 64 → 256 classes (TrackNet uses 256-bin heatmap encoding)
        self.conv18 = ConvBlock(64, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool(x)

        # Decoder
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv14(x)
        x = self.conv15(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv16(x)
        x = self.conv17(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv18(x)   # [B, 256, H, W]

        return x             # raw logits — softmax applied in inference.py