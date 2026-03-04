"""
TrackNet V2 architecture — matched to yastrebksv/TrackNet weights.

Source: https://github.com/yastrebksv/TrackNet (MIT License)

Key fix: the decoder uses explicit size targets for upsampling
instead of scale_factor=2, avoiding rounding errors when the
input height/width isn't perfectly divisible by 16.

Input  : [B, 9, H, W]  — 3 consecutive frames stacked channel-wise
Output : [B, 256, H, W] raw logits (same spatial size as input)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Conv2d(0) → ReLU(1) → BatchNorm2d(2)
    Order matches the yastrebksv/TrackNet weight file exactly.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class ConvBlock(nn.Module):
    """Named conv block matching weight-file key pattern."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = _conv_bn_relu(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TrackNetV2(nn.Module):
    """
    Encoder-decoder with 4× pooling and bilinear upsampling.

    Encoder: 9→64→64→pool→128→128→pool→256→256→256→pool→512→512→512→pool
    Decoder: 512→up→256→256→256→up→128→128→up→64→64→256
    """

    def __init__(self, input_frames: int = 3):
        super().__init__()
        in_ch = input_frames * 3

        # ── Encoder ───────────────────────────────────────────────
        self.conv1  = ConvBlock(in_ch, 64)
        self.conv2  = ConvBlock(64,   64)
        self.conv3  = ConvBlock(64,  128)
        self.conv4  = ConvBlock(128, 128)
        self.conv5  = ConvBlock(128, 256)
        self.conv6  = ConvBlock(256, 256)
        self.conv7  = ConvBlock(256, 256)
        self.conv8  = ConvBlock(256, 512)
        self.conv9  = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Decoder ───────────────────────────────────────────────
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.conv16 = ConvBlock(128,  64)
        self.conv17 = ConvBlock(64,   64)

        # ── Output head ───────────────────────────────────────────
        self.conv18 = ConvBlock(64, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save sizes at each pool stage for exact upsampling
        s0 = x.shape[2:]  # original H, W

        x = self.conv1(x)
        x = self.conv2(x)
        s1 = x.shape[2:]
        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        s2 = x.shape[2:]
        x = self.pool(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        s3 = x.shape[2:]
        x = self.pool(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool(x)

        # Decoder — upsample to exact pre-pool sizes
        x = F.interpolate(x, size=s3, mode="bilinear", align_corners=False)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = F.interpolate(x, size=s2, mode="bilinear", align_corners=False)
        x = self.conv14(x)
        x = self.conv15(x)

        x = F.interpolate(x, size=s1, mode="bilinear", align_corners=False)
        x = self.conv16(x)
        x = self.conv17(x)

        x = F.interpolate(x, size=s0, mode="bilinear", align_corners=False)
        x = self.conv18(x)

        return x