"""
TrackNet V2 architecture — matched to yastrebksv/TrackNet weights.
Source: https://github.com/yastrebksv/TrackNet (MIT License)

Input  : [B, 9, H, W]  — 3 frames stacked channel-wise
Output : [B, 256, H, W] — raw logits (same spatial size as input)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv -> ReLU -> BN (order matches weight file exactly)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = _conv_bn_relu(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TrackNetV2Model(nn.Module):
    """
    Encoder-decoder with 4x pooling and bilinear upsampling.
    Encoder: 9->64->64->pool->128->128->pool->256x3->pool->512x3->pool
    Decoder: 512->up->256x3->up->128x2->up->64x2->up->256
    """

    def __init__(self, input_frames: int = 3):
        super().__init__()
        in_ch = input_frames * 3

        # Encoder
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
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.conv16 = ConvBlock(128,  64)
        self.conv17 = ConvBlock(64,   64)
        self.conv18 = ConvBlock(64,  256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = x.shape[2:]
        x = self.conv2(self.conv1(x));  s1 = x.shape[2:]
        x = self.pool(x)
        x = self.conv4(self.conv3(x));  s2 = x.shape[2:]
        x = self.pool(x)
        x = self.conv7(self.conv6(self.conv5(x)));  s3 = x.shape[2:]
        x = self.pool(x)
        x = self.conv10(self.conv9(self.conv8(x)))
        x = self.pool(x)

        x = F.interpolate(x, size=s3, mode="bilinear", align_corners=False)
        x = self.conv13(self.conv12(self.conv11(x)))
        x = F.interpolate(x, size=s2, mode="bilinear", align_corners=False)
        x = self.conv15(self.conv14(x))
        x = F.interpolate(x, size=s1, mode="bilinear", align_corners=False)
        x = self.conv17(self.conv16(x))
        x = F.interpolate(x, size=s0, mode="bilinear", align_corners=False)
        return self.conv18(x)
