"""3D U-Net architecture for volumetric segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """Double 3D convolution block with batch norm and ReLU.

    Architecture: Conv3d -> BatchNorm3d -> ReLU -> Conv3d -> BatchNorm3d -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """Initialize double convolution block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution.

        Args:
            x: Input tensor [B, C_in, D, H, W]

        Returns:
            Output tensor [B, C_out, D, H, W]
        """
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with double convolution and max pooling.

    Consists of:
    - DoubleConv3D for feature extraction
    - MaxPool3d for downsampling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
    ):
        """Initialize encoder block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            pool_size: Max pooling kernel size and stride
        """
        super().__init__()

        self.conv = DoubleConv3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder block.

        Args:
            x: Input tensor [B, C_in, D, H, W]

        Returns:
            Tuple of (skip_connection, downsampled) tensors
            - skip_connection: [B, C_out, D, H, W] for decoder
            - downsampled: [B, C_out, D//2, H//2, W//2] for next encoder
        """
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class UNet3D(nn.Module):
    """3D U-Net for volumetric medical image segmentation.

    Based on the original U-Net architecture adapted for 3D data.
    Features:
    - Symmetric encoder-decoder structure
    - Skip connections for preserving spatial information
    - Batch normalization for stable training
    - Hardware-agnostic (works on CUDA, ROCm, Intel XPU, CPU)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 32,
        depth: int = 4,
    ):
        """Initialize 3D U-Net.

        Args:
            in_channels: Number of input channels (typically 1 for CT scans)
            out_channels: Number of output channels (task-dependent)
            init_features: Number of features in first encoder layer
            depth: Network depth (number of encoder/decoder stages)

        Example:
            >>> model = UNet3D(in_channels=1, out_channels=2, init_features=32, depth=4)
            >>> x = torch.randn(2, 1, 64, 64, 64)
            >>> output = model(x)
            >>> output.shape
            torch.Size([2, 2, 64, 64, 64])
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.depth = depth

        # Build encoder blocks
        self.encoders = nn.ModuleList()
        features = init_features

        for i in range(depth):
            in_ch = in_channels if i == 0 else features // 2
            self.encoders.append(EncoderBlock(in_ch, features))
            features *= 2

        # Bottleneck (deepest layer, no pooling)
        self.bottleneck = DoubleConv3D(features // 2, features)

        # TODO: Implement decoder blocks

        # Temporary output layer for testing encoder
        # Will be replaced with proper decoder
        self._temp_output_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Input tensor [B, in_channels, D, H, W]

        Returns:
            Output tensor [B, out_channels, D, H, W]

        Raises:
            ValueError: If input has wrong number of dimensions
        """
        if x.dim() != 5:
            msg = f"Expected 5D input [B, C, D, H, W], got {x.dim()}D tensor"
            raise ValueError(msg)

        # Store original spatial size
        original_size = x.shape[2:]

        # Encoder path with skip connections
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # TODO: Decoder path (will use skip_connections in reverse)
        # For now, return bottleneck output upsampled to original size
        # This maintains spatial size while we implement the encoder

        # Temporary: upsample back to original size for testing
        # This will be replaced by proper decoder implementation
        x = nn.functional.interpolate(x, size=original_size, mode="trilinear", align_corners=False)

        # Temporary: reduce channels to match output
        x = self._temp_output_conv(x)

        return x
