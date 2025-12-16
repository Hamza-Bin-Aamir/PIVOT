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


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution and skip connection fusion.

    Consists of:
    - ConvTranspose3d for upsampling
    - Concatenation with skip connection from encoder
    - DoubleConv3D for feature refinement
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_size: int = 2,
    ):
        """Initialize decoder block.

        Args:
            in_channels: Number of input channels from previous decoder/bottleneck
            out_channels: Number of output channels
            upsample_size: Upsampling factor (kernel size and stride)
        """
        super().__init__()

        # Transposed convolution for upsampling
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=upsample_size,
            stride=upsample_size,
        )

        # Double conv after concatenation with skip connection
        # Input channels: in_channels//2 (upsampled) + in_channels//2 (skip) = in_channels
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x: Input tensor from previous decoder/bottleneck [B, C_in, D, H, W]
            skip: Skip connection from encoder [B, C_in//2, D*2, H*2, W*2]

        Returns:
            Output tensor [B, C_out, D*2, H*2, W*2]
        """
        # Upsample
        x = self.upsample(x)

        # Handle potential size mismatch due to pooling/upsampling
        # This can happen with odd-sized inputs
        if x.shape != skip.shape:
            # Resize x to match skip connection
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=False
            )

        # Concatenate with skip connection
        x = torch.cat([skip, x], dim=1)

        # Refine features
        x = self.conv(x)

        return x


class BottleneckBlock(nn.Module):
    """Bottleneck block at the deepest layer of U-Net.

    The bottleneck processes features at the lowest spatial resolution
    and highest feature dimension. Unlike encoder blocks, it has no
    pooling, and unlike decoder blocks, it has no upsampling.

    Architecture: DoubleConv3D without pooling or upsampling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """Initialize bottleneck block.

        Args:
            in_channels: Number of input channels from last encoder
            out_channels: Number of output channels (typically 2x in_channels)
        """
        super().__init__()
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck.

        Args:
            x: Input tensor [B, in_channels, D, H, W]

        Returns:
            Output tensor [B, out_channels, D, H, W]
        """
        return self.conv(x)


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

        # Bottleneck (deepest layer, no pooling or upsampling)
        self.bottleneck = BottleneckBlock(features // 2, features)

        # Build decoder blocks
        self.decoders = nn.ModuleList()

        # Decoder mirrors encoder in reverse
        for _ in range(depth):
            self.decoders.append(DecoderBlock(features, features // 2))
            features //= 2

        # Final output convolution (1x1x1 conv to produce desired channels)
        # After all decoders, features == init_features
        self.output_conv = nn.Conv3d(init_features, out_channels, kernel_size=1)

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
            msg = f"Expected 5D input (B, C, D, H, W), got {x.dim()}D"
            raise ValueError(msg)

        # Encoder path with skip connections
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections (reverse order)
        for decoder, skip in zip(self.decoders, reversed(skip_connections), strict=True):
            x = decoder(x, skip)

        # Final output convolution
        x = self.output_conv(x)

        return x


class SegmentationHead(nn.Module):
    """Segmentation head for binary nodule mask prediction.

    Produces pixel-wise binary segmentation masks indicating nodule regions.
    Uses a simple 1x1x1 convolution to map backbone features to a single
    output channel, followed by sigmoid activation for probability output.

    Architecture: Conv3d(1x1x1) -> Sigmoid (optional, for inference)
    """

    def __init__(
        self,
        in_channels: int,
        apply_sigmoid: bool = False,
    ):
        """Initialize segmentation head.

        Args:
            in_channels: Number of input channels from backbone
            apply_sigmoid: Whether to apply sigmoid activation.
                          Set False during training (loss handles it),
                          True during inference for probability output.
        """
        super().__init__()

        self.in_channels = in_channels
        self.apply_sigmoid = apply_sigmoid

        # 1x1x1 convolution for segmentation logits
        self.conv = nn.Conv3d(
            in_channels,
            1,  # Binary segmentation (nodule vs background)
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through segmentation head.

        Args:
            x: Input features from backbone [B, in_channels, D, H, W]

        Returns:
            Segmentation output [B, 1, D, H, W]
            - Logits if apply_sigmoid=False (for training with BCE loss)
            - Probabilities if apply_sigmoid=True (for inference)

        Raises:
            ValueError: If input has wrong number of dimensions
        """
        if x.dim() != 5:
            msg = f"Expected 5D input (B, C, D, H, W), got {x.dim()}D"
            raise ValueError(msg)

        # Generate segmentation logits
        x = self.conv(x)

        # Optionally apply sigmoid for inference
        if self.apply_sigmoid:
            x = torch.sigmoid(x)

        return x


class CenterDetectionHead(nn.Module):
    """Center point detection head for nodule localization via heatmap.

    Produces a probability heatmap indicating nodule center locations.
    Uses a 1x1x1 convolution to map backbone features to a single output
    channel representing a center point heatmap with Gaussian peaks at
    nodule centers.

    Architecture: Conv3d(1x1x1) -> Sigmoid (optional, for inference)
    """

    def __init__(
        self,
        in_channels: int,
        apply_sigmoid: bool = False,
    ):
        """Initialize center detection head.

        Args:
            in_channels: Number of input channels from backbone
            apply_sigmoid: Whether to apply sigmoid activation.
                          Set False during training (focal loss handles it),
                          True during inference for probability heatmap.
        """
        super().__init__()

        self.in_channels = in_channels
        self.apply_sigmoid = apply_sigmoid

        # 1x1x1 convolution for center heatmap logits
        self.conv = nn.Conv3d(
            in_channels,
            1,  # Single channel heatmap
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through center detection head.

        Args:
            x: Input features from backbone [B, in_channels, D, H, W]

        Returns:
            Center heatmap output [B, 1, D, H, W]
            - Logits if apply_sigmoid=False (for training with focal loss)
            - Probabilities if apply_sigmoid=True (for inference/peak detection)

        Raises:
            ValueError: If input has wrong number of dimensions
        """
        if x.dim() != 5:
            msg = f"Expected 5D input (B, C, D, H, W), got {x.dim()}D"
            raise ValueError(msg)

        # Generate center heatmap logits
        x = self.conv(x)

        # Optionally apply sigmoid for inference
        if self.apply_sigmoid:
            x = torch.sigmoid(x)

        return x


class SizeRegressionHead(nn.Module):
    """Size regression head for predicting 3D nodule dimensions.

    Predicts diameter values in three dimensions (x, y, z) for detected nodules.
    Uses 1x1x1 convolution to transform feature maps to 3-channel size predictions.
    Designed for smooth L1 loss training.

    Args:
        in_channels (int): Number of input feature channels
        use_global_pool (bool): If True, applies global average pooling to convert
                                spatial feature maps to single size prediction per sample.
                                Default: True for standard size regression.

    Attributes:
        conv (nn.Conv3d): 1x1x1 convolution layer for size prediction
        pool (nn.AdaptiveAvgPool3d | None): Optional global pooling layer
    """

    def __init__(self, in_channels: int, use_global_pool: bool = True) -> None:
        """Initialize size regression head.

        Args:
            in_channels: Number of input feature channels
            use_global_pool: Whether to use global average pooling
        """
        super().__init__()

        # 1x1x1 conv: in_channels -> 3 (diameter_x, diameter_y, diameter_z)
        self.conv = nn.Conv3d(in_channels, 3, kernel_size=1)

        # Optional global pooling for spatial-to-vector prediction
        self.pool = nn.AdaptiveAvgPool3d(1) if use_global_pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for size regression.

        Args:
            x: Input feature tensor of shape (B, C, D, H, W)

        Returns:
            Size predictions:
            - (B, 3, 1, 1, 1) if use_global_pool=True
            - (B, 3, D, H, W) if use_global_pool=False

        Raises:
            ValueError: If input has wrong number of dimensions
        """
        if x.dim() != 5:
            msg = f"Expected 5D input (B, C, D, H, W), got {x.dim()}D"
            raise ValueError(msg)

        # Generate size regression values
        x = self.conv(x)

        # Optionally apply global pooling
        if self.pool is not None:
            x = self.pool(x)

        return x
