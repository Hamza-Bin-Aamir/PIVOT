"""Tests for model architecture."""

from __future__ import annotations

import pytest
import torch

from src.model.unet import DecoderBlock, DoubleConv3D, EncoderBlock, UNet3D


class TestDoubleConv3D:
    """Tests for DoubleConv3D building block."""

    def test_double_conv_output_shape(self):
        """Test DoubleConv3D produces correct output shape."""
        conv = DoubleConv3D(in_channels=1, out_channels=32)
        x = torch.randn(2, 1, 16, 16, 16)

        output = conv(x)

        assert output.shape == (2, 32, 16, 16, 16)

    def test_double_conv_channel_expansion(self):
        """Test DoubleConv3D can expand channels."""
        conv = DoubleConv3D(in_channels=16, out_channels=64)
        x = torch.randn(1, 16, 8, 8, 8)

        output = conv(x)

        assert output.shape == (1, 64, 8, 8, 8)

    def test_double_conv_preserves_spatial_dims(self):
        """Test DoubleConv3D preserves spatial dimensions with padding."""
        conv = DoubleConv3D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        x = torch.randn(1, 32, 24, 24, 24)

        output = conv(x)

        # Spatial dims should be preserved with padding=1
        assert output.shape[2:] == (24, 24, 24)

    def test_double_conv_no_padding(self):
        """Test DoubleConv3D reduces spatial dims without padding."""
        conv = DoubleConv3D(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        x = torch.randn(1, 16, 10, 10, 10)

        output = conv(x)

        # Two 3x3x3 convs without padding: 10 -> 8 -> 6
        assert output.shape == (1, 32, 6, 6, 6)

    def test_double_conv_contains_batch_norm(self):
        """Test DoubleConv3D includes batch normalization."""
        conv = DoubleConv3D(in_channels=1, out_channels=16)

        # Check that batch norm layers exist in the sequential
        has_batch_norm = any(
            isinstance(module, torch.nn.BatchNorm3d) for module in conv.double_conv.modules()
        )
        assert has_batch_norm is True

    def test_double_conv_contains_relu(self):
        """Test DoubleConv3D includes ReLU activations."""
        conv = DoubleConv3D(in_channels=1, out_channels=16)

        # Check that ReLU layers exist
        has_relu = any(isinstance(module, torch.nn.ReLU) for module in conv.double_conv.modules())
        assert has_relu is True


class TestEncoderBlock:
    """Tests for EncoderBlock."""

    def test_encoder_block_returns_tuple(self):
        """Test EncoderBlock returns both skip connection and pooled output."""
        encoder = EncoderBlock(in_channels=1, out_channels=32)
        x = torch.randn(2, 1, 16, 16, 16)

        skip, pooled = encoder(x)

        assert isinstance(skip, torch.Tensor)
        assert isinstance(pooled, torch.Tensor)

    def test_encoder_block_skip_connection_shape(self):
        """Test skip connection has same spatial dims as input."""
        encoder = EncoderBlock(in_channels=1, out_channels=32)
        x = torch.randn(2, 1, 16, 16, 16)

        skip, _ = encoder(x)

        assert skip.shape == (2, 32, 16, 16, 16)

    def test_encoder_block_pooled_shape(self):
        """Test pooled output has halved spatial dimensions."""
        encoder = EncoderBlock(in_channels=1, out_channels=32, pool_size=2)
        x = torch.randn(2, 1, 16, 16, 16)

        _, pooled = encoder(x)

        assert pooled.shape == (2, 32, 8, 8, 8)

    def test_encoder_block_channel_transformation(self):
        """Test encoder transforms channels correctly."""
        encoder = EncoderBlock(in_channels=64, out_channels=128)
        x = torch.randn(1, 64, 32, 32, 32)

        skip, pooled = encoder(x)

        assert skip.shape[1] == 128  # Output channels
        assert pooled.shape[1] == 128

    def test_encoder_block_different_pool_sizes(self):
        """Test encoder with different pooling sizes."""
        encoder_2 = EncoderBlock(in_channels=1, out_channels=16, pool_size=2)
        encoder_4 = EncoderBlock(in_channels=1, out_channels=16, pool_size=4)

        x = torch.randn(1, 1, 32, 32, 32)

        _, pooled_2 = encoder_2(x)
        _, pooled_4 = encoder_4(x)

        assert pooled_2.shape == (1, 16, 16, 16, 16)  # 32/2
        assert pooled_4.shape == (1, 16, 8, 8, 8)  # 32/4

    def test_encoder_block_skip_preserves_features(self):
        """Test skip connection preserves features before pooling."""
        encoder = EncoderBlock(in_channels=1, out_channels=32)
        x = torch.randn(1, 1, 16, 16, 16)

        skip, pooled = encoder(x)

        # Skip should have full resolution features
        assert skip.shape[2:] == x.shape[2:]
        # Pooled should be downsampled
        assert pooled.shape[2:] != x.shape[2:]


class TestDecoderBlock:
    """Tests for DecoderBlock."""

    def test_decoder_block_upsamples_input(self):
        """Test DecoderBlock upsamples input tensor."""
        decoder = DecoderBlock(in_channels=128, out_channels=64)
        x = torch.randn(2, 128, 8, 8, 8)
        skip = torch.randn(2, 64, 16, 16, 16)

        output = decoder(x, skip)

        # Should upsample to skip's spatial size
        assert output.shape[2:] == skip.shape[2:]

    def test_decoder_block_output_channels(self):
        """Test DecoderBlock produces correct output channels."""
        decoder = DecoderBlock(in_channels=256, out_channels=128)
        x = torch.randn(1, 256, 4, 4, 4)
        skip = torch.randn(1, 128, 8, 8, 8)

        output = decoder(x, skip)

        assert output.shape[1] == 128

    def test_decoder_block_concatenates_skip(self):
        """Test DecoderBlock concatenates with skip connection."""
        decoder = DecoderBlock(in_channels=64, out_channels=32)
        x = torch.randn(1, 64, 8, 8, 8)
        skip = torch.randn(1, 32, 16, 16, 16)

        # Manually trace through to verify concatenation
        upsampled = decoder.upsample(x)
        assert upsampled.shape == (1, 32, 16, 16, 16)  # Half channels, 2x spatial

        # After concat with skip: 32 + 32 = 64 channels
        output = decoder(x, skip)
        assert output.shape == (1, 32, 16, 16, 16)

    def test_decoder_block_handles_size_mismatch(self):
        """Test DecoderBlock handles odd-sized inputs with interpolation."""
        decoder = DecoderBlock(in_channels=128, out_channels=64)
        x = torch.randn(1, 128, 7, 7, 7)
        skip = torch.randn(1, 64, 15, 15, 15)  # Not exactly 2x

        output = decoder(x, skip)

        # Should match skip size
        assert output.shape == (1, 64, 15, 15, 15)

    def test_decoder_block_different_upsample_sizes(self):
        """Test DecoderBlock with different upsampling factors."""
        decoder_2 = DecoderBlock(in_channels=64, out_channels=32, upsample_size=2)
        decoder_4 = DecoderBlock(in_channels=64, out_channels=32, upsample_size=4)

        x = torch.randn(1, 64, 4, 4, 4)
        skip_2 = torch.randn(1, 32, 8, 8, 8)
        skip_4 = torch.randn(1, 32, 16, 16, 16)

        output_2 = decoder_2(x, skip_2)
        output_4 = decoder_4(x, skip_4)

        assert output_2.shape == (1, 32, 8, 8, 8)
        assert output_4.shape == (1, 32, 16, 16, 16)

    def test_decoder_block_batch_size(self):
        """Test DecoderBlock handles different batch sizes."""
        decoder = DecoderBlock(in_channels=128, out_channels=64)

        x_b1 = torch.randn(1, 128, 8, 8, 8)
        skip_b1 = torch.randn(1, 64, 16, 16, 16)

        x_b4 = torch.randn(4, 128, 8, 8, 8)
        skip_b4 = torch.randn(4, 64, 16, 16, 16)

        output_b1 = decoder(x_b1, skip_b1)
        output_b4 = decoder(x_b4, skip_b4)

        assert output_b1.shape[0] == 1
        assert output_b4.shape[0] == 4


class TestUNet3DEncoder:
    """Tests for UNet3D encoder path."""

    def test_unet_attributes_exposed(self):
        """Test UNet exposes configuration attributes."""
        model = UNet3D(in_channels=2, out_channels=3, init_features=16, depth=2)

        assert model.in_channels == 2
        assert model.out_channels == 3
        assert model.init_features == 16
        assert model.depth == 2

    def test_unet_encoder_blocks_created(self):
        """Test UNet creates correct number of encoder blocks."""
        model = UNet3D(depth=4)

        assert len(model.encoders) == 4

    def test_unet_encoder_feature_progression(self):
        """Test encoder features double at each level."""
        model = UNet3D(init_features=32, depth=3)

        # Check encoder channel progression
        assert model.encoders[0].conv.double_conv[0].out_channels == 32
        assert model.encoders[1].conv.double_conv[0].out_channels == 64
        assert model.encoders[2].conv.double_conv[0].out_channels == 128

    def test_unet_bottleneck_features(self):
        """Test bottleneck has correct number of features."""
        model = UNet3D(init_features=32, depth=3)

        # After 3 encoder levels: 32 -> 64 -> 128 -> 256 (bottleneck)
        assert model.bottleneck.double_conv[0].in_channels == 128
        assert model.bottleneck.double_conv[0].out_channels == 256

    def test_unet_forward_wrong_dimensions(self):
        """Test UNet raises error for wrong input dimensions."""
        model = UNet3D()

        # 4D input (missing batch or spatial dimension)
        x_4d = torch.randn(1, 1, 16, 16)

        with pytest.raises(ValueError, match="Expected 5D input"):
            model(x_4d)

        # 3D input
        x_3d = torch.randn(1, 16, 16)

        with pytest.raises(ValueError, match="Expected 5D input"):
            model(x_3d)

    def test_unet_forward_batch_size(self):
        """Test UNet handles different batch sizes."""
        model = UNet3D(in_channels=1, out_channels=2, depth=2)

        x_batch_1 = torch.randn(1, 1, 16, 16, 16)
        x_batch_4 = torch.randn(4, 1, 16, 16, 16)

        output_1 = model(x_batch_1)
        output_4 = model(x_batch_4)

        assert output_1.shape[0] == 1
        assert output_4.shape[0] == 4

    def test_unet_forward_preserves_spatial_size(self):
        """Test UNet output has same spatial dimensions as input."""
        model = UNet3D(in_channels=1, out_channels=2, depth=3)

        x = torch.randn(2, 1, 32, 32, 32)
        output = model(x)

        assert output.shape[2:] == x.shape[2:]  # D, H, W preserved

    def test_unet_forward_correct_output_channels(self):
        """Test UNet produces correct number of output channels."""
        model = UNet3D(in_channels=1, out_channels=5, depth=2)

        x = torch.randn(1, 1, 16, 16, 16)
        output = model(x)

        assert output.shape[1] == 5

    def test_unet_encoder_downsampling_progression(self):
        """Test encoder progressively downsamples input."""
        model = UNet3D(depth=3)

        x = torch.randn(1, 1, 64, 64, 64)

        # Manually trace through encoder
        skip_connections = []
        for encoder in model.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Check spatial downsampling at each level
        assert skip_connections[0].shape[2:] == (64, 64, 64)  # Level 0: no downsample yet
        assert skip_connections[1].shape[2:] == (32, 32, 32)  # Level 1: /2
        assert skip_connections[2].shape[2:] == (16, 16, 16)  # Level 2: /4

        # x after all encoders should be smallest
        assert x.shape[2:] == (8, 8, 8)  # /8 after 3 pooling ops

    def test_unet_encoder_channel_progression(self):
        """Test encoder progressively increases channels."""
        model = UNet3D(init_features=16, depth=3)

        x = torch.randn(1, 1, 32, 32, 32)

        skip_connections = []
        for encoder in model.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Check channel progression
        assert skip_connections[0].shape[1] == 16  # init_features
        assert skip_connections[1].shape[1] == 32  # init_features * 2
        assert skip_connections[2].shape[1] == 64  # init_features * 4

    def test_unet_different_input_sizes(self):
        """Test UNet handles different input spatial sizes."""
        model = UNet3D(depth=2)

        sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]

        for size in sizes:
            x = torch.randn(1, 1, *size)
            output = model(x)
            assert output.shape[2:] == size

    def test_unet_variable_depth(self):
        """Test UNet with different depths."""
        for depth in [2, 3, 4, 5]:
            model = UNet3D(depth=depth)
            assert len(model.encoders) == depth

    def test_unet_custom_init_features(self):
        """Test UNet with different initial feature counts."""
        model_16 = UNet3D(init_features=16, depth=2)
        model_64 = UNet3D(init_features=64, depth=2)

        assert model_16.encoders[0].conv.double_conv[0].out_channels == 16
        assert model_64.encoders[0].conv.double_conv[0].out_channels == 64

    def test_unet_multi_channel_input(self):
        """Test UNet handles multi-channel input."""
        model = UNet3D(in_channels=3, out_channels=2, depth=2)

        x = torch.randn(2, 3, 16, 16, 16)
        output = model(x)

        assert output.shape == (2, 2, 16, 16, 16)

    def test_unet_encoder_gradients(self):
        """Test gradients flow through encoder."""
        model = UNet3D(depth=2)
        x = torch.randn(1, 1, 16, 16, 16, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check input has gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_unet_encoder_eval_mode(self):
        """Test UNet encoder in evaluation mode."""
        model = UNet3D(depth=2)
        model.eval()

        x = torch.randn(1, 1, 16, 16, 16)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1, 16, 16, 16)

    def test_unet_encoder_reproducibility(self):
        """Test encoder produces consistent results with same input."""
        model = UNet3D(depth=2)
        model.eval()

        x = torch.randn(1, 1, 16, 16, 16)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7)


class TestUNet3DDecoder:
    """Tests for UNet3D decoder path and complete architecture."""

    def test_unet_decoder_blocks_created(self):
        """Test UNet creates correct number of decoder blocks."""
        model = UNet3D(depth=4)

        assert len(model.decoders) == 4

    def test_unet_decoder_channel_progression(self):
        """Test decoder features halve at each level."""
        model = UNet3D(init_features=32, depth=3)

        # Decoder channel progression (reverse of encoder)
        # Bottleneck: 256 -> 128 -> 64 -> 32
        assert model.decoders[0].conv.double_conv[0].out_channels == 128
        assert model.decoders[1].conv.double_conv[0].out_channels == 64
        assert model.decoders[2].conv.double_conv[0].out_channels == 32

    def test_unet_complete_forward_pass(self):
        """Test complete U-Net forward pass."""
        model = UNet3D(in_channels=1, out_channels=3, init_features=16, depth=3)

        x = torch.randn(2, 1, 32, 32, 32)
        output = model(x)

        assert output.shape == (2, 3, 32, 32, 32)

    def test_unet_symmetric_architecture(self):
        """Test U-Net has symmetric encoder-decoder structure."""
        model = UNet3D(depth=4)

        assert len(model.encoders) == len(model.decoders)

    def test_unet_output_conv_channels(self):
        """Test final output conv produces correct channels."""
        model = UNet3D(in_channels=1, out_channels=5, depth=2)

        x = torch.randn(1, 1, 16, 16, 16)
        output = model(x)

        assert output.shape[1] == 5

    def test_unet_various_output_channels(self):
        """Test U-Net with different output channel counts."""
        test_cases = [1, 2, 3, 5, 10]

        for out_ch in test_cases:
            model = UNet3D(in_channels=1, out_channels=out_ch, depth=2)
            x = torch.randn(1, 1, 16, 16, 16)
            output = model(x)

            assert output.shape[1] == out_ch

    def test_unet_decoder_upsampling_progression(self):
        """Test decoder progressively upsamples features."""
        model = UNet3D(init_features=16, depth=3)
        model.eval()

        x = torch.randn(1, 1, 32, 32, 32)

        # Trace through network manually
        skip_connections = []
        for encoder in model.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # After encoders: 32 -> 16 -> 8 -> 4
        assert x.shape[2:] == (4, 4, 4)

        x = model.bottleneck(x)

        # Decoder should upsample back
        for decoder, skip in zip(model.decoders, reversed(skip_connections), strict=True):
            x = decoder(x, skip)

        # Should be back to original size
        assert x.shape[2:] == (32, 32, 32)

    def test_unet_skip_connections_used(self):
        """Test skip connections are properly utilized."""
        model = UNet3D(init_features=16, depth=2)

        x = torch.randn(1, 1, 16, 16, 16)

        # Verify forward pass works (implicitly uses skip connections)
        output = model(x)

        # Output should be same spatial size as input
        assert output.shape[2:] == x.shape[2:]

    def test_unet_different_depths_complete(self):
        """Test complete U-Net with different depths."""
        for depth in [2, 3, 4]:
            model = UNet3D(depth=depth, init_features=16)
            x = torch.randn(1, 1, 32, 32, 32)

            output = model(x)

            assert output.shape[2:] == (32, 32, 32)

    def test_unet_large_input(self):
        """Test U-Net handles larger inputs."""
        model = UNet3D(init_features=16, depth=3)

        x = torch.randn(1, 1, 64, 64, 64)
        output = model(x)

        assert output.shape == (1, 1, 64, 64, 64)

    def test_unet_small_input(self):
        """Test U-Net handles smaller inputs."""
        model = UNet3D(init_features=16, depth=2)

        x = torch.randn(1, 1, 16, 16, 16)
        output = model(x)

        assert output.shape == (1, 1, 16, 16, 16)

    def test_unet_gradients_flow_through_decoder(self):
        """Test gradients flow through complete network."""
        model = UNet3D(depth=2, init_features=16)
        x = torch.randn(1, 1, 16, 16, 16, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist throughout network
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check decoder gradients
        for decoder in model.decoders:
            for param in decoder.parameters():
                if param.requires_grad:
                    assert param.grad is not None

    def test_unet_eval_mode_complete(self):
        """Test complete U-Net in evaluation mode."""
        model = UNet3D(depth=2)
        model.eval()

        x = torch.randn(2, 1, 16, 16, 16)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 1, 16, 16, 16)

    def test_unet_reproducibility_complete(self):
        """Test complete network produces consistent results."""
        model = UNet3D(depth=2, init_features=16)
        model.eval()

        x = torch.randn(1, 1, 16, 16, 16)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7)

    def test_unet_odd_sized_input(self):
        """Test U-Net handles odd-sized inputs with interpolation."""
        model = UNet3D(depth=2, init_features=16)

        # Odd sizes that don't divide evenly
        x = torch.randn(1, 1, 17, 17, 17)
        output = model(x)

        # Should still output same spatial size
        assert output.shape[2:] == (17, 17, 17)

    def test_unet_multi_channel_input_complete(self):
        """Test complete U-Net with multi-channel input."""
        model = UNet3D(in_channels=4, out_channels=3, depth=2, init_features=16)

        x = torch.randn(2, 4, 16, 16, 16)
        output = model(x)

        assert output.shape == (2, 3, 16, 16, 16)

    def test_unet_parameter_count_reasonable(self):
        """Test U-Net parameter count is reasonable."""
        model = UNet3D(init_features=32, depth=3)

        total_params = sum(p.numel() for p in model.parameters())

        # Should have millions of parameters but not too excessive
        assert 1_000_000 < total_params < 50_000_000

    def test_unet_no_nan_output(self):
        """Test U-Net doesn't produce NaN outputs."""
        model = UNet3D(depth=2, init_features=16)
        model.eval()

        x = torch.randn(1, 1, 16, 16, 16)

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
