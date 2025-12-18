"""Unit tests for TorchScript optimizer module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.inference.torchscript_optimizer import TorchScriptOptimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.squeeze()


class TestTorchScriptOptimizerInit:
    """Tests for TorchScriptOptimizer initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        optimizer = TorchScriptOptimizer()

        assert optimizer.method == "auto"
        assert optimizer.validate is True
        assert optimizer.tolerance == 1e-5
        assert optimizer.device == torch.device("cpu")

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        optimizer = TorchScriptOptimizer(
            method="script",
            validate=False,
            tolerance=1e-4,
            device="cpu",
        )

        assert optimizer.method == "script"
        assert optimizer.validate is False
        assert optimizer.tolerance == 1e-4

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Method must be"):
            TorchScriptOptimizer(method="invalid")  # type: ignore[arg-type]

    def test_negative_tolerance_raises_error(self) -> None:
        """Test that negative tolerance raises ValueError."""
        with pytest.raises(ValueError, match="Tolerance must be"):
            TorchScriptOptimizer(tolerance=-0.1)


class TestScriptCompilation:
    """Tests for torch.jit.script compilation."""

    def test_compile_with_script_method(self) -> None:
        """Test compilation using script method."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer(method="script", validate=False)

        compiled = optimizer.optimize(model)

        assert isinstance(compiled, torch.jit.ScriptModule)

    def test_script_compilation_preserves_functionality(self) -> None:
        """Test that scripted model produces same outputs."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer(method="script", validate=False)

        compiled = optimizer.optimize(model)

        # Test with example input
        example = torch.randn(1, 1, 16, 16, 16)
        with torch.no_grad():
            original_out = model(example)
            compiled_out = compiled(example)

        assert torch.allclose(original_out, compiled_out, atol=1e-5)


class TestTraceCompilation:
    """Tests for torch.jit.trace compilation."""

    def test_compile_with_trace_method(self) -> None:
        """Test compilation using trace method."""
        model = SimpleModel()
        example = torch.randn(1, 1, 16, 16, 16)
        optimizer = TorchScriptOptimizer(method="trace", validate=False)

        compiled = optimizer.optimize(model, example_inputs=[example])

        assert isinstance(compiled, torch.jit.ScriptModule)

    def test_trace_without_example_raises_error(self) -> None:
        """Test that trace without example inputs raises error."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer(method="trace")

        with pytest.raises(ValueError, match="requires example_inputs"):
            optimizer.optimize(model)

    def test_trace_compilation_preserves_functionality(self) -> None:
        """Test that traced model produces same outputs."""
        model = SimpleModel()
        example = torch.randn(1, 1, 16, 16, 16)
        optimizer = TorchScriptOptimizer(method="trace", validate=False)

        compiled = optimizer.optimize(model, example_inputs=[example])

        # Test with same input
        with torch.no_grad():
            original_out = model(example)
            compiled_out = compiled(example)

        assert torch.allclose(original_out, compiled_out, atol=1e-5)

    def test_trace_with_multiple_inputs(self) -> None:
        """Test tracing with multiple inputs."""

        class MultiInputModel(nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        model = MultiInputModel()
        example1 = torch.randn(4, 4)
        example2 = torch.randn(4, 4)
        optimizer = TorchScriptOptimizer(method="trace", validate=False)

        compiled = optimizer.optimize(model, example_inputs=[example1, example2])

        assert isinstance(compiled, torch.jit.ScriptModule)


class TestAutoCompilation:
    """Tests for auto compilation mode."""

    def test_auto_mode_tries_script_first(self) -> None:
        """Test that auto mode attempts script first."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer(method="auto", validate=False)

        compiled = optimizer.optimize(model)

        assert isinstance(compiled, torch.jit.ScriptModule)

    def test_auto_mode_falls_back_to_trace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that auto mode falls back to trace when script fails."""

        def failing_script(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Script failed")

        monkeypatch.setattr(torch.jit, "script", failing_script)

        model = SimpleModel()
        example = torch.randn(1, 1, 16, 16, 16)
        optimizer = TorchScriptOptimizer(method="auto", validate=False)

        compiled = optimizer.optimize(model, example_inputs=[example])

        assert isinstance(compiled, torch.jit.ScriptModule)

    def test_auto_mode_fails_without_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that auto mode fails if script fails and no example inputs."""

        def failing_script(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Script failed")

        monkeypatch.setattr(torch.jit, "script", failing_script)

        model = SimpleModel()
        optimizer = TorchScriptOptimizer(method="auto", validate=False)

        with pytest.raises(RuntimeError, match="Auto compilation failed"):
            optimizer.optimize(model)


class TestValidation:
    """Tests for output validation."""

    def test_validation_passes_for_matching_outputs(self) -> None:
        """Test that validation passes when outputs match."""
        model = SimpleModel()
        example = torch.randn(1, 1, 16, 16, 16)
        optimizer = TorchScriptOptimizer(method="script", validate=True, tolerance=1e-5)

        # Should not raise
        compiled = optimizer.optimize(model, example_inputs=[example])

        assert isinstance(compiled, torch.jit.ScriptModule)

    def test_validation_fails_for_different_outputs(self) -> None:
        """Test that validation fails when outputs differ."""

        class RandomModel(nn.Module):
            """Model that uses random operations which differ between runs."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Add random noise - this will be different each time
                return x.mean() + torch.randn(1)

        model = RandomModel()
        example = torch.randn(1, 1, 16, 16, 16)

        # Use script method so the random call is preserved
        optimizer = TorchScriptOptimizer(method="script", validate=True, tolerance=1e-5)

        # This should fail because random output differs between original and compiled
        with pytest.raises(RuntimeError, match="Validation failed"):
            optimizer.optimize(model, example_inputs=[example])

    def test_validation_skipped_without_example_inputs(self) -> None:
        """Test that validation is skipped when no example inputs provided."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer(method="script", validate=True)

        # Should not raise even though validation is enabled
        compiled = optimizer.optimize(model)

        assert isinstance(compiled, torch.jit.ScriptModule)


class TestSaveLoad:
    """Tests for saving and loading compiled models."""

    def test_save_compiled_model(self, tmp_path: Path) -> None:
        """Test saving a compiled model to file."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer(validate=False)
        example = torch.randn(1, 1, 16, 16, 16)

        compiled = optimizer.optimize(model, example_inputs=[example])
        filepath = tmp_path / "model.pt"
        optimizer.save(compiled, filepath)

        assert filepath.exists()

    def test_save_creates_directories(self, tmp_path: Path) -> None:
        """Test that save creates parent directories."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer(validate=False)
        example = torch.randn(1, 1, 16, 16, 16)

        compiled = optimizer.optimize(model, example_inputs=[example])
        filepath = tmp_path / "nested" / "dir" / "model.pt"
        optimizer.save(compiled, filepath)

        assert filepath.exists()

    def test_save_non_scriptmodule_raises_error(self, tmp_path: Path) -> None:
        """Test that saving non-ScriptModule raises TypeError."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer()
        filepath = tmp_path / "model.pt"

        with pytest.raises(TypeError, match="must be a torch.jit.ScriptModule"):
            optimizer.save(model, filepath)  # type: ignore[arg-type]

    def test_load_compiled_model(self, tmp_path: Path) -> None:
        """Test loading a compiled model from file."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer(method="script", validate=False)
        example = torch.randn(1, 1, 16, 16, 16)

        # Compile and save
        compiled = optimizer.optimize(model, example_inputs=[example])
        filepath = tmp_path / "model.pt"
        optimizer.save(compiled, filepath)

        # Load
        loaded = optimizer.load(filepath)

        assert isinstance(loaded, torch.jit.ScriptModule)

    def test_load_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        optimizer = TorchScriptOptimizer()
        filepath = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            optimizer.load(filepath)

    def test_save_load_preserves_functionality(self, tmp_path: Path) -> None:
        """Test that saved and loaded model works correctly."""
        model = SimpleModel()
        optimizer = TorchScriptOptimizer(method="script", validate=False)
        example = torch.randn(1, 1, 16, 16, 16)

        # Save
        compiled = optimizer.optimize(model, example_inputs=[example])
        filepath = tmp_path / "model.pt"
        optimizer.save(compiled, filepath)

        # Load
        loaded = optimizer.load(filepath)

        # Test
        with torch.no_grad():
            compiled_out = compiled(example)
            loaded_out = loaded(example)

        assert torch.allclose(compiled_out, loaded_out, atol=1e-6)


class TestOptimizeForInference:
    """Tests for inference optimization."""

    def test_optimize_for_inference_enabled(self) -> None:
        """Test that inference optimization is applied by default."""
        model = SimpleModel()
        example = torch.randn(1, 1, 16, 16, 16)
        optimizer = TorchScriptOptimizer(validate=False)

        compiled = optimizer.optimize(model, example_inputs=[example], optimize_for_inference=True)

        assert isinstance(compiled, torch.jit.ScriptModule)

    def test_optimize_for_inference_disabled(self) -> None:
        """Test disabling inference optimization."""
        model = SimpleModel()
        example = torch.randn(1, 1, 16, 16, 16)
        optimizer = TorchScriptOptimizer(validate=False)

        compiled = optimizer.optimize(model, example_inputs=[example], optimize_for_inference=False)

        assert isinstance(compiled, torch.jit.ScriptModule)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_compile_model_already_on_device(self) -> None:
        """Test compiling a model already on the target device."""
        model = SimpleModel().to("cpu")
        optimizer = TorchScriptOptimizer(method="script", device="cpu", validate=False)

        compiled = optimizer.optimize(model)

        assert isinstance(compiled, torch.jit.ScriptModule)

    def test_compile_with_very_small_input(self) -> None:
        """Test compilation with minimal input size."""
        model = SimpleModel()
        example = torch.randn(1, 1, 4, 4, 4)
        optimizer = TorchScriptOptimizer(method="trace", validate=False)

        compiled = optimizer.optimize(model, example_inputs=[example])

        assert isinstance(compiled, torch.jit.ScriptModule)

    def test_zero_tolerance_validation(self) -> None:
        """Test validation with very strict tolerance."""
        model = SimpleModel()
        example = torch.randn(1, 1, 16, 16, 16)
        # Use very small but non-zero tolerance due to floating point precision
        optimizer = TorchScriptOptimizer(method="script", validate=True, tolerance=1e-6)

        compiled = optimizer.optimize(model, example_inputs=[example])

        assert isinstance(compiled, torch.jit.ScriptModule)

    def test_compile_model_with_no_parameters(self) -> None:
        """Test compiling a model with no learnable parameters."""

        class ParameterlessModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        model = ParameterlessModel()
        optimizer = TorchScriptOptimizer(method="script", validate=False)

        compiled = optimizer.optimize(model)

        assert isinstance(compiled, torch.jit.ScriptModule)
