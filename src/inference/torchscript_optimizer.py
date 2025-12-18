"""TorchScript optimization for faster inference.

This module provides utilities to compile PyTorch models to TorchScript format
for improved inference performance. TorchScript models can run independently
of Python, benefit from optimizations, and have lower overhead.

Two compilation methods are supported:
1. torch.jit.script: Compiles the model directly from Python source
2. torch.jit.trace: Records operations during a forward pass with example inputs

The module includes automatic fallback mechanisms and validation to ensure
compiled models produce equivalent outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

__all__ = [
    "CompilationMethod",
    "TorchScriptOptimizer",
]

CompilationMethod = Literal["script", "trace", "auto"]

logger = logging.getLogger(__name__)


class TorchScriptOptimizer:
    """Optimize PyTorch models using TorchScript compilation.

    This class provides a unified interface for compiling models to TorchScript,
    with automatic method selection, validation, and fallback mechanisms.

    Args:
        method: Compilation method ("script", "trace", or "auto")
        validate: Whether to validate compiled model outputs match original
        tolerance: Maximum allowed difference for validation (L-infinity norm)
        device: Device for compilation ("cpu", "cuda", or torch.device)

    Attributes:
        method: The compilation method being used
        validate: Whether validation is enabled
        tolerance: Validation tolerance threshold
        device: Target device for compilation

    Examples:
        >>> # Script compilation
        >>> optimizer = TorchScriptOptimizer(method="script")
        >>> scripted_model = optimizer.optimize(model)
        >>>
        >>> # Trace compilation with validation
        >>> optimizer = TorchScriptOptimizer(method="trace", validate=True)
        >>> example = torch.randn(1, 1, 64, 64, 64)
        >>> traced_model = optimizer.optimize(model, example_inputs=[example])
        >>>
        >>> # Auto mode (tries script, falls back to trace)
        >>> optimizer = TorchScriptOptimizer(method="auto")
        >>> optimized = optimizer.optimize(model, example_inputs=[example])
    """

    def __init__(
        self,
        method: CompilationMethod = "auto",
        validate: bool = True,
        tolerance: float = 1e-5,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize TorchScript optimizer.

        Args:
            method: Compilation method ("script", "trace", or "auto")
            validate: Whether to validate compiled outputs
            tolerance: Maximum allowed output difference for validation
            device: Target device (None uses model's current device)

        Raises:
            ValueError: If method is not recognized
            ValueError: If tolerance is negative
        """
        if method not in {"script", "trace", "auto"}:
            msg = f"Method must be 'script', 'trace', or 'auto', got '{method}'"
            raise ValueError(msg)

        if tolerance < 0:
            msg = f"Tolerance must be non-negative, got {tolerance}"
            raise ValueError(msg)

        self.method = method
        self.validate = validate
        self.tolerance = tolerance
        self.device = device if device is not None else torch.device("cpu")

    def optimize(
        self,
        model: nn.Module,
        example_inputs: list[torch.Tensor] | None = None,
        *,
        optimize_for_inference: bool = True,
    ) -> torch.jit.ScriptModule:
        """Compile model to TorchScript.

        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing (required for trace/auto methods)
            optimize_for_inference: Whether to apply inference optimizations

        Returns:
            Compiled TorchScript model

        Raises:
            ValueError: If trace method requires example_inputs but none provided
            RuntimeError: If compilation fails and no fallback available
            RuntimeError: If validation fails (when enabled)
        """
        # Move model to target device
        model = model.to(self.device)
        model.eval()

        # Compile based on method
        if self.method == "script":
            compiled = self._compile_script(model)
        elif self.method == "trace":
            if example_inputs is None:
                msg = "Trace method requires example_inputs"
                raise ValueError(msg)
            compiled = self._compile_trace(model, example_inputs)
        else:  # auto
            compiled = self._compile_auto(model, example_inputs)

        # Apply freeze optimization if requested (freeze is safe for serialization)
        if optimize_for_inference:
            try:
                logger.info("Freezing model (converts parameters to constants)")
                compiled = torch.jit.freeze(compiled)
                logger.info("Successfully froze model")
            except Exception as e:
                logger.warning(f"Failed to freeze model: {e}")

        # Validate if requested
        if self.validate and example_inputs is not None:
            self._validate_outputs(model, compiled, example_inputs)

        return compiled

    def save(
        self,
        model: torch.jit.ScriptModule,
        filepath: str | Path,
        *,
        compress: bool = False,
    ) -> None:
        """Save compiled model to file.

        Args:
            model: Compiled TorchScript model
            filepath: Output file path
            compress: Whether to compress saved model

        Raises:
            TypeError: If model is not a ScriptModule
        """
        if not isinstance(model, torch.jit.ScriptModule):
            msg = "Model must be a torch.jit.ScriptModule"
            raise TypeError(msg)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save with optional compression
        extra_files: dict[str, Any] = {}
        if compress:
            torch.jit.save(model, str(filepath), _extra_files=extra_files)
        else:
            torch.jit.save(model, str(filepath))

        logger.info(f"Saved TorchScript model to {filepath}")

    def load(self, filepath: str | Path) -> torch.jit.ScriptModule:
        """Load compiled model from file.

        Args:
            filepath: Path to saved TorchScript model

        Returns:
            Loaded TorchScript model

        Raises:
            FileNotFoundError: If filepath does not exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"Model file not found: {filepath}"
            raise FileNotFoundError(msg)

        model = torch.jit.load(str(filepath), map_location=self.device)  # type: ignore[no-any-return]
        logger.info(f"Loaded TorchScript model from {filepath}")

        # Apply optimize_for_inference after loading (not serializable, so apply at runtime)
        try:
            logger.info("Applying optimize_for_inference (runtime optimization)")
            optimized = torch.jit.optimize_for_inference(model)
            assert isinstance(optimized, torch.jit.ScriptModule)
            model = optimized
            logger.info("Successfully optimized model for inference")
        except Exception as e:
            logger.warning(f"Failed to apply optimize_for_inference: {e}")

        return model  # type: ignore[no-any-return]

    def _compile_script(self, model: nn.Module) -> torch.jit.ScriptModule:
        """Compile using torch.jit.script.

        Args:
            model: Model to compile

        Returns:
            Scripted model

        Raises:
            RuntimeError: If scripting fails
        """
        try:
            logger.info("Compiling model using torch.jit.script")
            compiled = torch.jit.script(model)
            logger.info("Successfully compiled with torch.jit.script")
            return compiled  # type: ignore[no-any-return]
        except Exception as e:
            msg = f"Failed to compile with torch.jit.script: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _compile_trace(
        self,
        model: nn.Module,
        example_inputs: list[torch.Tensor],
    ) -> torch.jit.ScriptModule:
        """Compile using torch.jit.trace.

        Args:
            model: Model to compile
            example_inputs: Example inputs for tracing

        Returns:
            Traced model

        Raises:
            RuntimeError: If tracing fails
        """
        try:
            logger.info("Compiling model using torch.jit.trace")
            # Move example inputs to device
            example_inputs = [x.to(self.device) for x in example_inputs]

            # Trace the model
            with torch.no_grad():
                if len(example_inputs) == 1:
                    compiled = torch.jit.trace(model, example_inputs[0])
                else:
                    compiled = torch.jit.trace(model, example_inputs)

            logger.info("Successfully compiled with torch.jit.trace")
            return compiled  # type: ignore[no-any-return]
        except Exception as e:
            msg = f"Failed to compile with torch.jit.trace: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _compile_auto(
        self,
        model: nn.Module,
        example_inputs: list[torch.Tensor] | None,
    ) -> torch.jit.ScriptModule:
        """Compile using auto method (try script, fallback to trace).

        Args:
            model: Model to compile
            example_inputs: Example inputs for tracing fallback

        Returns:
            Compiled model

        Raises:
            RuntimeError: If both methods fail
        """
        # Try script first
        try:
            logger.info("Auto mode: attempting torch.jit.script")
            return self._compile_script(model)
        except Exception as e:
            logger.warning(f"Script compilation failed: {e}")

            # Fall back to trace if example inputs provided
            if example_inputs is not None:
                logger.info("Auto mode: falling back to torch.jit.trace")
                return self._compile_trace(model, example_inputs)

            # No fallback available
            msg = "Auto compilation failed: script failed and no example_inputs for trace"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _validate_outputs(
        self,
        original: nn.Module,
        compiled: torch.jit.ScriptModule,
        example_inputs: list[torch.Tensor],
    ) -> None:
        """Validate that compiled model produces same outputs as original.

        Args:
            original: Original PyTorch model
            compiled: Compiled TorchScript model
            example_inputs: Inputs to test with

        Raises:
            RuntimeError: If outputs differ by more than tolerance
        """
        logger.info("Validating compiled model outputs")

        # Move inputs to device
        example_inputs = [x.to(self.device) for x in example_inputs]

        # Run both models
        with torch.no_grad():
            if len(example_inputs) == 1:
                original_out = original(example_inputs[0])
                compiled_out = compiled(example_inputs[0])
            else:
                original_out = original(*example_inputs)
                compiled_out = compiled(*example_inputs)

        # Convert to tensors if needed
        if not isinstance(original_out, torch.Tensor):
            original_out = torch.stack(list(original_out)) if isinstance(original_out, (list, tuple)) else original_out
        if not isinstance(compiled_out, torch.Tensor):
            compiled_out = torch.stack(list(compiled_out)) if isinstance(compiled_out, (list, tuple)) else compiled_out

        # Compute maximum difference
        max_diff = torch.max(torch.abs(original_out - compiled_out)).item()

        logger.info(f"Maximum output difference: {max_diff:.2e} (tolerance: {self.tolerance:.2e})")

        if max_diff > self.tolerance:
            msg = f"Validation failed: max difference {max_diff:.2e} exceeds tolerance {self.tolerance:.2e}"
            raise RuntimeError(msg)

        logger.info("Validation passed: outputs match within tolerance")
