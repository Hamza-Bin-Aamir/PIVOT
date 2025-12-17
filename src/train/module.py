"""PyTorch Lightning module for multi-task nodule detection training.

This module implements the training, validation, and testing logic using
PyTorch Lightning's LightningModule interface.
"""

from __future__ import annotations

from typing import Any, Literal

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.loss import HardNegativeMiningLoss, MultiTaskLoss
from src.model import MultiTaskUNet3D


class LitNoduleDetection(L.LightningModule):
    """PyTorch Lightning module for multi-task lung nodule detection.

    This module integrates the MultiTaskUNet3D model with MultiTaskLoss for
    training, validation, and testing. It handles:
    - Forward pass through the model
    - Loss computation for all tasks
    - Metric logging
    - Optimizer and scheduler configuration
    - Mixed precision training (FP16, BF16, or FP32)
    - Hard negative mining for center detection

    Args:
        model_depth: Depth of the U-Net encoder/decoder. Default: 4
        init_features: Number of features in first encoder block. Default: 32
        seg_weight: Weight for segmentation loss. Default: 1.0
        center_weight: Weight for center detection loss. Default: 1.0
        size_weight: Weight for size regression loss. Default: 1.0
        triage_weight: Weight for malignancy triage loss. Default: 1.0
        learning_rate: Initial learning rate for AdamW. Default: 1e-4
        weight_decay: Weight decay for AdamW. Default: 1e-5
        max_epochs: Maximum number of training epochs for scheduler. Default: 100
        precision: Training precision mode. Options: '32', '16-mixed', 'bf16-mixed'.
            - '32': Full FP32 precision (default, most stable)
            - '16-mixed': Mixed precision FP16 (faster, NVIDIA GPUs)
            - 'bf16-mixed': Mixed precision BF16 (faster, AMD/Intel GPUs, more stable than FP16)
            Default: '32'
        use_hard_negative_mining: Enable hard negative mining for center detection.
            Helps address extreme class imbalance. Default: False
        hard_negative_ratio: Ratio of hard negatives to positives when mining is enabled.
            Only used if use_hard_negative_mining=True. Default: 3.0
        min_negative_samples: Minimum hard negatives to mine regardless of positive count.
            Only used if use_hard_negative_mining=True. Default: 100
        seg_loss_kwargs: Keyword arguments for DiceLoss. Default: None
        center_loss_kwargs: Keyword arguments for FocalLoss. Default: None
        size_loss_kwargs: Keyword arguments for SmoothL1Loss. Default: None
        triage_loss_kwargs: Keyword arguments for WeightedBCELoss. Default: None

    Example:
        >>> # Basic usage with FP32
        >>> model = LitNoduleDetection(model_depth=4, init_features=32)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(model, train_dataloader, val_dataloader)

        >>> # Mixed precision FP16 (NVIDIA GPUs)
        >>> model = LitNoduleDetection(precision='16-mixed')
        >>> trainer = L.Trainer(max_epochs=100, precision='16-mixed')
        >>> trainer.fit(model, train_dataloader, val_dataloader)

        >>> # Mixed precision BF16 (AMD/Intel GPUs)
        >>> model = LitNoduleDetection(precision='bf16-mixed')
        >>> trainer = L.Trainer(max_epochs=100, precision='bf16-mixed')
        >>> trainer.fit(model, train_dataloader, val_dataloader)

        >>> # With hard negative mining for center detection
        >>> model = LitNoduleDetection(
        ...     use_hard_negative_mining=True,
        ...     hard_negative_ratio=3.0,
        ...     min_negative_samples=100
        ... )
        >>> trainer.fit(model, train_dataloader, val_dataloader)

        >>> # Custom task weights
        >>> model = LitNoduleDetection(
        ...     seg_weight=1.0,
        ...     center_weight=2.0,
        ...     size_weight=0.5,
        ...     triage_weight=3.0
        ... )

        >>> # Custom loss configurations
        >>> model = LitNoduleDetection(
        ...     seg_loss_kwargs={'smooth': 1.0},
        ...     center_loss_kwargs={'alpha': 0.25, 'gamma': 2.0},
        ...     size_loss_kwargs={'beta': 1.0},
        ...     triage_loss_kwargs={'pos_weight': 3.0}
        ... )
    """

    def __init__(
        self,
        model_depth: int = 4,
        init_features: int = 32,
        seg_weight: float = 1.0,
        center_weight: float = 1.0,
        size_weight: float = 1.0,
        triage_weight: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        precision: Literal["32", "16-mixed", "bf16-mixed"] = "32",
        use_hard_negative_mining: bool = False,
        hard_negative_ratio: float = 3.0,
        min_negative_samples: int = 100,
        seg_loss_kwargs: dict[str, Any] | None = None,
        center_loss_kwargs: dict[str, Any] | None = None,
        size_loss_kwargs: dict[str, Any] | None = None,
        triage_loss_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # Validate precision parameter
        valid_precisions = {"32", "16-mixed", "bf16-mixed"}
        if precision not in valid_precisions:
            raise ValueError(f"Invalid precision '{precision}'. Must be one of {valid_precisions}")

        # Validate hard negative mining parameters
        if use_hard_negative_mining:
            if hard_negative_ratio <= 0:
                raise ValueError(f"hard_negative_ratio must be positive, got {hard_negative_ratio}")
            if min_negative_samples < 0:
                raise ValueError(f"min_negative_samples must be >= 0, got {min_negative_samples}")

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Initialize model
        self.model = MultiTaskUNet3D(
            depth=model_depth,
            init_features=init_features,
        )

        # Initialize loss function
        self.loss_fn = MultiTaskLoss(
            seg_weight=seg_weight,
            center_weight=center_weight,
            size_weight=size_weight,
            triage_weight=triage_weight,
            seg_loss_kwargs=seg_loss_kwargs,
            center_loss_kwargs=center_loss_kwargs,
            size_loss_kwargs=size_loss_kwargs,
            triage_loss_kwargs=triage_loss_kwargs,
            reduction="mean",
        )

        # Wrap center loss with hard negative mining if enabled
        if use_hard_negative_mining:
            # Get the base center loss from MultiTaskLoss
            base_center_loss = self.loss_fn.center_loss

            # Wrap it with hard negative mining
            self.loss_fn.center_loss = HardNegativeMiningLoss(  # type: ignore[assignment]
                base_loss=base_center_loss,
                hard_negative_ratio=hard_negative_ratio,
                min_negative_samples=min_negative_samples,
                reduction="mean",
            )

        # Store optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.precision = precision
        self.use_hard_negative_mining = use_hard_negative_mining

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input CT volume of shape (B, 1, D, H, W)

        Returns:
            Dictionary with predictions for all tasks:
                - 'segmentation': (B, 1, D, H, W)
                - 'center': (B, 1, D, H, W)
                - 'size': (B, 3, 1, 1, 1)
                - 'triage': (B, 1, 1, 1, 1)
        """
        return self.model(x)  # type: ignore[no-any-return]

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dictionary containing:
                - 'image': Input CT volume (B, 1, D, H, W)
                - 'segmentation': Ground truth segmentation (B, 1, D, H, W)
                - 'center': Ground truth center heatmap (B, 1, D, H, W)
                - 'size': Ground truth size (B, 3, 1, 1, 1)
                - 'triage': Ground truth malignancy score (B, 1, 1, 1, 1)
            batch_idx: Index of the batch

        Returns:
            Total training loss
        """
        # Get predictions
        predictions = self(batch["image"])

        # Prepare targets
        targets = {
            "segmentation": batch["segmentation"],
            "center": batch["center"],
            "size": batch["size"],
            "triage": batch["triage"],
        }

        # Compute loss with individual task breakdown
        self.loss_fn.reduction = "none"
        losses = self.loss_fn(predictions, targets)
        self.loss_fn.reduction = "mean"

        # Log individual task losses
        self.log("train/loss", losses["total"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/loss_seg",
            losses["segmentation_unweighted"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/loss_center",
            losses["center_unweighted"],
            on_step=False,
            on_epoch=True,
        )
        self.log("train/loss_size", losses["size_unweighted"], on_step=False, on_epoch=True)
        self.log(
            "train/loss_triage",
            losses["triage_unweighted"],
            on_step=False,
            on_epoch=True,
        )

        return losses["total"]  # type: ignore[no-any-return]

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Dictionary containing:
                - 'image': Input CT volume (B, 1, D, H, W)
                - 'segmentation': Ground truth segmentation (B, 1, D, H, W)
                - 'center': Ground truth center heatmap (B, 1, D, H, W)
                - 'size': Ground truth size (B, 3, 1, 1, 1)
                - 'triage': Ground truth malignancy score (B, 1, 1, 1, 1)
            batch_idx: Index of the batch

        Returns:
            Total validation loss
        """
        # Get predictions
        predictions = self(batch["image"])

        # Prepare targets
        targets = {
            "segmentation": batch["segmentation"],
            "center": batch["center"],
            "size": batch["size"],
            "triage": batch["triage"],
        }

        # Compute loss with individual task breakdown
        self.loss_fn.reduction = "none"
        losses = self.loss_fn(predictions, targets)
        self.loss_fn.reduction = "mean"

        # Log individual task losses
        self.log("val/loss", losses["total"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_seg", losses["segmentation_unweighted"], on_step=False, on_epoch=True)
        self.log("val/loss_center", losses["center_unweighted"], on_step=False, on_epoch=True)
        self.log("val/loss_size", losses["size_unweighted"], on_step=False, on_epoch=True)
        self.log("val/loss_triage", losses["triage_unweighted"], on_step=False, on_epoch=True)

        return losses["total"]  # type: ignore[no-any-return]

    def test_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch: Dictionary containing:
                - 'image': Input CT volume (B, 1, D, H, W)
                - 'segmentation': Ground truth segmentation (B, 1, D, H, W)
                - 'center': Ground truth center heatmap (B, 1, D, H, W)
                - 'size': Ground truth size (B, 3, 1, 1, 1)
                - 'triage': Ground truth malignancy score (B, 1, 1, 1, 1)
            batch_idx: Index of the batch

        Returns:
            Total test loss
        """
        # Get predictions
        predictions = self(batch["image"])

        # Prepare targets
        targets = {
            "segmentation": batch["segmentation"],
            "center": batch["center"],
            "size": batch["size"],
            "triage": batch["triage"],
        }

        # Compute loss with individual task breakdown
        self.loss_fn.reduction = "none"
        losses = self.loss_fn(predictions, targets)
        self.loss_fn.reduction = "mean"

        # Log individual task losses
        self.log("test/loss", losses["total"], on_step=False, on_epoch=True)
        self.log("test/loss_seg", losses["segmentation_unweighted"], on_step=False, on_epoch=True)
        self.log("test/loss_center", losses["center_unweighted"], on_step=False, on_epoch=True)
        self.log("test/loss_size", losses["size_unweighted"], on_step=False, on_epoch=True)
        self.log("test/loss_triage", losses["triage_unweighted"], on_step=False, on_epoch=True)

        return losses["total"]  # type: ignore[no-any-return]

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # AdamW optimizer with weight decay
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.learning_rate * 0.01,  # Minimum LR is 1% of initial
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        """Hook called at the end of each training epoch."""
        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=False, on_epoch=True)
