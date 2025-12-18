"""PyTorch Lightning module for multi-task nodule detection training.

This module implements the training, validation, and testing logic using
PyTorch Lightning's LightningModule interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data import LUNADataset
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
    - Training and validation data loading
    - Model checkpointing

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
        data_dir: Directory containing preprocessed data. Default: "data/processed"
        batch_size: Batch size for training and validation. Default: 2
        num_workers: Number of data loading workers. Default: 4
        pin_memory: Pin memory for faster GPU transfer. Default: True
        patch_size: Size of patches extracted from volumes (D, H, W). Default: (96, 96, 96)
        patches_per_volume: Number of patches to sample per volume per epoch. Default: 16
        positive_fraction: Fraction of patches centered on nodules. Default: 0.5
        cache_size: Number of volumes to keep in memory cache. Default: 4
        checkpoint_dir: Directory to save checkpoints. Default: "checkpoints"
        checkpoint_monitor: Metric to monitor for checkpointing. Default: "val/loss"
        checkpoint_mode: Mode for monitored metric ('min' or 'max'). Default: "min"
        checkpoint_save_top_k: Number of best models to save. Default: 3
        checkpoint_save_last: Save the last checkpoint. Default: True
        checkpoint_every_n_epochs: Save checkpoint every N epochs. Default: 1
        checkpoint_filename: Filename pattern for checkpoints. Default: "epoch={epoch:02d}-val_loss={val/loss:.4f}"
        early_stopping_monitor: Metric to monitor for early stopping. Default: "val/loss"
        early_stopping_patience: Number of epochs with no improvement before stopping. Default: 10
        early_stopping_mode: Mode for monitored metric ('min' or 'max'). Default: "min"
        early_stopping_min_delta: Minimum change to qualify as improvement. Default: 0.0
        wandb_project: Weights & Biases project name. Default: "lung-nodule-detection"
        wandb_name: Weights & Biases run name. Default: None (auto-generated)
        wandb_log_model: Log model checkpoints to W&B. Default: False
        wandb_offline: Run W&B in offline mode. Default: False
        seg_loss_kwargs: Keyword arguments for DiceLoss. Default: None
        center_loss_kwargs: Keyword arguments for FocalLoss. Default: None
        size_loss_kwargs: Keyword arguments for SmoothL1Loss. Default: None
        triage_loss_kwargs: Keyword arguments for WeightedBCELoss. Default: None

    Example:
        >>> # Basic usage with automatic data loading
        >>> model = LitNoduleDetection(data_dir="data/processed")
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(model)

        >>> # Custom data loading configuration
        >>> model = LitNoduleDetection(
        ...     data_dir="data/processed",
        ...     batch_size=4,
        ...     num_workers=8,
        ...     patch_size=(128, 128, 128),
        ...     patches_per_volume=32,
        ...     positive_fraction=0.7
        ... )
        >>> trainer.fit(model)

        >>> # Mixed precision FP16 (NVIDIA GPUs)
        >>> model = LitNoduleDetection(precision='16-mixed')
        >>> trainer = L.Trainer(max_epochs=100, precision='16-mixed')
        >>> trainer.fit(model)

        >>> # Mixed precision BF16 (AMD/Intel GPUs)
        >>> model = LitNoduleDetection(precision='bf16-mixed')
        >>> trainer = L.Trainer(max_epochs=100, precision='bf16-mixed')
        >>> trainer.fit(model)

        >>> # With hard negative mining for center detection
        >>> model = LitNoduleDetection(
        ...     use_hard_negative_mining=True,
        ...     hard_negative_ratio=3.0,
        ...     min_negative_samples=100
        ... )
        >>> trainer.fit(model)

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
        data_dir: str | Path = "data/processed",
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        patches_per_volume: int = 16,
        positive_fraction: float = 0.5,
        cache_size: int = 4,
        checkpoint_dir: str = "checkpoints",
        checkpoint_monitor: str = "val/loss",
        checkpoint_mode: Literal["min", "max"] = "min",
        checkpoint_save_top_k: int = 3,
        checkpoint_save_last: bool = True,
        checkpoint_every_n_epochs: int = 1,
        checkpoint_filename: str = "epoch={epoch:02d}-val_loss={val/loss:.4f}",
        early_stopping_monitor: str = "val/loss",
        early_stopping_patience: int = 10,
        early_stopping_mode: Literal["min", "max"] = "min",
        early_stopping_min_delta: float = 0.0,
        wandb_project: str = "lung-nodule-detection",
        wandb_name: str | None = None,
        wandb_log_model: bool = False,
        wandb_offline: bool = False,
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

        # Validate data loading parameters
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        if patches_per_volume <= 0:
            raise ValueError(f"patches_per_volume must be positive, got {patches_per_volume}")
        if not 0 < positive_fraction <= 1:
            raise ValueError(f"positive_fraction must be in (0, 1], got {positive_fraction}")
        if cache_size < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size}")
        if len(patch_size) != 3 or any(s <= 0 for s in patch_size):
            raise ValueError(f"patch_size must be 3 positive integers, got {patch_size}")

        # Validate checkpoint parameters
        valid_modes = {"min", "max"}
        if checkpoint_mode not in valid_modes:
            raise ValueError(
                f"Invalid checkpoint_mode '{checkpoint_mode}'. Must be one of {valid_modes}"
            )
        if checkpoint_save_top_k < 1:
            raise ValueError(f"checkpoint_save_top_k must be >= 1, got {checkpoint_save_top_k}")
        if checkpoint_every_n_epochs < 1:
            raise ValueError(
                f"checkpoint_every_n_epochs must be >= 1, got {checkpoint_every_n_epochs}"
            )

        # Validate early stopping parameters
        if early_stopping_mode not in valid_modes:
            raise ValueError(
                f"Invalid early_stopping_mode '{early_stopping_mode}'. Must be one of {valid_modes}"
            )
        if early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be >= 1, got {early_stopping_patience}")
        if early_stopping_min_delta < 0:
            raise ValueError(
                f"early_stopping_min_delta must be >= 0, got {early_stopping_min_delta}"
            )

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

        # Store checkpoint parameters
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        self.checkpoint_save_top_k = checkpoint_save_top_k
        self.checkpoint_save_last = checkpoint_save_last
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.checkpoint_filename = checkpoint_filename

        # Store early stopping parameters
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_mode = early_stopping_mode
        self.early_stopping_min_delta = early_stopping_min_delta

        # Store W&B parameters
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.wandb_log_model = wandb_log_model
        self.wandb_offline = wandb_offline

        # Store data loading parameters
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.positive_fraction = positive_fraction
        self.cache_size = cache_size

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

    def configure_callbacks(self) -> list[L.Callback]:
        """Configure Lightning callbacks for training.

        Returns:
            List of configured callbacks including ModelCheckpoint and EarlyStopping
        """
        callbacks: list[L.Callback] = []

        # Add model checkpointing callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=self.checkpoint_filename,
            monitor=self.checkpoint_monitor,
            mode=self.checkpoint_mode,
            save_top_k=self.checkpoint_save_top_k,
            save_last=self.checkpoint_save_last,
            every_n_epochs=self.checkpoint_every_n_epochs,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        # Add early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor=self.early_stopping_monitor,
            patience=self.early_stopping_patience,
            mode=self.early_stopping_mode,
            min_delta=self.early_stopping_min_delta,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)

        return callbacks

    def configure_loggers(self) -> WandbLogger | bool:
        """Configure Weights & Biases logger for experiment tracking.

        Returns:
            WandbLogger instance for experiment tracking, or False to disable logging
        """
        # Create WandbLogger for experiment tracking
        logger = WandbLogger(
            project=self.wandb_project,
            name=self.wandb_name,
            log_model=self.wandb_log_model,
            offline=self.wandb_offline,
        )

        return logger

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

    def train_dataloader(self) -> DataLoader:
        """Create training data loader.

        Returns:
            DataLoader for training set with shuffle=True and drop_last=True
        """
        dataset = LUNADataset(
            data_dir=self.data_dir,
            split="train",
            patch_size=self.patch_size,
            patches_per_volume=self.patches_per_volume,
            positive_fraction=self.positive_fraction,
            cache_size=self.cache_size,
            seed=1337,  # Fixed seed for reproducibility
            include_mask=True,
            include_heatmap=True,
            transform=None,  # Augmentation handled separately if needed
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # Drop last incomplete batch
            persistent_workers=self.num_workers > 0,  # Keep workers alive between epochs
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader.

        Returns:
            DataLoader for validation set with shuffle=False and drop_last=False
        """
        dataset = LUNADataset(
            data_dir=self.data_dir,
            split="val",
            patch_size=self.patch_size,
            patches_per_volume=self.patches_per_volume,
            positive_fraction=self.positive_fraction,
            cache_size=self.cache_size,
            seed=42,  # Different seed from training
            include_mask=True,
            include_heatmap=True,
            transform=None,  # No augmentation for validation
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # Keep all validation samples
            persistent_workers=self.num_workers > 0,  # Keep workers alive between epochs
        )
