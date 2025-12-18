"""Checkpoint management endpoints for model checkpoints.

This module provides REST API endpoints for managing model checkpoints,
including upload, download, listing, and deletion.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..database import CheckpointMetadata, get_db_session

router = APIRouter(prefix="/checkpoints", tags=["checkpoints"])

# Checkpoint storage directory
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


class CheckpointInfo(BaseModel):
    """Checkpoint information response."""

    id: int
    session_id: str
    filename: str
    epoch: int | None = None
    metric_value: float | None = None
    metric_name: str | None = None
    created_at: datetime
    is_best: bool = False
    size_bytes: int = 0

    class Config:
        from_attributes = True


class CheckpointUploadResponse(BaseModel):
    """Response after uploading a checkpoint."""

    id: int
    filename: str
    size_bytes: int
    message: str = "Checkpoint uploaded successfully"


class CheckpointList(BaseModel):
    """List of checkpoints."""

    checkpoints: list[CheckpointInfo]
    total: int


@router.post("/", response_model=CheckpointUploadResponse, status_code=201)
async def upload_checkpoint(
    file: Annotated[UploadFile, File(description="Checkpoint file (.pth or .pt)")],
    session_id: Annotated[str, Form(description="Training session ID")],
    epoch: Annotated[int | None, Form(description="Epoch number")] = None,
    metric_value: Annotated[float | None, Form(description="Metric value (e.g., validation loss)")] = None,
    metric_name: Annotated[str | None, Form(description="Metric name (e.g., 'val_loss')")] = None,
    is_best: Annotated[bool, Form(description="Whether this is the best checkpoint")] = False,
) -> CheckpointUploadResponse:
    """Upload a model checkpoint.

    Args:
        file: Checkpoint file to upload
        session_id: Training session identifier
        epoch: Optional epoch number
        metric_value: Optional metric value
        metric_name: Optional metric name
        is_best: Whether this is the best checkpoint

    Returns:
        Upload confirmation with checkpoint info

    Raises:
        HTTPException: If file type is invalid or upload fails
    """
    # Validate file extension
    if not file.filename or not (file.filename.endswith('.pth') or file.filename.endswith('.pt')):
        raise HTTPException(status_code=400, detail="File must be a .pth or .pt checkpoint file")

    # Create session directory
    session_dir = CHECKPOINT_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    filepath = session_dir / file.filename
    try:
        with filepath.open('wb') as f:
            content = await file.read()
            f.write(content)
        size_bytes = len(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save checkpoint: {e!s}")

    # Store metadata in database
    db = get_db_session()
    try:
        checkpoint = CheckpointMetadata(
            session_id=session_id,
            filename=file.filename,
            filepath=str(filepath),
            epoch=epoch,
            metric_value=metric_value,
            metric_name=metric_name,
            is_best=is_best,
        )
        db.add(checkpoint)
        db.commit()
        db.refresh(checkpoint)

        return CheckpointUploadResponse(
            id=checkpoint.id,
            filename=file.filename,
            size_bytes=size_bytes,
        )
    finally:
        db.close()


@router.get("/", response_model=CheckpointList)
async def list_checkpoints(
    session_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> CheckpointList:
    """List all checkpoints.

    Args:
        session_id: Optional filter by session ID
        limit: Maximum number of checkpoints to return
        offset: Number of checkpoints to skip

    Returns:
        List of checkpoint information
    """
    db = get_db_session()
    try:
        query = db.query(CheckpointMetadata)

        if session_id:
            query = query.filter(CheckpointMetadata.session_id == session_id)

        total = query.count()
        checkpoints = query.order_by(CheckpointMetadata.created_at.desc()).limit(limit).offset(offset).all()

        # Get file sizes
        checkpoint_infos = []
        for cp in checkpoints:
            filepath = Path(cp.filepath)
            size_bytes = filepath.stat().st_size if filepath.exists() else 0

            checkpoint_infos.append(CheckpointInfo(
                id=cp.id,
                session_id=cp.session_id,
                filename=cp.filename,
                epoch=cp.epoch,
                metric_value=cp.metric_value,
                metric_name=cp.metric_name,
                created_at=cp.created_at,
                is_best=cp.is_best,
                size_bytes=size_bytes,
            ))

        return CheckpointList(checkpoints=checkpoint_infos, total=total)
    finally:
        db.close()


@router.get("/{checkpoint_id}")
async def download_checkpoint(checkpoint_id: int) -> FileResponse:
    """Download a checkpoint file.

    Args:
        checkpoint_id: Checkpoint ID

    Returns:
        Checkpoint file

    Raises:
        HTTPException: If checkpoint not found or file missing
    """
    db = get_db_session()
    try:
        checkpoint = db.query(CheckpointMetadata).filter(CheckpointMetadata.id == checkpoint_id).first()

        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        filepath = Path(checkpoint.filepath)
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Checkpoint file not found on disk")

        return FileResponse(
            path=filepath,
            filename=checkpoint.filename,
            media_type='application/octet-stream',
        )
    finally:
        db.close()


@router.delete("/{checkpoint_id}", status_code=204)
async def delete_checkpoint(checkpoint_id: int) -> None:
    """Delete a checkpoint.

    Args:
        checkpoint_id: Checkpoint ID

    Raises:
        HTTPException: If checkpoint not found
    """
    db = get_db_session()
    try:
        checkpoint = db.query(CheckpointMetadata).filter(CheckpointMetadata.id == checkpoint_id).first()

        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        # Delete file
        filepath = Path(checkpoint.filepath)
        if filepath.exists():
            filepath.unlink()

        # Delete metadata
        db.delete(checkpoint)
        db.commit()
    finally:
        db.close()
