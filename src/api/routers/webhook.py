"""Webhook endpoints for managing webhook notifications.

This module provides REST API endpoints for configuring webhooks
and viewing delivery status.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..webhooks import WebhookConfig, webhook_manager

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class WebhookConfigModel(BaseModel):
    """Webhook configuration model."""

    url: str
    enabled: bool = True
    secret: str | None = None
    timeout_seconds: int = 10
    max_retries: int = 3
    retry_delay_seconds: int = 60


class WebhookInfo(BaseModel):
    """Webhook information response."""

    name: str
    url: str
    enabled: bool
    timeout_seconds: int
    max_retries: int
    retry_delay_seconds: int


class WebhookListResponse(BaseModel):
    """List of webhooks response."""

    webhooks: list[WebhookInfo]
    total: int


class WebhookDeliveryInfo(BaseModel):
    """Webhook delivery information."""

    id: str
    webhook_url: str
    status: str
    attempts: int
    created_at: str
    last_attempt_at: str | None
    delivered_at: str | None
    error_message: str | None


class DeliveryListResponse(BaseModel):
    """List of deliveries response."""

    deliveries: list[WebhookDeliveryInfo]
    total: int


class SendWebhookRequest(BaseModel):
    """Request to send a webhook."""

    webhook_name: str
    payload: dict[str, Any]


class SendWebhookResponse(BaseModel):
    """Response after sending webhook."""

    delivery_id: str
    status: str
    message: str = "Webhook sent"


@router.post("/", status_code=201)
async def create_webhook(
    name: str,
    config: WebhookConfigModel,
) -> dict[str, str]:
    """Create or update a webhook configuration.

    Args:
        name: Webhook name/identifier
        config: Webhook configuration

    Returns:
        Confirmation message
    """
    webhook_config = WebhookConfig(
        url=config.url,
        enabled=config.enabled,
        secret=config.secret,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        retry_delay_seconds=config.retry_delay_seconds,
    )

    webhook_manager.add_webhook(name, webhook_config)
    return {"message": f"Webhook {name} created"}


@router.get("/", response_model=WebhookListResponse)
async def list_webhooks() -> WebhookListResponse:
    """List all webhook configurations.

    Returns:
        List of webhooks
    """
    webhook_infos = [
        WebhookInfo(
            name=name,
            url=config.url,
            enabled=config.enabled,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            retry_delay_seconds=config.retry_delay_seconds,
        )
        for name, config in webhook_manager.webhooks.items()
    ]

    return WebhookListResponse(webhooks=webhook_infos, total=len(webhook_infos))


@router.get("/{name}", response_model=WebhookInfo)
async def get_webhook(name: str) -> WebhookInfo:
    """Get webhook configuration.

    Args:
        name: Webhook name

    Returns:
        Webhook information

    Raises:
        HTTPException: If webhook not found
    """
    config = webhook_manager.get_webhook(name)
    if not config:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return WebhookInfo(
        name=name,
        url=config.url,
        enabled=config.enabled,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        retry_delay_seconds=config.retry_delay_seconds,
    )


@router.delete("/{name}", status_code=204)
async def delete_webhook(name: str) -> None:
    """Delete a webhook configuration.

    Args:
        name: Webhook name

    Raises:
        HTTPException: If webhook not found
    """
    if not webhook_manager.remove_webhook(name):
        raise HTTPException(status_code=404, detail="Webhook not found")


@router.post("/send", response_model=SendWebhookResponse)
async def send_webhook(request: SendWebhookRequest) -> SendWebhookResponse:
    """Send a webhook notification.

    Args:
        request: Webhook send request

    Returns:
        Delivery information

    Raises:
        HTTPException: If webhook not found or disabled
    """
    try:
        delivery = await webhook_manager.send_webhook(
            request.webhook_name,
            request.payload,
        )

        return SendWebhookResponse(
            delivery_id=delivery.id,
            status=delivery.status.value,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/deliveries", response_model=DeliveryListResponse)
async def list_deliveries(limit: int = 100) -> DeliveryListResponse:
    """List recent webhook deliveries.

    Args:
        limit: Maximum number of deliveries to return

    Returns:
        List of deliveries
    """
    deliveries = webhook_manager.get_recent_deliveries(limit)

    delivery_infos = [
        WebhookDeliveryInfo(
            id=d.id,
            webhook_url=d.webhook_url,
            status=d.status.value,
            attempts=d.attempts,
            created_at=d.created_at.isoformat(),
            last_attempt_at=d.last_attempt_at.isoformat() if d.last_attempt_at else None,
            delivered_at=d.delivered_at.isoformat() if d.delivered_at else None,
            error_message=d.error_message,
        )
        for d in deliveries
    ]

    return DeliveryListResponse(deliveries=delivery_infos, total=len(delivery_infos))


@router.get("/deliveries/{delivery_id}", response_model=WebhookDeliveryInfo)
async def get_delivery_status(delivery_id: str) -> WebhookDeliveryInfo:
    """Get webhook delivery status.

    Args:
        delivery_id: Delivery ID

    Returns:
        Delivery information

    Raises:
        HTTPException: If delivery not found
    """
    delivery = webhook_manager.get_delivery_status(delivery_id)
    if not delivery:
        raise HTTPException(status_code=404, detail="Delivery not found")

    return WebhookDeliveryInfo(
        id=delivery.id,
        webhook_url=delivery.webhook_url,
        status=delivery.status.value,
        attempts=delivery.attempts,
        created_at=delivery.created_at.isoformat(),
        last_attempt_at=delivery.last_attempt_at.isoformat() if delivery.last_attempt_at else None,
        delivered_at=delivery.delivered_at.isoformat() if delivery.delivered_at else None,
        error_message=delivery.error_message,
    )
