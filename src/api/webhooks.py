"""Webhook notifications for sending alerts to external services.

This module provides webhook functionality for delivering alerts
to external HTTP endpoints with retry logic.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx


class WebhookStatus(str, Enum):
    """Webhook delivery status."""

    PENDING = 'pending'
    DELIVERED = 'delivered'
    FAILED = 'failed'
    RETRYING = 'retrying'


@dataclass
class WebhookConfig:
    """Webhook configuration."""

    url: str
    enabled: bool = True
    secret: str | None = None
    timeout_seconds: int = 10
    max_retries: int = 3
    retry_delay_seconds: int = 60


@dataclass
class WebhookDelivery:
    """Webhook delivery record."""

    id: str
    webhook_url: str
    payload: dict[str, Any]
    status: WebhookStatus = WebhookStatus.PENDING
    attempts: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_attempt_at: datetime | None = None
    delivered_at: datetime | None = None
    error_message: str | None = None


class WebhookManager:
    """Manager for webhook notifications."""

    def __init__(self) -> None:
        """Initialize webhook manager."""
        self.webhooks: dict[str, WebhookConfig] = {}
        self.deliveries: dict[str, WebhookDelivery] = {}
        self._delivery_counter = 0

    def add_webhook(self, name: str, config: WebhookConfig) -> None:
        """Add a webhook configuration.

        Args:
            name: Webhook name/identifier
            config: Webhook configuration
        """
        self.webhooks[name] = config

    def remove_webhook(self, name: str) -> bool:
        """Remove a webhook configuration.

        Args:
            name: Webhook name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self.webhooks:
            del self.webhooks[name]
            return True
        return False

    def get_webhook(self, name: str) -> WebhookConfig | None:
        """Get webhook configuration.

        Args:
            name: Webhook name

        Returns:
            Webhook config or None if not found
        """
        return self.webhooks.get(name)

    async def send_webhook(
        self,
        webhook_name: str,
        payload: dict[str, Any],
    ) -> WebhookDelivery:
        """Send a webhook notification.

        Args:
            webhook_name: Name of webhook to use
            payload: Data to send

        Returns:
            Delivery record

        Raises:
            ValueError: If webhook not found or disabled
        """
        webhook = self.webhooks.get(webhook_name)
        if not webhook:
            raise ValueError(f'Webhook {webhook_name} not found')

        if not webhook.enabled:
            raise ValueError(f'Webhook {webhook_name} is disabled')

        # Create delivery record
        self._delivery_counter += 1
        delivery_id = f'webhook_{self._delivery_counter}'
        delivery = WebhookDelivery(
            id=delivery_id,
            webhook_url=webhook.url,
            payload=payload,
        )
        self.deliveries[delivery_id] = delivery

        # Attempt delivery
        await self._attempt_delivery(webhook, delivery)

        return delivery

    async def _attempt_delivery(
        self,
        webhook: WebhookConfig,
        delivery: WebhookDelivery,
    ) -> None:
        """Attempt to deliver a webhook.

        Args:
            webhook: Webhook configuration
            delivery: Delivery record to update
        """
        delivery.attempts += 1
        delivery.last_attempt_at = datetime.now(timezone.utc)
        delivery.status = WebhookStatus.PENDING

        headers = {'Content-Type': 'application/json'}
        if webhook.secret:
            headers['X-Webhook-Secret'] = webhook.secret

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook.url,
                    json=delivery.payload,
                    headers=headers,
                    timeout=webhook.timeout_seconds,
                )
                response.raise_for_status()

            # Success
            delivery.status = WebhookStatus.DELIVERED
            delivery.delivered_at = datetime.now(timezone.utc)

        except Exception as e:
            delivery.error_message = str(e)

            # Retry logic
            if delivery.attempts < webhook.max_retries:
                delivery.status = WebhookStatus.RETRYING
                # Schedule retry (simplified - in production use a task queue)
                asyncio.create_task(self._retry_delivery(webhook, delivery))
            else:
                delivery.status = WebhookStatus.FAILED

    async def _retry_delivery(
        self,
        webhook: WebhookConfig,
        delivery: WebhookDelivery,
    ) -> None:
        """Retry a failed delivery after delay.

        Args:
            webhook: Webhook configuration
            delivery: Delivery record
        """
        await asyncio.sleep(webhook.retry_delay_seconds)
        await self._attempt_delivery(webhook, delivery)

    def get_delivery_status(self, delivery_id: str) -> WebhookDelivery | None:
        """Get delivery status.

        Args:
            delivery_id: Delivery ID

        Returns:
            Delivery record or None if not found
        """
        return self.deliveries.get(delivery_id)

    def get_recent_deliveries(self, limit: int = 100) -> list[WebhookDelivery]:
        """Get recent webhook deliveries.

        Args:
            limit: Maximum number of deliveries to return

        Returns:
            List of recent deliveries
        """
        deliveries = sorted(
            self.deliveries.values(),
            key=lambda d: d.created_at,
            reverse=True,
        )
        return deliveries[:limit]


# Global webhook manager
webhook_manager = WebhookManager()
