"""Alerting endpoints for managing alert rules.

This module provides REST API endpoints for managing alert rules,
viewing alerts, and acknowledging/resolving alerts.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..alerting import AlertStatus, alert_engine

router = APIRouter(prefix="/alerts", tags=["alerting"])


class AlertInfo(BaseModel):
    """Alert information response."""

    rule_id: str
    level: str
    message: str
    timestamp: str
    status: str
    extra_data: dict[str, Any] = {}


class AlertRuleInfo(BaseModel):
    """Alert rule information."""

    id: str
    name: str
    description: str
    level: str
    enabled: bool
    cooldown_seconds: int


class AlertListResponse(BaseModel):
    """List of alerts response."""

    alerts: list[AlertInfo]
    total: int


class RuleListResponse(BaseModel):
    """List of alert rules response."""

    rules: list[AlertRuleInfo]
    total: int


@router.get("/", response_model=AlertListResponse)
async def list_alerts(
    status: str | None = None,
    limit: int = 100,
) -> AlertListResponse:
    """List alerts.

    Args:
        status: Optional filter by status (active/resolved/acknowledged)
        limit: Maximum number of alerts to return

    Returns:
        List of alerts
    """
    alerts = alert_engine.alerts

    # Filter by status
    if status:
        try:
            status_enum = AlertStatus(status)
            alerts = [a for a in alerts if a.status == status_enum]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    # Apply limit
    alerts = alerts[-limit:]

    alert_infos = [
        AlertInfo(
            rule_id=a.rule_id,
            level=a.level.value,
            message=a.message,
            timestamp=a.timestamp.isoformat(),
            status=a.status.value,
            extra_data=a.extra_data,
        )
        for a in alerts
    ]

    return AlertListResponse(alerts=alert_infos, total=len(alert_infos))


@router.post("/check")
async def check_alerts() -> AlertListResponse:
    """Manually trigger alert rule evaluation.

    Returns:
        Newly triggered alerts
    """
    new_alerts = alert_engine.check_rules()

    alert_infos = [
        AlertInfo(
            rule_id=a.rule_id,
            level=a.level.value,
            message=a.message,
            timestamp=a.timestamp.isoformat(),
            status=a.status.value,
            extra_data=a.extra_data,
        )
        for a in new_alerts
    ]

    return AlertListResponse(alerts=alert_infos, total=len(alert_infos))


@router.post("/{rule_id}/acknowledge")
async def acknowledge_alert(rule_id: str) -> dict[str, str]:
    """Acknowledge an alert.

    Args:
        rule_id: Rule ID of alert to acknowledge

    Returns:
        Confirmation message
    """
    if alert_engine.acknowledge_alert(rule_id):
        return {"message": f"Alert {rule_id} acknowledged"}
    raise HTTPException(status_code=404, detail="Active alert not found")


@router.post("/{rule_id}/resolve")
async def resolve_alert(rule_id: str) -> dict[str, str]:
    """Resolve an alert.

    Args:
        rule_id: Rule ID of alert to resolve

    Returns:
        Confirmation message
    """
    if alert_engine.resolve_alert(rule_id):
        return {"message": f"Alert {rule_id} resolved"}
    raise HTTPException(status_code=404, detail="Alert not found")


@router.get("/rules", response_model=RuleListResponse)
async def list_rules() -> RuleListResponse:
    """List all alert rules.

    Returns:
        List of alert rules
    """
    rule_infos = [
        AlertRuleInfo(
            id=r.id,
            name=r.name,
            description=r.description,
            level=r.level.value,
            enabled=r.enabled,
            cooldown_seconds=r.cooldown_seconds,
        )
        for r in alert_engine.rules.values()
    ]

    return RuleListResponse(rules=rule_infos, total=len(rule_infos))


@router.post("/rules/{rule_id}/enable")
async def enable_rule(rule_id: str) -> dict[str, str]:
    """Enable an alert rule.

    Args:
        rule_id: Rule ID to enable

    Returns:
        Confirmation message
    """
    if alert_engine.enable_rule(rule_id):
        return {"message": f"Rule {rule_id} enabled"}
    raise HTTPException(status_code=404, detail="Rule not found")


@router.post("/rules/{rule_id}/disable")
async def disable_rule(rule_id: str) -> dict[str, str]:
    """Disable an alert rule.

    Args:
        rule_id: Rule ID to disable

    Returns:
        Confirmation message
    """
    if alert_engine.disable_rule(rule_id):
        return {"message": f"Rule {rule_id} disabled"}
    raise HTTPException(status_code=404, detail="Rule not found")
