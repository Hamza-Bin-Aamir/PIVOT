"""Alerting rules engine for monitoring thresholds.

This module provides alerting functionality for system metrics,
allowing rules to be defined for CPU, memory, disk, and GPU usage.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .monitoring import monitor


class AlertLevel(str, Enum):
    """Alert severity level."""

    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


class AlertStatus(str, Enum):
    """Alert status."""

    ACTIVE = 'active'
    RESOLVED = 'resolved'
    ACKNOWLEDGED = 'acknowledged'


@dataclass
class AlertRule:
    """Alert rule definition."""

    id: str
    name: str
    description: str
    condition: Callable[[dict[str, Any]], bool]
    level: AlertLevel = AlertLevel.WARNING
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes


@dataclass
class Alert:
    """Alert instance."""

    rule_id: str
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: AlertStatus = AlertStatus.ACTIVE
    extra_data: dict[str, Any] = field(default_factory=dict)


class AlertEngine:
    """Alert engine for monitoring system metrics."""

    def __init__(self) -> None:
        """Initialize alert engine with default rules."""
        self.rules: dict[str, AlertRule] = {}
        self.alerts: list[Alert] = []
        self.last_trigger: dict[str, datetime] = {}

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default monitoring rules."""
        # CPU rules
        self.add_rule(AlertRule(
            id='cpu_high',
            name='High CPU Usage',
            description='CPU usage above 90%',
            condition=lambda m: m.get('cpu', {}).get('available', False) and m['cpu']['percent'] > 90,
            level=AlertLevel.WARNING,
        ))

        self.add_rule(AlertRule(
            id='cpu_critical',
            name='Critical CPU Usage',
            description='CPU usage above 95%',
            condition=lambda m: m.get('cpu', {}).get('available', False) and m['cpu']['percent'] > 95,
            level=AlertLevel.CRITICAL,
        ))

        # Memory rules
        self.add_rule(AlertRule(
            id='memory_high',
            name='High Memory Usage',
            description='Memory usage above 80%',
            condition=lambda m: m.get('memory', {}).get('available', False) and m['memory']['virtual']['percent'] > 80,
            level=AlertLevel.WARNING,
        ))

        self.add_rule(AlertRule(
            id='memory_critical',
            name='Critical Memory Usage',
            description='Memory usage above 90%',
            condition=lambda m: m.get('memory', {}).get('available', False) and m['memory']['virtual']['percent'] > 90,
            level=AlertLevel.CRITICAL,
        ))

        # Disk rules
        self.add_rule(AlertRule(
            id='disk_high',
            name='High Disk Usage',
            description='Disk usage above 85%',
            condition=lambda m: any(
                p['percent'] > 85
                for p in m.get('disk', {}).get('partitions', [])
            ) if m.get('disk', {}).get('available', False) else False,
            level=AlertLevel.WARNING,
        ))

        self.add_rule(AlertRule(
            id='disk_critical',
            name='Critical Disk Usage',
            description='Disk usage above 95%',
            condition=lambda m: any(
                p['percent'] > 95
                for p in m.get('disk', {}).get('partitions', [])
            ) if m.get('disk', {}).get('available', False) else False,
            level=AlertLevel.CRITICAL,
        ))

        # GPU rules (if available)
        self.add_rule(AlertRule(
            id='gpu_high',
            name='High GPU Usage',
            description='GPU usage above 90%',
            condition=lambda m: (
                m.get('gpu', {}).get('available', False) and
                any(g['load'] > 90 for g in m['gpu'].get('gpus', []))
            ),
            level=AlertLevel.WARNING,
        ))

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule.

        Args:
            rule_id: Rule ID to remove

        Returns:
            True if rule was removed, False if not found
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule.

        Args:
            rule_id: Rule ID to enable

        Returns:
            True if rule was enabled, False if not found
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule.

        Args:
            rule_id: Rule ID to disable

        Returns:
            True if rule was disabled, False if not found
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            return True
        return False

    def check_rules(self) -> list[Alert]:
        """Check all rules against current metrics.

        Returns:
            List of new alerts triggered
        """
        metrics = monitor.get_all_metrics()
        new_alerts = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.id in self.last_trigger:
                time_since_last = (datetime.now(timezone.utc) - self.last_trigger[rule.id]).total_seconds()
                if time_since_last < rule.cooldown_seconds:
                    continue

            # Evaluate condition
            try:
                if rule.condition(metrics):
                    alert = Alert(
                        rule_id=rule.id,
                        level=rule.level,
                        message=f'{rule.name}: {rule.description}',
                        extra_data={'metrics': metrics},
                    )
                    self.alerts.append(alert)
                    new_alerts.append(alert)
                    self.last_trigger[rule.id] = datetime.now(timezone.utc)
            except Exception:
                # Silently skip rules that fail to evaluate
                continue

        return new_alerts

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts.

        Returns:
            List of active alerts
        """
        return [a for a in self.alerts if a.status == AlertStatus.ACTIVE]

    def acknowledge_alert(self, rule_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            rule_id: Rule ID of alert to acknowledge

        Returns:
            True if alert was acknowledged, False if not found
        """
        for alert in self.alerts:
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                return True
        return False

    def resolve_alert(self, rule_id: str) -> bool:
        """Resolve an alert.

        Args:
            rule_id: Rule ID of alert to resolve

        Returns:
            True if alert was resolved, False if not found
        """
        for alert in self.alerts:
            if alert.rule_id == rule_id and alert.status != AlertStatus.RESOLVED:
                alert.status = AlertStatus.RESOLVED
                return True
        return False

    def clear_alerts(self, before: datetime | None = None) -> int:
        """Clear resolved alerts.

        Args:
            before: Optional datetime to clear alerts before

        Returns:
            Number of alerts cleared
        """
        if before is None:
            before = datetime.now(timezone.utc)

        initial_count = len(self.alerts)
        self.alerts = [
            a for a in self.alerts
            if not (a.status == AlertStatus.RESOLVED and a.timestamp < before)
        ]
        return initial_count - len(self.alerts)


# Global alert engine instance
alert_engine = AlertEngine()
