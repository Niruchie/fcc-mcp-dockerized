"""Shared MCP event bus instance for the API layer."""

from __future__ import annotations

from core.usage_tracker import McpEventBus

_event_bus: McpEventBus | None = None


def get_event_bus() -> McpEventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = McpEventBus()
    return _event_bus
