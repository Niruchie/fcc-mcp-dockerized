"""Usage tracking, provider metrics, and internal event bus."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderMetrics:
    request_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    last_request_at: float = 0.0
    last_reset_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class EventRecord:
    event_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class McpEventBus:
    """Lightweight pub-sub bus for MCP notifications pushed via resources/subscribe."""

    def __init__(self) -> None:
        self._latest: dict[str, EventRecord] = {}

    def emit(self, event_type: str, **data: Any) -> None:
        self._latest[event_type] = EventRecord(event_type=event_type, data=data)

    def latest(self, event_type: str) -> EventRecord | None:
        return self._latest.get(event_type)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            key: {"type": rec.event_type, "data": rec.data, "timestamp": rec.timestamp}
            for key, rec in self._latest.items()
        }


class UsageTracker:
    """Singleton, thread-safe usage tracker per provider."""

    _instance: UsageTracker | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._metrics: dict[str, ProviderMetrics] = {}
        self._inner_lock = threading.Lock()
        self.event_bus = McpEventBus()

    @classmethod
    def get_instance(cls) -> UsageTracker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def record(self, provider_id: str, input_tokens: int, output_tokens: int) -> None:
        with self._inner_lock:
            m = self._metrics.get(provider_id)
            if m is None:
                m = ProviderMetrics()
                self._metrics[provider_id] = m
            m.request_count += 1
            m.total_input_tokens += input_tokens
            m.total_output_tokens += output_tokens
            m.last_request_at = time.time()
        self.event_bus.emit(
            "metrics.updated",
            provider_id=provider_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def record_error(self, provider_id: str, error_type: str) -> None:
        self.event_bus.emit(
            "provider.error",
            provider_id=provider_id,
            error_type=error_type,
        )

    def get(self, provider_id: str | None = None) -> dict[str, Any]:
        with self._inner_lock:
            if provider_id is not None:
                m = self._metrics.get(provider_id)
                if m is None:
                    return {"provider_id": provider_id, "requests": 0}
                return {
                    "provider_id": provider_id,
                    "request_count": m.request_count,
                    "total_input_tokens": m.total_input_tokens,
                    "total_output_tokens": m.total_output_tokens,
                    "last_request_at": m.last_request_at,
                    "last_reset_at": m.last_reset_at,
                }
            return {
                pid: {
                    "provider_id": pid,
                    "request_count": m.request_count,
                    "total_input_tokens": m.total_input_tokens,
                    "total_output_tokens": m.total_output_tokens,
                    "last_request_at": m.last_request_at,
                    "last_reset_at": m.last_reset_at,
                }
                for pid, m in self._metrics.items()
            }

    def reset(self, provider_id: str | None = None) -> dict[str, Any]:
        with self._inner_lock:
            if provider_id is not None:
                self._metrics.pop(provider_id, None)
                result = {"provider_id": provider_id, "reset": True}
            else:
                self._metrics.clear()
                result = {"reset_all": True}
        self.event_bus.emit("metrics.reset", **result)
        return result
