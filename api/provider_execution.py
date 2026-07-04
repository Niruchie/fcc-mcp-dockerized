"""Shared provider execution primitive for API product handlers."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count
from core.trace import api_messages_request_snapshot, trace_event, traced_async_stream
from core.usage_tracker import UsageTracker
from providers.base import BaseProvider

from .model_router import RoutedMessagesRequest

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]
ProviderGetter = Callable[[str], BaseProvider]


async def _track_usage_stream(
    raw_stream: AsyncIterator[str],
    provider_id: str,
) -> AsyncIterator[str]:
    """Wrap provider SSE events to count usage and detect errors via UsageTracker."""
    tracker = UsageTracker.get_instance()
    input_tokens = 0
    async for event_str in raw_stream:
        data = _extract_sse_data(event_str)
        if data is not None:
            t = data.get("type")
            if t == "message_start":
                usage = data.get("message", {}).get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
            elif t == "message_delta":
                usage = data.get("usage", {})
                out = usage.get("output_tokens", 0)
                tracker.record(provider_id, input_tokens, out)
        yield event_str


def _extract_sse_data(chunk: str) -> dict[str, Any] | None:
    """Extract JSON data from an Anthropic-format SSE event chunk.

    The stream yields chunks like::
        event: message_start
        data: {"type":"message_start",...}
    """
    for line in chunk.splitlines():
        line_s = line.strip()
        if line_s.startswith("data: "):
            try:
                return json.loads(line_s[6:])
            except json.JSONDecodeError:
                return None
    return None


class ProviderExecutionService:
    """Resolve a provider and execute one routed Anthropic Messages stream."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        *,
        token_counter: TokenCounter = get_token_count,
    ) -> None:
        self._settings = settings
        self._provider_getter = provider_getter
        self._token_counter = token_counter

    def stream(
        self,
        routed: RoutedMessagesRequest,
        *,
        wire_api: str,
        raw_log_label: str,
        raw_log_payload: Any,
    ) -> AsyncIterator[str]:
        provider = self._provider_getter(routed.resolved.provider_id)
        provider.preflight_stream(
            routed.request,
            thinking_enabled=routed.resolved.thinking_enabled,
        )

        route_trace: dict[str, Any] = {
            "stage": "routing",
            "event": "api.route.resolved",
            "source": "api",
            "provider_id": routed.resolved.provider_id,
            "provider_model": routed.resolved.provider_model,
            "provider_model_ref": routed.resolved.provider_model_ref,
            "gateway_model": routed.request.model,
            "thinking_enabled": routed.resolved.thinking_enabled,
        }
        if wire_api == "responses":
            route_trace["wire_api"] = "responses"
        trace_event(**route_trace)

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        trace_event(
            stage="ingress",
            event=(
                "api.responses.request.received"
                if wire_api == "responses"
                else "api.request.received"
            ),
            source="api",
            message_count=len(routed.request.messages),
            snapshot=api_messages_request_snapshot(routed.request),
            request_id=request_id,
        )

        if self._settings.log_raw_api_payloads:
            logger.debug(f"{raw_log_label} [{{}}]: {{}}", request_id, raw_log_payload)

        input_tokens = self._token_counter(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )
        raw = provider.stream_response(
            routed.request,
            input_tokens=input_tokens,
            request_id=request_id,
            thinking_enabled=routed.resolved.thinking_enabled,
        )
        return traced_async_stream(
            _track_usage_stream(raw, routed.resolved.provider_id),
            stage="egress",
            source="api",
            complete_event=(
                "api.responses.stream_completed"
                if wire_api == "responses"
                else "api.response.stream_completed"
            ),
            interrupted_event=(
                "api.responses.stream_interrupted"
                if wire_api == "responses"
                else "api.response.stream_interrupted"
            ),
            chunk_event=None,
            extra={
                "request_id": request_id,
                "provider_id": routed.resolved.provider_id,
                "gateway_model": routed.request.model,
            },
        )
