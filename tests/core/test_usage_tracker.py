"""Unit tests for UsageTracker, McpEventBus, ProviderMetrics."""

from __future__ import annotations

import threading
import time

import pytest

from core.usage_tracker import EventRecord, McpEventBus, ProviderMetrics, UsageTracker


class TestProviderMetrics:
    def test_defaults(self) -> None:
        m = ProviderMetrics()
        assert m.request_count == 0
        assert m.total_input_tokens == 0
        assert m.total_output_tokens == 0
        assert m.last_request_at == 0.0
        assert m.last_reset_at > 0

    def test_slots(self) -> None:
        m = ProviderMetrics()
        with pytest.raises(AttributeError):
            object.__setattr__(m, "nonexistent", 1)


class TestMcpEventBus:
    def test_emit_and_latest(self) -> None:
        bus = McpEventBus()
        assert bus.latest("foo") is None

        bus.emit("foo", bar=1)
        rec = bus.latest("foo")
        assert rec is not None
        assert rec.event_type == "foo"
        assert rec.data == {"bar": 1}

    def test_snapshot(self) -> None:
        bus = McpEventBus()
        bus.emit("a", x=1)
        bus.emit("b", y=2)
        snap = bus.snapshot()
        assert "a" in snap
        assert "b" in snap
        assert snap["a"]["data"] == {"x": 1}
        assert snap["b"]["data"] == {"y": 2}

    def test_latest_returns_none_for_missing(self) -> None:
        bus = McpEventBus()
        assert bus.latest("nonexistent") is None

    def test_emit_overwrites_previous(self) -> None:
        bus = McpEventBus()
        bus.emit("ev", version=1)
        ev1 = bus.latest("ev")
        assert ev1 is not None
        assert ev1.data == {"version": 1}
        bus.emit("ev", version=2)
        ev2 = bus.latest("ev")
        assert ev2 is not None
        assert ev2.data == {"version": 2}


class TestEventRecord:
    def test_timestamp_set_on_creation(self) -> None:
        before = time.time()
        rec = EventRecord(event_type="t", data={"k": "v"})
        after = time.time()
        assert before <= rec.timestamp <= after


class TestUsageTracker:
    def teardown_method(self) -> None:
        UsageTracker._instance = None

    def test_singleton(self) -> None:
        a = UsageTracker.get_instance()
        b = UsageTracker.get_instance()
        assert a is b

    def test_singleton_thread_safety(self) -> None:
        instances: list[UsageTracker] = []
        errors: list[Exception] = []

        def get() -> None:
            try:
                instances.append(UsageTracker.get_instance())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(i is instances[0] for i in instances)

    def test_record_creates_metrics_on_first_call(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        t.record("p1", 10, 20)
        m = t._metrics["p1"]
        assert m.request_count == 1
        assert m.total_input_tokens == 10
        assert m.total_output_tokens == 20

    def test_record_accumulates(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        t.record("p1", 10, 20)
        t.record("p1", 30, 40)
        m = t._metrics["p1"]
        assert m.request_count == 2
        assert m.total_input_tokens == 40
        assert m.total_output_tokens == 60

    def test_record_updates_timestamp(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        now = time.time()
        t.record("p1", 1, 1)
        assert t._metrics["p1"].last_request_at >= now

    def test_record_emits_metrics_updated_event(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        t.record("p1", 5, 15)
        ev = t.event_bus.latest("metrics.updated")
        assert ev is not None
        assert ev.data["provider_id"] == "p1"
        assert ev.data["input_tokens"] == 5
        assert ev.data["output_tokens"] == 15

    def test_record_error(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        t.record_error("p1", "timeout")
        ev = t.event_bus.latest("provider.error")
        assert ev is not None
        assert ev.data["provider_id"] == "p1"
        assert ev.data["error_type"] == "timeout"

    def test_get_all_empty(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        assert t.get() == {}

    def test_get_specific_missing(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        result = t.get("nonexistent")
        assert result == {"provider_id": "nonexistent", "requests": 0}

    def test_get_specific_with_data(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        t.record("p1", 100, 200)
        result = t.get("p1")
        assert result["provider_id"] == "p1"
        assert result["request_count"] == 1
        assert result["total_input_tokens"] == 100
        assert result["total_output_tokens"] == 200

    def test_get_all(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        t.record("p1", 10, 20)
        t.record("p2", 30, 40)
        result = t.get()
        assert "p1" in result
        assert "p2" in result
        assert result["p1"]["request_count"] == 1
        assert result["p2"]["request_count"] == 1

    def test_reset_single(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        t.record("p1", 1, 1)
        t.record("p2", 2, 2)
        result = t.reset("p1")
        assert result == {"provider_id": "p1", "reset": True}
        assert "p1" not in t._metrics
        assert "p2" in t._metrics

    def test_reset_all(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        t.record("p1", 1, 1)
        t.record("p2", 2, 2)
        result = t.reset()
        assert result == {"reset_all": True}
        assert t._metrics == {}

    def test_reset_emits_event(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()
        t.record("p1", 1, 1)
        t.reset("p1")
        ev = t.event_bus.latest("metrics.reset")
        assert ev is not None
        assert ev.data == {"provider_id": "p1", "reset": True}

    def test_thread_safety(self) -> None:
        UsageTracker._instance = None
        t = UsageTracker.get_instance()

        def record_many() -> None:
            for _ in range(100):
                t.record("p1", 1, 1)

        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        m = t.get("p1")
        assert m["request_count"] == 1000
        assert m["total_input_tokens"] == 1000
        assert m["total_output_tokens"] == 1000


class TestEmitModelsEmpty:
    def setup_method(self) -> None:
        import api.mcp_events as mcp_events_module

        mcp_events_module._event_bus = None

    def test_emit_models_empty_when_no_cached_models(self) -> None:
        from unittest import mock

        from api.mcp_events import get_event_bus
        from api.mcp_server import _emit_models_empty_if_needed

        mock_runtime = mock.MagicMock()
        mock_runtime.cached_model_ids.return_value = {}

        _emit_models_empty_if_needed(mock_runtime)

        bus = get_event_bus()
        ev = bus.latest("models.empty")
        assert ev is not None
        assert "hint" in ev.data

    def test_emit_models_empty_when_all_providers_have_no_models(self) -> None:
        from unittest import mock

        from api.mcp_events import get_event_bus
        from api.mcp_server import _emit_models_empty_if_needed

        mock_runtime = mock.MagicMock()
        mock_runtime.cached_model_ids.return_value = {
            "nvidia_nim": frozenset(),
            "openrouter": frozenset(),
        }

        _emit_models_empty_if_needed(mock_runtime)

        bus = get_event_bus()
        ev = bus.latest("models.empty")
        assert ev is not None
        assert "hint" in ev.data

    def test_no_models_empty_when_runtime_is_none(self) -> None:
        from api.mcp_events import get_event_bus
        from api.mcp_server import _emit_models_empty_if_needed

        _emit_models_empty_if_needed(None)

        bus = get_event_bus()
        ev = bus.latest("models.empty")
        assert ev is None

    def test_no_models_empty_when_providers_have_models(self) -> None:
        from unittest import mock

        from api.mcp_events import get_event_bus
        from api.mcp_server import _emit_models_empty_if_needed

        mock_runtime = mock.MagicMock()
        mock_runtime.cached_model_ids.return_value = {
            "nvidia_nim": frozenset(["gpt-4o"]),
        }

        _emit_models_empty_if_needed(mock_runtime)

        bus = get_event_bus()
        ev = bus.latest("models.empty")
        assert ev is None


@pytest.mark.asyncio
async def test_track_usage_stream_parses_messages() -> None:
    from api.provider_execution import _track_usage_stream
    from core.usage_tracker import UsageTracker

    UsageTracker._instance = None
    tracker = UsageTracker.get_instance()

    async def source() -> list[str]:
        return [
            'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":50}}}\n\n',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0}\n\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hello"}}\n\n',
            'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":30}}\n\n',
            "data: [DONE]\n",
        ]

    # Build an async generator from the list
    async def gen():
        for item in await source():
            yield item

    chunks = [chunk async for chunk in _track_usage_stream(gen(), "p1")]
    assert len(chunks) == 5
    assert chunks[-1] == "data: [DONE]\n"

    metrics = tracker.get("p1")
    assert metrics["request_count"] == 1
    assert metrics["total_input_tokens"] == 50
    assert metrics["total_output_tokens"] == 30


@pytest.mark.asyncio
async def test_track_usage_stream_skips_malformed() -> None:
    from api.provider_execution import _track_usage_stream
    from core.usage_tracker import UsageTracker

    UsageTracker._instance = None

    async def gen():
        yield "not data: json\n"
        yield 'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":10}}\n\n'

    chunks = [chunk async for chunk in _track_usage_stream(gen(), "p1")]
    assert len(chunks) == 2
    tracker = UsageTracker.get_instance()
    metrics = tracker.get("p1")
    assert metrics["total_output_tokens"] == 10


@pytest.mark.asyncio
async def test_track_usage_stream_handles_no_message_start() -> None:
    from api.provider_execution import _track_usage_stream
    from core.usage_tracker import UsageTracker

    UsageTracker._instance = None

    async def gen():
        yield 'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":10}}\n\n'

    chunks = [chunk async for chunk in _track_usage_stream(gen(), "p1")]
    assert len(chunks) == 1
    tracker = UsageTracker.get_instance()
    metrics = tracker.get("p1")
    assert metrics["total_input_tokens"] == 0
    assert metrics["total_output_tokens"] == 10
