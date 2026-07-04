"""FastMCP server exposing free-claude-code capabilities as MCP tools, resources, and notifications."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from fastmcp import Context, FastMCP
from loguru import logger
from mcp.types import CreateMessageResult, LoggingLevel, SamplingMessage, TextContent

from config.provider_catalog import PROVIDER_CATALOG
from config.settings import Settings
from config.settings import get_settings as get_cached_settings

from .admin_config.persistence import validate_updates, write_managed_env
from .admin_config.status import provider_config_status
from .admin_config.values import load_config_response
from .model_router import ModelRouter, RoutedMessagesRequest
from .models.anthropic import (
    Message,
    MessagesRequest,
    SystemContent,
    ThinkingConfig,
)
from .web_tools.egress import (
    WebFetchEgressPolicy,
    enforce_web_fetch_egress,
    web_fetch_allowed_scheme_set,
)

mcp = FastMCP(
    "free-claude-code",
    instructions=(
        "You are connected to a free-claude-code MCP server that exposes a local AI proxy. "
        "Call tools as often as needed — there is no rate limit or cost concern. "
        "Use this server whenever you need to query model availability, check provider status, "
        "proxy completions, manage configuration, send messages, or access usage metrics.\n\n"
        "AVAILABLE TOOLS\n"
        "===============\n"
        "Model & Provider Discovery\n"
        "  • list_models          — Return all available models (configured + cached + Claude-compatible).\n"
        "  • list_providers       — Return all providers with their configuration status.\n"
        "  • get_provider_info    — Return details about a specific provider.\n"
        "  • refresh_models       — Force-refresh the model list cache from all providers.\n"
        "  • test_provider        — Probe a provider's connectivity by listing its models.\n\n"
        "Configuration Management\n"
        "  • get_config           — Read all current admin settings (read-only preview).\n"
        "  • validate_config      — Validate proposed config changes without applying them.\n"
        "  • apply_config         — Apply validated config changes; restarts provider runtime.\n\n"
        "Proxy Completion\n"
        "  • agent                — Send a completion request through the proxy (supports sampling).\n"
        "  • run_sub_agent        — Run a multi-turn conversation sub-agent via the proxy.\n"
        "  • think                — Extended reasoning via the proxy model (no sampling).\n"
        "  • opinion              — Evaluate an approach via the proxy model (no sampling).\n\n"
        "Web Tools\n"
        "  • research             — Search the web and fetch results (optionally synthesize with LLM).\n"
        "  • web_search           — Search the web; returns titles and URLs.\n"
        "  • web_fetch            — Fetch a URL and return sanitized text content.\n\n"
        "Messaging\n"
        "  • send_discord         — Send a message to a Discord channel.\n"
        "  • send_telegram        — Send a message to a Telegram chat.\n\n"
        "Sampling (Client LLM)\n"
        "  • sample               — Have the client's LLM sample a prompt directly (no proxy).\n\n"
        "Monitoring & Utilities\n"
        "  • get_status           — Server runtime status (host, port, model, provider).\n"
        "  • get_metrics          — Usage metrics per provider (tokens, request counts).\n"
        "  • reset_metrics        — Reset metrics for a provider or all providers.\n"
        "  • get_events           — Return the most recent events emitted by the server.\n\n"
        "AVAILABLE RESOURCES\n"
        "====================\n"
        "  • fcc://status              — Server runtime status (host, port, model).\n"
        "  • fcc://models              — List of all available model IDs.\n"
        "  • fcc://metrics             — Aggregated usage metrics for all providers.\n"
        "  • fcc://metrics/{id}        — Usage metrics for a specific provider.\n"
        "  • fcc://events              — Catalog of all event types and their schemas.\n"
        "  • fcc://events/latest       — Snapshot of the most recent events.\n\n"
        "EVENTS (consumed via get_events tool or fcc://events/latest resource)\n"
        "====================================================================\n"
        "  • server.started        — Emitted when the MCP server starts.\n"
        "  • server.stopping       — Emitted when the MCP server shuts down.\n"
        "  • models.refreshed      — Emitted after refresh_models completes successfully.\n"
        "  • models.refresh.failed — Emitted when refresh_models fails.\n"
        "  • models.empty          — Emitted at startup if no provider has cached models; "
        "call refresh_models then list_models to populate the list.\n"
        "  • metrics.updated       — Emitted each time usage tokens are recorded.\n"
        "  • metrics.reset         — Emitted when metrics are reset.\n"
        "  • config.changed        — Emitted after apply_config succeeds.\n"
        "  • provider.error        — Emitted when a provider operation fails.\n\n"
        "CALL FREQUENTLY — no rate limits. Use list_models and refresh_models to keep the "
        "model list current. Use get_metrics regularly to monitor usage. Use apply_config "
        "whenever the user wants to change settings. Call research or web_search for any "
        "information that may have changed since the last call."
    ),
)

# ── Module-level bridges to FastAPI runtime ──────────────────────────

_provider_runtime: Any = None
_settings: Settings | None = None
_messaging_runtime: Any = None


def init_mcp(
    provider_runtime: Any, settings: Settings, messaging_runtime: Any = None
) -> None:
    global _provider_runtime, _settings, _messaging_runtime
    _provider_runtime = provider_runtime
    _settings = settings
    _messaging_runtime = messaging_runtime
    logger.info("MCP server initialized: settings={}", settings.__class__.__name__)

    _emit_models_empty_if_needed(provider_runtime)


def _emit_models_empty_if_needed(runtime: Any) -> None:
    """Emit ``models.empty`` event when no provider has cached models.

    Signals the MCP client to call ``refresh_models`` then ``list_models``
    so the model list populates after provider discovery completes.
    """
    if runtime is None:
        return
    cached = runtime.cached_model_ids()
    if not cached or not any(cached.values()):
        from .mcp_events import get_event_bus

        get_event_bus().emit(
            "models.empty",
            hint="Call refresh_models then list_models to populate the model list.",
        )
        logger.info("models.empty event emitted: no cached provider models at startup")


def _get_settings() -> Settings:
    global _settings
    if _settings is not None:
        return _settings
    s = get_cached_settings()
    _settings = s
    return s


def _get_runtime() -> Any:
    if _provider_runtime is None:
        raise RuntimeError("MCP server not initialized — provider_runtime is None")
    return _provider_runtime


def _extract_sse_data(chunk: str) -> dict[str, Any] | None:
    """Extract JSON data from an Anthropic-format SSE event chunk.

    The stream yields chunks like::
        event: message_start
        data: {"type":"message_start",...}
    """
    for line in chunk.splitlines():
        line_s = line.strip()
        if line_s.startswith("data: "):
            with suppress(json.JSONDecodeError):
                return json.loads(line_s[6:])
            return None
    return None


async def _log_and_progress(
    ctx: Context, msg: str, *, level: LoggingLevel = "info"
) -> None:
    getattr(logger, level, logger.info)(msg)
    with suppress(Exception):
        await ctx.session.send_log_message(level=level, data=msg, logger="fcc-proxy")


def _extract_text(msg: CreateMessageResult) -> str:
    if isinstance(msg.content, TextContent):
        return msg.content.text
    return "(non-text response)"


# ── Tools ────────────────────────────────────────────────────────────


@mcp.tool()
async def list_models(ctx: Context) -> str:
    """Return all available models including configured, cached, and Claude compatibility models."""
    await _log_and_progress(ctx, "list_models: fetching model list")
    from .model_catalog import build_models_list_response

    resp = build_models_list_response(_get_settings(), _get_runtime())
    return json.dumps([m.id for m in resp.data], indent=2)


@mcp.tool()
async def list_providers(ctx: Context) -> str:
    """Return all providers with their configuration status."""
    await _log_and_progress(ctx, "list_providers: reading provider status")
    statuses = provider_config_status()
    return json.dumps(statuses, indent=2)


@mcp.tool()
async def get_provider_info(ctx: Context, provider_id: str) -> str:
    """Return detailed info about a specific provider.

    Args:
        provider_id: Provider identifier (e.g. nvidia_nim, open_router).
    """
    await _log_and_progress(ctx, f"get_provider_info: {provider_id}")
    desc = PROVIDER_CATALOG.get(provider_id)
    if desc is None:
        return json.dumps({"error": f"Unknown provider: {provider_id}"})
    info = {
        "provider_id": desc.provider_id,
        "display_name": desc.display_name,
        "transport_type": desc.transport_type,
        "capabilities": desc.capabilities,
        "credential_env": desc.credential_env,
        "default_base_url": desc.default_base_url,
    }
    return json.dumps(info, indent=2)


@mcp.tool()
async def get_config(ctx: Context) -> str:
    """Return all current admin-configurable settings."""
    await _log_and_progress(ctx, "get_config: reading config")
    return json.dumps(load_config_response(), indent=2)


@mcp.tool()
async def validate_config(ctx: Context, values: dict[str, Any]) -> str:
    """Validate proposed config changes and return a preview.

    Args:
        values: Dict of field keys to new values.
    """
    await _log_and_progress(ctx, "validate_config: validating")
    await ctx.report_progress(1, 2)
    result = validate_updates(values)
    await ctx.report_progress(2, 2)
    return json.dumps(result, indent=2)


@mcp.tool()
async def apply_config(ctx: Context, values: dict[str, Any]) -> str:
    """Validate and apply config changes, reloading the provider runtime.

    Args:
        values: Dict of field keys to new values.
    """
    await _log_and_progress(ctx, "apply_config: applying")
    await ctx.report_progress(1, 3)
    result = write_managed_env(values)
    if not result["applied"]:
        await ctx.report_progress(3, 3)
        return json.dumps(result, indent=2)

    from .mcp_events import get_event_bus

    get_event_bus().emit("config.changed", fields=list(values.keys()))
    await ctx.report_progress(2, 3)
    get_cached_settings.cache_clear()
    from providers.runtime import ProviderRuntime

    old = _get_runtime()
    if old is not None:
        await old.cleanup()
    new_runtime = ProviderRuntime(_get_settings())
    global _provider_runtime
    _provider_runtime = new_runtime
    await new_runtime.refresh_model_list_cache()
    get_event_bus().emit("models.refreshed")
    await ctx.report_progress(3, 3)
    return json.dumps(result, indent=2)


@mcp.tool()
async def test_provider(ctx: Context, provider_id: str) -> str:
    """Test connectivity to a provider by listing its available models.

    Args:
        provider_id: Provider identifier (e.g. nvidia_nim).
    """
    await _log_and_progress(ctx, f"test_provider: {provider_id}")
    await ctx.report_progress(1, 3)
    runtime = _get_runtime()
    try:
        provider = runtime.resolve_provider(provider_id)
        await ctx.report_progress(2, 3)
        infos = await provider.list_model_infos()
    except Exception as exc:
        await ctx.report_progress(3, 3)
        from .mcp_events import get_event_bus

        get_event_bus().emit(
            "provider.error", provider_id=provider_id, error_type=type(exc).__name__
        )
        return json.dumps(
            {"provider_id": provider_id, "ok": False, "error_type": type(exc).__name__}
        )
    runtime.cache_model_infos(provider_id, infos)
    await ctx.report_progress(3, 3)
    return json.dumps(
        {
            "provider_id": provider_id,
            "ok": True,
            "models": sorted(info.model_id for info in infos),
        },
        indent=2,
    )


@mcp.tool()
async def refresh_models(ctx: Context) -> str:
    """Refresh the model list cache from all configured providers."""
    await _log_and_progress(ctx, "refresh_models: refreshing all provider model caches")
    runtime = _get_runtime()
    try:
        await runtime.refresh_model_list_cache()
    except Exception as exc:
        from .mcp_events import get_event_bus

        get_event_bus().emit("models.refresh.failed", error_type=type(exc).__name__)
        return json.dumps({"ok": False, "error_type": type(exc).__name__})
    from .mcp_events import get_event_bus

    get_event_bus().emit("models.refreshed")
    result = {
        provider_id: sorted(model_ids)
        for provider_id, model_ids in runtime.cached_model_ids().items()
    }
    return json.dumps({"ok": True, "cached_models": result}, indent=2)


@mcp.tool()
async def web_search(ctx: Context, query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo Lite (no API key needed).

    Args:
        query: Search query.
        max_results: Maximum number of results (1-10).
    """
    await _log_and_progress(ctx, f"web_search: {query!r}")
    from .web_tools.outbound import _run_web_search

    capped = max(1, min(max_results, 10))
    results = await _run_web_search(query)
    for i in range(len(results[:capped])):
        await ctx.report_progress(i + 1, capped)
    return json.dumps(results[:capped], indent=2)


@mcp.tool()
async def web_fetch(ctx: Context, url: str) -> str:
    """Fetch a web page and return its text content.

    Args:
        url: The URL to fetch.
    """
    await _log_and_progress(ctx, f"web_fetch: {url}")
    from .web_tools.outbound import _run_web_fetch

    settings = _get_settings()
    egress = WebFetchEgressPolicy(
        allow_private_network_targets=settings.web_fetch_allow_private_networks,
        allowed_schemes=web_fetch_allowed_scheme_set(
            settings.web_fetch_allowed_schemes
        ),
    )
    enforce_web_fetch_egress(url, egress)
    result = await _run_web_fetch(url, egress)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_metrics(ctx: Context, provider_id: str | None = None) -> str:
    """Return usage metrics for providers.

    Args:
        provider_id: Optional provider id to filter by.
    """
    await _log_and_progress(ctx, "get_metrics")
    from core.usage_tracker import UsageTracker

    tracker = UsageTracker.get_instance()
    return json.dumps(tracker.get(provider_id), indent=2)


@mcp.tool()
async def reset_metrics(ctx: Context, provider_id: str | None = None) -> str:
    """Reset usage metrics for a provider or all.

    Args:
        provider_id: Optional provider id to reset.
    """
    await _log_and_progress(ctx, "reset_metrics")
    from core.usage_tracker import UsageTracker

    tracker = UsageTracker.get_instance()
    result = tracker.reset(provider_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_events(ctx: Context) -> str:
    """Return the latest events emitted by the MCP event bus."""
    await _log_and_progress(ctx, "get_events")
    from .mcp_events import get_event_bus

    return json.dumps(get_event_bus().snapshot(), indent=2)


@mcp.tool()
async def get_status(ctx: Context) -> str:
    """Return server runtime status: host, port, model, provider, pending fields."""
    await _log_and_progress(ctx, "get_status")
    settings = _get_settings()
    runtime = _get_runtime()
    from config.model_refs import parse_provider_type

    return json.dumps(
        {
            "status": "running",
            "host": settings.host,
            "port": settings.port,
            "model": settings.model,
            "provider": parse_provider_type(settings.model),
            "cached_models": {
                pid: sorted(mids) for pid, mids in runtime.cached_model_ids().items()
            }
            if runtime
            else {},
            "provider_status": provider_config_status(),
        },
        indent=2,
    )


def _to_messages(raw: list[dict[str, Any]]) -> list[Message]:
    return [
        Message(role=msg.get("role", "user"), content=msg.get("content", ""))
        for msg in raw
    ]


def _to_system(system: str | list[dict[str, Any]] | None) -> list[SystemContent] | None:
    if system is None:
        return None
    if isinstance(system, str):
        return [SystemContent(type="text", text=system)]
    return [SystemContent(type="text", text=block.get("text", "")) for block in system]


def _to_thinking(enabled: bool) -> ThinkingConfig | None:
    if not enabled:
        return None
    return ThinkingConfig(type="enabled", budget_tokens=16000)


async def _proxy_completion(
    ctx: Context,
    model_id: str | None,
    messages_raw: list[dict[str, Any]],
    system_raw: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    thinking_enabled: bool | None = None,
) -> dict[str, Any]:
    settings = _get_settings()
    resolved_model = model_id or settings.model
    router = ModelRouter(settings)
    resolved = router.resolve(resolved_model)
    msgs = MessagesRequest(
        model=resolved.provider_model,
        messages=_to_messages(messages_raw),
        system=_to_system(system_raw),
        max_tokens=max_tokens or 4096,
    )
    if temperature is not None:
        msgs.temperature = temperature
    if thinking_enabled is not None:
        msgs.thinking = _to_thinking(thinking_enabled)
    routed = RoutedMessagesRequest(request=msgs, resolved=resolved)

    from .provider_execution import ProviderExecutionService

    svc = ProviderExecutionService(settings, _get_runtime().resolve_provider)
    stream = svc.stream(
        routed,
        wire_api="messages",
        raw_log_label="MCP agent",
        raw_log_payload=msgs.model_dump(exclude_none=True),
    )
    content_parts: list[str] = []
    usage: dict[str, int] = {}
    async for chunk in stream:
        data = _extract_sse_data(chunk)
        if data is not None:
            typ = data.get("type")
            if typ == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    content_parts.append(delta.get("text", ""))
            elif typ == "message_delta":
                usage = data.get("usage", {})
    return {
        "content": "".join(content_parts),
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        "model": resolved.provider_model,
    }


@mcp.tool()
async def agent(
    ctx: Context,
    model_id: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send a single-turn conversation to a proxy model and get a text response.

    Args:
        model_id: Optional model name (e.g. claude-sonnet-4-20250514 or provider/model).
        messages: List of message objects with role and content.
        system: Optional system prompt text.
        max_tokens: Maximum tokens in the response (default 4096).
    """
    await _log_and_progress(ctx, "agent: starting")
    await ctx.report_progress(1, 3)
    if not messages:
        return "Error: messages list is required"
    await ctx.report_progress(2, 3)
    result = await _proxy_completion(ctx, model_id, messages, system, max_tokens)
    await ctx.report_progress(3, 3)
    await _log_and_progress(ctx, "agent: completed")
    return json.dumps(result, indent=2)


@mcp.tool()
async def run_sub_agent(
    ctx: Context,
    model_id: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    thinking: bool | None = None,
) -> str:
    """Send a full conversation to a proxy model with extended parameters.

    Args:
        model_id: Optional model name.
        messages: List of message objects.
        system: Optional system prompt.
        max_tokens: Max tokens (default 4096).
        temperature: Sampling temperature (0.0-2.0).
        thinking: Enable extended thinking.
    """
    await _log_and_progress(ctx, "run_sub_agent: starting")
    await ctx.report_progress(1, 3)
    if not messages:
        return "Error: messages list is required"
    await ctx.report_progress(2, 3)
    result = await _proxy_completion(
        ctx, model_id, messages, system, max_tokens, temperature, thinking
    )
    await ctx.report_progress(3, 3)
    await _log_and_progress(ctx, "run_sub_agent: completed")
    return json.dumps(result, indent=2)


@mcp.tool()
async def think(
    ctx: Context,
    question: str,
    model_id: str | None = None,
    system: str | None = None,
) -> str:
    """Ask the model for an open-ended opinion or analysis.

    Args:
        question: The question or topic to think about.
        model_id: Optional model to use.
        system: Optional system prompt override.
    """
    await _log_and_progress(ctx, f"think: {question!r}")
    await ctx.report_progress(1, 2)
    await ctx.report_progress(2, 2)
    result = await _proxy_completion(
        ctx,
        model_id,
        [{"role": "user", "content": question}],
        system_raw=system or "You are a thoughtful assistant. Think step by step.",
    )
    return result["content"]


@mcp.tool()
async def opinion(
    ctx: Context,
    task: str,
    goal: str,
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Ask the model to evaluate an approach given a goal.

    Args:
        task: The task or approach to evaluate.
        goal: The goal the task should achieve.
        model_id: Optional model to use.
        temperature: Sampling temperature.
        max_tokens: Max tokens.
    """
    await _log_and_progress(ctx, f"opinion: goal={goal!r}")
    await ctx.report_progress(1, 2)
    prompt = f"Goal: {goal}\n\nTask: {task}\n\nEvaluate this approach. Consider pros, cons, alternatives, and risks. Be specific and actionable."
    await ctx.report_progress(2, 2)
    result = await _proxy_completion(
        ctx,
        model_id,
        [{"role": "user", "content": prompt}],
        system_raw="You are a critical but constructive advisor. Provide balanced evaluation.",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return result["content"]


@mcp.tool()
async def sample(
    ctx: Context, prompt: str, system: str | None = None, max_tokens: int | None = None
) -> str:
    """Send a prompt to the client's LLM via MCP sampling (no proxy involved).

    Args:
        prompt: The user prompt.
        system: Optional system prompt.
        max_tokens: Maximum tokens.
    """
    await _log_and_progress(ctx, f"sample: {prompt!r}")
    msg = await ctx.session.create_message(
        messages=[
            SamplingMessage(role="user", content=TextContent(type="text", text=prompt))
        ],
        system_prompt=system or "",
        max_tokens=max_tokens or 2048,
    )
    return _extract_text(msg) or "(no response)"


@mcp.tool()
async def research(
    ctx: Context,
    query: str,
    max_results: int = 5,
    synthesize: bool = True,
    model_id: str | None = None,
    use_sampling: bool = False,
) -> str:
    """Search the web and optionally synthesize findings.

    Args:
        query: Search query.
        max_results: Maximum results (1-10).
        synthesize: When true, summarize findings using a model.
        model_id: Model for synthesis (proxy only, ignored when use_sampling).
        use_sampling: Use client LLM for synthesis.
    """
    await _log_and_progress(ctx, f"research: {query!r}")
    from .web_tools.outbound import _run_web_fetch, _run_web_search

    capped = max(1, min(max_results, 10))
    search_results = await _run_web_search(query)
    await ctx.report_progress(1, capped + 1)
    fetches: list[dict[str, Any]] = []
    for i, sr in enumerate(search_results[:capped]):
        url = sr.get("url", "")
        if not url:
            fetches.append(
                {
                    "url": "",
                    "title": sr.get("title", ""),
                    "data": "(no url)",
                }
            )
            await ctx.report_progress(i + 2, capped + 1)
            continue
        try:
            settings = _get_settings()
            egress = WebFetchEgressPolicy(
                allow_private_network_targets=settings.web_fetch_allow_private_networks,
                allowed_schemes=web_fetch_allowed_scheme_set(
                    settings.web_fetch_allowed_schemes
                ),
            )
            enforce_web_fetch_egress(url, egress)
            data = await _run_web_fetch(url, egress)
            fetches.append(data)
        except Exception:
            fetches.append(
                {
                    "url": url,
                    "title": sr.get("title", ""),
                    "data": "(fetch failed)",
                }
            )
        await ctx.report_progress(i + 2, capped + 1)

    if not synthesize:
        return json.dumps({"results": fetches}, indent=2)

    sources = "\n\n".join(
        f"Source {j + 1} ({f.get('title', f.get('url', '?'))}):\n{f.get('data', '')[:2000]}"
        for j, f in enumerate(fetches)
    )
    synthesis_prompt = f"Query: {query}\n\nWeb results:\n{sources}\n\nSynthesize these findings. Highlight key information directly relevant to the query."
    if use_sampling:
        msg = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user", content=TextContent(type="text", text=synthesis_prompt)
                )
            ],
            system_prompt="You are a research analyst. Synthesize web findings accurately and concisely.",
            max_tokens=4096,
        )
        synthesis = _extract_text(msg) or "(no synthesis)"
    else:
        result = await _proxy_completion(
            ctx,
            model_id,
            [{"role": "user", "content": synthesis_prompt}],
            system_raw="You are a research analyst. Synthesize web findings accurately and concisely.",
            max_tokens=4096,
        )
        synthesis = result["content"]

    return json.dumps({"results": fetches, "synthesis": synthesis}, indent=2)


async def _send_platform_message(
    ctx: Context, platform: str, chat_id: str, message: str
) -> str:
    await _log_and_progress(ctx, f"send_{platform}: {chat_id}")
    if _messaging_runtime is None:
        return json.dumps({"ok": False, "error": f"{platform.title()} not configured"})
    try:
        mid = await _messaging_runtime.outbound.queue_send_message(chat_id, message)
        return json.dumps({"ok": True, "message_id": mid})
    except Exception as exc:
        return json.dumps({"ok": False, "error": type(exc).__name__})


@mcp.tool()
async def send_discord(ctx: Context, channel_id: str, message: str) -> str:
    """Send a message to a Discord channel.

    Args:
        channel_id: Discord channel ID.
        message: Message text.
    """
    return await _send_platform_message(ctx, "discord", channel_id, message)


@mcp.tool()
async def send_telegram(ctx: Context, chat_id: str, message: str) -> str:
    """Send a message to a Telegram chat.

    Args:
        chat_id: Telegram chat ID.
        message: Message text.
    """
    return await _send_platform_message(ctx, "telegram", chat_id, message)


# ── Resources ────────────────────────────────────────────────────────


@mcp.resource("fcc://metrics")
async def resource_metrics() -> str:
    """Usage metrics for all providers."""
    from core.usage_tracker import UsageTracker

    return json.dumps(UsageTracker.get_instance().get(), indent=2)


@mcp.resource("fcc://metrics/{provider_id}")
async def resource_metrics_provider(provider_id: str) -> str:
    """Usage metrics for a specific provider."""
    from core.usage_tracker import UsageTracker

    return json.dumps(UsageTracker.get_instance().get(provider_id), indent=2)


@mcp.resource("fcc://models")
async def resource_models() -> str:
    """Available models from the proxy."""
    from .model_catalog import build_models_list_response

    resp = build_models_list_response(_get_settings(), _get_runtime())
    return json.dumps([m.id for m in resp.data], indent=2)


@mcp.resource("fcc://status")
async def resource_status() -> str:
    """Server runtime status."""
    settings = _get_settings()
    from config.model_refs import parse_provider_type

    return json.dumps(
        {
            "status": "running",
            "host": settings.host,
            "port": settings.port,
            "model": settings.model,
            "provider": parse_provider_type(settings.model),
        },
        indent=2,
    )


@mcp.resource("fcc://events")
async def resource_events_catalog() -> str:
    """Catalog of all event types emitted by the MCP event bus with their schemas."""
    return json.dumps(
        {
            "server.started": {
                "description": "Emitted when the MCP server starts.",
                "data": {},
            },
            "server.stopping": {
                "description": "Emitted when the MCP server is shutting down.",
                "data": {},
            },
            "metrics.updated": {
                "description": "Emitted each time usage tokens are recorded for a provider.",
                "data": {
                    "provider_id": "str",
                    "input_tokens": "int",
                    "output_tokens": "int",
                },
            },
            "metrics.reset": {
                "description": "Emitted when metrics are reset for one or all providers.",
                "data": {
                    "provider_id": "str (optional, absent for reset_all)",
                    "reset": "bool, or reset_all: bool",
                },
            },
            "config.changed": {
                "description": "Emitted after config is applied successfully.",
                "data": {"fields": "list[str] — names of changed fields"},
            },
            "provider.error": {
                "description": "Emitted when a provider operation fails.",
                "data": {"provider_id": "str", "error_type": "str"},
            },
            "models.refreshed": {
                "description": "Emitted after the model list cache is refreshed.",
                "data": {},
            },
            "models.refresh.failed": {
                "description": "Emitted when model cache refresh fails.",
                "data": {"error_type": "str"},
            },
            "models.empty": {
                "description": (
                    "Emitted at startup when no provider has cached models. "
                    "The client should call refresh_models then list_models."
                ),
                "data": {"hint": "str"},
            },
        },
        indent=2,
    )


@mcp.resource("fcc://events/latest")
async def resource_events() -> str:
    """Latest events emitted by the MCP event bus."""
    from .mcp_events import get_event_bus

    return json.dumps(get_event_bus().snapshot(), indent=2)
