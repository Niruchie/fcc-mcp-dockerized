# Free Claude Code — MCP-Enabled and Dockerized

> **This is an  fork** of [free-claude-code](https://github.com/Alishahryar1/free-claude-code) repository. The original project is licensed under the MIT Licence.

---

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-%5E0.11.0-blue?style=for-the-badge)](https://github.com/astral-sh/uv)
[![FastMCP 3.4.2](https://img.shields.io/badge/FastMCP-3.4.2-6B8AFD?style=for-the-badge)](https://github.com/jlowin/fastmcp)
[![Type checking: Ty](https://img.shields.io/badge/type%20checking-ty-ffcc00?style=for-the-badge)](https://pypi.org/project/ty/)

</div>

## Table of Contents

1. [Overview](#overview)
2. [Protocol Reference](#protocol-reference)
3. [MCP Server Architecture](#mcp-server-architecture)
4. [MCP Tools](#mcp-tools)
5. [MCP Resources](#mcp-resources)
6. [MCP Events and Notifications](#mcp-events-and-notifications)
7. [MCP Sampling — Client LLM Delegation](#mcp-sampling--client-llm-delegation)
8. [Model Variants and Catalog](#model-variants-and-catalog)
9. [Docker Deployment](#docker-deployment)
10. [Configuration](#configuration)
11. [Metrics and Usage Tracking](#metrics-and-usage-tracking)
12. [CI / Quality Gates](#ci--quality-gates)
13. [Repository Structure](#repository-structure)
14. [Licence](#licence)

---

## Overview

This project is a fork of [free-claude-code](https://github.com/Alishahryar1/free-claude-code).  adds a full-featured **MCP (Model Context Protocol) server** to the existing FastAPI proxy, enabling any MCP-compatible client — including OpenCode, Claude Desktop, and custom clients — to interact with the proxy through the MCP wire protocol rather than (or in addition to) the raw Anthropic/OpenAI HTTP APIs. **Also dockerizes the proxy for easy deployment.**

The MCP server exposes:

- **23 tools** for model discovery, proxy completion, web search, messaging, and monitoring
- **6 resources** at `fcc://` URI templates
- **9 server-side event types** emitted via `McpEventBus`
- **MCP sampling** — delegate reasoning to the connecting client's own LLM directly
- **SSE transport** via FastMCP 3.4.2

The proxy itself routes Anthropic Messages API traffic (Claude Code) and OpenAI Responses API traffic (Codex) to any configured provider backend. The MCP layer sits on top and is independently accessible — clients can use MCP-only features (e.g., `think`, `opinion`, `research`) without routing proxy traffic.

---

## Protocol Reference

### MCP Transport: Server-Sent Events (SSE)

The MCP server uses **SSE (Server-Sent Events)** as its transport layer, implemented via FastMCP 3.4.2's built-in SSE support. Clients connect over two HTTP endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/mcp/sse` | `GET` | Establish SSE stream from server to client |
| `/mcp/messages/` | `POST` | Send JSON-RPC messages from client to server |

SSE events from the server follow this format:

```
event: {type}
data: {json}

```

The MCP server is a **sub-application** mounted inside the FastAPI app at `/mcp`, not a separate process:

```python
# FastMCP app with SSE transport, registered as a FastAPI sub-app
mcp_http_app = mcp.http_app(transport="sse")
app.mount("/mcp", mcp_http_app)
```

### MCP Protocol Lifecycle

The MCP protocol follows the standard initialization → operation → shutdown lifecycle:

1. **Handshake / Initialization** — Client sends `initialize` to the server. Server responds with its name, version, capabilities, and a description of available tools. This response carries the `instructions` parameter (see below).
2. **Tools** — Client invokes tools via `tools/call`. Server executes the corresponding Python function and returns a JSON result.
3. **Resources** — Client reads resources via `resources/read`. Server returns content at the requested `fcc://` URI.
4. **Sampling** — Client requests that the server ask the *client's own LLM* to sample a prompt via `SamplingMessage` / `create_message()`.
5. **Notifications / Events** — Server emits events via `McpEventBus.emit()`. Clients poll `get_events` or subscribe to `fcc://events/latest`.
6. **Shutdown** — Client sends `notifications/initialized` once it has processed the initialize response, then the connection stays open until either side closes.

---

## MCP Server Architecture

### Components

| File | Role |
|---|---|
| `api/mcp_server.py` | FastMCP server instance (`FastMCP(...)`), all tools, all resources, initialization logic |
| `api/mcp_events.py` | Singleton `McpEventBus` (`get_event_bus()`) shared across the codebase |
| `core/usage_tracker.py` | `McpEventBus` class definition, `UsageTracker` singleton, `ProviderMetrics`, `EventRecord` |
| `api/model_catalog.py` | `build_models_list_response()` — builds the combined model list from configured + cached + Claude-compatible models |
| `api/gateway_model_ids.py` | `gateway_model_id()` / `no_thinking_gateway_model_id()` — encodes provider/model refs into gateway-safe IDs |
| `api/web_tools/outbound.py` | `_run_web_search()` (returns `list[dict[str, str]]` with `"url"` and `"title"` keys), `_run_web_fetch()` |
| `api/admin_config/` | Config persistence, validation, and status reporting |

### Initialization Flow

```
FastAPI app starts
  → init_mcp(provider_runtime, settings, messaging_runtime)
    → _emit_models_empty_if_needed()
      → McpEventBus.emit("models.empty") if no cached provider models
    → Module globals set: _provider_runtime, _settings, _messaging_runtime
```

`init_mcp()` is called from **both** the `lifespan` context manager and the `GracefulLifespanApp` path — each path initializes the MCP server, so `models.empty` may be emitted twice on startup (harmless duplicate; same data).

### Key Design Decisions

- **No `@mcp.notification()` decorator** — FastMCP 3.x removed this. Server-side notifications are driven entirely through `McpEventBus.emit()` and consumed by clients via `get_events` or the `fcc://events/latest` resource.
- **`ctx: Context` typing** — All tool functions accept `ctx: Context` from `fastmcp` (not `Any`). This allows FastMCP's dependency injection to exclude `ctx` from tool input schemas sent to clients.
- **`send_log_message()` uses `data=`** — FastMCP's API uses `data=` not `message=`.
- **`create_message()` uses `SamplingMessage` + `TextContent`** — Request body uses `messages=[SamplingMessage(role=..., content=TextContent(type="text", text=...))]`, parameters are snake_case.
- **`research` uses `"url"` not `"href"`** — `_run_web_search()` returns dicts with `"url"` and `"title"` keys. The `research` tool was previously buggy (accessing `sr["href"]`) and has been corrected.
- **`{param}` URI templates** — Resource templates use `{param}` not `{param:path}` per MCP specification.

---

## MCP Tools

There are **23 tools**, registered with `@mcp.tool()` in `api/mcp_server.py`.

### Model & Provider Discovery

| Tool | Signature | Description |
|---|---|---|
| `list_models` | `ctx: Context` | Returns all available model IDs (configured + cached + Claude-compatible variants). |
| `list_providers` | `ctx: Context` | Returns all providers with their configuration status. |
| `get_provider_info` | `ctx: Context, provider_id: str` | Returns detailed info (provider_id, display_name, transport_type, capabilities, credential_env, default_base_url) for a specific provider. |
| `refresh_models` | `ctx: Context` | Force-refreshes the model list cache from all configured providers. Emits `models.refreshed` or `models.refresh.failed`. |
| `test_provider` | `ctx: Context, provider_id: str` | Probes a provider's connectivity by listing its models. Emits `provider.error` on failure. |

### Configuration Management

| Tool | Signature | Description |
|---|---|---|
| `get_config` | `ctx: Context` | Returns all current admin settings (read-only preview, no side effects). |
| `validate_config` | `ctx: Context, values: dict[str, Any]` | Validates proposed config changes without applying them. |
| `apply_config` | `ctx: Context, values: dict[str, Any]` | Validates, applies, clears settings cache, re-creates `ProviderRuntime`, re-emits `config.changed` and `models.refreshed`. |

### Proxy Completion (Anthropic Messages API)

| Tool | Signature | Description |
|---|---|---|
| `agent` | `ctx: Context, model_id?, messages, system?, max_tokens?` | Single-turn conversation through the proxy. |
| `run_sub_agent` | `ctx: Context, model_id?, messages, system?, max_tokens?, temperature?, thinking?` | Multi-turn-capable conversation with extended parameters. |
| `think` | `ctx: Context, question: str, model_id?, system?` | Extended reasoning via the proxy model. Never delegates to client LLM. |
| `opinion` | `ctx: Context, task: str, goal: str, model_id?, temperature?, max_tokens?` | Evaluate an approach via the proxy model. Never delegates to client LLM. |

### Web Tools

| Tool | Signature | Description |
|---|---|---|
| `research` | `ctx: Context, query: str, max_results?, synthesize?, model_id?, use_sampling?` | Search + fetch + optional LLM synthesis. When `synthesize=True` and `use_sampling=True`, delegates to client's LLM via `create_message()`. Otherwise uses proxy. |
| `web_search` | `ctx: Context, query: str, max_results?` | Search the web via DuckDuckGo Lite (no API key). Returns `list[dict[str, str]]` with `"url"` and `"title"` keys. |
| `web_fetch` | `ctx: Context, url: str` | Fetch a URL and return sanitized text content. Respects `web_fetch_allow_private_networks` and `web_fetch_allowed_schemes` settings. |

### Messaging

| Tool | Signature | Description |
|---|---|---|
| `send_discord` | `ctx: Context, channel_id: str, message: str` | Send a message to a Discord channel via the messaging runtime. |
| `send_telegram` | `ctx: Context, chat_id: str, message: str` | Send a message to a Telegram chat via the messaging runtime. |

### MCP Sampling (Client LLM)

| Tool | Signature | Description |
|---|---|---|
| `sample` | `ctx: Context, prompt: str, system?, max_tokens?` | Have the client's own LLM sample a prompt directly. No proxy traffic. Uses `ctx.session.create_message()` with `SamplingMessage` + `TextContent`. |

### Monitoring & Utilities

| Tool | Signature | Description |
|---|---|---|
| `get_status` | `ctx: Context` | Server runtime status (host, port, model, provider, cached_models, provider_status). |
| `get_metrics` | `ctx: Context, provider_id?` | Usage metrics per provider (request_count, input/output tokens, last_request_at). |
| `reset_metrics` | `ctx: Context, provider_id?` | Reset metrics for one provider or all. Emits `metrics.reset`. |
| `get_events` | `ctx: Context` | Return the latest events emitted by the `McpEventBus`. |

---

## MCP Resources

Six resources are registered at `fcc://` URI templates:

| URI | Description |
|---|---|
| `fcc://status` | Server runtime status (host, port, model, provider). |
| `fcc://models` | List of all available model IDs. |
| `fcc://metrics` | Aggregated usage metrics for all providers. |
| `fcc://metrics/{provider_id}` | Usage metrics for a specific provider. |
| `fcc://events` | **Catalog** of all event types and their schemas (not the latest events — see below). |
| `fcc://events/latest` | Snapshot of the most recent events from the `McpEventBus`. |

> **Note:** `fcc://events` returns the *event type catalog* (static schema description), while `fcc://events/latest` returns the *live event snapshot* (most recent `EventRecord` per type).

All resource templates use `{param}` not `{param:path}`.

---

## MCP Events and Notifications

Events are emitted server-side via `McpEventBus.emit()` in `core/usage_tracker.py`. Clients consume them via the `get_events` tool or `fcc://events/latest` resource. There are **9 event types**:

| Event | When Emitted | Data Fields |
|---|---|---|
| `server.started` | MCP server initialized | — |
| `server.stopping` | MCP server shutting down | — |
| `models.refreshed` | `refresh_models()` completes successfully | — |
| `models.refresh.failed` | `refresh_models()` raises an exception | `error_type: str` |
| `models.empty` | At startup if no provider has cached models | `hint: str` ("Call refresh_models then list_models...") |
| `metrics.updated` | `UsageTracker.record()` is called | `provider_id`, `input_tokens`, `output_tokens` |
| `metrics.reset` | `reset_metrics()` is called | `provider_id?` or `reset_all: bool` |
| `config.changed` | `apply_config()` succeeds | `fields: list[str]` |
| `provider.error` | A provider operation fails | `provider_id`, `error_type` |

### `models.empty` Event

Emitted at startup by `_emit_models_empty_if_needed()` inside `init_mcp()`. Checks `runtime.cached_model_ids()`. If no provider has cached models, emits `models.empty` with a hint to call `refresh_models` then `list_models`. The event bus singleton (`api/mcp_events.py::get_event_bus()`) is shared across the codebase — all tools and the initialization path use the same instance.

---

## MCP Sampling — Client LLM Delegation

The MCP server supports **MCP sampling** — a mechanism by which the server asks the *connecting MCP client* to have its own LLM sample a prompt. This is useful when the proxy model is not needed or when the client's LLM should reason about data the proxy has access to.

Two tools support sampling:

- **`sample`** — Direct prompt to the client's LLM. Uses `ctx.session.create_message(messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))], ...)`.
- **`research`** with `use_sampling=True` — The synthesis step is delegated to the client LLM instead of the proxy.

**Critical constraint:** `think`, `opinion`, and the default `research` synthesis path (`use_sampling=False`) **never** delegate to the client LLM. They always route through the proxy provider.

MCP sampling requires the client to support the `create_message` capability. The `instructions` string sent in the `initialize` response documents this for clients.

---

## Model Variants and Catalog

### Gateway Model ID Encoding

Each provider/model reference is encoded into two gateway-safe model IDs:

| Prefix | Format | Purpose |
|---|---|---|
| `anthropic/` | `anthropic/{provider_id}/{provider_model}` | Normal variant — client may enable thinking |
| `claude-3-freecc-no-thinking/` | `claude-3-freecc-no-thinking/{provider_id}/{provider_model}` | Disables client-side thinking |

Encoding/decoding is in `api/gateway_model_ids.py`:

```python
gateway_model_id("nvidia/nemotron-3-super-120b-a12b")
# → "anthropic/nvidia/nemotron-3-super-120b-a12b"

no_thinking_gateway_model_id("openrouter/anthropic/claude-sonnet-4")
# → "claude-3-freecc-no-thinking/openrouter/anthropic/claude-sonnet-4"

decode_gateway_model_id("anthropic/nvidia/nemotron-3-super-120b-a12b")
# → DecodedGatewayModelId(provider_id="nvidia", provider_model="nemotron-3-super-120b-a12b", force_thinking_enabled=None)
```

### Model List Construction

`build_models_list_response()` in `api/model_catalog.py` builds the combined model list by:

1. Adding configured chat model refs with their two variants (thinking / no-thinking)
2. Adding all cached provider model infos with their two variants
3. Adding the 7 supported Claude compatibility models (claude-opus-4, claude-sonnet-4, etc.)

The resulting list is returned in `ModelsListResponse` format (`.data` is a list of `ModelResponse` objects with `.id` and `.display_name`). The `list_models` MCP tool serialises this as a JSON array of model IDs.

### Provider Runtime Caching

`_provider_runtime` is a module-level global in `mcp_server.py` set by `init_mcp()`. MCP tools access it via `_get_runtime()`, which raises `RuntimeError` if accessed before initialization.

---

## Docker Deployment

### Dockerfile

The project ships a `Dockerfile` that builds the application into a container. After building and running the container, the MCP server is accessible at `/mcp` inside the container network.

### Startup Sequence

1. Container starts; `main.py` runs FastAPI app with `GracefulLifespanApp`
2. On startup, `init_mcp()` is called, which:
   - Sets the module-level `_provider_runtime`, `_settings`, `_messaging_runtime` globals
   - Calls `_emit_models_empty_if_needed()` which may emit `models.empty`
3. `ProviderRuntime.refresh_model_list_cache()` is called to populate model cache
4. MCP SSE endpoints are ready at `/mcp/sse` and `/mcp/messages/`

### Connecting an MCP Client (e.g. OpenCode)

1. Start the Docker container
2. Restart the MCP client to pick up the new SSE connection
3. The client sends `initialize` — the server responds with name, version, capabilities, and the **instructions string** describing all tools, resources, and events

### Multi-Provider Configuration

When multiple providers are active (e.g., `nvidia_nim`, `open_router`, `mistral`, `zai`), each provider's models are cached separately and reflected in the combined model catalog. The `refresh_models` tool refreshes all provider caches at once.

> Please refer to the original [free-claude-code](https://github.com/Alishahryar1/free-claude-code) repository for more information about configuration, provider setup, and API key management.

---

## Configuration

Configuration is driven by environment variables documented in `.env.example`. Key variables:

| Variable | Description |
|---|---|
| `FCC_PROVIDERS` | Comma-separated list of provider IDs to enable |
| `NVIDIA_NIM_API_KEY`, `OPENROUTER_API_KEY`, etc. | Per-provider API keys |
| `FCC_MODEL` | Default model to use |
| `FCC_PORT` | HTTP server port |
| `FCC_ADMIN_TOKEN` | Admin UI authentication token |

Configuration is read by `config/settings.py::Settings` (a Pydantic `BaseSettings` subclass) and cached with `functools.lru_cache` / `cache_clear()`. The `apply_config` tool clears this cache and re-creates the `ProviderRuntime`.

---

## Metrics and Usage Tracking

`UsageTracker` in `core/usage_tracker.py` is a **thread-safe singleton** that records:

- `request_count` — number of proxy requests
- `total_input_tokens` / `total_output_tokens` — cumulative token counts
- `last_request_at` / `last_reset_at` — timestamps

It is updated by `_track_usage_stream()` in `provider_execution.py` after each streaming response completes. On every `record()` call, it emits a `metrics.updated` event via its internal `McpEventBus`.

Metrics are accessible via:
- `get_metrics` MCP tool
- `fcc://metrics` resource
- `fcc://metrics/{provider_id}` resource

`get_events` / `fcc://events/latest` show the latest `metrics.updated` event data per provider.

---

## CI / Quality Gates

All commits run through five parallel CI jobs (configured in `.github/workflows/tests.yml`):

| Check | Command | Purpose |
|---|---|---|
| **Ban type ignores** | Grep for `# type: ignore` / `# ty: ignore` | Zero-tolerance for suppressions |
| **ruff-format** | `ruff format --check` | Code formatting (py314 target) |
| **ruff-check** | `ruff check` | Linting (E, F, I, UP, B, C4, SIM, PERF, RUF) |
| **ty** | `ty check` | Static type checking (Python 3.14) |
| **pytest** | `pytest -v --tb=short` | Unit + contract tests (parallel with pytest-xdist) |

Locally, use `.\scripts\ci.ps1` (Windows) or `./scripts/ci.sh` (macOS/Linux) to run the full sequence. The scripts run `ruff format` / `ruff check --fix` first in repair mode, then type checking and tests in check-only mode.

---

## Repository Structure

```
fcc-proxy/
├── api/
│   ├── mcp_server.py          # FastMCP server: 23 tools, 6 resources, instructions
│   ├── mcp_events.py          # get_event_bus() singleton
│   ├── model_catalog.py       # build_models_list_response()
│   ├── gateway_model_ids.py   # gateway_model_id() / no_thinking_gateway_model_id()
│   ├── model_router.py        # ModelRouter — resolves model refs to providers
│   ├── provider_execution.py  # ProviderExecutionService, usage tracking
│   ├── models/                # Anthropic / OpenAI response models
│   ├── web_tools/
│   │   ├── outbound.py        # _run_web_search() (returns {"url", "title"}),
│   │   │                      # _run_web_fetch()
│   │   ├── egress.py          # WebFetchEgressPolicy
│   │   └── parsers.py         # SearchResultParser
│   └── admin_config/          # Config validation, persistence, status
├── core/
│   └── usage_tracker.py       # McpEventBus, UsageTracker, ProviderMetrics,
│                              # EventRecord
├── config/
│   ├── settings.py            # Pydantic Settings, get_settings() with cache_clear
│   ├── provider_catalog.py    # ProviderCatalog
│   └── model_refs.py          # configured_chat_model_refs(), parse_provider_type()
├── providers/
│   └── runtime.py             # ProviderRuntime — manages provider instances and cache
├── cli/
│   └── entrypoints.py         # serve(), fcc-server entrypoint
├── tests/
│   └── core/test_usage_tracker.py  # TestEmitModelsEmpty (4 tests)
├── smoke/                     # Live smoke tests (opt-in)
├── scripts/
│   ├── ci.sh / ci.ps1         # Local CI sequence
│   └── install.sh / install.ps1
├── pyproject.toml             # version 2.4.3 (fork — not updated)
├── .env.example
└── Dockerfile
```

---

## Licence

> **This is an  fork** of [free-claude-code](https://github.com/Alishahryar1/free-claude-code) repository. The original project is licensed under the MIT Licence.

The MIT Licence permits use, copying, modification, merging, publishing, distribution, sublicensing, and/or sale of copies of the Software, subject to the condition that the above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. I provide this fork under the same terms, as is, and without any warranty.