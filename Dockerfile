FROM python:3.14-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md .env.example ./
COPY api/ api/
COPY cli/ cli/
COPY config/ config/
COPY core/ core/
COPY messaging/ messaging/
COPY providers/ providers/
COPY server.py .

RUN uv sync --frozen --no-dev

FROM python:3.14-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8082/admin', timeout=5)" || exit 1

CMD ["fcc-server"]
