# syntax=docker/dockerfile:1.7
# Production image for the MyVista LLM Fabric gateway.
# Secrets are never baked in. Configuration is environment-driven.

FROM python:3.12-slim-bookworm AS build
WORKDIR /src
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin fabric
WORKDIR /app
COPY --from=build /src/.venv /app/.venv
COPY --from=build /src/src /app/src
COPY --from=build /src/config /app/config
COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER 10001
EXPOSE 47317
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:47317/healthz')"
STOPSIGNAL SIGTERM
CMD ["python", "-m", "llm_fabric"]
