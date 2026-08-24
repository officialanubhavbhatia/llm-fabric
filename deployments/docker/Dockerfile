# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS dependency-builder
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /usr/local/bin/
WORKDIR /build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

FROM dependency-builder AS application-build
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable \
    && .venv/bin/python -c \
      "from importlib.resources import files; assert files('llm_fabric.gateway').joinpath('static/command_center.html').is_file()"

FROM python:3.12-slim-bookworm AS runtime
ARG VERSION=0.1.0
ARG REVISION=unknown
ARG SOURCE=https://github.com/officialanubhavbhatia/llm-fabric
LABEL org.opencontainers.image.title="MyVista LLM Fabric" \
      org.opencontainers.image.description="Cloud-neutral LLM routing and policy gateway" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="${SOURCE}" \
      org.opencontainers.image.licenses="Proprietary"
RUN groupadd --gid 10001 fabric \
    && useradd --no-create-home --uid 10001 --gid 10001 --shell /usr/sbin/nologin fabric
WORKDIR /app
COPY --from=application-build --chown=10001:10001 /build/.venv /app/.venv
COPY --from=application-build --chown=10001:10001 /build/alembic.ini /app/alembic.ini
COPY --from=application-build --chown=10001:10001 /build/alembic /app/alembic
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    TMPDIR=/tmp
USER 10001:10001
EXPOSE 47317
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:47317/healthz')"
STOPSIGNAL SIGTERM
CMD ["python", "-m", "llm_fabric"]
