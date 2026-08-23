"""Entry point: `python -m llm_fabric` or the `llm-fabric` console script."""

from __future__ import annotations

import uvicorn

from llm_fabric.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "llm_fabric.gateway.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
