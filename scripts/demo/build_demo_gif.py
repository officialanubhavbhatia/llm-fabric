#!/usr/bin/env python3
"""Assemble the README demo GIF from real Command Center PNG captures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parents[2] / "docs" / "assets"
ORDER = (
    "sdk-request.png",
    "command-center.png",
    "command-center-requests.png",
    "command-center-traces.png",
    "command-center-intents.png",
    "command-center-context.png",
    "command-center-models.png",
    "command-center-kv_cache.png",
)
WIDTH = 960
DURATION_MS = 3500


def main() -> None:
    frames = []
    for name in ORDER:
        path = ASSETS / name
        if not path.is_file():
            raise SystemExit(f"missing capture {path}")
        image = Image.open(path).convert("RGB")
        width, height = image.size
        frames.append(
            image.resize((WIDTH, int(height * WIDTH / width)), Image.Resampling.LANCZOS).convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=64
            )
        )
    out = ASSETS / "myvista-demo.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(out, out.stat().st_size, "bytes", len(frames), "frames", len(frames) * DURATION_MS, "ms")


if __name__ == "__main__":
    main()
