#!/usr/bin/env python3
"""Capture real Command Center screenshots via Chrome DevTools Protocol.

Requires Google Chrome. Does not fabricate dashboard values.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

BASE_URL = os.environ.get("MYVISTA_BASE_URL", "http://127.0.0.1:47317").rstrip("/")
OUT = Path(__file__).resolve().parents[2] / "docs" / "assets"
VIEWS = (
    "overview",
    "requests",
    "traces",
    "intents",
    "routing",
    "economics",
    "reliability",
)
CHROME = os.environ.get(
    "CHROME_BIN",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
PORT = int(os.environ.get("CDP_PORT", "9229"))


async def cdp(ws, method: str, params: dict | None = None, session_id: str | None = None) -> dict:
    payload = {"id": int(time.time() * 1000) % 1_000_000, "method": method, "params": params or {}}
    if session_id:
        payload["sessionId"] = session_id
    await ws.send(json.dumps(payload))
    while True:
        message = json.loads(await ws.recv())
        if message.get("id") == payload["id"]:
            if "error" in message:
                raise RuntimeError(f"{method}: {message['error']}")
            return message.get("result") or {}


async def capture() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=5))
    page = next((item for item in pages if item.get("type") == "page"), None)
    if page is None:
        raise SystemExit("no Chrome page target")
    paths: list[Path] = []
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=20_000_000) as ws:
        await cdp(ws, "Page.enable")
        await cdp(ws, "Runtime.enable")
        await cdp(ws, "Page.navigate", {"url": f"{BASE_URL}/command-center"})
        await asyncio.sleep(2.5)
        for view in VIEWS:
            await cdp(
                ws,
                "Runtime.evaluate",
                {"expression": f"load({view!r})", "awaitPromise": True},
            )
            await asyncio.sleep(1.2)
            shot = await cdp(ws, "Page.captureScreenshot", {"format": "png", "fromSurface": True})
            name = "command-center.png" if view == "overview" else f"command-center-{view}.png"
            dest = OUT / name
            dest.write_bytes(base64.b64decode(shot["data"]))
            paths.append(dest)
            print(dest, dest.stat().st_size)
    return paths


def start_chrome(profile: Path) -> subprocess.Popen:
    if not Path(CHROME).exists():
        raise SystemExit(f"Chrome not found: {CHROME}")
    return subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--window-size=1440,900",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={profile}",
            f"{BASE_URL}/command-center",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    profile = Path(tempfile.mkdtemp(prefix="myvista-cc-"))
    proc = start_chrome(profile)
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1).read()
                break
            except OSError:
                time.sleep(0.25)
        else:
            raise SystemExit("Chrome DevTools port did not open")
        asyncio.run(capture())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
