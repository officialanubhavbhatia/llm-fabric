"""Kind rolling restart during an in-cluster SSE stream.

Skipped unless LLM_FABRIC_KIND_TEST=1. This is the cluster proof; the local
SIGTERM drain is tests/unit/test_stream_shutdown_live.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import pytest

pytestmark = [
    pytest.mark.system,
    pytest.mark.skipif(
        os.environ.get("LLM_FABRIC_KIND_TEST") != "1",
        reason="kind rolling-stream test requires LLM_FABRIC_KIND_TEST=1",
    ),
]


def _kubectl(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", "--context", "kind-llm-fabric", "-n", "llm-fabric", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _api_key() -> str:
    import base64

    result = _kubectl(
        "get",
        "secret",
        "llm-fabric",
        "-o",
        "jsonpath={.data.LLM_FABRIC_API_KEYS}",
    )
    assert result.returncode == 0, result.stderr
    raw = base64.b64decode(result.stdout.strip() or "Cg==").decode("utf-8")
    key = raw.strip().split(",")[0].strip()
    assert len(key) >= 16, "kind secret LLM_FABRIC_API_KEYS is missing"
    return key


def _gateway_image() -> str:
    result = _kubectl(
        "get",
        "deploy",
        "llm-fabric",
        "-o",
        "jsonpath={.spec.template.spec.containers[0].image}",
    )
    assert result.returncode == 0, result.stderr
    image = result.stdout.strip()
    assert image, result.stderr
    return image


def test_kind_rolling_restart_drains_in_cluster_stream() -> None:
    delay = _kubectl("set", "env", "deploy/llm-fabric", "LLM_FABRIC_MOCK_DELAY_S=2")
    assert delay.returncode == 0, delay.stderr
    status = _kubectl("rollout", "status", "deploy/llm-fabric", "--timeout=180s", timeout=200)
    assert status.returncode == 0, status.stderr + status.stdout

    key = _api_key()
    models = _kubectl(
        "exec",
        "deploy/llm-fabric",
        "--",
        "python",
        "-c",
        (
            "import json,os,urllib.request;"
            "req=urllib.request.Request('http://127.0.0.1:47317/v1/models',"
            f"headers={{'Authorization':'Bearer {key}'}});"
            "print(urllib.request.urlopen(req,timeout=10).read().decode())"
        ),
        timeout=30,
    )
    assert models.returncode == 0, models.stderr
    payload = json.loads(models.stdout)
    model_id = payload["data"][0]["id"]

    client_name = f"stream-client-{uuid.uuid4().hex[:8]}"
    script = (
        "import json,sys,urllib.request;"
        f"key={key!r};"
        f"model={model_id!r};"
        "body=json.dumps({'model':model,'stream':True,"
        "'messages':[{'role':'user','content':'alpha beta gamma delta'}]}).encode();"
        "req=urllib.request.Request('http://llm-fabric:47317/v1/chat/completions',"
        "data=body,method='POST',"
        "headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'});"
        "raw=urllib.request.urlopen(req,timeout=90).read().decode();"
        "sys.stdout.write(raw);"
        "sys.exit(0 if '[DONE]' in raw else 1)"
    )
    started = _kubectl(
        "run",
        client_name,
        f"--image={_gateway_image()}",
        "--image-pull-policy=IfNotPresent",
        "--restart=Never",
        "--command",
        "--",
        "python",
        "-c",
        script,
        timeout=30,
    )
    assert started.returncode == 0, started.stderr + started.stdout
    try:
        time.sleep(2)
        restart = _kubectl("rollout", "restart", "deploy/llm-fabric")
        assert restart.returncode == 0, restart.stderr
        rollout = _kubectl("rollout", "status", "deploy/llm-fabric", "--timeout=180s", timeout=200)
        assert rollout.returncode == 0, rollout.stderr + rollout.stdout
        phase = ""
        deadline = time.time() + 90
        while time.time() < deadline:
            got = _kubectl(
                "get",
                "pod",
                client_name,
                "-o",
                "jsonpath={.status.phase}",
            )
            phase = got.stdout.strip()
            if phase in {"Succeeded", "Failed"}:
                break
            time.sleep(1)
        logs = _kubectl("logs", client_name, timeout=30)
        assert phase == "Succeeded", f"phase={phase} logs={logs.stdout} err={logs.stderr}"
        assert "[DONE]" in logs.stdout
    finally:
        _kubectl("delete", "pod", client_name, "--ignore-not-found=true", "--wait=false")
