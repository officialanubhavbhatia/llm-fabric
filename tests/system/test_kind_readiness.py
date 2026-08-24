"""Kind/Helm Ready → NotReady → Ready. Skipped unless LLM_FABRIC_KIND_TEST=1."""

from __future__ import annotations

import os
import subprocess
import time

import pytest

pytestmark = [
    pytest.mark.system,
    pytest.mark.skipif(
        os.environ.get("LLM_FABRIC_KIND_TEST") != "1",
        reason="kind readiness test requires LLM_FABRIC_KIND_TEST=1 and a live cluster",
    ),
]


def _kubectl(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", "--context", "kind-llm-fabric", "-n", "llm-fabric", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _pod_name() -> str:
    result = _kubectl(
        "get",
        "pods",
        "-l",
        "app.kubernetes.io/name=llm-fabric",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    )
    assert result.returncode == 0, result.stderr
    name = result.stdout.strip()
    assert name, result.stderr or result.stdout
    return name


def _ready(pod: str) -> bool:
    result = _kubectl(
        "get",
        "pod",
        pod,
        "-o",
        'jsonpath={.status.conditions[?(@.type=="Ready")].status}',
    )
    return result.stdout.strip() == "True"


def _exec(pod: str, *command: str) -> subprocess.CompletedProcess[str]:
    return _kubectl("exec", pod, "--", *command, timeout=20)


def test_kind_pod_transitions_notready_on_postgres_outage_and_recovers() -> None:
    pod = _pod_name()
    assert _ready(pod), f"{pod} was not Ready at start"
    live = _exec(
        pod,
        "python",
        "-c",
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:47317/healthz')",
    )
    assert live.returncode == 0, live.stderr

    postgres = _kubectl(
        "get",
        "pods",
        "-l",
        "app=postgres",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    )
    if postgres.returncode != 0 or not postgres.stdout.strip():
        postgres = _kubectl(
            "get",
            "pods",
            "-l",
            "app.kubernetes.io/name=postgres",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        )
    pg = postgres.stdout.strip()
    assert pg, "could not find a postgres pod to isolate"

    scaled = _kubectl("scale", "deploy/postgres", "--replicas=0")
    assert scaled.returncode == 0, scaled.stderr
    try:
        started = time.monotonic()
        deadline = time.time() + 60
        became_not_ready = False
        while time.time() < deadline:
            if not _ready(pod):
                became_not_ready = True
                break
            time.sleep(1)
        not_ready_s = time.monotonic() - started
        assert became_not_ready, f"{pod} stayed Ready after postgres isolation"
        print(f"kind_not_ready_s={not_ready_s:.3f} pod={pod}")
        # Process must still be the same pod (not a restart storm).
        still = _kubectl("get", "pod", pod, "-o", "jsonpath={.metadata.name}")
        assert still.stdout.strip() == pod
        restarts = _kubectl(
            "get",
            "pod",
            pod,
            "-o",
            "jsonpath={.status.containerStatuses[0].restartCount}",
        )
        assert int(restarts.stdout.strip() or "99") == 0
        healthz = _exec(
            pod,
            "python",
            "-c",
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:47317/healthz')",
        )
        assert healthz.returncode == 0, healthz.stderr
        readyz = _exec(
            pod,
            "python",
            "-c",
            (
                "import urllib.request; "
                "r=urllib.request.urlopen('http://127.0.0.1:47317/readyz'); "
                "raise SystemExit(0)"
            ),
        )
        # urlopen raises on 503; that is the expected NotReady body.
        assert readyz.returncode != 0
    finally:
        _kubectl("scale", "deploy/postgres", "--replicas=1")
        recover_started = time.monotonic()
        deadline = time.time() + 90
        while time.time() < deadline:
            if _ready(pod):
                print(f"kind_ready_again_s={time.monotonic() - recover_started:.3f} pod={pod}")
                break
            time.sleep(2)
        else:
            pytest.fail(f"{pod} did not return to Ready after postgres recovered")
    assert _pod_name()  # still discoverable
    assert _ready(pod)
