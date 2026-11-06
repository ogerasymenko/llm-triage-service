"""Goes green with stage 1. The routes are thin, but the mapping from failure to
status code is the contract telling a caller what to do next -- 503 try later,
402 raise a limit, 422 do not retry. Collapsing them into 500 makes every caller
guess."""

from __future__ import annotations

import json

import pytest

from app.schema import TriageResult
from tests.conftest import SAMPLE_TEXT
from tests.conftest import requires_adapter

pytestmark = requires_adapter


async def test_health_is_green(app_client, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    client, _ = app_client
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "anthropic"


@pytest.mark.chaos
async def test_triage_returns_a_valid_result(app_client, env):
    env(FAULT_RATE="0.4", TEMPERATURE="0.7")
    client, _ = app_client

    response = await client.post("/triage", json={"text": SAMPLE_TEXT})

    assert response.status_code == 200, response.text
    result = TriageResult.model_validate(response.json())
    assert result.component
    assert response.headers["x-idempotent-replay"] == "false"


async def test_idempotency_key_header_replays_the_same_bytes(app_client, mock_server, env):
    """The HTTP contract for idempotency, including the replay header."""
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    client, _ = app_client
    mock_server.reset_stats()

    headers = {"idempotency-key": "http-order-77"}
    first = await client.post("/triage", json={"text": SAMPLE_TEXT}, headers=headers)
    second = await client.post("/triage", json={"text": SAMPLE_TEXT}, headers=headers)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.headers["x-idempotent-replay"] == "false"
    assert second.headers["x-idempotent-replay"] == "true"
    assert mock_server.stats()["calls"] == 1


async def test_empty_text_is_rejected_by_validation(app_client, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    client, _ = app_client
    response = await client.post("/triage", json={"text": ""})
    assert response.status_code == 422


async def test_unknown_field_is_rejected(app_client, env):
    """`extra="forbid"` on the request too: typos should be loud."""
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    client, _ = app_client
    response = await client.post("/triage", json={"text": "hello", "urgent": True})
    assert response.status_code == 422


async def test_spend_ceiling_surfaces_as_402(app_client, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    client, service = app_client
    service.limits.ledger.ceiling_usd = service.limits.ledger.worst_case_usd() * 0.5

    response = await client.post("/triage", json={"text": SAMPLE_TEXT})

    assert response.status_code == 402
    assert response.json()["error"] == "spend_ceiling_exceeded"


@pytest.mark.chaos
async def test_exhausted_retries_surface_as_503_with_retry_after(app_client, env):
    env(FAULT_RATE="1.0", FAULT_MODE="rate_limit", TEMPERATURE="0.7", MAX_RETRIES="1")
    client, _ = app_client

    response = await client.post("/triage", json={"text": "this will never succeed"})

    assert response.status_code == 503
    assert response.headers.get("retry-after") is not None
    assert response.json()["error"] == "upstream_unavailable"


async def test_stream_endpoint_emits_sse(app_client, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    client, _ = app_client

    async with client.stream(
        "POST", "/triage/stream", json={"text": SAMPLE_TEXT}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert "event: start" in body
    assert "event: delta" in body
    assert "event: done" in body

    final = [
        line[len("data: "):]
        for line in body.splitlines()
        if line.startswith("data: ")
    ][-1]
    payload = json.loads(final)
    assert payload["type"] == "done"
    TriageResult.model_validate(payload["result"])


async def test_stream_truncation_ends_in_an_error_event(app_client, env):
    """No partial object over HTTP either."""
    env(FAULT_RATE="1.0", FAULT_MODE="truncated_stream", TEMPERATURE="0.7")
    client, _ = app_client

    async with client.stream(
        "POST", "/triage/stream", json={"text": SAMPLE_TEXT}
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert "event: error" in body
    assert "event: done" not in body


async def test_limits_endpoint_reports_the_configuration(app_client, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_CONCURRENCY="4", MAX_RETRIES="3")
    client, _ = app_client
    response = await client.get("/limits")
    assert response.status_code == 200
    body = response.json()
    assert body["max_concurrency"] == 4
    assert body["max_retries"] == 3
    assert body["breaker_state"] == "closed"
