"""The mock's rate limit is what the stage-2 rate tests read, so it is checked
here directly over HTTP, with no adapter in the way. A throttle that miscounts
would fail a correct limiter, or pass a missing one, with equal confidence."""

from __future__ import annotations

import time

import httpx
import pytest


def _body(i: int, *, stream: bool = False) -> dict:
    return {
        "model": "claude-haiku-4-5",
        "max_tokens": 256,
        "stream": stream,
        "messages": [{"role": "user", "content": f"Throttle probe {i}: the export fails."}],
    }


def _post(mock_server, i: int) -> httpx.Response:
    return httpx.post(f"{mock_server.base_url}/v1/messages", json=_body(i), timeout=10.0)


def test_the_call_past_the_limit_is_refused_with_retry_after(mock_server, env):
    env(FAULT_RATE="0.0", MOCK_RATE_LIMIT_REQUESTS="2", MOCK_RATE_LIMIT_WINDOW_S="1.0")
    mock_server.reset_stats()

    codes = [_post(mock_server, i).status_code for i in range(2)]
    refused = _post(mock_server, 2)

    assert codes == [200, 200]
    assert refused.status_code == 429
    assert refused.json()["error"]["type"] == "rate_limit_error"
    assert refused.headers["retry-after"] == "1"

    stats = mock_server.stats()
    assert stats["calls"] == 3, "a throttled call still reached upstream and counts"
    assert stats["calls_by_outcome"] == {"ok": 2, "throttled": 1}


def test_waiting_out_the_window_is_enough(mock_server, env):
    """Refusals do not occupy the window, so a client that waits gets back in."""
    env(FAULT_RATE="0.0", MOCK_RATE_LIMIT_REQUESTS="2", MOCK_RATE_LIMIT_WINDOW_S="1.0")
    mock_server.reset_stats()

    for i in range(2):
        assert _post(mock_server, i).status_code == 200
    for i in range(2, 5):
        assert _post(mock_server, i).status_code == 429

    time.sleep(1.0)
    assert _post(mock_server, 5).status_code == 200


def test_the_openai_dialect_shares_the_limit(mock_server, env):
    """One provider, one limit: two dialects must not each get their own."""
    env(FAULT_RATE="0.0", MOCK_RATE_LIMIT_REQUESTS="1", MOCK_RATE_LIMIT_WINDOW_S="1.0")
    mock_server.reset_stats()

    assert _post(mock_server, 0).status_code == 200
    refused = httpx.post(
        f"{mock_server.base_url}/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "probe"}]},
        timeout=10.0,
    )
    assert refused.status_code == 429
    assert refused.json()["error"]["type"] == "rate_limit_exceeded"
    assert refused.headers["retry-after"] == "1"


@pytest.mark.parametrize("value", ["0", ""])
def test_no_limit_means_no_throttling(mock_server, env, value):
    env(FAULT_RATE="0.0", MOCK_RATE_LIMIT_REQUESTS=value, MOCK_RATE_LIMIT_WINDOW_S="1.0")
    mock_server.reset_stats()

    assert all(_post(mock_server, i).status_code == 200 for i in range(6))
    assert "throttled" not in mock_server.stats()["calls_by_outcome"]
