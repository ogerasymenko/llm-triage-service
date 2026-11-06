"""Stage 5 (surviving a bad provider) -- a stall must not become the caller's.

Red at checkout: the adapter does not exist and nothing bounds the wait.

The mock's timeout fault holds a connection for 30 seconds. A client with no
timeout waits all 30; one with REQUEST_TIMEOUT_S=8 gives up at 8 -- and must
give up for the whole logical request. Retrying a stall three times at 8 seconds
each is a 24-second wait wearing a resilience costume, so the budget below
bounds the entire call including retries.
"""

from __future__ import annotations

import time

import pytest

from app.reliability.retry import RetryExhaustedError


@pytest.mark.latency
async def test_stalled_upstream_returns_within_the_timeout_budget(
    make_service, mock_server, env
):
    """A 30s stall must surface as a failure in well under 30s."""
    env(
        FAULT_RATE="1.0",
        FAULT_MODE="timeout",
        TEMPERATURE="0.7",
        REQUEST_TIMEOUT_S="2.0",
        MAX_RETRIES="1",
    )
    service = make_service()
    mock_server.reset_stats()

    started = time.monotonic()
    with pytest.raises(RetryExhaustedError):
        await service.triage("this request will stall upstream for 30 seconds")
    elapsed = time.monotonic() - started

    # 2 attempts x 2s, plus backoff and overhead. The mock would hold for 30s.
    assert elapsed < 8.0, (
        f"a stalled upstream took {elapsed:.1f}s to fail. REQUEST_TIMEOUT_S bounds "
        f"one attempt; the retry budget must bound the whole request."
    )


@pytest.mark.latency
async def test_single_attempt_respects_request_timeout(make_service, mock_server, env):
    env(
        FAULT_RATE="1.0",
        FAULT_MODE="timeout",
        TEMPERATURE="0.7",
        REQUEST_TIMEOUT_S="1.5",
        MAX_RETRIES="0",
    )
    service = make_service()

    started = time.monotonic()
    with pytest.raises(RetryExhaustedError):
        await service.triage("single attempt against a stalling upstream")
    elapsed = time.monotonic() - started

    assert 0.5 < elapsed < 5.0, f"expected to give up near 1.5s, took {elapsed:.1f}s"


@pytest.mark.chaos
async def test_timeout_is_transient_and_gets_retried(make_service, mock_server, env):
    """A timeout is worth retrying -- and the retry is bounded by MAX_RETRIES."""
    env(
        FAULT_RATE="1.0",
        FAULT_MODE="timeout",
        TEMPERATURE="0.7",
        REQUEST_TIMEOUT_S="1.0",
        MAX_RETRIES="2",
    )
    service = make_service()
    mock_server.reset_stats()

    with pytest.raises(RetryExhaustedError):
        await service.triage("a stall should be retried, not given up on immediately")

    calls = mock_server.stats()["calls"]
    assert calls == 3, f"MAX_RETRIES=2 should give 3 attempts against a stall, saw {calls}"


@pytest.mark.chaos
async def test_recovers_when_the_stall_clears(make_service, mock_server, env):
    """A transient stall followed by a healthy upstream must succeed.

    FAULT_RATE=0.5 with FAULT_MODE=timeout means some attempts stall and some
    do not, which is what a real degraded dependency looks like.
    """
    env(
        FAULT_RATE="0.5",
        FAULT_MODE="timeout",
        TEMPERATURE="0.7",
        REQUEST_TIMEOUT_S="1.0",
        MAX_RETRIES="5",
    )
    service = make_service()
    mock_server.reset_stats()

    started = time.monotonic()
    result, _ = await service.triage("Checkout stalls intermittently under load.")
    elapsed = time.monotonic() - started

    assert result.component
    assert elapsed < 10.0, f"recovery took {elapsed:.1f}s"
