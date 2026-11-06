"""The breaker through the full stack, rather than on its own.

These depend on the adapter and the retry layer, so at checkout they would be red
for someone else's reason -- which is why they are not in test_breaker.py, a
stage 6 file the spec ships green.
"""

from __future__ import annotations

import asyncio

import pytest

from app.providers.base import PermanentProviderError, TransientProviderError
from app.reliability.breaker import CircuitBreaker, CircuitOpenError
from app.reliability.retry import RetryPolicy, with_retry
from tests.conftest import requires_adapter

pytestmark = requires_adapter



@pytest.mark.chaos
async def test_open_breaker_stops_calls_reaching_upstream(make_service, mock_server, env):
    """The point of the whole thing: upstream sees nothing while open."""
    env(FAULT_RATE="1.0", FAULT_MODE="rate_limit", TEMPERATURE="0.7", MAX_RETRIES="2")
    service = make_service()
    service.breaker.failure_threshold = 3
    service.breaker.recovery_time_s = 60.0
    mock_server.reset_stats()

    from app.reliability.retry import RetryExhaustedError

    with pytest.raises((RetryExhaustedError, CircuitOpenError)):
        await service.triage("first failing request")

    calls_after_first = mock_server.stats()["calls"]
    assert service.breaker.state == "open"

    with pytest.raises(CircuitOpenError):
        await service.triage("second request, breaker should be open")

    assert mock_server.stats()["calls"] == calls_after_first, (
        "a call reached upstream while the breaker was open"
    )


async def test_permanent_errors_do_not_trip_the_breaker():
    """One bad input must not take the service down for everyone."""
    breaker = CircuitBreaker(failure_threshold=2)
    policy = RetryPolicy(max_retries=2, jitter=False, base_delay_s=0.0)

    async def permanently_bad():
        raise PermanentProviderError("this input is malformed")

    for _ in range(5):
        with pytest.raises(PermanentProviderError):
            await with_retry(permanently_bad, policy=policy, breaker=breaker)

    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 0


async def test_open_breaker_stops_the_retry_loop():
    breaker = CircuitBreaker(failure_threshold=2)
    policy = RetryPolicy(max_retries=10, jitter=False, base_delay_s=0.0)
    attempts = 0

    async def always_transient():
        nonlocal attempts
        attempts += 1
        raise TransientProviderError("upstream is down")

    with pytest.raises(CircuitOpenError):
        await with_retry(always_transient, policy=policy, breaker=breaker)

    assert attempts == 2, (
        f"the retry loop made {attempts} attempts against an open breaker; it should "
        f"stop at the threshold (2), not run its full budget of 11"
    )


async def test_api_returns_503_with_retry_after(app_client, make_service, env):
    """The HTTP surface of an open breaker: 503 carrying Retry-After.

    A 503 without Retry-After tells the caller to back off but not for how
    long, so every client invents its own answer and they all come back at
    once. The breaker knows exactly when it will next admit a call, so it says.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    client, service = app_client

    service.breaker.failure_threshold = 2
    service.breaker.recovery_time_s = 30.0
    await service.breaker.record_failure()
    await service.breaker.record_failure()
    assert service.breaker.state == "open"

    response = await client.post("/triage", json={"text": "anything at all"})

    assert response.status_code == 503
    assert response.json()["error"] == "circuit_open"
    retry_after = response.headers.get("retry-after")
    assert retry_after is not None, "503 from an open breaker must carry Retry-After"
    assert int(retry_after) >= 1
