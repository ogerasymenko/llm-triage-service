"""Stage 6 (degrade and report) -- the circuit breaker. GREEN at checkout.

breaker.py ships complete, so this file passes before you have written a line of
stage 1. It is the worked example: the shape the modules you write in stages 1-5
should end up with.

Everything here exercises breaker.py alone, which is why it stays green while
the adapter and the retry layer are still stubs. The tests that check the same
component through the full stack live in support/test_breaker_integration.py.
"""

from __future__ import annotations

import asyncio

import pytest

from app.providers.base import PermanentProviderError, TransientProviderError
from app.reliability.breaker import CircuitBreaker, CircuitOpenError
from app.reliability.retry import RetryPolicy, with_retry


async def test_opens_after_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=3, recovery_time_s=30.0)
    assert breaker.state == "closed"

    for _ in range(3):
        await breaker.record_failure()

    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError) as excinfo:
        await breaker.before_call()
    assert excinfo.value.retry_after > 0


async def test_success_resets_the_failure_count():
    """Consecutive means consecutive: one success clears the tally."""
    breaker = CircuitBreaker(failure_threshold=3)
    await breaker.record_failure()
    await breaker.record_failure()
    await breaker.record_success()
    await breaker.record_failure()

    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 1


async def test_half_open_lets_exactly_one_trial_through():
    breaker = CircuitBreaker(failure_threshold=2, recovery_time_s=0.05)
    await breaker.record_failure()
    await breaker.record_failure()
    assert breaker.state == "open"

    await asyncio.sleep(0.08)
    await breaker.before_call()          # transitions to half_open
    assert breaker.state == "half_open"

    await breaker.record_success()
    assert breaker.state == "closed"


async def test_failed_trial_reopens_the_circuit():
    breaker = CircuitBreaker(failure_threshold=2, recovery_time_s=0.05)
    await breaker.record_failure()
    await breaker.record_failure()
    await asyncio.sleep(0.08)
    await breaker.before_call()
    assert breaker.state == "half_open"

    await breaker.record_failure()
    assert breaker.state == "open", "a failed trial call must re-open, not close"


@pytest.mark.parametrize("threshold", [1, 3, 5])
async def test_threshold_is_respected(threshold):
    breaker = CircuitBreaker(failure_threshold=threshold)
    for i in range(threshold - 1):
        await breaker.record_failure()
        assert breaker.state == "closed", f"opened early at failure {i + 1}"
    await breaker.record_failure()
    assert breaker.state == "open"
