"""Stage 5 (surviving a bad provider) -- retry, and the rule that exactly ONE
layer may do it.

Red at checkout: app/reliability/retry.py is a stub.

The anthropic SDK ships max_retries=2. Write a retry loop, leave that alone, and
one request becomes up to (1 + MAX_RETRIES) x 3 upstream calls -- twelve at the
shipped defaults -- while MAX_RETRIES looks inert because the SDK is tripling it.

That bug does not change the outcome. The request still succeeds; it just costs
three times as much and hammers the upstream you were being careful with. So
every assertion here counts the mock's calls rather than checking the result. A
test that only checks the outcome passes just as happily with two retry layers,
which makes it worse than no test.
"""

from __future__ import annotations

import asyncio

import pytest

from app.reliability.retry import RetryExhaustedError
from app.schema import TriageResult
from tests.conftest import SAMPLE_TEXT, SAMPLES_DIR

# Distinct text per test so each gets its own fault sequence and its own
# cache entry -- a shared body would let one test's cached result satisfy
# another and hide a real regression.
RATE_LIMITED_TEXT = "Payment capture returns 429 under load on the checkout path."


@pytest.mark.chaos
async def test_succeeds_under_chaos_within_the_retry_budget(make_service, mock_server, env):
    """At FAULT_RATE=0.4 the request goes through, and the cost is bounded."""
    env(FAULT_RATE="0.4", TEMPERATURE="0.7", MAX_RETRIES="3")
    service = make_service()
    mock_server.reset_stats()

    result, cached = await service.triage(SAMPLE_TEXT)

    assert cached is False
    assert result.component
    calls = mock_server.stats()["calls"]
    assert calls <= 1 + 3, (
        f"one logical request cost {calls} upstream calls with MAX_RETRIES=3; "
        f"the ceiling is 1 + MAX_RETRIES = 4"
    )


@pytest.mark.parametrize("max_retries", [0, 1, 2, 3])
@pytest.mark.chaos
async def test_max_retries_is_the_only_thing_controlling_attempts(
    make_service, mock_server, env, max_retries
):
    """MAX_RETRIES must bound upstream calls exactly -- at every value.

    This is the assertion that catches a second retry layer. With the SDK also
    retrying, MAX_RETRIES=0 still produces 3 upstream calls, so the
    `max_retries=0` case fails loudly and specifically.

    `FAULT_MODE=rate_limit` at `FAULT_RATE=1.0` guarantees every attempt
    fails, so the observed count is the attempt budget itself rather than
    however many attempts luck happened to require. Note this makes the fault
    worse, not milder -- and it is set through the environment, not through a
    request header.
    """
    env(
        FAULT_RATE="1.0",
        FAULT_MODE="rate_limit",
        TEMPERATURE="0.7",
        MAX_RETRIES=str(max_retries),
    )
    service = make_service()
    mock_server.reset_stats()

    with pytest.raises(RetryExhaustedError):
        await service.triage(RATE_LIMITED_TEXT + f" case={max_retries}")

    calls = mock_server.stats()["calls"]
    expected = 1 + max_retries
    assert calls == expected, (
        f"MAX_RETRIES={max_retries} should produce exactly {expected} upstream call(s), "
        f"saw {calls}. More than that means a second layer is retrying too -- check "
        f"that the anthropic client is built with max_retries=0."
    )


async def test_sdk_retries_are_disabled(make_service):
    """The adapter's SDK client must not retry on its own.

    Asserted directly as well as by call count, because this is the single
    setting the whole arrangement depends on, and a direct check names the fix.
    """
    service = make_service()
    assert service.provider.client.max_retries == 0, (
        "the anthropic client is configured to retry; with app-level retry also "
        "active that multiplies attempts. Exactly one layer may retry."
    )


async def test_permanent_errors_are_not_retried(make_service, mock_server, env):
    """A permanent failure must cost exactly one call.

    Retrying a 400 or a schema violation cannot help: the same request fails
    the same way. Four identical failures cost four times as much and tell you
    nothing new.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_RETRIES="3", MAX_TOKENS="1024")
    service = make_service()
    mock_server.reset_stats()

    from app.providers.base import PermanentProviderError

    async def permanent(*args, **kwargs):
        raise PermanentProviderError("schema violation")

    service.provider.extract_with_usage = permanent

    with pytest.raises(PermanentProviderError):
        await service.triage("some text that will fail permanently")

    assert mock_server.stats()["calls"] == 0


@pytest.mark.chaos
async def test_retries_observed_matches_our_own_count(make_service, mock_server, env):
    """/stats must agree with the client about how many retries happened.

    `retries_observed` counts repeat calls carrying the same identity. If our
    retry budget is 4 attempts and all 4 fail, the mock should see 4 calls of
    which 3 are retries.
    """
    env(FAULT_RATE="1.0", FAULT_MODE="rate_limit", TEMPERATURE="0.7", MAX_RETRIES="3")
    service = make_service()
    mock_server.reset_stats()

    with pytest.raises(RetryExhaustedError):
        await service.triage("stats agreement check for retries")

    stats = mock_server.stats()
    assert stats["calls"] == 4
    assert stats["retries_observed"] == 3, (
        f"the mock counted {stats['retries_observed']} retries for 4 calls of one "
        f"request; expected 3"
    )


@pytest.mark.chaos
async def test_every_sample_survives_chaos(make_service, env):
    """Stage 1's samples again, now at FAULT_RATE=0.4.

    Stage 1 checked extraction with no faults, because surviving them is this
    stage's job. One request per sample, so a fault that retry fails to absorb
    fails the test.
    """
    env(FAULT_RATE="0.4", TEMPERATURE="0.7", MAX_RETRIES="5")
    service = make_service()

    for sample in sorted(SAMPLES_DIR.glob("*.txt")):
        result, _ = await service.triage(sample.read_text())
        assert isinstance(result, TriageResult), sample.name


REPORTS = [
    "Login fails with a 500 for every user after the deploy.",
    "Refund is charged twice when the button is pressed twice.",
    "The settings modal is misaligned by 40px on the dashboard.",
    "Order page takes 9 seconds intermittently, possibly a deadlock.",
    "Search returns unrelated articles for two-word queries.",
    "Comment notification emails arrive up to 20 minutes late.",
    "Uploading a 200MB attachment fails at the very end.",
    "Worker cannot connect to postgres, OperationalError in the log.",
]


@pytest.mark.chaos
@pytest.mark.latency
async def test_retries_count_against_the_rate_limit(make_service, mock_server, env):
    """A retry is a request like any other; the provider counts it.

    Stage 2's rate limiter, now with retries in flight. `truncated_stream` on a
    unary call is a fast 500, so retries pile up right behind the first
    attempts, and each one has to take a slot of its own.

    MAX_RETRIES is raised because surviving is not what this test is about: at
    the default of 3, one of these bodies draws four faults in a row and the
    run dies on that before it can say anything about the limit. More retries
    also means more pressure on the limiter, not less.
    """
    env(
        RATE_LIMIT_REQUESTS="4",
        RATE_LIMIT_WINDOW_S="1.0",
        MOCK_RATE_LIMIT_REQUESTS="4",
        MOCK_RATE_LIMIT_WINDOW_S="1.0",
        TEMPERATURE="0.7",
        REQUEST_TIMEOUT_S="20.0",
        FAULT_RATE="0.4",
        FAULT_MODE="truncated_stream",
        MAX_CONCURRENCY="10",
        MAX_RETRIES="6",
    )
    service = make_service()
    mock_server.reset_stats()

    results = await asyncio.gather(*(service.triage(text) for text in REPORTS))

    stats = mock_server.stats()
    throttled = stats["calls_by_outcome"].get("throttled", 0)
    assert len(results) == 8
    assert stats["calls_by_outcome"].get("server_error", 0) > 0, (
        "no faults were injected, so this run proved nothing about retries"
    )
    assert throttled == 0, (
        f"the mock throttled {throttled} call(s) with retries in flight. Every "
        f"attempt must take a rate slot, not only the first."
    )
