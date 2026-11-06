"""Stage 2 (async and backpressure) -- the concurrency semaphore and the rate
limiter.

Red at checkout: ConcurrencyLimiter and RateLimiter in
app/reliability/limits.py are stubs.

Every assertion here reads the mock, not our own counters. peak_concurrency is
what actually arrived upstream; asserting on the limiter's own number instead
would be marking our own homework, since the two agree only if the permit is
genuinely held around the call.

The wall-clock assertion is the other half. A semaphore that bounds concurrency
by serialising everything is not a limiter, it is a bottleneck.

The rate tests turn on the mock's own limit -- N calls in any window, 429 with
Retry-After past it -- and set ours to match. The retry layer would recover from
those 429s eventually, so success proves nothing; what the tests read is the
mock's `throttled` count, which a limiter that waits keeps at zero.
"""

from __future__ import annotations

import asyncio
import time

import pytest

REPORTS = [
    "Login fails with a 500 for every user after the deploy.",
    "Refund is charged twice when the button is pressed twice.",
    "The settings modal is misaligned by 40px on the dashboard.",
    "Order page takes 9 seconds intermittently, possibly a deadlock.",
    "Search returns unrelated articles for two-word queries.",
    "Comment notification emails arrive up to 20 minutes late.",
    "Uploading a 200MB attachment fails at the very end.",
    "Worker cannot connect to postgres, OperationalError in the log.",
    "TypeError on checkout: NoneType multiplied by int in totals.",
    "The export button does nothing on the reports screen.",
]


@pytest.mark.latency
async def test_ten_concurrent_requests_finish_promptly(make_service, mock_server, env):
    """10 requests against a slow upstream finish in batches, not in series.

    A note on the numbers. 10 requests at MAX_CONCURRENCY=4 is
    `ceil(10/4) = 3` batches, so against a 1s upstream the floor is 3.0s of
    pure waiting -- the spec's "< 3s" leaves exactly zero room for scheduling,
    HTTP and JSON. The property being tested is "batched, not serialised", so
    the upstream is set to 0.8s: three batches is 2.4s and the budget below has
    real headroom, while a serialising implementation takes ~8s and misses it
    by a mile. Tightening the upstream rather than loosening the assertion
    keeps the test honest.
    """
    env(
        FAULT_RATE="0.0",
        TEMPERATURE="0.7",
        MAX_CONCURRENCY="4",
        MOCK_LATENCY_S="0.8",
        REQUEST_TIMEOUT_S="20.0",
    )
    service = make_service()
    mock_server.reset_stats()

    started = time.monotonic()
    results = await asyncio.gather(*(service.triage(text) for text in REPORTS))
    elapsed = time.monotonic() - started

    assert len(results) == 10
    assert all(result.component for result, _ in results)
    assert elapsed < 3.0, (
        f"10 requests against a 0.8s upstream took {elapsed:.2f}s. At "
        f"MAX_CONCURRENCY=4 that is 3 batches, so ~2.4s. Serialising them would "
        f"take ~8s -- a semaphore is a limit, not a queue of one."
    )


@pytest.mark.latency
async def test_mock_never_saw_more_than_max_concurrency(make_service, mock_server, env):
    """The mock's own peak must never exceed the configured limit."""
    env(
        FAULT_RATE="0.0",
        TEMPERATURE="0.7",
        MAX_CONCURRENCY="4",
        MOCK_LATENCY_S="0.4",
        REQUEST_TIMEOUT_S="20.0",
    )
    service = make_service()
    mock_server.reset_stats()

    await asyncio.gather(*(service.triage(text) for text in REPORTS))

    peak = mock_server.stats()["peak_concurrency"]
    assert peak <= 4, (
        f"the mock saw {peak} requests in flight at once with MAX_CONCURRENCY=4. "
        f"The semaphore must be held around the upstream call itself."
    )
    assert peak > 1, (
        f"the mock never saw more than {peak} request in flight; the requests were "
        f"serialised rather than limited, which is a different bug"
    )


@pytest.mark.parametrize("limit", [1, 2, 5])
@pytest.mark.latency
async def test_peak_tracks_the_configured_limit(make_service, mock_server, env, limit):
    """Changing MAX_CONCURRENCY must actually change what upstream observes."""
    env(
        FAULT_RATE="0.0",
        TEMPERATURE="0.7",
        MAX_CONCURRENCY=str(limit),
        MOCK_LATENCY_S="0.3",
        REQUEST_TIMEOUT_S="20.0",
    )
    service = make_service()
    mock_server.reset_stats()

    await asyncio.gather(*(service.triage(f"{text} limit={limit}") for text in REPORTS))

    peak = mock_server.stats()["peak_concurrency"]
    assert peak <= limit, f"MAX_CONCURRENCY={limit} but the mock saw {peak} in flight"


@pytest.mark.latency
async def test_limiter_agrees_with_the_mock(make_service, mock_server, env):
    """Our counter and the mock's must tell the same story.

    They can only disagree if the permit is held around something other than
    the upstream call -- which is the mistake this catches.
    """
    env(
        FAULT_RATE="0.0",
        TEMPERATURE="0.7",
        MAX_CONCURRENCY="3",
        MOCK_LATENCY_S="0.3",
        REQUEST_TIMEOUT_S="20.0",
    )
    service = make_service()
    mock_server.reset_stats()

    await asyncio.gather(*(service.triage(text) for text in REPORTS))

    ours = service.limits.limiter.peak_in_flight
    theirs = mock_server.stats()["peak_concurrency"]
    assert ours <= 3
    assert theirs <= ours, (
        f"the mock saw {theirs} concurrent requests but our limiter peaked at {ours}; "
        f"work is escaping the semaphore"
    )
    assert service.limits.limiter.in_flight == 0, "a permit was leaked"


def _rate_env(env, *, requests: int, window_s: float, **extra: object) -> None:
    env(
        RATE_LIMIT_REQUESTS=str(requests),
        RATE_LIMIT_WINDOW_S=str(window_s),
        MOCK_RATE_LIMIT_REQUESTS=str(requests),
        MOCK_RATE_LIMIT_WINDOW_S=str(window_s),
        TEMPERATURE="0.7",
        REQUEST_TIMEOUT_S="20.0",
        **extra,
    )


@pytest.mark.latency
async def test_a_burst_waits_for_the_rate_limit(make_service, mock_server, env):
    """Ten requests at once against 4 per second: the excess waits, not 429s.

    Concurrency is set above the burst on purpose. With MAX_CONCURRENCY=4 the
    semaphore alone would spread the calls out and hide a missing limiter.
    """
    _rate_env(env, requests=4, window_s=1.0, FAULT_RATE="0.0", MAX_CONCURRENCY="10")
    service = make_service()
    mock_server.reset_stats()

    started = time.monotonic()
    results = await asyncio.gather(*(service.triage(text) for text in REPORTS))
    elapsed = time.monotonic() - started

    stats = mock_server.stats()
    throttled = stats["calls_by_outcome"].get("throttled", 0)
    assert len(results) == 10
    assert throttled == 0, (
        f"the mock throttled {throttled} call(s): more than 4 started inside one "
        f"1s window. The limiter must make the excess wait before it is sent -- "
        f"a 429 means the provider enforced the limit for you."
    )
    assert stats["calls"] == 10, (
        f"10 requests made {stats['calls']} upstream calls with no faults injected"
    )
    assert elapsed < 3.5, (
        f"10 requests at 4 per second took {elapsed:.2f}s; three windows is ~2s. "
        f"The limiter is waiting longer than the limit requires."
    )
