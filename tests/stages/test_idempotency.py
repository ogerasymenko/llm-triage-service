"""Stage 5 (surviving a bad provider) -- idempotency with a RESULT cache.

Red at checkout: app/reliability/idempotency.py is a stub.

The distinction this file enforces: cache the result, not merely the call.

Against a live model at TEMPERATURE=0.7 the same prompt twice returns two
different answers, both valid. A caller retrying a request whose response it
never saw must get back the one it already paid for, byte for byte. So "one
upstream call" is half the assertion; "the identical object" is the other half,
and an implementation with the first but not the second has not solved the
problem the exercise poses.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import SAMPLE_TEXT


async def test_same_key_returns_byte_identical_object(make_service, mock_server, env):
    """Two calls, one key: identical bytes, and exactly one upstream call."""
    env(TEMPERATURE="0.7", FAULT_RATE="0.0")
    service = make_service()
    mock_server.reset_stats()

    first, first_cached = await service.triage(SAMPLE_TEXT, idem_key="order-1234")
    calls_after_first = mock_server.stats()["calls"]

    second, second_cached = await service.triage(SAMPLE_TEXT, idem_key="order-1234")
    calls_after_second = mock_server.stats()["calls"]

    assert first_cached is False, "the first call cannot be a cache hit"
    assert second_cached is True, "the second call with the same key must be served from cache"

    # Byte-identical, not merely equal: the caller may be comparing checksums,
    # and `==` on a model would hide a re-serialisation that reordered keys.
    assert first.model_dump_json() == second.model_dump_json()
    assert first == second

    # The second call must not reach upstream AT ALL. Not "reached it and threw
    # the answer away", not "reached it with a dedupe header" -- not at all.
    assert calls_after_first == 1, f"first call should make exactly 1 upstream call, saw {calls_after_first}"
    assert calls_after_second == 1, (
        f"the second call with the same idempotency key reached upstream: "
        f"{calls_after_second} total calls, expected 1. The result must be cached, "
        f"not merely the side effect deduplicated."
    )


async def test_cached_result_is_stable_under_a_nondeterministic_upstream(
    make_service, mock_server, env
):
    """The cache must beat a genuinely varying upstream.

    The mock's answer depends on the request body, so to prove the cache is
    doing the work we ask for the SAME key with DIFFERENT text. A correct
    implementation returns the first answer; an implementation that merely
    dedupes on the body would call upstream again and return the new answer.
    """
    env(TEMPERATURE="0.7", FAULT_RATE="0.0")
    service = make_service()
    mock_server.reset_stats()

    first, _ = await service.triage(SAMPLE_TEXT, idem_key="ticket-99")
    other_text = "Search returns no results for any query since this morning."
    second, cached = await service.triage(other_text, idem_key="ticket-99")

    assert cached is True
    assert second.model_dump_json() == first.model_dump_json(), (
        "the same idempotency key returned a different object; the key, not the "
        "body, defines the logical request"
    )
    assert mock_server.stats()["calls"] == 1


async def test_different_keys_do_not_share_a_result(make_service, mock_server, env):
    """The cache must not be so eager that it merges distinct requests."""
    env(TEMPERATURE="0.7", FAULT_RATE="0.0")
    service = make_service()
    mock_server.reset_stats()

    auth_text = "Login returns a 500 for all users since the deploy."
    pay_text = "Refunds are issued twice when the button is pressed twice."

    first, _ = await service.triage(auth_text, idem_key="key-a")
    second, cached = await service.triage(pay_text, idem_key="key-b")

    assert cached is False
    assert first.component != second.component or first != second
    assert mock_server.stats()["calls"] == 2


async def test_concurrent_duplicates_collapse_to_one_upstream_call(
    make_service, mock_server, env
):
    """Eight simultaneous retries of one request must make ONE upstream call.

    This is the case a bare check-then-set cache fails: every caller misses the
    empty cache at the same instant and every caller calls upstream. It is also
    the realistic case -- duplicate requests arrive together precisely because
    something upstream just retried.
    """
    env(TEMPERATURE="0.7", FAULT_RATE="0.0", MOCK_LATENCY_S="0.2")
    service = make_service()
    mock_server.reset_stats()

    results = await asyncio.gather(
        *(service.triage(SAMPLE_TEXT, idem_key="burst-1") for _ in range(8))
    )

    payloads = {result.model_dump_json() for result, _ in results}
    assert len(payloads) == 1, "concurrent duplicates returned different objects"

    cached_flags = [cached for _, cached in results]
    assert cached_flags.count(False) == 1, (
        f"exactly one of the 8 concurrent calls should have done the work, "
        f"{cached_flags.count(False)} did"
    )
    assert mock_server.stats()["calls"] == 1, (
        f"8 concurrent duplicates made {mock_server.stats()['calls']} upstream calls; "
        f"the cache needs a per-key lock, not a bare check-then-set"
    )


async def test_no_key_means_no_cache_across_different_text(make_service, mock_server, env):
    """Without an explicit key, identity falls back to the request body."""
    env(TEMPERATURE="0.7", FAULT_RATE="0.0")
    service = make_service()
    mock_server.reset_stats()

    await service.triage("Upload of a 200MB attachment fails at the end.")
    _, cached = await service.triage("Upload of a 200MB attachment fails at the end.")
    assert cached is True, "identical text with no key should still hit the cache"
    assert mock_server.stats()["calls"] == 1
