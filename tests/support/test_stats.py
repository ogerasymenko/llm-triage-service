"""Participants debug with these counters, so each test here hand-counts a
controlled run and compares -- a number that is approximately right teaches a
wrong lesson with total confidence."""

from __future__ import annotations

import asyncio

import pytest

from app.reliability.limits import cost_usd
from tests.conftest import SAMPLE_TEXT
from tests.conftest import requires_adapter

pytestmark = requires_adapter


@pytest.mark.parametrize("n", [1, 3, 7])
async def test_calls_matches_a_hand_count(make_service, mock_server, env, n):
    """N distinct requests, no faults: exactly N calls, none of them retries.

    Parametrised so the counter has to track the number rather than happen to
    agree with one of them. A counter that is off by a constant, saturates, or
    counts a fresh request as a repeat fails here.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()
    mock_server.reset_stats()

    for i in range(n):
        await service.triage(f"Distinct report {i} about subsystem {i} failing.")

    stats = mock_server.stats()
    assert stats["calls"] == n
    assert stats["retries_observed"] == 0, (
        f"{n} distinct requests produced {stats['retries_observed']} retries; "
        f"distinct bodies are not repeats of one another"
    )
    assert stats["calls_by_outcome"].get("ok") == n


async def test_one_request_costs_what_the_retry_budget_allows(
    make_service, mock_server, env
):
    """The counter must reflect the configured budget, at more than one value.

    This is the check a participant can watch while working: set MAX_RETRIES,
    run one request against a failing upstream, and `calls` should be
    1 + MAX_RETRIES. If it reads higher, something is making more upstream
    requests than the budget allows -- which is the counter doing its job.
    """
    from app.reliability.retry import RetryExhaustedError

    for max_retries in (0, 1, 3):
        env(
            FAULT_RATE="1.0",
            FAULT_MODE="rate_limit",
            TEMPERATURE="0.7",
            MAX_RETRIES=str(max_retries),
        )
        service = make_service()
        mock_server.reset_stats()
        with pytest.raises(RetryExhaustedError):
            await service.triage(f"one logical request, budget={max_retries}")

        stats = mock_server.stats()
        assert stats["calls"] == 1 + max_retries, (
            f"MAX_RETRIES={max_retries} should cost {1 + max_retries} upstream "
            f"call(s); /stats counted {stats['calls']}"
        )
        assert stats["retries_observed"] == max_retries


@pytest.mark.chaos
async def test_retries_observed_counts_repeat_identities(make_service, mock_server, env):
    """4 attempts of one request = 4 calls, 3 of them retries."""
    env(FAULT_RATE="1.0", FAULT_MODE="rate_limit", TEMPERATURE="0.7", MAX_RETRIES="3")
    service = make_service()
    mock_server.reset_stats()

    from app.reliability.retry import RetryExhaustedError

    with pytest.raises(RetryExhaustedError):
        await service.triage("a request that will exhaust its retry budget")

    stats = mock_server.stats()
    assert stats["calls"] == 4
    assert stats["retries_observed"] == 3
    assert stats["calls_by_outcome"].get("rate_limit") == 4


async def test_token_and_cost_accounting_is_consistent(make_service, mock_server, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()
    mock_server.reset_stats()

    for i in range(4):
        await service.triage(f"Report {i}: the widget service returns 500s.")

    stats = mock_server.stats()
    assert stats["input_tokens"] > 0 and stats["output_tokens"] > 0
    expected = cost_usd(stats["input_tokens"], stats["output_tokens"])
    assert stats["estimated_cost_usd"] == pytest.approx(expected, rel=1e-6), (
        f"/stats priced {stats['input_tokens']}in/{stats['output_tokens']}out at "
        f"${stats['estimated_cost_usd']}, expected ${expected}"
    )


@pytest.mark.latency
async def test_peak_concurrency_is_observed_not_assumed(make_service, mock_server, env):
    """A serialised run must report peak 1; a parallel one must report more."""
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_CONCURRENCY="4", MOCK_LATENCY_S="0.2")
    service = make_service()

    mock_server.reset_stats()
    for i in range(4):
        await service.triage(f"Sequential report {i} about a failing job.")
    assert mock_server.stats()["peak_concurrency"] == 1

    mock_server.reset_stats()
    await asyncio.gather(
        *(service.triage(f"Parallel report {i} about a failing job.") for i in range(4))
    )
    peak = mock_server.stats()["peak_concurrency"]
    assert 1 < peak <= 4, f"parallel run reported peak_concurrency={peak}"


async def test_reset_clears_every_counter(mock_server, make_service, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()
    await service.triage("something to count")
    assert mock_server.stats()["calls"] > 0

    mock_server.reset_stats()
    stats = mock_server.stats()
    assert stats["calls"] == 0
    assert stats["retries_observed"] == 0
    assert stats["peak_concurrency"] == 0
    assert stats["input_tokens"] == 0
    assert stats["output_tokens"] == 0
    assert stats["estimated_cost_usd"] == 0
