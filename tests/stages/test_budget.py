"""Stage 5 (surviving a bad provider) -- MAX_TOKENS everywhere, and a ceiling
that bites before the money moves.

Red at checkout: SpendLedger in app/reliability/limits.py is a stub.

The load-bearing word is before. A ceiling that notices the overrun afterwards
is a report, not a limit. Since a call's true cost is unknown until it returns,
the ledger reserves the worst case up front and reconciles after -- which is
what lets it refuse the call that would cross the line rather than the one that
already did.

Checked against the mock's own estimated_cost_usd, which prices what actually
arrived. If our ledger says $1.00 and the mock says $1.40, our ledger is wrong.
"""

from __future__ import annotations

import asyncio

import pytest

from app.reliability.limits import SpendCeilingExceeded, SpendLedger, cost_usd
from tests.conftest import SAMPLE_TEXT


async def test_max_tokens_is_sent_on_every_request(make_service, mock_server, env):
    """Every upstream call must carry MAX_TOKENS.

    Omitting it is how a runaway generation bills for a context window. The
    mock reports what it received, so this checks the wire, not our intent.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_TOKENS="512")
    service = make_service()

    seen: list[dict] = []
    original = service.provider.client.messages.create

    async def spy(**kwargs):
        seen.append(kwargs)
        return await original(**kwargs)

    service.provider.client.messages.create = spy

    await service.triage("First report: login is broken for everyone.")
    await service.triage("Second report: refunds are duplicated.")

    assert len(seen) == 2
    for call in seen:
        assert call["max_tokens"] == 512, f"max_tokens missing or wrong: {call.get('max_tokens')}"


async def test_temperature_reaches_the_wire(make_service, env):
    """TEMPERATURE must actually be sent, not merely stored in config.

    This SDK version dropped `temperature` from its typed signature, so the
    adapter passes it through `extra_body`. If that ever silently stops
    working, TEMPERATURE=0.7 becomes decorative and the gate stops meaning what
    it says.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()

    seen: list[dict] = []
    original = service.provider.client.messages.create

    async def spy(**kwargs):
        seen.append(kwargs)
        return await original(**kwargs)

    service.provider.client.messages.create = spy
    await service.triage("Some report about a broken export button.")

    assert seen[0]["extra_body"]["temperature"] == 0.7


async def test_ceiling_raises_before_the_call_that_would_cross_it(
    make_service, mock_server, env
):
    """The refusal happens BEFORE spending, and the refused call never goes out.

    The ceiling is set to *less than one call's worst case*, so the very first
    request cannot be guaranteed to fit and must be refused outright -- with
    zero upstream calls. This is the sharpest form of the property: not "we
    noticed afterwards", but "it never happened".
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_TOKENS="1024")
    service = make_service()
    ledger = service.limits.ledger
    ledger.ceiling_usd = ledger.worst_case_usd() * 0.5
    mock_server.reset_stats()

    with pytest.raises(SpendCeilingExceeded) as excinfo:
        await service.triage("Report one: login is broken for every user.")

    assert mock_server.stats()["calls"] == 0, (
        f"the refused call still reached upstream ({mock_server.stats()['calls']} calls). "
        f"The ceiling must refuse before spending, not report afterwards."
    )
    assert ledger.spent_usd == 0.0
    assert mock_server.stats()["estimated_cost_usd"] == 0.0
    assert excinfo.value.spent_usd <= excinfo.value.ceiling_usd
    assert ledger.committed_usd == pytest.approx(0.0, abs=1e-12), (
        "the refused call left a reservation behind"
    )


async def test_ceiling_refuses_the_crossing_call_mid_run(make_service, mock_server, env):
    """Some calls succeed, then one is refused -- and it is refused, not billed.

    Headroom is computed from the ledger's own numbers rather than guessed, so
    the assertion says what it means: the run stops at the first request whose
    worst case would not fit, and upstream never sees that request.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_TOKENS="1024")
    service = make_service()
    ledger = service.limits.ledger
    mock_server.reset_stats()

    # Enough room for a handful of worst-case calls.
    ledger.ceiling_usd = ledger.worst_case_usd() * 3

    made = 0
    for i in range(100):
        calls_before = mock_server.stats()["calls"]
        spent_before = ledger.spent_usd
        try:
            await service.triage(f"Report {i}: module {i} fails intermittently.")
            made += 1
        except SpendCeilingExceeded:
            assert mock_server.stats()["calls"] == calls_before, (
                "the refused call reached upstream anyway"
            )
            assert ledger.spent_usd == pytest.approx(spent_before), (
                "spend moved while refusing the call"
            )
            break
    else:
        pytest.fail("the ceiling never fired across 100 calls")

    assert made > 0, "the ceiling refused everything; it should admit what fits"
    assert ledger.spent_usd <= ledger.ceiling_usd
    assert mock_server.stats()["estimated_cost_usd"] <= ledger.ceiling_usd


async def test_spend_never_exceeds_the_ceiling_over_a_long_run(
    make_service, mock_server, env
):
    """Drive many calls at a tight ceiling; the mock's own bill must stay under it.

    This is the assertion that would catch a ledger which is merely usually
    right: it keeps going until the ceiling stops it, then compares against
    what the mock says it actually served.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_TOKENS="1024")
    service = make_service()
    ledger = service.limits.ledger
    ledger.ceiling_usd = ledger.worst_case_usd() * 4
    mock_server.reset_stats()

    made = 0
    for i in range(200):
        try:
            await service.triage(f"Report {i}: subsystem {i} misbehaves under load.")
            made += 1
        except SpendCeilingExceeded:
            break
    else:
        pytest.fail("the ceiling never fired across 200 calls")

    assert made > 0
    billed = mock_server.stats()["estimated_cost_usd"]
    assert billed <= ledger.ceiling_usd, (
        f"the mock billed ${billed} against a ${ledger.ceiling_usd} ceiling after {made} calls"
    )
    assert ledger.spent_usd <= ledger.ceiling_usd


async def test_ceiling_holds_under_concurrency(make_service, mock_server, env):
    """Concurrent callers must not each see room for themselves.

    Reserving before the call is what makes this work: with an
    "add up the cost afterwards" ledger, ten simultaneous calls all read a
    spend of $0 and all proceed.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_TOKENS="1024", MOCK_LATENCY_S="0.2")
    service = make_service()
    worst_case = service.limits.ledger.worst_case_usd()
    service.limits.ledger.ceiling_usd = worst_case * 3
    mock_server.reset_stats()

    outcomes = await asyncio.gather(
        *(service.triage(f"Concurrent report {i} about a failing widget.") for i in range(10)),
        return_exceptions=True,
    )

    succeeded = [o for o in outcomes if not isinstance(o, BaseException)]
    refused = [o for o in outcomes if isinstance(o, SpendCeilingExceeded)]

    assert len(succeeded) == 3, f"{len(succeeded)} calls got through a 3x ceiling"
    assert len(refused) == 7
    assert mock_server.stats()["calls"] == 3
    assert service.limits.ledger.spent_usd <= service.limits.ledger.ceiling_usd


async def test_ledger_reserves_then_settles_to_the_true_cost():
    """A unit check on the reserve/settle cycle itself."""
    ledger = SpendLedger(ceiling_usd=1.0, max_tokens=1000, estimated_input_tokens=1000)
    worst = ledger.worst_case_usd()
    assert worst == pytest.approx(cost_usd(1000, 1000))

    reserved = await ledger.reserve()
    assert ledger.committed_usd == pytest.approx(worst)
    assert ledger.spent_usd == 0.0, "reserving is not spending"

    await ledger.settle(reserved, 100, 50)
    assert ledger.spent_usd == pytest.approx(cost_usd(100, 50))
    assert ledger.committed_usd == pytest.approx(cost_usd(100, 50)), "reservation not released"


@pytest.mark.chaos
async def test_failed_call_releases_its_reservation(make_service, mock_server, env):
    """A call that fails must not permanently consume budget.

    Otherwise a burst of retries exhausts the ceiling with money nobody spent.
    """
    env(FAULT_RATE="1.0", FAULT_MODE="rate_limit", TEMPERATURE="0.7", MAX_RETRIES="2")
    service = make_service()
    from app.reliability.retry import RetryExhaustedError

    with pytest.raises(RetryExhaustedError):
        await service.triage("this will fail every attempt")

    assert service.limits.ledger.committed_usd == pytest.approx(0.0, abs=1e-9), (
        "reservations from failed attempts were never released"
    )


async def test_cached_result_costs_nothing(make_service, mock_server, env):
    """A cache hit must not be charged: no upstream call, no spend."""
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()
    mock_server.reset_stats()

    await service.triage(SAMPLE_TEXT, idem_key="billed-once")
    spent_after_first = service.limits.ledger.spent_usd

    await service.triage(SAMPLE_TEXT, idem_key="billed-once")
    assert service.limits.ledger.spent_usd == pytest.approx(spent_after_first), (
        "the cached call was charged; a cache hit costs nothing"
    )
    assert mock_server.stats()["calls"] == 1
