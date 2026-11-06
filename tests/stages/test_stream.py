"""Stage 3 (streaming) -- fast first byte, and never a partial object.

Red at checkout: Provider.stream does not exist.

The second property is the one that matters. Half a TriageResult looks exactly
like a whole one -- same shape, same fields, wrong answer, no indication
anything went wrong -- so a truncated stream must end in error and never in
done. Silent partial data beats a loud failure only in the sense that a quiet
fire beats a smoke alarm.

This is the suite's one legitimate use of x-mock-fault-mode: it needs a
truncated stream on demand and asks for that ONE fault type, never a lower rate.
The graded run refuses the header, so those tests skip there rather than pass
vacuously, and the same property is covered again through FAULT_MODE.

The rate-limit test lives here rather than in stage 2 because it needs stream()
to exist; it needs stage 2's RateLimiter too, which by stage 3 is done.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.schema import StreamEvent, TriageResult
from tests.conftest import SAMPLE_TEXT, header_overrides_allowed

TRUNCATE_HEADER = {"x-mock-fault-mode": "truncated_stream"}


@pytest.mark.latency
async def test_first_event_arrives_quickly(make_service, mock_server, env):
    """First byte in under 1.5s: a stream that buffers is not a stream."""
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()

    started = time.monotonic()
    first_delta_at = None
    events: list[StreamEvent] = []

    async for event in service.triage_stream(SAMPLE_TEXT):
        events.append(event)
        if event.type == "delta" and first_delta_at is None:
            first_delta_at = time.monotonic() - started

    assert first_delta_at is not None, "no delta events were produced"
    assert first_delta_at < 1.5, f"first delta took {first_delta_at:.2f}s"
    assert events[0].type == "start"
    assert events[-1].type == "done"


async def test_stream_produces_a_valid_result(make_service, env):
    """The assembled object must validate, and match what the unary path gives."""
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()

    events = [event async for event in service.triage_stream(SAMPLE_TEXT)]
    done = [event for event in events if event.type == "done"]

    assert len(done) == 1
    assert isinstance(done[0].result, TriageResult)
    assert not [event for event in events if event.type == "error"]

    text = "".join(event.text or "" for event in events if event.type == "delta")
    assert TriageResult.model_validate_json(text) == done[0].result


@pytest.mark.skipif(
    not header_overrides_allowed(),
    reason="graded run refuses x-mock-fault-* headers; covered by FAULT_MODE instead",
)
async def test_truncated_stream_yields_an_error_not_a_partial_object(
    make_service, mock_server, env
):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()
    service.provider.client = service.provider.client.with_options(
        default_headers=TRUNCATE_HEADER
    )

    events = [event async for event in service.triage_stream(SAMPLE_TEXT)]
    kinds = [event.type for event in events]

    assert "error" in kinds, f"a truncated stream produced no error event: {kinds}"
    assert "done" not in kinds, (
        "a truncated stream produced a `done` event. A partial object is not a "
        "result -- it looks like an answer and is not one."
    )
    assert events[-1].type == "error"
    assert events[-1].result is None


@pytest.mark.skipif(
    not header_overrides_allowed(),
    reason="graded run refuses x-mock-fault-* headers; covered by FAULT_MODE instead",
)
async def test_partial_text_is_never_presented_as_a_result(make_service, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()
    service.provider.client = service.provider.client.with_options(
        default_headers=TRUNCATE_HEADER
    )

    events = [event async for event in service.triage_stream(SAMPLE_TEXT)]
    partial = "".join(event.text or "" for event in events if event.type == "delta")

    assert partial, "expected some deltas before the truncation"
    with pytest.raises(Exception):
        TriageResult.model_validate_json(partial)
    assert all(event.result is None for event in events)


async def test_truncation_via_environment_not_headers(make_service, mock_server, env):
    """The same property, driven by FAULT_MODE so it runs in the graded suite too.

    Streams are the only path where truncation applies, so with
    `FAULT_MODE=truncated_stream` at rate 1.0 every stream attempt is severed
    -- no header required, and nothing silenced.
    """
    env(FAULT_RATE="1.0", FAULT_MODE="truncated_stream", TEMPERATURE="0.7")
    service = make_service()

    events = [event async for event in service.triage_stream(SAMPLE_TEXT)]
    kinds = [event.type for event in events]

    assert "error" in kinds, f"expected an error event, got {kinds}"
    assert "done" not in kinds
    assert all(event.result is None for event in events)


STREAM_REPORTS = [
    "Login fails with a 500 for every user after the deploy.",
    "Refund is charged twice when the button is pressed twice.",
    "The settings modal is misaligned by 40px on the dashboard.",
    "Order page takes 9 seconds intermittently, possibly a deadlock.",
    "Search returns unrelated articles for two-word queries.",
    "Comment notification emails arrive up to 20 minutes late.",
    "Uploading a 200MB attachment fails at the very end.",
]


@pytest.mark.latency
async def test_streams_respect_the_rate_limit(make_service, mock_server, env):
    """The streaming path reaches the same provider and spends the same limit."""
    env(
        RATE_LIMIT_REQUESTS="3",
        RATE_LIMIT_WINDOW_S="1.0",
        MOCK_RATE_LIMIT_REQUESTS="3",
        MOCK_RATE_LIMIT_WINDOW_S="1.0",
        TEMPERATURE="0.7",
        REQUEST_TIMEOUT_S="20.0",
        FAULT_RATE="0.0",
        MAX_CONCURRENCY="10",
    )
    service = make_service()
    mock_server.reset_stats()

    async def drain(text: str) -> list[str]:
        return [event.type async for event in service.triage_stream(text)]

    runs = await asyncio.gather(*(drain(text) for text in STREAM_REPORTS))

    throttled = mock_server.stats()["calls_by_outcome"].get("throttled", 0)
    assert throttled == 0, (
        f"the mock throttled {throttled} stream(s): streams must wait for a rate "
        f"slot like unary calls do"
    )
    assert all(events and events[-1] == "done" for events in runs), runs


async def test_a_severed_stream_is_marked_transient(make_service, env):
    """The same request could succeed a moment later, and the event must say so."""
    env(FAULT_RATE="1.0", FAULT_MODE="truncated_stream", TEMPERATURE="0.7")
    service = make_service()

    events = [event async for event in service.triage_stream(SAMPLE_TEXT)]
    errors = [event for event in events if event.type == "error"]

    assert len(errors) == 1, [event.type for event in events]
    assert errors[0].transient is True, (
        "a truncated stream is transient: set transient=True on its error event"
    )


async def test_a_refusal_is_permanent_and_spares_the_breaker(make_service, env):
    """A refusal is the request's fault, not upstream's.

    More refusals than the breaker's threshold, back to back. Counted as upstream
    failures they would open the circuit, and one hostile input would turn every
    other caller away with 503s.
    """
    env(FAULT_RATE="1.0", FAULT_MODE="refusal", TEMPERATURE="0.7")
    service = make_service()

    for i in range(service.breaker.failure_threshold + 1):
        events = [event async for event in service.triage_stream(f"{SAMPLE_TEXT} #{i}")]
        assert events[-1].type == "error", [event.type for event in events]
        assert events[-1].transient is False, (
            "a refusal will refuse again: set transient=False on its error event"
        )
        assert all(event.result is None for event in events)

    assert service.breaker.state == "closed"
    assert service.breaker.consecutive_failures == 0
