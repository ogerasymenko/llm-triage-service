"""Stage 7 (a second provider) -- the OpenAI adapter against the same mock.

Red at checkout: OpenAIProvider in app/providers/openai_p.py is a stub.

What this checks is not "the adapter returns JSON". It is that the SAME
reliability layers, untouched, produce the same guarantees over a different
wire format: the retry budget still bounds upstream calls, an idempotency key
still replays byte-identical, and a truncated stream still ends in an error
rather than a partial object.
"""

from __future__ import annotations

import pytest

from app.schema import TriageResult


@pytest.fixture
def openai_service(make_service, env, mock_server):
    """A service configured for the OpenAI dialect, against the same mock."""

    def _build(**overrides):
        env(PROVIDER="openai", MODEL="gpt-4o-mini", **overrides)
        return make_service()

    return _build


async def test_the_factory_picks_the_openai_adapter(openai_service):
    service = openai_service(FAULT_RATE="0.0", TEMPERATURE="0.7")
    assert type(service.provider).__name__ == "OpenAIProvider"


async def test_extract_returns_a_valid_result(openai_service):
    service = openai_service(FAULT_RATE="0.0", TEMPERATURE="0.7")
    result, cached = await service.triage("Login returns a 500 for every user.")
    assert isinstance(result, TriageResult)
    assert result.component
    assert cached is False


async def test_the_client_uses_the_openai_key(openai_service):
    """Each vendor gets its own credential, never the other's."""
    service = openai_service(
        FAULT_RATE="0.0",
        OPENAI_API_KEY="sk-openai-test",
        ANTHROPIC_API_KEY="sk-ant-must-not-leak",
    )
    assert service.provider.client.api_key == "sk-openai-test"


async def test_sdk_retries_are_disabled_here_too(openai_service):
    """Divergence 4: the openai client also defaults to max_retries=2.

    Porting the adapter without porting this puts the double-retry bug back for
    the second dialect, and MAX_RETRIES stops meaning what it says again.
    """
    service = openai_service(FAULT_RATE="0.0")
    assert service.provider.client.max_retries == 0


@pytest.mark.chaos
@pytest.mark.parametrize("max_retries", [0, 2])
async def test_retry_budget_still_bounds_upstream_calls(
    openai_service, mock_server, max_retries
):
    """The reliability layer is untouched, so its guarantee must be too."""
    from app.reliability.retry import RetryExhaustedError

    service = openai_service(
        FAULT_RATE="1.0", FAULT_MODE="rate_limit", TEMPERATURE="0.7",
        MAX_RETRIES=str(max_retries),
    )
    mock_server.reset_stats()
    with pytest.raises(RetryExhaustedError):
        await service.triage(f"always rate limited, budget={max_retries}")

    calls = mock_server.stats()["calls"]
    assert calls == 1 + max_retries, (
        f"MAX_RETRIES={max_retries} should cost {1 + max_retries} calls over the "
        f"OpenAI dialect too; saw {calls}"
    )


async def test_idempotency_replays_byte_identical(openai_service, mock_server):
    service = openai_service(FAULT_RATE="0.0", TEMPERATURE="0.7")
    mock_server.reset_stats()

    first, c1 = await service.triage("Refunds are charged twice.", idem_key="oa-1")
    second, c2 = await service.triage("Refunds are charged twice.", idem_key="oa-1")

    assert (c1, c2) == (False, True)
    assert first.model_dump_json() == second.model_dump_json()
    assert mock_server.stats()["calls"] == 1


async def test_stream_produces_one_terminal_done(openai_service):
    service = openai_service(FAULT_RATE="0.0", TEMPERATURE="0.7")
    events = [e async for e in service.triage_stream("The export button does nothing.")]

    assert events[0].type == "start"
    assert [e for e in events if e.type == "delta"], "no deltas"
    terminal = [e for e in events if e.type in {"done", "error"}]
    assert len(terminal) == 1 and terminal[0] is events[-1]
    assert isinstance(terminal[0].result, TriageResult)


@pytest.mark.chaos
async def test_truncated_stream_yields_error_not_partial(openai_service):
    """Divergence 2: no typed terminal event, so truncation must still be caught.

    An OpenAI stream that dies mid-object looks like a slow one until the
    connection drops -- there is no message_stop whose absence gives it away.
    The adapter must still refuse to hand back a partial object.
    """
    service = openai_service(
        FAULT_RATE="1.0", FAULT_MODE="truncated_stream", TEMPERATURE="0.7"
    )
    events = [e async for e in service.triage_stream("Checkout stalls under load.")]
    kinds = [e.type for e in events]

    assert "error" in kinds, f"expected an error event, got {kinds}"
    assert "done" not in kinds
    assert all(e.result is None for e in events)


async def test_max_tokens_truncation_is_permanent(openai_service):
    """finish_reason='length' is OpenAI's spelling of stop_reason='max_tokens'."""
    from app.providers.base import PermanentProviderError

    service = openai_service(FAULT_RATE="0.0", TEMPERATURE="0.7")
    original = service.provider.client.chat.completions.create

    async def truncating(**kwargs):
        completion = await original(**kwargs)
        completion.choices[0].finish_reason = "length"
        return completion

    service.provider.client.chat.completions.create = truncating

    with pytest.raises(PermanentProviderError, match="(?i)max_tokens"):
        await service.triage("an answer that will be cut off")


async def test_both_dialects_share_one_set_of_counters(openai_service, mock_server, env):
    """One mock, one fault decision, one /stats -- across two wire formats."""
    mock_server.reset_stats()

    service = openai_service(FAULT_RATE="0.0", TEMPERATURE="0.7")
    await service.triage("Search returns nothing for two-word queries.")
    after_openai = mock_server.stats()["calls"]

    env(PROVIDER="anthropic", MODEL="claude-haiku-4-5")
    from app.service import TriageService
    import app.config as config_module

    anthropic_service = TriageService(settings=config_module.get_settings())
    try:
        await anthropic_service.triage("Search returns nothing for two-word queries.")
    finally:
        await anthropic_service.aclose()

    assert after_openai == 1
    assert mock_server.stats()["calls"] == 2, (
        "the two dialects must accumulate into the same counters, or /stats is a "
        "lie the moment someone runs both"
    )


@pytest.mark.chaos
async def test_a_severed_stream_is_marked_transient(openai_service):
    """Stage 3's contract holds over the second dialect too."""
    service = openai_service(
        FAULT_RATE="1.0", FAULT_MODE="truncated_stream", TEMPERATURE="0.7"
    )
    events = [e async for e in service.triage_stream("Checkout stalls under load.")]
    errors = [e for e in events if e.type == "error"]

    assert len(errors) == 1, [e.type for e in events]
    assert errors[0].transient is True
