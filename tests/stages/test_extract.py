"""Stage 1 (structured output) -- every sample yields a valid TriageResult.

Red at checkout: providers/anthropic_p.py is a stub.

Runs without fault injection. Surviving chaos needs the retry layer, which is
stage 5; until then its stub passes each call straight through, so one injected
fault would fail a request here for a reason that has nothing to do with
extraction. test_retry.py runs every sample again at FAULT_RATE=0.4 once retry
exists. The refusal and max_tokens cases are the ones worth reading: a failure
the schema cannot fix must be reported as itself, not crash the process and not
be retried three times first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.providers.base import PermanentProviderError
from app.schema import TriageResult
from tests.conftest import SAMPLES_DIR

SAMPLE_FILES = sorted(SAMPLES_DIR.glob("*.txt"))


def test_ten_samples_are_present():
    assert len(SAMPLE_FILES) == 10, f"expected 10 samples, found {len(SAMPLE_FILES)}"


@pytest.mark.parametrize("sample", SAMPLE_FILES, ids=lambda p: p.stem)
async def test_every_sample_extracts_a_valid_result(make_service, sample, env):
    """All 10 samples produce a valid object."""
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()

    result, _ = await service.triage(sample.read_text())

    assert isinstance(result, TriageResult)
    assert result.severity in {"trivial", "low", "medium", "high", "critical"}
    assert result.component and isinstance(result.component, str)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.steps, list)
    assert all(isinstance(step, str) for step in result.steps)


async def test_max_tokens_truncation_is_reported_not_crashed(make_service, env):
    """`stop_reason=max_tokens` must raise a clear, permanent error.

    The naive failure mode is a JSON parse error from a half-written object,
    which sends the participant hunting in the wrong place. The right response
    names the actual cause and does not retry -- the same call would truncate
    identically.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()

    original = service.provider.client.messages.create

    async def truncating(**kwargs):
        message = await original(**kwargs)
        message.stop_reason = "max_tokens"
        return message

    service.provider.client.messages.create = truncating

    with pytest.raises(PermanentProviderError) as excinfo:
        await service.triage("a report whose answer will be cut off")

    assert "max_tokens" in str(excinfo.value).lower()


async def test_refusal_is_reported_not_crashed(make_service, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7")
    service = make_service()

    original = service.provider.client.messages.create

    async def refusing(**kwargs):
        message = await original(**kwargs)
        message.stop_reason = "refusal"
        return message

    service.provider.client.messages.create = refusing

    with pytest.raises(PermanentProviderError) as excinfo:
        await service.triage("a report the model declines to answer")

    assert "refus" in str(excinfo.value).lower()


async def test_schema_violation_is_permanent(make_service, mock_server, env):
    """Output that will not validate must fail permanently, costing one call."""
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_RETRIES="3")
    service = make_service()
    mock_server.reset_stats()

    original = service.provider.client.messages.create

    async def malformed(**kwargs):
        message = await original(**kwargs)
        message.content[0].text = '{"severity": "apocalyptic", "component": 42}'
        return message

    service.provider.client.messages.create = malformed

    with pytest.raises(PermanentProviderError):
        await service.triage("a report whose answer will not validate")

    assert mock_server.stats()["calls"] == 1, "a schema violation must not be retried"


async def test_bad_content_still_validates(make_service, env):
    """The `bad_content` fault is schema-valid and WRONG -- on purpose.

    This is the failure a schema cannot catch: right shape, wrong answer, high
    confidence. It must not raise. Catching it is what the golden set is for.
    """
    env(FAULT_RATE="1.0", FAULT_MODE="bad_content", TEMPERATURE="0.7")
    service = make_service()

    result, _ = await service.triage(SAMPLE_FILES[0].read_text())

    assert isinstance(result, TriageResult)
    assert result.confidence > 0.9, "the bad_content fault is confidently wrong by design"
