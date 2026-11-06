"""The anti-cheat check on the suite itself.

The mock's x-mock-fault-* headers are a useful tool and a tempting exit: a red
reliability test can be made green by sending `x-mock-fault-mode: none` instead
of writing the logic it asks for. make test-graded refuses them, and this checks
the refusal is real rather than assumed -- production has no mock to refuse them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from tests.conftest import APP_DIR, TESTS_DIR, header_overrides_allowed, requires_adapter



def test_no_source_file_sends_a_mock_fault_header():
    """Nothing under app/ may mention the mock's fault headers.

    A grep, deliberately: this is a property of the source, not of a run, and
    it must hold on every path including the ones no test exercises.
    """
    offenders = []
    for path in APP_DIR.rglob("*.py"):
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            if "x-mock-fault" in lowered or "mock_allow_header_override" in lowered:
                # A comment explaining the prohibition is fine; sending one is not.
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("*"):
                    continue
                offenders.append(f"{path.relative_to(APP_DIR.parent)}:{lineno}: {stripped}")

    assert not offenders, (
        "app/ code references the mock's fault-control headers:\n  "
        + "\n  ".join(offenders)
        + "\nThese are a mock-only extension; the real API ignores them."
    )


def test_no_test_lowers_the_fault_rate_by_header():
    """Tests may isolate a fault type by header, never reduce the rate.

    `x-mock-fault-rate` and `x-mock-fault-mode: none` both make the run
    easier than the gate. Naming a specific mode does not -- it makes one
    fault certain, which is stricter, and it is how test_stream.py gets a
    truncated stream on demand.
    """
    tests_dir = TESTS_DIR
    this_file = Path(__file__).resolve()
    offenders = []
    for path in tests_dir.rglob("test_*.py"):
        # This file necessarily contains the strings it is scanning for -- it is
        # the scanner, and the two tests below deliberately attempt the cheat in
        # order to prove the mock refuses it.
        if path.resolve() == this_file:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            lowered = stripped.lower()
            if "x-mock-fault-rate" in lowered:
                offenders.append(f"{path.name}:{lineno}: {stripped}")
            if "x-mock-fault-mode" in lowered and '"none"' in lowered:
                offenders.append(f"{path.name}:{lineno}: {stripped}")

    assert not offenders, (
        "a test is turning chaos down instead of surviving it:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.skipif(
    header_overrides_allowed(),
    reason="only meaningful when MOCK_ALLOW_HEADER_OVERRIDE=false (make test-graded)",
)
@requires_adapter
async def test_graded_mode_refuses_the_headers(make_service, mock_server, env):
    env(FAULT_RATE="1.0", FAULT_MODE="rate_limit", TEMPERATURE="0.7", MAX_RETRIES="0")
    service = make_service()
    mock_server.reset_stats()

    # Try to silence the chaos the way a participant might.
    service.provider.client = service.provider.client.with_options(
        default_headers={"x-mock-fault-mode": "none"}
    )

    from app.reliability.retry import RetryExhaustedError

    with pytest.raises(RetryExhaustedError):
        await service.triage("attempting to silence the fault injector")

    stats = mock_server.stats()
    assert stats["header_override_allowed"] is False
    assert stats["header_overrides_refused"] >= 1, (
        "the mock did not record the refused override"
    )


@pytest.mark.skipif(
    not header_overrides_allowed(),
    reason="only meaningful when overrides are allowed (the default run)",
)
@requires_adapter
async def test_dev_mode_honours_the_headers(make_service, mock_server, env):
    """The headers work outside graded mode -- that is what they are for."""
    env(FAULT_RATE="1.0", FAULT_MODE="rate_limit", TEMPERATURE="0.7", MAX_RETRIES="0")
    service = make_service()
    mock_server.reset_stats()

    service.provider.client = service.provider.client.with_options(
        default_headers={"x-mock-fault-mode": "none"}
    )
    result, _ = await service.triage("this one is allowed to be silenced")
    assert result.component
    assert mock_server.stats()["header_overrides_refused"] == 0
