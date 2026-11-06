"""Stage 4 (testing a non-deterministic dependency) -- invariants and
properties. YOU WRITE THIS FILE.

Every other test file in `tests/stages/` is written for you. This one is the
assignment: the placeholders below describe a property and leave you to assert
it.

Why bother, when the suite is already green
-------------------------------------------

Everything you have written so far checks examples: this input gives that
output. Examples are cheap to write and they pin down real bugs, but they pin
down one point each. A property holds for every input, and properties are
what survive a model swap, a temperature change or a prompt rewrite.

When the vendor ships a new checkpoint next quarter, your example tests go red
and tell you almost nothing -- of course the wording changed. A property like
"a stream ends in exactly one terminal event" or "upstream calls never exceed
1 + MAX_RETRIES" is still exactly as true, and still exactly as load-bearing.
That is the difference this file is teaching.

How to work
-----------

Each placeholder calls `pytest.fail` with the property spelled out. Replace
the call with real assertions. `test_schema_forbids_unknown_fields` at the top
ships green as the worked example of the shape to aim for.

You have the whole suite as raw material: `make_service`, `mock_server`,
`env` and `SAMPLES` are all available (see `tests/conftest.py`), and the
mock's `/stats` is how you observe what actually reached upstream rather than
what you hoped would.

Mark anything that drives fault injection `@pytest.mark.chaos` and anything
that measures wall-clock time `@pytest.mark.latency`, so `make test-fast`
stays fast.
"""

from __future__ import annotations

import asyncio

import pytest

from app.schema import TriageResult
from tests.conftest import SAMPLE_TEXT, SAMPLES_DIR

SAMPLES = sorted(SAMPLES_DIR.glob("*.txt"))
SEVERITIES = ["trivial", "low", "medium", "high", "critical"]


def test_schema_forbids_unknown_fields():
    """`extra="forbid"` is load-bearing: silent field drift is the enemy.

    GREEN at checkout -- the worked example. A property stated as a law about
    the contract, not as a fact about one response.
    """
    with pytest.raises(Exception):
        TriageResult.model_validate(
            {
                "severity": "high",
                "component": "auth",
                "steps": [],
                "confidence": 0.5,
                "urgency": "asap",          # not in the contract
            }
        )


# --- write these -------------------------------------------------------------


async def test_output_always_satisfies_the_contract(make_service, env):
    """PROPERTY: every successful extraction validates, for every input.

    Stage 5 behaviour: at FAULT_RATE=0.4 some samples need retry, so with the
    pass-through stubs this is red until stage 5 is done. Write it now.

    Not "the ten samples give sensible answers" -- that is the golden set's job.
    This says the shape is guaranteed regardless of what came in. Run it over
    every file in SAMPLES, at FAULT_RATE=0.4 so it holds under chaos too.

    Consider also: does a result survive a round-trip through JSON unchanged?
    The idempotency cache promises byte-identical replays, which is only true
    if serialisation is lossless.
    """
    pytest.fail(
        "stage 4: assert that triage() returns a valid TriageResult for every "
        "sample -- severity in range, component non-empty, confidence in [0,1], "
        "steps a list of strings -- and that model_validate_json(model_dump_json()) "
        "round-trips to an equal object."
    )


async def test_hostile_input_never_crashes_the_service(make_service, env):
    """PROPERTY: no input produces an unhandled exception.

    Either a valid TriageResult or a typed error the service layer knows how
    to turn into a status code. A KeyError or a bare ValueError escaping means
    some input can take the process down -- and inputs come from users.

    Feed it the nasty ones: empty, whitespace, a single character, 5000
    characters, emoji, "null", "{}", a partial JSON object, HTML, SQL, control
    characters.
    """
    pytest.fail(
        "stage 4: parametrise over hostile inputs and assert each either returns "
        "a TriageResult or raises one of ProviderError / RetryExhaustedError / "
        "CircuitOpenError / SpendCeilingExceeded. Anything else is a crash."
    )


async def test_idempotency_is_reflexive(make_service, env):
    """PROPERTY: for any key, triage(text, key) == triage(text, key).

    Stage 5 behaviour: with its pass-through stubs this is red until stage 5
    is done. Write it now; that is the point of a property.

    The cache's contract stated as a law rather than as one example. Check the
    bytes, not just equality -- `model_dump_json()`.
    """
    pytest.fail(
        "stage 4: for several samples, call triage() twice with the same idem_key "
        "and assert the second is a cache hit returning byte-identical JSON."
    )


async def test_retry_never_exceeds_its_budget(make_service, mock_server, env):
    """PROPERTY: upstream calls per request <= 1 + MAX_RETRIES, for any input.

    Stage 5 behaviour: with its pass-through stubs this is red until stage 5
    is done. Write it now; that is the point of a property.

    State it over several inputs AND several values of MAX_RETRIES, so it is a
    law about the retry layer rather than a fact about one request. Failing the
    request is allowed; overspending the budget is not.

    `mock_server.stats()["calls"]` is your evidence.
    """
    pytest.fail(
        "stage 4: for MAX_RETRIES in (0, 2), against an upstream that always fails "
        "transiently (FAULT_RATE=1.0, FAULT_MODE=rate_limit), reset /stats, expect "
        "RetryExhaustedError, and assert calls == 1 + MAX_RETRIES. 'At most' alone "
        "is satisfied by a retry that never retries."
    )


async def test_concurrency_never_exceeds_the_limit(make_service, mock_server, env):
    """PROPERTY: the mock's peak concurrency <= MAX_CONCURRENCY, at any limit.

    Assert against `mock_server.stats()["peak_concurrency"]`, not your own
    counter -- the mock is the only witness to what actually arrived.
    """
    pytest.fail(
        "stage 4: for a couple of values of MAX_CONCURRENCY, gather several "
        "concurrent triage() calls and assert the mock's peak_concurrency never "
        "exceeded the limit."
    )


async def test_a_stream_ends_in_exactly_one_terminal_event(make_service, env):
    """PROPERTY: every stream ends in exactly one of `done` or `error`.

    Never both, never neither, and the terminal event must be last. A stream
    with no terminal event leaves the caller waiting forever; one with both is
    telling two different stories. An `error` must carry no result.

    Run it at FAULT_RATE=0.4 so both outcomes actually occur.
    """
    pytest.fail(
        "stage 4: for several samples, collect the events from triage_stream() and "
        "assert exactly one terminal event, that it is last, and that a 'done' "
        "carries a TriageResult while an 'error' carries none."
    )


async def test_spend_is_monotonic_and_bounded(make_service, env):
    """PROPERTY: spend only ever increases, and never passes the ceiling.

    Stage 5 behaviour: with its pass-through stubs this is red until stage 5
    is done. Write it now; that is the point of a property.

    Monotonicity is the interesting half: a ledger that reserves and settles has
    two moving numbers, and it is easy to write one that dips when a reservation
    is released.
    """
    pytest.fail(
        "stage 4: run several triage() calls, asserting after each that "
        "ledger.spent_usd went UP (a call costs money) and has not exceeded "
        "ledger.ceiling_usd. 'Never decreased' alone is satisfied by a ledger that "
        "records nothing."
    )
