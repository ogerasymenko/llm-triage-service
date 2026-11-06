"""Retry with backoff. STUB — stage 5.

Your job: retry transient failures, and be the ONLY layer that does.

The stubs below are PASS-THROUGH: they run, and do nothing useful. Stage 1
drives requests through the whole service, so a stub that raised would block
stage 1 on this one. Each `# stage 5:` comment says what replaces it.

The trap
--------

The `anthropic` SDK ships `max_retries=2`. If you write this module and
leave that default alone, one logical request becomes up to
`(1 + MAX_RETRIES) * (1 + 2)` upstream calls -- twelve at the shipped
defaults -- while `MAX_RETRIES` appears to do nothing, because the SDK is
quietly tripling whatever you set.

The bug does not change the outcome. The request still succeeds; it just costs
three times as much, takes three times as long, and hammers an upstream you were
trying to be gentle with. That is why `test_retry.py` asserts on the mock's
call count rather than on whether the call returned, and why the MAX_RETRIES=0
case is the one that catches you. Run `make stats` and read `calls` and
`retries_observed` -- the mock will show you the multiplication directly.

Decide who retries, disable the other, and make `MAX_RETRIES` control whichever
one is live. (The reference solution retries here and builds the SDK client with
`max_retries=0`, because this layer can see the circuit breaker, the spend
ledger and the idempotency cache, and the SDK can see none of them.)

What to implement
-----------------

`RetryPolicy.delay_for(attempt, retry_after=None)` -- exponential backoff
capped at `max_delay_s`. A server-supplied `Retry-After` should win over your
own computation: the server is the only party that actually knows when it will
accept traffic again. Cap it anyway, so a hostile header cannot wedge the
request past the client timeout. Add jitter -- without it, N clients that failed
together retry together and rebuild the thundering herd that caused the failure.

`with_retry(operation, policy=..., breaker=...)` -- run `operation` until it
succeeds or the budget runs out. `operation` must perform exactly one upstream
attempt; if it retries internally you have two layers again and this function
cannot tell.

  * Only `TransientProviderError` is worth retrying. A `PermanentProviderError`
    -- schema violation, refusal, 400 -- must propagate immediately: the same
    call fails the same way, and four identical failures cost four times as much
    to tell you the same thing.
  * Ask the breaker before every attempt. `CircuitOpenError` means stop, not
    try harder, so it must NOT be caught by your transient handler.
  * Record success and failure on the breaker, but only for transient failures --
    a malformed input says nothing about upstream health, and tripping on it
    would let one bad request take the service down for everyone.
  * When the budget is exhausted, raise `RetryExhaustedError` carrying the last
    error.

`max_retries` counts retries, so total attempts is `max_retries + 1`.
`MAX_RETRIES=0` means one attempt and no retrying.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from app.providers.base import TransientProviderError
from app.reliability.breaker import CircuitBreaker, CircuitOpenError

T = TypeVar("T")

BASE_DELAY_S = 0.05
MAX_DELAY_S = 2.0


@dataclass
class RetryPolicy:
    """How many times to try, and how long to wait between attempts."""

    max_retries: int = 3
    base_delay_s: float = BASE_DELAY_S
    max_delay_s: float = MAX_DELAY_S
    jitter: bool = True

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Backoff before retry number `attempt` (1-based)."""
        # stage 5: exponential backoff capped at max_delay_s, with jitter. A
        # server-supplied retry_after should win over your own computation (but
        # still be capped).
        return 0.0


class RetryExhaustedError(RuntimeError):
    """Every attempt failed transiently. Carries the last error."""

    def __init__(self, message: str, *, attempts: int, last_error: BaseException | None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    breaker: CircuitBreaker | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run `operation` until it succeeds, or until the retries run out."""
    # stage 5: retry `operation` up to policy.max_attempts times. Retry only
    # TransientProviderError; let PermanentProviderError propagate at once. Ask
    # `breaker.before_call()` before each attempt and let CircuitOpenError
    # through -- an open breaker means stop, not try harder. Raise
    # RetryExhaustedError when the budget is spent. And make sure exactly ONE
    # layer is retrying: the SDK client must be built with max_retries=0, or
    # MAX_RETRIES will not mean what it says. Check `make stats`.
    return await operation()
