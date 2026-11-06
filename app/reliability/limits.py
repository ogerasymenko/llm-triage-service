"""Three limits: concurrency, rate, and spend. STUB — stages 2 and 5.

This file spans two homework stages. `ConcurrencyLimiter` and `RateLimiter`
are stage 2 (async and backpressure); `SpendLedger` is stage 5 (surviving a bad
provider). Do stage 2's half first -- `test_concurrency.py` needs only those.

Your job: bound how many calls are in flight, how many may start per window,
and how much they may cost.

Every class here is a PASS-THROUGH stub: it runs, and limits nothing. That is
deliberate -- stage 1 drives requests through the whole service, so a stub
that raised would block stage 1 on stages 2 and 5. Each `# stage N:` comment
says what to replace the pass-through with. Your stage's tests are red until
you do, failing on what actually reached the mock.

The semaphore
-------------

Bound concurrency to `MAX_CONCURRENCY`. Hold the permit around the upstream
call ONLY -- not the cache lookup, not response serialisation. A permit held
during work that is not the scarce resource is a permit nobody else can use.

`test_concurrency.py` does not take your word for it. It fires ten requests
and reads `peak_concurrency` off the mock's `/stats`: the highest number the
mock ever had in flight at once. With no faults injected -- which is how the
test runs -- a peak above `MAX_CONCURRENCY` means the semaphore did not bound
what actually reached upstream, whatever the code looks like. It also checks
the run was not merely serialised -- a semaphore is a limit, not a queue of one.

Under timeout faults the reading changes. When the client gives up at
REQUEST_TIMEOUT_S, the permit is rightly released, but the mock is still
holding the abandoned connection until its stall ends. The next call then
overlaps it, and the mock briefly counts one more than your limit. That is not
a leak in your semaphore; it is the upstream still paying for a request you
stopped waiting for -- worth noticing in /stats, and in REPORT.md.

The rate limiter
----------------

A provider bounds how many requests may start in a window and answers 429 past
it. `RateLimiter` keeps us under that limit on our side of the wire: at most
`max_requests` calls start in any `window_s` seconds, and the excess WAITS
rather than being sent. `service.py` already calls `acquire()` immediately
before each upstream attempt; you only build the limiter.

Do not trust a request succeeding as proof. The retry layer recovers from 429s
on its own, just slower and at more cost. The tests switch on the mock's own
limit and read `calls_by_outcome.throttled` from `/stats`: every throttled call
is a request the provider had to refuse for you. Read the mock's limit as a
contract -- "no more than N in any window" -- and check that your design keeps
it, rather than merely averaging to it.

The spend ceiling
-----------------

Raise before the call that would exceed `SPEND_CEILING_USD`, never after.
Noticing you are over budget once the money is gone is a report, not a limit.

The wrinkle: a call's true cost is unknown until the response arrives. So
reserve the worst case up front (`max_tokens` billed at the output rate),
then reconcile to the real usage once you have it. Reserving optimistically and
correcting later lets a burst of concurrent calls sail past the ceiling
together, each one seeing room for itself.

Three operations:

  * `reserve()` -- check `settled + reserved + worst_case` against the
    ceiling and raise `SpendCeilingExceeded` if it would cross. Check against
    committed spend, not just settled, or concurrency defeats you.
  * `settle(reserved, in_tokens, out_tokens)` -- release the reservation and
    add what the call really cost.
  * `release(reserved)` -- give a reservation back when the call failed or
    never happened. Forget this and a burst of retries exhausts the ceiling with
    money nobody spent.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

from app.config import (
    PRICE_INPUT_USD_PER_MTOK,
    PRICE_OUTPUT_USD_PER_MTOK,
    Settings,
    get_settings,
)


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Price one call at the configured rates (Haiku 4.5 list price).

    GIVEN -- the mock's /stats uses the same arithmetic, so both sides agree on
    what a call costs.
    """
    return (
        input_tokens / 1_000_000 * PRICE_INPUT_USD_PER_MTOK
        + output_tokens / 1_000_000 * PRICE_OUTPUT_USD_PER_MTOK
    )


class SpendCeilingExceeded(RuntimeError):
    """The next call would cross the ceiling, so it was not made."""

    def __init__(self, message: str, *, spent_usd: float, ceiling_usd: float) -> None:
        super().__init__(message)
        self.spent_usd = spent_usd
        self.ceiling_usd = ceiling_usd


class SpendLedger:
    """Tracks spend against a ceiling, reserving before each call.

    Reserve -> call -> settle. The reservation is the worst case the call could
    cost; settling replaces it with what it actually cost.
    """

    def __init__(self, ceiling_usd: float, *, max_tokens: int, estimated_input_tokens: int = 1000):
        self.ceiling_usd = ceiling_usd
        self.max_tokens = max_tokens
        self.estimated_input_tokens = estimated_input_tokens
        # stage 5: track settled spend and outstanding reservations separately,
        # behind an asyncio.Lock so concurrent callers cannot each see room for
        # themselves.

    @property
    def spent_usd(self) -> float:
        """What has actually been charged so far."""
        # stage 5: settled spend only, excluding reservations.
        return 0.0

    @property
    def committed_usd(self) -> float:
        """Settled spend plus outstanding reservations."""
        # stage 5: settled + reserved.
        return 0.0

    def worst_case_usd(self) -> float:
        """The most one call could cost: max_tokens out, an estimate in."""
        # stage 5: cost_usd(self.estimated_input_tokens, self.max_tokens).
        return 0.0

    async def reserve(self) -> float:
        """Reserve worst-case budget for one call, or raise before spending it."""
        # stage 5: if settled + reserved + worst_case would exceed the ceiling,
        # raise SpendCeilingExceeded BEFORE the call goes out. Otherwise add the
        # reservation and return it. Checking settled spend alone lets concurrent
        # callers blow past the ceiling together.
        return 0.0

    async def settle(self, reserved: float, input_tokens: int, output_tokens: int) -> float:
        """Replace a reservation with the call's true cost."""
        # stage 5: drop `reserved` from the outstanding total, add
        # cost_usd(input_tokens, output_tokens) to settled spend, return it.
        return 0.0

    async def release(self, reserved: float) -> None:
        """Give a reservation back: the call failed or never happened."""
        # stage 5: drop `reserved` from the outstanding total.

    async def reset(self) -> None:
        # stage 5: zero both settled and reserved.
        pass


class ConcurrencyLimiter:
    """An asyncio semaphore with the peak recorded, for symmetry with /stats."""

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self.peak_in_flight = 0
        # stage 2: hold an asyncio.Semaphore(max_concurrency) and a counter of
        # how many calls are in flight.

    def __call__(self):
        """Use as `async with limiter():` around the upstream call."""
        # stage 2: return an async context manager that acquires a permit on
        # entry and releases it on exit, updating in_flight and peak_in_flight.
        # The mock's peak_concurrency is what the test reads, and it can only
        # agree with your counter if the permit is held around the call itself.
        return contextlib.nullcontext()

    @property
    def in_flight(self) -> int:
        # stage 2: how many permits are currently held.
        return 0


class RateLimiter:
    """At most `max_requests` upstream calls start in any `window_s` seconds."""

    def __init__(self, max_requests: int, window_s: float) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self.max_requests = max_requests
        self.window_s = window_s
        # Total seconds callers spent waiting for a slot; /limits shows it.
        self.waited_s = 0.0
        # stage 2: keep whatever state you need to know how many calls started
        # within the last window_s seconds.

    async def acquire(self) -> None:
        """Return once this call may start; called right before each attempt."""
        # stage 2: if fewer than max_requests calls started within the last
        # window_s, record this one and return. Otherwise wait -- with asyncio,
        # never time.sleep -- until one ages out, and check again. Several
        # callers can be waiting at once; the mock's `throttled` count tells you
        # whether they kept to the limit together.
        return None

    @property
    def in_window(self) -> int:
        # stage 2: how many calls started within the last window_s.
        return 0


@dataclass
class Limits:
    """The limiters, built from settings. GIVEN."""

    limiter: ConcurrencyLimiter
    ledger: SpendLedger
    # None when RATE_LIMIT_REQUESTS=0, so an unconfigured limit costs nothing
    # and cannot fail.
    rate: RateLimiter | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Limits":
        settings = settings or get_settings()
        return cls(
            limiter=ConcurrencyLimiter(settings.max_concurrency),
            ledger=SpendLedger(settings.spend_ceiling_usd, max_tokens=settings.max_tokens),
            rate=(
                RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_s)
                if settings.rate_limit_requests > 0
                else None
            ),
        )
