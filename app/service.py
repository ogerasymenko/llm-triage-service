"""The composition root: the order the reliability layers wrap the provider.

The order is the lesson. From outside in:

    idempotency  ->  spend ceiling  ->  breaker + retry  ->  semaphore  ->  SDK

When the primary is down (retries exhausted or breaker open) the same stack runs
again around the fallback provider, with a breaker of its own.

  * Idempotency outermost. A cached result must cost nothing: no permit, no
    budget, no upstream call. Put it inside the semaphore and a burst of
    duplicate retries queues up to be told something we already knew.

  * Spend ceiling next. It reserves before the attempt, so the refusal
    happens before the money is spent. Note it wraps the *retry loop*: every
    attempt is charged, because every attempt costs real tokens. Charging only
    the successful one is how a retry storm goes unnoticed.

  * Breaker and retry together. Retry asks the breaker before each attempt,
    so an outage stops the loop instead of feeding it.

  * Semaphore innermost, held around the upstream call alone. It bounds what
    reaches the provider, which is the scarce thing.

  * Rate limit inside the semaphore, immediately before the call. Every
    attempt takes a slot, retries included, because the provider counts
    every request it receives.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from app.config import Settings, get_settings
from app.providers.base import Provider
from app.reliability.breaker import CircuitBreaker, CircuitOpenError
from app.reliability.idempotency import IdempotencyStore, build_redis, make_key
from app.reliability.limits import Limits, SpendCeilingExceeded
from app.reliability.retry import RetryExhaustedError, RetryPolicy, with_retry
from app.schema import StreamEvent, TriageResult


def build_provider(settings: Settings, dialect: str | None = None) -> Provider:
    """The one place a dialect name becomes a concrete adapter.

    Imports are local so a process imports only the SDK of the dialect it is
    configured for. Until stage 7 is done, the OpenAI stub never loads for a
    service running the Anthropic dialect.

    Note what is NOT here: no reliability logic, no branching anywhere above
    this function. Everything from the semaphore upward is written against the
    Provider Protocol, so adding a dialect costs one adapter and one line here.
    """
    if (dialect or settings.provider) == "openai":
        from app.providers.openai_p import OpenAIProvider

        return OpenAIProvider(settings)
    from app.providers.anthropic_p import AnthropicProvider

    return AnthropicProvider(settings)


class TriageService:
    """Holds the provider and the reliability layers for the process."""

    def __init__(
        self,
        *,
        provider: Provider | None = None,
        settings: Settings | None = None,
        store: IdempotencyStore | None = None,
        limits: Limits | None = None,
        breaker: CircuitBreaker | None = None,
        fallback: Provider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or build_provider(self.settings)
        self.store = store or IdempotencyStore(build_redis(self.settings))
        self.limits = limits or Limits.from_settings(self.settings)
        self.breaker = breaker or CircuitBreaker(name="anthropic")
        if fallback is None and self.settings.fallback_provider:
            fallback = build_provider(self.settings, self.settings.fallback_provider)
        self.fallback = fallback
        # Its own breaker: the fallback's health is a separate fact. Sharing the
        # primary's would open it on the primary's failures and cut off the very
        # provider we are falling back to.
        self.fallback_breaker = CircuitBreaker(name="fallback")
        self.fallbacks_served = 0
        self.policy = RetryPolicy(max_retries=self.settings.max_retries)
        # Salts the cache key with the settings that change the answer, so a
        # run at a different temperature or model does not serve the previous
        # run's cached result.
        self._key_salt = f"{self.settings.model}:{self.settings.temperature}"

    # --- unary ------------------------------------------------------------

    async def triage(self, text: str, *, idem_key: str | None = None) -> tuple[TriageResult, bool]:
        """Triage `text`. Returns `(result, served_from_cache)`.

        With an idempotency key, the second call returns the stored object and
        makes no upstream call at all.
        """
        key = make_key(text, idem_key=idem_key, salt=self._key_salt)
        return await self.store.get_or_call(
            key, lambda: self._attempt_with_limits(text, idem_key)
        )

    async def _attempt_with_limits(self, text: str, idem_key: str | None) -> TriageResult:
        """One logical request: the primary with its retries, then the fallback.

        Only an outage switches vendors: retries exhausted, or the breaker
        already open. A permanent error (a refusal, output that will not
        validate) propagates untouched -- it is the request's problem, and the
        other vendor would refuse it too.
        """
        try:
            return await with_retry(
                self._attempt_via(self.provider, text, idem_key),
                policy=self.policy,
                breaker=self.breaker,
            )
        except (RetryExhaustedError, CircuitOpenError):
            if self.fallback is None:
                raise
        # Same layers around the fallback: every attempt is reserved against the
        # ceiling and takes a permit and a rate slot, so the second vendor is
        # not a way around any limit.
        result = await with_retry(
            self._attempt_via(self.fallback, text, idem_key),
            policy=self.policy,
            breaker=self.fallback_breaker,
        )
        # Counted on success only: a fallback that failed too served nobody.
        self.fallbacks_served += 1
        return result

    def _attempt_via(self, provider: Provider, text: str, idem_key: str | None):
        async def one_attempt() -> TriageResult:
            # Reserve worst-case budget BEFORE the call. Raises rather than
            # reporting an overrun after the fact.
            reserved = await self.limits.ledger.reserve()
            settled = False
            try:
                async with self.limits.limiter():
                    await self._rate_slot()
                    result, in_tok, out_tok = await provider.extract_with_usage(
                        text, idem_key=idem_key
                    )
                await self.limits.ledger.settle(reserved, in_tok, out_tok)
                settled = True
                return result
            finally:
                if not settled:
                    await self.limits.ledger.release(reserved)

        return one_attempt

    # --- streaming --------------------------------------------------------

    async def triage_stream(
        self, text: str, *, use_fallback: bool = False
    ) -> AsyncIterator[StreamEvent]:
        """Stream a triage result.

        No retry and no cache on this path: a stream has already started
        delivering bytes to the caller, so a transparent retry would replay a
        different answer mid-response. A truncated stream surfaces as an
        `error` event, which is what `test_stream.py` asserts.

        Charged against the spend ceiling like a unary call, but once: no
        retry means one reservation. The worst case is reserved before anything
        is sent (so the refusal raises on the first iteration, before the
        provider is touched), settled from the usage the `done` event carries,
        and released if the stream ends any other way -- an error event, an
        exception, or the client walking away mid-stream.

        The caller is expected to have run `breaker.before_call()` already --
        the route does, so it can answer 503 with a real status code before any
        bytes go out. Once a StreamingResponse has started there is no status
        code left to change. The same moment is the only safe one to switch
        vendors, so the route passes `use_fallback=True` when the primary's
        breaker is open; a stream that fails after it started is not switched.
        """
        provider, breaker = self.provider, self.breaker
        if use_fallback:
            if self.fallback is None:
                raise ValueError("use_fallback=True but no fallback is configured")
            provider, breaker = self.fallback, self.fallback_breaker
        reserved = await self.limits.ledger.reserve()
        settled = False
        upstream_failed = False
        try:
            async with self.limits.limiter():
                await self._rate_slot()
                async for event in provider.stream(text):
                    if event.type == "error" and event.transient is not False:
                        upstream_failed = True
                    if event.type == "done" and not settled:
                        # Settle before yielding: a consumer that stops reading
                        # right after `done` must not leave the reservation held.
                        await self.limits.ledger.settle(
                            reserved, event.input_tokens or 0, event.output_tokens or 0
                        )
                        settled = True
                        if use_fallback:
                            self.fallbacks_served += 1
                    yield event
        finally:
            if not settled:
                await self.limits.ledger.release(reserved)
        # A permanent error still counts as success here: upstream answered,
        # and it was the request that could not be served. Only a transient
        # error is evidence that upstream is sick.
        if upstream_failed:
            await breaker.record_failure()
        else:
            await breaker.record_success()

    async def _rate_slot(self) -> None:
        if self.limits.rate is not None:
            await self.limits.rate.acquire()

    async def aclose(self) -> None:
        await self.provider.aclose()
        if self.fallback is not None:
            await self.fallback.aclose()
