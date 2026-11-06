"""A circuit breaker.

GIVEN, complete. Participants read this rather than write it: after N
consecutive failures it stops calls reaching upstream at all. Retrying a dead
dependency is how one outage becomes two.

Only transient failures count toward the threshold. A malformed request says
nothing about upstream health, and tripping on one would let a single bad input
take the service down for everyone.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

State = Literal["closed", "open", "half_open"]


class CircuitOpenError(RuntimeError):
    """Carries retry_after so the 503 can tell callers when to come back."""

    def __init__(self, message: str, *, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class CircuitBreaker:
    """Trips after `failure_threshold` consecutive transient failures."""

    failure_threshold: int = 5
    recovery_time_s: float = 30.0
    name: str = "upstream"

    _state: State = field(default="closed", init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> State:
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def _now(self) -> float:
        return time.monotonic()

    async def before_call(self) -> None:
        async with self._lock:
            if self._state == "open":
                elapsed = self._now() - self._opened_at
                remaining = self.recovery_time_s - elapsed
                if remaining > 0:
                    raise CircuitOpenError(
                        f"circuit breaker open for {self.name}; upstream is failing",
                        retry_after=max(1.0, round(remaining, 3)),
                    )
                # Recovery window elapsed: let exactly one call through to see
                # whether upstream came back.
                self._state = "half_open"

    async def record_success(self) -> None:
        async with self._lock:
            self._state = "closed"
            self._consecutive_failures = 0

    async def record_failure(self) -> None:
        async with self._lock:
            self._consecutive_failures += 1
            if self._state == "half_open":
                # The trial call failed: upstream is still sick. Re-open and
                # restart the clock rather than letting a stampede through.
                self._state = "open"
                self._opened_at = self._now()
                return
            if self._consecutive_failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = self._now()

    async def reset(self) -> None:
        """For tests and for an operator who knows something we do not."""
        async with self._lock:
            self._state = "closed"
            self._consecutive_failures = 0
            self._opened_at = 0.0
