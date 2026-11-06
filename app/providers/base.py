"""GIVEN. The service layer is written against Provider, never a concrete SDK,
which is what lets stage 7 add an OpenAI adapter without touching it."""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from app.schema import StreamEvent, TriageResult


@runtime_checkable
class Provider(Protocol):

    async def extract(self, text: str, *, idem_key: str | None = None) -> TriageResult:
        """Return a validated `TriageResult` for `text`.

        `idem_key`, when supplied, marks two calls as the same logical
        request: the same key must yield the identical object and must not
        produce a second upstream call.
        """
        ...

    def stream(self, text: str) -> AsyncIterator[StreamEvent]:
        """Yield `StreamEvent`s as the result is produced.

        A stream that dies mid-object must surface an `error` event; it must
        never yield a `done` event carrying a partial or unvalidated object.
        """
        ...


class ProviderError(RuntimeError):
    """Base class for adapter failures the service layer knows how to handle."""


class TransientProviderError(ProviderError):
    """A failure worth retrying: rate limit, timeout, 5xx, truncated stream."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentProviderError(ProviderError):
    """A failure retrying cannot fix: bad request, refusal, schema violation."""
