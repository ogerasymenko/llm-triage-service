"""The Anthropic adapter. STUB — stages 1 and 3.

Your job: drive the real `anthropic` SDK and satisfy `providers.base.Provider`.
This is the only file in `app/` that may import `anthropic`; everything above
it is written against the Protocol, which is what lets stage 7 add an OpenAI
adapter without touching the reliability layers.

Run `make test-fast` while you work, `make test` to check you are done.

What the tests are asking for
-----------------------------

`extract_with_usage(text, idem_key=...)` -> `(TriageResult, input_tokens,
output_tokens)`. One call in, exactly ONE upstream request out. The token
counts feed the spend ledger, so they must be the real ones off the response.

`extract(text, idem_key=...)` -> `TriageResult`. The same thing without the
usage numbers.

`stream(text)` -> async iterator of `StreamEvent`. Yield `start`, then a
`delta` per text chunk, then exactly one terminal event: `done` carrying a
validated result, or `error`. A stream that dies mid-object must yield
`error`. Never `done` with a partial object -- half a TriageResult has the
same shape as a whole one and the caller cannot tell.

Four things that are easy to get wrong
--------------------------------------

1. Build the client with `max_retries=0`. The SDK's default is 2. Leave it
   alone and you have two retry layers multiplying: `(1 + MAX_RETRIES) x 3`
   upstream calls for one request, with MAX_RETRIES appearing to do nothing.
   `reliability/retry.py` owns retrying because only it can see the breaker,
   the ledger and the cache. Watch `make stats` if you doubt it.

2. Translate SDK exceptions at this boundary. The service layer decides what
   to retry from `TransientProviderError` vs `PermanentProviderError` and
   must never see an `anthropic.*` type. Transient: rate limit, timeout,
   connection failure, 5xx, overload -- carry `retry_after` when the server
   sent one. Permanent: 4xx, refusal, output that will not validate.

3. `stop_reason` is not always `end_turn`. `max_tokens` means the
   object is cut off mid-JSON; say so, rather than letting it surface as a
   baffling JSON parse error. `refusal` means the model declined. Both are
   permanent -- retrying produces the identical result at four times the cost.

4. `temperature` is not in this SDK version's typed signature. `anthropic
   ==1.1.0` dropped it from `messages.create()`, but the wire API still takes
   it and the workshop turns it. Pass it through `extra_body` or `TEMPERATURE`
   silently stops meaning anything.

Ask the mock for the schema with `output_config`:

    {"format": {"type": "json_schema", "schema": triage_result_json_schema()}}
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from app.config import Settings, get_settings
from app.schema import StreamEvent, TriageResult, triage_result_json_schema

SYSTEM_PROMPT = (
    "You are a bug triage assistant. Read the report and return a single JSON "
    "object matching the provided schema. Judge severity by user impact, name "
    "the one component most likely at fault, and give the repro steps you can "
    "actually see in the text. Set confidence to how sure you are; a vague or "
    "noisy report should score low. Return the JSON object and nothing else."
)


class AnthropicProvider:
    """`Provider` implementation driving the real `anthropic` SDK.

    Holds no reliability logic of its own beyond error translation: one call in,
    one upstream request out. Retry, idempotency, concurrency and spend live in
    `app/reliability/` so every future adapter shares them.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        raise NotImplementedError(
            "stage 1: build an anthropic.AsyncAnthropic client here. Use "
            "settings.provider_base_url as base_url, settings.request_timeout_s "
            "as timeout, and max_retries=0 -- reliability/retry.py owns retrying, "
            "and leaving the SDK's default of 2 on gives you both."
        )

    async def aclose(self) -> None:
        raise NotImplementedError("stage 1: close the underlying SDK client")

    async def extract(self, text: str, *, idem_key: str | None = None) -> TriageResult:
        raise NotImplementedError(
            "stage 1: one upstream call returning a validated TriageResult. No "
            "retrying and no caching here -- those live above you."
        )

    async def extract_with_usage(
        self, text: str, *, idem_key: str | None = None
    ) -> tuple[TriageResult, int, int]:
        raise NotImplementedError(
            "stage 1: like extract(), but also return (input_tokens, output_tokens) "
            "from the response's usage block so the spend ledger can charge the call."
        )

    async def stream(self, text: str) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError(
            "stage 3: yield StreamEvent(type='start'), then 'delta' per chunk, then "
            "exactly one of 'done' (with a validated result) or 'error'. A truncated "
            "stream must end in 'error' -- never a partial object dressed as an answer. "
            "Set `transient` on the error: True if the same request could succeed "
            "later (truncation, 429, stall), False if it never will (refusal, "
            "max_tokens, invalid output). The breaker counts only transient ones. "
            "Note: a stream cut off mid-response raises httpx2.RemoteProtocolError "
            "(both SDKs sit on httpx2, not httpx), and the SDK does not wrap it."
        )
        yield  # pragma: no cover - makes this an async generator
