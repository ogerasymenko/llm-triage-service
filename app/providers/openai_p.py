"""The OpenAI adapter. **STUB — stage 7.**

Your job: satisfy the same `providers.base.Provider` Protocol the Anthropic
adapter satisfies, over a different wire format. `tests/stages/test_openai_port.py`
sets `PROVIDER=openai` and `MODEL=gpt-4o-mini` itself; the mock serves both
dialects on the same port.

The exercise is not "write another adapter". It is to find out how much of what
you built in stages 1-5 was really about Anthropic. If this costs you one new
file, the abstraction held. If you reach for `service.py` or anything under
`reliability/`, it did not -- and that is the more interesting finding.

The five divergences, which are the actual content of this stage
----------------------------------------------------------------

1. Structured output. Anthropic takes output_config={"format": {...}}. OpenAI
   takes response_format={"type": "json_schema", "json_schema": {"name": ...,
   "strict": ..., "schema": ...}} -- the schema sits under a named wrapper.

2. The stream event model. Anthropic emits named typed events, so "finished" is
   a fact the wire states. OpenAI emits opaque chunks carrying
   choices[].delta.content and you infer completion from finish_reason plus a
   literal `data: [DONE]`. So a truncated OpenAI stream looks exactly like a
   slow one until the connection drops: there is no missing message_stop to
   notice, and your error path carries more weight than it did in stage 3.

3. Rate limits. Both give 429 with Retry-After, but the envelopes and the
   exception classes differ. The service layer must see neither -- translate
   both to TransientProviderError here, or the retry logic starts branching on
   vendor, which is the thing the Protocol exists to prevent.

4. Retry defaults. The openai client ships max_retries=2, exactly as the
   anthropic one does. The trap you disarmed in stage 5 is waiting unchanged in
   the second SDK. Watch `make stats`.

5. Who owns the "/v1" in base_url. anthropic appends /v1/messages, so it wants
   http://host:8081. openai appends only chat/completions, so it wants
   http://host:8081/v1. Aim either client at the other's base URL and you get a
   bare 404 that names nothing.

   The most instructive one, because it is in neither vendor's documentation --
   each is internally consistent and has no occasion to mention the other. It
   only shows up when both clients point at one server, which is what this stage
   does and what a real multi-provider service does. The mock answers both
   spellings; the real APIs will not.
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


class OpenAIProvider:
    """Provider implementation driving the real openai SDK."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        raise NotImplementedError(
            "stage 7: build an openai.AsyncOpenAI client here, with "
            "api_key from settings.openai_api_key, base_url from "
            "settings.provider_base_url and max_retries=0 -- the "
            "second SDK ships the same default of 2 that stage 5 caught."
        )

    async def aclose(self) -> None:
        raise NotImplementedError("stage 7: close the underlying SDK client")

    async def extract(self, text: str, *, idem_key: str | None = None) -> TriageResult:
        raise NotImplementedError(
            "stage 7: one chat.completions call with response_format "
            "carrying the JSON schema, validated into a TriageResult."
        )

    async def extract_with_usage(
        self, text: str, *, idem_key: str | None = None
    ) -> tuple[TriageResult, int, int]:
        raise NotImplementedError(
            "stage 7: like extract(), plus (prompt_tokens, "
            "completion_tokens) off usage -- note OpenAI's names differ from "
            "Anthropic's input_tokens/output_tokens."
        )

    async def stream(self, text: str) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError(
            "stage 7: yield start, then delta per chunk, then exactly one "
            "of done or error, with `transient` set on an error as in stage 3 "
            "(a severed stream raises httpx2.RemoteProtocolError here too). "
            "There is no typed terminal event here -- see divergence 2 in this "
            "file's docstring."
        )
        yield  # pragma: no cover - makes this an async generator
