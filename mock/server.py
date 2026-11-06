"""An Anthropic-dialect HTTP mock.

The real `anthropic` client, pointed here with base_url, has to work without
complaint -- correct event names, correct payload shapes, a plausible usage
block. That cannot be eyeballed, so the suite drives the actual SDK against it
rather than asserting on what this file emits.

/stats is the teaching instrument. Participants read those counters to find
their own double-retry bugs, so a racy increment teaches a wrong lesson with
total confidence. Hence the lock around numbers nobody would otherwise contend.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from collections import Counter, deque
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mock.scenarios import (
    decide_fault,
    estimate_tokens,
    extract_text,
    generate_result,
    request_fingerprint,
)

# Haiku 4.5 list price, USD per million tokens.
PRICE_IN_PER_MTOK = 1.0
PRICE_OUT_PER_MTOK = 5.0

# Window in which two identical requests count as a retry, not a fresh call.
RETRY_WINDOW_S = 60.0

# How long the `timeout` fault holds the connection. Longer than any sane
# client timeout, short enough that a wedged test eventually dies.
#
# Overridable via MOCK_TIMEOUT_HOLD_S because the client giving up does not
# make the server let go: the connection stays held for the full duration, and
# anything that waits on it at teardown (closing an httpx client, for instance)
# inherits that wait. The suite shortens it to a couple of seconds -- still far
# longer than the sub-second REQUEST_TIMEOUT_S those tests set, so the fault is
# every bit as real, just not 30 seconds of dead time per stalled request.
DEFAULT_TIMEOUT_FAULT_HOLD_S = 30.0


# A call that arrives a hair inside the window still counts as outside it. The
# client stamps a call when it sends it and the mock when it arrives, and those
# two clocks differ by however long the request spent in transit. Without the
# grace a correct client limiter, sending its Nth call exactly one window after
# its first, would be throttled whenever that first request happened to travel
# slower -- a test that fails on scheduling noise teaches nothing.
THROTTLE_GRACE_S = 0.1


def _rate_limit() -> tuple[int, float]:
    """MOCK_RATE_LIMIT_REQUESTS per MOCK_RATE_LIMIT_WINDOW_S; 0 requests = off.

    Read per request, like MOCK_LATENCY_S, so a test can change it without
    restarting the mock.
    """
    try:
        limit = max(0, int(os.environ.get("MOCK_RATE_LIMIT_REQUESTS", "") or 0))
        window = float(os.environ.get("MOCK_RATE_LIMIT_WINDOW_S", "") or 60.0)
    except ValueError:
        return 0, 60.0
    return limit, max(window, THROTTLE_GRACE_S * 2)


def _timeout_hold_s() -> float:
    try:
        return max(0.0, float(os.environ.get("MOCK_TIMEOUT_HOLD_S", "") or DEFAULT_TIMEOUT_FAULT_HOLD_S))
    except ValueError:
        return DEFAULT_TIMEOUT_FAULT_HOLD_S


class Stats:
    """Counters behind a lock: accuracy matters more than speed here."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.calls_by_outcome: Counter[str] = Counter()
            self.retries_observed = 0
            self.peak_concurrency = 0
            self._in_flight = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.header_overrides_refused = 0
            # fingerprint/idem-key -> (last_seen_monotonic, count)
            self._seen: dict[str, tuple[float, int]] = {}
            # Arrival times of calls the rate limit let through.
            self._admitted: deque[float] = deque()

    def enter(self) -> None:
        with self._lock:
            self._in_flight += 1
            self.peak_concurrency = max(self.peak_concurrency, self._in_flight)

    def leave(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def record_call(self, key: str, now: float) -> int:
        """Count a call; return which attempt of `key` this is (1-based).

        A repeat within the window is a retry, and the attempt number it returns
        is what makes the mock's faults transient: attempt 1 may be rate
        limited while attempt 2 of the same body succeeds, deterministically.
        """
        with self._lock:
            self.calls += 1
            last = self._seen.get(key)
            if last is not None and (now - last[0]) <= RETRY_WINDOW_S:
                self.retries_observed += 1
                attempt = last[1] + 1
                self._seen[key] = (now, attempt)
                return attempt
            self._seen[key] = (now, 1)
            return 1

    def admit(self, now: float, limit: int, window_s: float) -> float | None:
        """Let a call through, or return how long until the window has room.

        Throttled calls do not occupy the window. A provider that counted its
        own refusals would punish a client for retrying, and the client then
        could never get back under the limit by waiting.
        """
        if limit <= 0:
            return None
        with self._lock:
            horizon = window_s - THROTTLE_GRACE_S
            while self._admitted and now - self._admitted[0] >= horizon:
                self._admitted.popleft()
            if len(self._admitted) < limit:
                self._admitted.append(now)
                return None
            return self._admitted[0] + horizon - now

    def record_override_refused(self) -> None:
        with self._lock:
            self.header_overrides_refused += 1

    def record_outcome(self, outcome: str) -> None:
        with self._lock:
            self.calls_by_outcome[outcome] += 1

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            cost = (
                self.input_tokens / 1_000_000 * PRICE_IN_PER_MTOK
                + self.output_tokens / 1_000_000 * PRICE_OUT_PER_MTOK
            )
            return {
                "calls": self.calls,
                "calls_by_outcome": dict(self.calls_by_outcome),
                "retries_observed": self.retries_observed,
                "peak_concurrency": self.peak_concurrency,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "estimated_cost_usd": round(cost, 6),
                "header_overrides_refused": self.header_overrides_refused,
                "header_override_allowed": _header_override_allowed(),
            }


STATS = Stats()
app = FastAPI(title="anthropic-mock", version="0.1.0")


def _header_override_allowed() -> bool:
    return _bool_env("MOCK_ALLOW_HEADER_OVERRIDE", True)


def _bool_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _fault_config(request: Request | None = None) -> tuple[float, str, int]:
    """Env supplies the defaults; x-mock-fault-* headers override per call.

    Those headers are a MOCK-ONLY extension -- the real API ignores them, so an
    adapter that depends on one is not exercising the protocol it claims to
    speak. They let one running mock serve both the clean path and a specific
    fault, which is why the gate script and a couple of tests use them.

    MOCK_ALLOW_HEADER_OVERRIDE=false (the graded run) makes the mock ignore them
    entirely. Otherwise a failing reliability test could be turned green by
    sending `x-mock-fault-mode: none` instead of writing the retry logic.
    """
    try:
        rate = float(os.environ.get("FAULT_RATE", "") or 0.0)
    except ValueError:
        rate = 0.0
    mode = os.environ.get("FAULT_MODE", "") or "all"
    try:
        seed = int(os.environ.get("SEED", "") or 1337)
    except ValueError:
        seed = 1337

    if request is not None and not _header_override_allowed():
        # Ignored, but never silently: /stats reports the count and the
        # response carries x-mock-header-override: refused.
        if request.headers.get("x-mock-fault-mode") or request.headers.get("x-mock-fault-rate"):
            STATS.record_override_refused()

    if request is not None and _header_override_allowed():
        header_mode = request.headers.get("x-mock-fault-mode")
        if header_mode == "none":
            return 0.0, "all", seed  # explicit opt-out: this call must not fault
        if header_mode:
            mode = header_mode
            rate = 1.0  # naming a mode per-request means "definitely do this"
        header_rate = request.headers.get("x-mock-fault-rate")
        if header_rate:
            try:
                rate = float(header_rate)
            except ValueError:
                pass

    return max(0.0, min(1.0, rate)), mode, seed


def _schema_from(body: dict[str, Any]) -> dict[str, Any] | None:
    config = body.get("output_config") or {}
    fmt = config.get("format") or {}
    if fmt.get("type") == "json_schema":
        return fmt.get("schema")
    return None


def _message_payload(
    body: dict[str, Any], result: dict[str, Any], input_tokens: int
) -> tuple[dict[str, Any], str, int]:
    """Build a wire-shaped `Message` plus the serialised text and its cost."""
    text = json.dumps(result, separators=(",", ":"))
    output_tokens = estimate_tokens(text)
    fingerprint = request_fingerprint(body)
    message = {
        "id": f"msg_{fingerprint[:24]}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "claude-haiku-4-5"),
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
    return message, text, output_tokens


def _error(status: int, err_type: str, message: str, headers: dict | None = None):
    """An Anthropic-shaped error envelope, which the SDK parses into its
    typed exceptions (APIStatusError subclasses)."""
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": err_type, "message": message}},
        headers=headers or {},
    )


def _sse(event: str, data: dict[str, Any]) -> bytes:
    """Exact wire form the SDK's parser expects; spacing is not cosmetic."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    now = time.monotonic()

    # Idempotency key, when the client sends one, is the better identity for
    # retry detection than the body hash: it is exactly what it means.
    idem_key = request.headers.get("idempotency-key") or request.headers.get(
        "anthropic-idempotency-key"
    )
    fingerprint = request_fingerprint(body)
    # Retry identity includes the transport: a unary call and a streamed call
    # of the same prompt are two different requests, not a retry of one. The
    # fingerprint itself stays stream-agnostic so both paths get the same
    # content and the same fault verdict.
    retry_identity = idem_key or f"{fingerprint}:{'stream' if body.get('stream') else 'unary'}"
    attempt = STATS.record_call(retry_identity, now)

    wait_s = _throttle(now)
    if wait_s is not None:
        return _error(
            429,
            "rate_limit_error",
            "Number of requests has exceeded your rate limit.",
            headers={"retry-after": _retry_after(wait_s)},
        )

    rate, mode, seed = _fault_config(request)
    fault = decide_fault(
        fingerprint, fault_rate=rate, fault_mode=mode, seed=seed, attempt=attempt
    )
    streaming = bool(body.get("stream"))

    prompt_text = extract_text(body)
    input_tokens = estimate_tokens(prompt_text) + 12  # + system/scaffold overhead

    if fault == "rate_limit":
        STATS.record_outcome("rate_limit")
        return _error(
            429,
            "rate_limit_error",
            "Number of requests has exceeded your rate limit.",
            headers={"retry-after": "1"},
        )

    if fault == "timeout":
        STATS.record_outcome("timeout")
        STATS.enter()
        try:
            await asyncio.sleep(_timeout_hold_s())
        finally:
            STATS.leave()
        return _error(504, "timeout_error", "Upstream stalled.")

    bad_content = fault == "bad_content"
    result = generate_result(body, seed=seed, bad_content=bad_content)
    message, text, output_tokens = _message_payload(body, result, input_tokens)
    if fault == "refusal":
        # What the real API does when the model declines: a normal 200 whose
        # stop_reason says so. Permanent -- the same input refuses again.
        text = "I can't help with that request."
        output_tokens = estimate_tokens(text)
        message["content"] = [{"type": "text", "text": text}]
        message["stop_reason"] = "refusal"
        message["usage"]["output_tokens"] = output_tokens

    if streaming:
        truncate = fault == "truncated_stream"
        STATS.record_outcome("truncated_stream" if truncate else fault if fault == "refusal" else "ok")
        return StreamingResponse(
            _stream_body(
                message, text, input_tokens, output_tokens,
                truncate=truncate, stop_reason=message["stop_reason"],
            ),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-mock-fault": fault or "none"},
        )

    if fault == "truncated_stream":
        # Nothing to truncate on a unary call; degrade to a retryable 500 so the
        # fault still exercises the client's retry path.
        STATS.record_outcome("server_error")
        return _error(500, "api_error", "Internal server error.")

    override_state = "allowed" if _header_override_allowed() else "refused"
    STATS.record_outcome("bad_content" if bad_content else fault if fault == "refusal" else "ok")
    STATS.record_tokens(input_tokens, output_tokens)
    STATS.enter()
    try:
        await asyncio.sleep(_latency_s())
    finally:
        STATS.leave()
    return JSONResponse(
        content=message,
        headers={"x-mock-fault": fault or "none", "x-mock-header-override": override_state},
    )


def _throttle(now: float) -> float | None:
    """Apply the mock's rate limit; the seconds to wait if this call is over it.

    Checked after record_call and before the fault decision: a throttled call
    did reach upstream, so it counts in `calls`, but it is the provider's
    limit speaking, not an injected fault, so it gets its own outcome.
    """
    limit, window_s = _rate_limit()
    wait_s = STATS.admit(now, limit, window_s)
    if wait_s is not None:
        STATS.record_outcome("throttled")
    return wait_s


def _retry_after(wait_s: float) -> str:
    # Whole seconds, as the real APIs send it, rounded up: a client that honours
    # the header must never arrive back early.
    return str(max(1, math.ceil(wait_s)))


def _latency_s() -> float:
    """Defaults to 0 so unit tests stay fast; the concurrency stage sets it."""
    try:
        return max(0.0, float(os.environ.get("MOCK_LATENCY_S", "") or 0.0))
    except ValueError:
        return 0.0


async def _stream_body(
    message: dict[str, Any],
    text: str,
    input_tokens: int,
    output_tokens: int,
    *,
    truncate: bool,
    stop_reason: str = "end_turn",
) -> AsyncIterator[bytes]:
    """message_start, content_block_start, delta*, stop, message_delta, stop.

    Truncation severs the connection mid-object rather than returning cleanly:
    a clean return closes the body normally and the SDK reads the short stream
    as a complete one, which is the opposite of the fault we want.
    """
    STATS.enter()
    try:
        start_message = dict(message)
        start_message["content"] = []
        start_message["stop_reason"] = None
        start_message["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        yield _sse("message_start", {"type": "message_start", "message": start_message})
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )

        latency = _latency_s()
        chunks = _chunk(text, 24)
        cutoff = max(1, len(chunks) // 2) if truncate else len(chunks)
        for i, chunk in enumerate(chunks):
            if i >= cutoff:
                # Truncated: sever the connection mid-object. Returning cleanly
                # here would close the body normally and the SDK would treat the
                # short stream as a complete one; raising makes the ASGI server
                # drop the connection, which is what a real truncation looks
                # like on the wire. No content_block_stop, message_delta or
                # message_stop is ever emitted.
                raise RuntimeError("injected truncated_stream fault")
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk},
                },
            )
            if latency:
                await asyncio.sleep(latency / max(1, len(chunks)))

        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
        STATS.record_tokens(input_tokens, output_tokens)
    finally:
        STATS.leave()


def _chunk(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


@app.get("/stats")
async def stats() -> dict[str, Any]:
    return STATS.snapshot()


@app.post("/stats/reset")
async def stats_reset() -> dict[str, str]:
    STATS.reset()
    return {"status": "reset"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "anthropic-mock"}


# --- OpenAI dialect ---------------------------------------------------------
#
# The same mock, a second wire format. One fault decision, one set of counters,
# two dialects on top -- which is the point of stage 7: the
# adapter changes, the reliability layers above it do not.
#
# Four things genuinely differ from the Anthropic side, and the port is really
# an exercise in noticing them:
#
#   1. structured output   response_format={"type":"json_schema", ...} rather
#                          than output_config
#   2. streaming           opaque chunks carrying choices[].delta.content, with
#                          termination signalled by finish_reason and a literal
#                          `data: [DONE]`, rather than named typed events
#   3. rate limits         same 429 + Retry-After, but the error envelope is
#                          {"error": {...}} with different field names
#   4. usage               absent from stream chunks unless you ask for it with
#                          stream_options={"include_usage": true}


def _openai_error(status: int, err_type: str, message: str, headers: dict | None = None):
    """An OpenAI-shaped error envelope.

    Shaped for the openai SDK's parser, which is not the Anthropic one: it reads
    error.message and error.type off a top-level "error" object.
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": None}},
        headers=headers or {},
    )


def _chat_completion(body: dict[str, Any], text: str, input_tokens: int,
                     output_tokens: int) -> dict[str, Any]:
    fingerprint = request_fingerprint(body)
    return {
        "id": f"chatcmpl-{fingerprint[:24]}",
        "object": "chat.completion",
        "created": 1700000000,
        "model": body.get("model", "gpt-4o-mini"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "refusal": None},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _chat_chunk(body: dict[str, Any], *, delta: dict[str, Any] | None = None,
                finish_reason: str | None = None,
                usage: dict[str, Any] | None = None) -> dict[str, Any]:
    fingerprint = request_fingerprint(body)
    chunk: dict[str, Any] = {
        "id": f"chatcmpl-{fingerprint[:24]}",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": body.get("model", "gpt-4o-mini"),
        "choices": [
            {
                "index": 0,
                "delta": delta if delta is not None else {},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        # Only present when the caller asked for it. Getting this wrong is how a
        # ported spend ledger silently reads zero.
        chunk["usage"] = usage
        chunk["choices"] = []
    return chunk


async def _chat_stream_body(body: dict[str, Any], text: str, input_tokens: int,
                            output_tokens: int, *, truncate: bool,
                            include_usage: bool) -> AsyncIterator[bytes]:
    """Emit OpenAI-dialect SSE: opaque chunks, then `data: [DONE]`.

    Truncation severs the connection mid-object exactly as the Anthropic path
    does -- no finish_reason, no [DONE] -- because returning cleanly would let
    the SDK read a short stream as a complete one.
    """
    STATS.enter()
    try:
        yield f"data: {json.dumps(_chat_chunk(body, delta={'role': 'assistant', 'content': ''}), separators=(',', ':'))}\n\n".encode()

        latency = _latency_s()
        chunks = _chunk(text, 24)
        cutoff = max(1, len(chunks) // 2) if truncate else len(chunks)
        for i, piece in enumerate(chunks):
            if i >= cutoff:
                raise RuntimeError("injected truncated_stream fault")
            payload = _chat_chunk(body, delta={"content": piece})
            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
            if latency:
                await asyncio.sleep(latency / max(1, len(chunks)))

        yield f"data: {json.dumps(_chat_chunk(body, delta={}, finish_reason='stop'), separators=(',', ':'))}\n\n".encode()
        if include_usage:
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            yield f"data: {json.dumps(_chat_chunk(body, usage=usage), separators=(',', ':'))}\n\n".encode()
        yield b"data: [DONE]\n\n"
        STATS.record_tokens(input_tokens, output_tokens)
    finally:
        STATS.leave()


# Divergence worth knowing before you port anything: the two SDKs disagree
# about who owns the "/v1". The anthropic client appends "/v1/messages" to
# base_url, so it wants http://host:8081. The openai client appends only
# "chat/completions", so it wants http://host:8081/v1. Serving both spellings
# means one PROVIDER_BASE_URL works for either dialect and nobody loses an hour
# to a 404 that says nothing.
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """The OpenAI dialect of the same service.

    Deliberately shares decide_fault, the fingerprint and STATS with
    /v1/messages: a participant must be able to point PROVIDER at either dialect
    and read one set of counters. Two counters would make /stats a lie the
    moment someone ran both.
    """
    body = await request.json()
    now = time.monotonic()

    idem_key = request.headers.get("idempotency-key")
    fingerprint = request_fingerprint(body)
    retry_identity = idem_key or f"{fingerprint}:{'stream' if body.get('stream') else 'unary'}"
    attempt = STATS.record_call(retry_identity, now)

    wait_s = _throttle(now)
    if wait_s is not None:
        return _openai_error(
            429,
            "rate_limit_exceeded",
            "Rate limit reached for requests.",
            headers={"retry-after": _retry_after(wait_s)},
        )

    rate, mode, seed = _fault_config(request)
    fault = decide_fault(
        fingerprint, fault_rate=rate, fault_mode=mode, seed=seed, attempt=attempt
    )
    if fault == "refusal":
        # Modelled on the Anthropic side only; OpenAI signals a refusal in a
        # different field, and no stage-7 test needs it yet.
        fault = None
    streaming = bool(body.get("stream"))

    prompt_text = extract_text(body)
    input_tokens = estimate_tokens(prompt_text) + 12

    if fault == "rate_limit":
        STATS.record_outcome("rate_limit")
        return _openai_error(
            429,
            "rate_limit_exceeded",
            "Rate limit reached for requests.",
            headers={"retry-after": "1"},
        )

    if fault == "timeout":
        STATS.record_outcome("timeout")
        STATS.enter()
        try:
            await asyncio.sleep(_timeout_hold_s())
        finally:
            STATS.leave()
        return _openai_error(504, "server_error", "Upstream stalled.")

    bad_content = fault == "bad_content"
    result = generate_result(body, seed=seed, bad_content=bad_content)
    text = json.dumps(result, separators=(",", ":"))
    output_tokens = estimate_tokens(text)

    if streaming:
        truncate = fault == "truncated_stream"
        STATS.record_outcome("truncated_stream" if truncate else "ok")
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(
            _chat_stream_body(
                body, text, input_tokens, output_tokens,
                truncate=truncate, include_usage=include_usage,
            ),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-mock-fault": fault or "none"},
        )

    if fault == "truncated_stream":
        STATS.record_outcome("server_error")
        return _openai_error(500, "server_error", "Internal server error.")

    override_state = "allowed" if _header_override_allowed() else "refused"
    STATS.record_outcome("bad_content" if bad_content else "ok")
    STATS.record_tokens(input_tokens, output_tokens)
    STATS.enter()
    try:
        await asyncio.sleep(_latency_s())
    finally:
        STATS.leave()
    return JSONResponse(
        content=_chat_completion(body, text, input_tokens, output_tokens),
        headers={"x-mock-fault": fault or "none", "x-mock-header-override": override_state},
    )
