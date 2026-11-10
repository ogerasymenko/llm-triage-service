"""FastAPI application: GET /health, POST /triage, POST /triage/stream.

The routes are thin on purpose. Everything interesting -- retry, idempotency,
the semaphore, the spend ceiling, the breaker -- lives in `app/service.py`
and `app/reliability/`. A route's job is to translate between HTTP and that
service, including translating each failure mode into the status code that
tells the caller what to do about it:

    503 + Retry-After   breaker open, or retries exhausted -- come back later
    402                 spend ceiling -- a human must raise the limit
    422                 the model's output did not validate -- do not retry
    504                 upstream stalled past REQUEST_TIMEOUT_S
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings, get_settings
from app.providers.base import PermanentProviderError, TransientProviderError
from app.reliability.breaker import CircuitOpenError
from app.reliability.limits import SpendCeilingExceeded
from app.reliability.retry import RetryExhaustedError
from app.schema import TriageRequest, TriageResult
from app.service import TriageService

router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resolve settings at boot; build the service on first use.

    Settings stay eager, so a bad environment still fails loudly at boot rather
    than on some unlucky first request.

    The service does NOT, and that is the whole point: on a fresh checkout the
    provider is a stub that raises NotImplementedError from its constructor.
    Building it here would take the process down before uvicorn finished
    starting, so `make dev` would never serve anything and `/health` -- which
    deliberately does not touch the provider -- would be unreachable. A
    participant would hit that on the Quick start, before writing a line.
    """
    app.state.settings = get_settings()
    app.state.service = None
    try:
        yield
    finally:
        if app.state.service is not None:
            await app.state.service.aclose()


app = FastAPI(
    title="triage-service",
    version="0.1.0",
    summary="Unstructured text in, a strict TriageResult out.",
    lifespan=lifespan,
)


class ProviderNotReady(RuntimeError):
    """The adapter is still a stub. Carries the message its author left."""


def get_service(request: Request) -> TriageService:
    """The service, built on first use.

    Raises ProviderNotReady while the adapter is unimplemented, which the routes
    turn into a 501 naming the stage. Anything else propagates: a genuinely
    broken configuration should not masquerade as unfinished homework.
    """
    service = getattr(request.app.state, "service", None)
    if service is not None:
        return service
    try:
        service = TriageService(settings=get_settings())
    except NotImplementedError as exc:
        raise ProviderNotReady(str(exc)) from exc
    request.app.state.service = service
    return service


@router.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness probe. Green means the process booted with valid settings.

    Deliberately does not touch the provider or Redis: this endpoint answers
    "is the service up", not "is upstream healthy". Confusing the two is how
    you get a health check that flaps whenever a vendor sneezes.
    """
    settings: Settings = get_settings()
    return {
        "status": "ok",
        "service": "triage-service",
        "version": app.version,
        "provider": settings.provider,
        "model": settings.model,
    }


@router.get("/limits", tags=["ops"])
async def limits(request: Request) -> dict:
    """A teaching surface: what the reliability layers currently believe."""
    service = get_service(request)
    return {
        "max_concurrency": service.limits.limiter.max_concurrency,
        "peak_in_flight": service.limits.limiter.peak_in_flight,
        "in_flight": service.limits.limiter.in_flight,
        "spend_ceiling_usd": service.limits.ledger.ceiling_usd,
        "spent_usd": round(service.limits.ledger.spent_usd, 6),
        "committed_usd": round(service.limits.ledger.committed_usd, 6),
        "rate_limit": (
            None
            if service.limits.rate is None
            else {
                "max_requests": service.limits.rate.max_requests,
                "window_s": service.limits.rate.window_s,
                "in_window": service.limits.rate.in_window,
                "waited_s": round(service.limits.rate.waited_s, 3),
            }
        ),
        "max_retries": service.policy.max_retries,
        "breaker_state": service.breaker.state,
        "breaker_consecutive_failures": service.breaker.consecutive_failures,
        "fallback_provider": type(service.fallback).__name__ if service.fallback else None,
        "fallback_breaker_state": service.fallback_breaker.state if service.fallback else None,
        "fallbacks_served": service.fallbacks_served,
    }


@app.exception_handler(ProviderNotReady)
async def provider_not_ready(request: Request, exc: ProviderNotReady) -> JSONResponse:
    """501 while the adapter is a stub -- the stage-1 contract.

    Registered as a handler rather than a try/except at each call site so a route
    added later cannot forget it and reintroduce the crash.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "detail": (
                "This endpoint needs the provider adapter, which is stage 1. "
                "Implement app/providers/anthropic_p.py; /health stays green "
                "meanwhile because it does not touch the provider."
            ),
            "stage": 1,
            "from_the_stub": str(exc),
        },
    )


def _error(status: int, code: str, detail: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": code, "detail": detail}, headers=headers or {}
    )


@router.post("/triage", tags=["triage"], responses={200: {"model": TriageResult}})
async def triage(
    request: Request,
    body: TriageRequest,
    idempotency_key: str | None = Header(default=None, alias="idempotency-key"),
) -> JSONResponse:
    """Triage one piece of text into a strict `TriageResult`.

    Send an `Idempotency-Key` header to make the call safe to retry: the same
    key returns the identical object and makes no second upstream call. The
    `x-idempotent-replay` response header says which of the two you got.
    """
    service = get_service(request)
    try:
        result, cached = await service.triage(body.text, idem_key=idempotency_key)
    except CircuitOpenError as exc:
        return _error(
            503,
            "circuit_open",
            str(exc),
            headers={"retry-after": str(int(max(1, exc.retry_after)))},
        )
    except SpendCeilingExceeded as exc:
        return _error(402, "spend_ceiling_exceeded", str(exc))
    except RetryExhaustedError as exc:
        return _error(
            503,
            "upstream_unavailable",
            f"{exc} (last error: {exc.last_error})",
            headers={"retry-after": "1"},
        )
    except PermanentProviderError as exc:
        return _error(422, "invalid_provider_output", str(exc))
    except TransientProviderError as exc:
        return _error(503, "upstream_unavailable", str(exc), headers={"retry-after": "1"})

    return JSONResponse(
        content=json.loads(result.model_dump_json()),
        headers={"x-idempotent-replay": "true" if cached else "false"},
    )


async def _chain(first, rest):
    if first is not None:
        yield first
    async for event in rest:
        yield event


@router.post("/triage/stream", tags=["triage"])
async def triage_stream(request: Request, body: TriageRequest):
    """Stream the triage result as server-sent events.

    Events mirror `StreamEvent`: `start`, `delta`*, then exactly one of
    `done` (carrying a validated object) or `error`. A truncated upstream
    stream ends in `error` -- never a partial object dressed up as an answer.
    """
    service = get_service(request)

    try:
        await service.breaker.before_call()
    except CircuitOpenError as exc:
        return _error(
            503,
            "circuit_open",
            str(exc),
            headers={"retry-after": str(int(max(1, exc.retry_after)))},
        )

    # Pull the first event before committing to a 200: the spend ceiling is
    # checked on it, and once a StreamingResponse starts there is no status
    # code left to change.
    events = service.triage_stream(body.text)
    try:
        first = await events.__anext__()
    except SpendCeilingExceeded as exc:
        return _error(402, "spend_ceiling_exceeded", str(exc))
    except StopAsyncIteration:
        first = None

    async def body_iter():
        try:
            async for event in _chain(first, events):
                payload = json.loads(event.model_dump_json(exclude_none=True))
                yield f"event: {event.type}\ndata: {json.dumps(payload)}\n\n".encode()
        except Exception as exc:  # a failure after headers went out
            payload = {"type": "error", "error": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode()

    return StreamingResponse(
        body_iter(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


app.include_router(router)
