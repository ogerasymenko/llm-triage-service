"""Booting must not depend on the provider.

This is the check that a fresh clone actually starts. It caught a real
regression: step 3 wired the real provider into the lifespan, so on a stubbed
tree uvicorn died during startup with NotImplementedError. `make dev` served
nothing, `/health` was unreachable, and the api container under compose exited
3 -- all before a participant had written a line.

Every test here runs against a DELIBERATELY STUBBED provider, whatever state
app/providers/anthropic_p.py happens to be in, so it keeps its meaning on both
branches: on main the adapter is already a stub, and on solution these force one.
"""

from __future__ import annotations

import httpx
import pytest

import app.config as config_module
from app.main import ProviderNotReady, app as fastapi_app


@pytest.fixture
def stub_provider(monkeypatch, mock_server):
    """Make the adapter behave as it does at checkout, whatever branch we are on."""

    def unimplemented(*args, **kwargs):
        raise NotImplementedError(
            "stage 1: build an anthropic.AsyncAnthropic client here."
        )

    monkeypatch.setattr("app.service.build_provider", unimplemented)
    fastapi_app.state.service = None
    fastapi_app.state.settings = config_module.get_settings()
    yield
    fastapi_app.state.service = None


async def run_lifespan_startup(app) -> tuple[bool, str]:
    """Drive the ASGI lifespan protocol and report whether startup succeeded.

    httpx's ASGITransport does NOT run lifespan, so a test that only issues
    requests through it never executes the startup path -- which is exactly
    where the regression lived. An earlier version of this file made that
    mistake and passed happily against the bug it was written to catch. So we
    speak the protocol directly.
    """
    received: list[dict] = []
    to_send = [{"type": "lifespan.startup"}]

    async def receive():
        return to_send.pop(0) if to_send else {"type": "lifespan.shutdown"}

    async def send(message):
        received.append(message)

    try:
        await app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send)
    except Exception as exc:                     # startup raised outright
        return False, f"{type(exc).__name__}: {exc}"

    for message in received:
        if message["type"] == "lifespan.startup.failed":
            return False, message.get("message", "")
    return any(m["type"] == "lifespan.startup.complete" for m in received), ""


@pytest.fixture
def stubbed_app(stub_provider):
    transport = httpx.ASGITransport(app=fastapi_app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def test_startup_completes_with_every_stub_in_place(stub_provider):
    """The regression, stated where it actually happened: ASGI startup.

    On a stubbed tree uvicorn must finish starting. When this fails, `make dev`
    serves nothing and the api container exits 3 -- the participant's first two
    commands both die.
    """
    ok, detail = await run_lifespan_startup(fastapi_app)
    assert ok, (
        f"ASGI startup failed on a stubbed tree: {detail}\n"
        f"Boot must not construct the provider; build the service lazily."
    )


async def test_health_is_green_with_every_stub_in_place(stubbed_app):
    """/health answers 200 without touching the provider.

    If this goes red, boot has been made to depend on the adapter again and a
    fresh clone will not start.
    """
    async with stubbed_app as client:
        response = await client.get("/health")

    assert response.status_code == 200, (
        f"/health returned {response.status_code} on a stubbed tree. Boot must not "
        f"construct the provider -- see the lifespan in app/main.py."
    )
    assert response.json()["status"] == "ok"


async def test_triage_answers_501_rather_than_killing_the_process(stubbed_app):
    """POST /triage names the stage instead of crashing."""
    async with stubbed_app as client:
        response = await client.post("/triage", json={"text": "login is broken"})

    assert response.status_code == 501, f"expected 501, got {response.status_code}"
    body = response.json()
    assert body["error"] == "not_implemented"
    assert body["stage"] == 1
    assert "anthropic_p.py" in body["detail"]


async def test_stream_route_also_answers_501(stubbed_app):
    async with stubbed_app as client:
        response = await client.post("/triage/stream", json={"text": "x"})
    assert response.status_code == 501


async def test_lifespan_does_not_build_the_service(stub_provider):
    """A request may build the service lazily; startup may not."""
    ok, detail = await run_lifespan_startup(fastapi_app)
    assert ok, detail
    assert getattr(fastapi_app.state, "service", None) is None, (
        "the TriageService was constructed during boot; it must be built on "
        "first use so a stubbed tree still starts"
    )


async def test_a_broken_config_still_fails_loudly(monkeypatch, mock_server):
    """Settings stay eager -- an unimplemented adapter is not a licence to
    swallow a genuinely bad environment."""
    monkeypatch.setenv("MAX_CONCURRENCY", "0")
    config_module.reset_settings()
    with pytest.raises(ValueError, match="MAX_CONCURRENCY"):
        config_module.get_settings()
    config_module.reset_settings()


async def test_provider_not_ready_only_catches_notimplemented(monkeypatch, mock_server):
    """A real failure must not masquerade as unfinished homework."""
    from app.main import get_service

    def exploding(*args, **kwargs):
        raise RuntimeError("redis is on fire")

    monkeypatch.setattr("app.service.build_provider", exploding)
    monkeypatch.setattr(fastapi_app.state, "service", None, raising=False)

    class FakeRequest:
        app = fastapi_app

    with pytest.raises(RuntimeError, match="redis is on fire"):
        get_service(FakeRequest())
