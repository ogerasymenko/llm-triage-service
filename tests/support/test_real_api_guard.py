"""Pointing either dialect at a vendor's real host costs money, so the config
refuses it without ALLOW_REAL_API=true. Checked here because stage 7 made the
OpenAI dialect required, and the guard used to know only Anthropic's host."""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.anthropic.com",
        "https://api.openai.com/v1",
        "https://eu.api.openai.com/v1",
        "",
    ],
)
def test_real_hosts_need_explicit_consent(monkeypatch, provider, base_url):
    monkeypatch.setenv("PROVIDER", provider)
    monkeypatch.setenv("PROVIDER_BASE_URL", base_url)
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    with pytest.raises(ValueError, match="ALLOW_REAL_API"):
        Settings()

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    assert Settings().targets_real_api


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8081", "http://127.0.0.1:8081/v1", "http://api.openai.com.evil.test"],
)
def test_anything_else_is_not_the_real_api(monkeypatch, provider, base_url):
    monkeypatch.setenv("PROVIDER", provider)
    monkeypatch.setenv("PROVIDER_BASE_URL", base_url)
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    assert not Settings().targets_real_api
