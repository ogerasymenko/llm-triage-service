"""Same request in, same result out: determinism comes from SEED plus a hash of
the request body, never wall-clock time or the random module's global state."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

# Fault modes the mock knows how to inject.
FAULT_MODES = ("rate_limit", "timeout", "truncated_stream", "bad_content")

# Modes injected only when named. A refusal is permanent, not chaos: in `all`
# it would fail requests no retry can rescue and break every chaos test that
# expects to survive, and it would reshuffle every seed's fault sequence.
EXPLICIT_ONLY_MODES = ("refusal",)

# Component keywords, checked in order. First hit wins, so put the specific
# terms ahead of the generic ones.
_COMPONENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("auth", ("login", "logout", "signin", "sign in", "password", "oauth", "token",
              "session", "credential", "2fa", "sso")),
    ("payments", ("payment", "checkout", "invoice", "billing", "charge", "refund",
                  "stripe", "card", "subscription")),
    ("database", ("query", "sql", "deadlock", "migration", "postgres", "index",
                  "transaction", "connection pool")),
    ("api", ("endpoint", "http", "rest", "request", "response", "timeout", "500",
             "502", "503", "rate limit", "payload")),
    ("ui", ("button", "modal", "render", "css", "layout", "screen", "click",
            "dropdown", "form", "page")),
    ("search", ("search", "index", "relevance", "query results", "facet")),
    ("notifications", ("email", "notification", "webhook", "push", "sms", "alert")),
    ("storage", ("upload", "download", "file", "s3", "bucket", "disk", "attachment")),
]

_SEVERITY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("critical", ("data loss", "corrupt", "security", "breach", "outage", "cannot log in",
                  "all users", "production down", "crash on startup", "leak")),
    ("high", ("crash", "500", "fails", "broken", "blocked", "cannot", "unable",
              "error", "exception", "regression")),
    ("medium", ("slow", "intermittent", "sometimes", "delay", "incorrect", "wrong",
                "unexpected")),
    ("low", ("minor", "cosmetic", "typo", "misaligned", "occasionally")),
    ("trivial", ("nit", "whitespace", "wording", "suggestion", "polish")),
]


def request_fingerprint(body: dict[str, Any]) -> str:
    """A stable hash of the semantically meaningful parts of a request.

    Deliberately ignores `stream`: the same prompt asked for streamed and
    non-streamed must land on the same fault decision and the same content, so
    that a participant comparing the two paths is comparing like with like.
    """
    payload = {
        "model": body.get("model"),
        "messages": body.get("messages"),
        "system": body.get("system"),
        # Both dialects' token cap, and both dialects' structured-output
        # request. Naming both keys keeps one request identity across the two
        # wire formats, so a fault decision does not depend on which SDK asked.
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
        "temperature": body.get("temperature"),
        "output_config": body.get("output_config"),
        "response_format": body.get("response_format"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def extract_text(body: dict[str, Any]) -> str:
    chunks: list[str] = []
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chunks.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    chunks.append(block)
    return "\n".join(chunks)


def _unit_interval(fingerprint: str, seed: int, salt: str = "") -> float:
    digest = hashlib.sha256(f"{seed}:{salt}:{fingerprint}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def decide_fault(
    fingerprint: str, *, fault_rate: float, fault_mode: str, seed: int, attempt: int = 1
) -> str | None:
    """Return the fault to inject for this attempt, or `None` for success.

    Deterministic in (fingerprint, seed, attempt). Note the third term: the
    verdict is fixed per attempt, not per request. Attempt 1 of a given body
    always gets the same answer, attempt 2 always gets the same answer, and a
    fresh process replays the whole sequence identically -- so runs are still
    perfectly reproducible.

    Keying on the body alone would make every fault permanent: a rate-limited
    request would be rate-limited on every retry forever, and no correct retry
    implementation could ever go green. That is not what upstreams do. A 429, a
    stalled connection and a severed stream are all transient by nature -- the
    same request a moment later succeeds. Retry only means anything against an
    upstream that can recover, so the mock models one that does.
    """
    if fault_rate <= 0:
        return None
    salt = f"fault:{attempt}"
    if _unit_interval(fingerprint, seed, salt) >= fault_rate:
        return None
    if fault_mode == "all":
        pick = _unit_interval(fingerprint, seed, f"mode:{attempt}")
        return FAULT_MODES[min(int(pick * len(FAULT_MODES)), len(FAULT_MODES) - 1)]
    if fault_mode in FAULT_MODES or fault_mode in EXPLICIT_ONLY_MODES:
        return fault_mode
    return None


def _match(text: str, rules: list[tuple[str, tuple[str, ...]]], fallback: str) -> str:
    lowered = text.lower()
    for label, keywords in rules:
        if any(keyword in lowered for keyword in keywords):
            return label
    return fallback


# A reporter who hedges leaves room for more than one reading, and a sampled
# model takes it some of the time. Unhedged reports, however many keywords they
# brush against, get the first reading every time -- which keeps the
# unambiguous golden cases unambiguous.
_HEDGE = re.compile(r"\b(maybe|not sure|sometimes|randomly|weird|somehow)\b")

# Chance that a hedged report gets an alternative reading, per unit of
# temperature. At TEMPERATURE=0 it is zero: greedy decoding is deterministic.
_MISREAD_PER_UNIT_TEMPERATURE = 0.6


def _component(text: str, body: dict[str, Any], fingerprint: str, seed: int) -> str:
    """The first matching rule, or -- on a hedged report -- sometimes another.

    This is what gives the golden set something to measure. Without it every
    seed scores the same, and "set the threshold by measuring" measures nothing.
    The draw is keyed on (seed, request), so one seed always gives one score and
    different seeds give a spread, the way reruns against a live model would.
    """
    lowered = text.lower()
    readings = [label for label, keywords in _COMPONENT_RULES
                if any(keyword in lowered for keyword in keywords)]
    if not readings:
        return "unknown"
    temperature = float(body.get("temperature") or 0.0)
    misread = min(1.0, _MISREAD_PER_UNIT_TEMPERATURE * temperature)
    if len(readings) > 1 and _HEDGE.search(lowered) \
            and _unit_interval(fingerprint, seed, "reading") < misread:
        alternatives = readings[1:]
        pick = int(_unit_interval(fingerprint, seed, "alternative") * len(alternatives))
        return alternatives[min(pick, len(alternatives) - 1)]
    return readings[0]


def _steps(text: str) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(\d+[.)]\s+|[-*•]\s+)", stripped):
            cleaned = re.sub(r"^(\d+[.)]\s+|[-*•]\s+)", "", stripped).strip()
            if cleaned:
                steps.append(cleaned)
    if steps:
        return steps[:6]
    first = next((l.strip() for l in text.splitlines() if l.strip()), "Reproduce the report")
    return [first[:120]]


def _confidence(text: str, fingerprint: str, seed: int) -> float:
    """Higher when the text is specific, lower when it is vague or noisy.

    Jittered deterministically so the golden set has to cope with a spread
    rather than a single magic number.
    """
    score = 0.55
    lowered = text.lower()
    if len(text) > 200:
        score += 0.10
    if re.search(r"\d+[.)]\s", text):          # has numbered steps
        score += 0.10
    if re.search(r"\b(error|exception|traceback|stack)\b", lowered):
        score += 0.10
    if re.search(r"\b(maybe|not sure|sometimes|randomly|weird|somehow)\b", lowered):
        score -= 0.20
    if len(text) < 80:
        score -= 0.10
    score += (_unit_interval(fingerprint, seed, "conf") - 0.5) * 0.08
    return round(min(0.99, max(0.05, score)), 2)


def generate_result(
    body: dict[str, Any], *, seed: int, bad_content: bool = False
) -> dict[str, Any]:
    """Build the `TriageResult`-shaped payload for this request.

    `bad_content=True` produces the `bad_content` fault: still schema-valid,
    but naming the wrong component with unearned confidence. That combination is
    the point — it is the failure a schema cannot catch for you, and the reason
    the golden set exists.
    """
    text = extract_text(body)
    fingerprint = request_fingerprint(body)

    severity = _match(text, _SEVERITY_RULES, "medium")

    if bad_content:
        component = _match(text, _COMPONENT_RULES, "unknown")
        wrong = [c for c, _ in _COMPONENT_RULES if c != component]
        pick = int(_unit_interval(fingerprint, seed, "wrong") * len(wrong))
        return {
            "severity": severity,
            "component": wrong[min(pick, len(wrong) - 1)],
            "duplicate_of": None,
            "steps": _steps(text),
            "confidence": 0.97,
        }

    component = _component(text, body, fingerprint, seed)
    duplicate_of = None
    match = re.search(r"\b(?:dup(?:licate)?(?:\s+of)?|see|same as)\s+(BUG-\d+|#\d+)", text, re.I)
    if match:
        duplicate_of = match.group(1)

    return {
        "severity": severity,
        "component": component,
        "duplicate_of": duplicate_of,
        "steps": _steps(text),
        "confidence": _confidence(text, fingerprint, seed),
    }


def estimate_tokens(text: str) -> int:
    """Crude but stable: the tests need reproducible numbers, not real ones."""
    return max(1, len(text) // 4)
