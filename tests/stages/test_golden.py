"""Stage 4 (testing a non-deterministic dependency) -- the golden set.

Red at checkout: extraction does not work yet, and the threshold is unset.

A golden set grades a probabilistic system, so it measures quality rather than
demanding agreement with its author. Each case lists every answer a careful
reader could defend, and severity is graded by distance on the ordered scale --
calling a high bug critical is a far smaller error than calling it trivial.

The threshold lives in samples/golden.yaml and is committed, so changing the bar
is a change to a reviewed file rather than a number quietly edited in a test
until the suite goes green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, SAMPLES_DIR
from tests.support.goldenio import load_golden

GOLDEN = load_golden(SAMPLES_DIR / "golden.yaml")
SCALE = GOLDEN["severity_scale"]
CASES = GOLDEN["cases"]


def severity_distance(a: str, b: str) -> int:
    return abs(SCALE.index(a) - SCALE.index(b))


def grade(result, accept: dict) -> tuple[bool, list[str]]:
    """Score one result against a case's accepted answers."""
    problems: list[str] = []

    if result.component not in accept["component"]:
        problems.append(
            f"component {result.component!r} not in {accept['component']}"
        )

    allowed = accept["severity"]
    if result.severity not in allowed:
        nearest = min(severity_distance(result.severity, s) for s in allowed)
        if nearest > GOLDEN["severity_tolerance"]:
            problems.append(
                f"severity {result.severity!r} is {nearest} steps from {allowed}"
            )

    floor = accept.get("min_confidence")
    if floor is not None and result.confidence < floor:
        problems.append(f"confidence {result.confidence} below floor {floor}")

    cap = accept.get("max_confidence")
    if cap is not None and result.confidence > cap:
        problems.append(
            f"confidence {result.confidence} above cap {cap} -- overconfident on a vague report"
        )

    if "duplicate_of" in accept and result.duplicate_of not in accept["duplicate_of"]:
        problems.append(
            f"duplicate_of {result.duplicate_of!r} not in {accept['duplicate_of']}"
        )

    return (not problems), problems


def require_threshold() -> float:
    """The committed threshold, or a failure that says what to do.

    Unset at checkout: stage 4 is where you measure the set and commit a bar.
    Without this the tests below would die on a TypeError comparing a float to
    None, which tells a participant nothing about what is actually missing.
    """
    threshold = GOLDEN.get("threshold")
    if threshold is None:
        pytest.fail(
            "stage 4: samples/golden.yaml has no committed threshold. Run `make "
            "golden` and `make golden MODE=bad_content` to see what the set actually "
            "scores across seeds, then set `threshold:` to a bar every clean seed "
            "clears and no bad_content seed does. It lives in the YAML, not in this test, so that moving "
            "the bar is a reviewable change."
        )
    return threshold


def test_golden_file_is_well_formed():
    assert len(CASES) == 10
    threshold = require_threshold()
    assert 0.0 < threshold <= 1.0
    for case in CASES:
        assert (SAMPLES_DIR / case["file"]).exists(), f"missing {case['file']}"
        assert case["accept"]["component"], f"case {case['id']} accepts no component"
        assert case["kind"] in {"unambiguous", "borderline", "noisy"}


async def test_golden_set_meets_threshold(make_service, env):
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_RETRIES="5")
    service = make_service()

    passed, report = 0, []
    for case in CASES:
        text = (SAMPLES_DIR / case["file"]).read_text()
        result, _ = await service.triage(text)
        ok, problems = grade(result, case["accept"])
        passed += ok
        if not ok:
            report.append(f"  {case['id']} ({case['kind']}): " + "; ".join(problems))

    threshold = require_threshold()
    score = passed / len(CASES)
    assert score >= threshold, (
        f"golden score {score:.2f} is below the committed threshold "
        f"{threshold}\n" + "\n".join(report)
    )


@pytest.mark.parametrize(
    "case", [c for c in CASES if c["kind"] == "unambiguous"], ids=lambda c: c["id"]
)
async def test_unambiguous_cases_are_always_right(make_service, env, case):
    """The three clear-cut inputs must pass individually, not on average.

    A threshold is a budget for the hard cases. Spending it on the easy ones
    means the extraction is broken rather than merely uncertain.
    """
    env(FAULT_RATE="0.0", TEMPERATURE="0.7", MAX_RETRIES="5")
    service = make_service()

    text = (SAMPLES_DIR / case["file"]).read_text()
    result, _ = await service.triage(text)
    ok, problems = grade(result, case["accept"])

    assert ok, f"unambiguous case {case['id']} failed: " + "; ".join(problems)


async def test_bad_content_fault_is_caught_by_the_golden_set(make_service, env):
    """The golden set must actually catch confidently-wrong output.

    This is the test that proves the golden set has teeth. Under
    `FAULT_MODE=bad_content` every answer is schema-valid and names the wrong
    component with 0.97 confidence -- exactly what a schema check waves
    through. If the golden set still scores above threshold here, it is not
    measuring anything.
    """
    env(FAULT_RATE="1.0", FAULT_MODE="bad_content", TEMPERATURE="0.7", MAX_RETRIES="5")
    service = make_service()

    passed = 0
    for case in CASES:
        text = (SAMPLES_DIR / case["file"]).read_text()
        result, _ = await service.triage(text)
        ok, _ = grade(result, case["accept"])
        passed += ok

    threshold = require_threshold()
    score = passed / len(CASES)
    assert score < threshold, (
        f"deliberately wrong output scored {score:.2f}, at or above the "
        f"{threshold} threshold. The golden set is not discriminating."
    )
