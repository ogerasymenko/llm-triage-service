"""A stage file that loses its marker vanishes from the grouped summary and the
participant stops being told to fix it, so the split is checked rather than
trusted."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import TESTS_DIR
from tests.plugin_stages import STAGE_GOALS, STAGE_OF_FILE, STAGE_TITLES

# The nine files in the spec's table, test_openai_port.py for stage 7, and
# test_breaker.py which ships green as the worked example. These are the
# participant-facing suite.
SPEC_TABLE_FILES = {
    "test_extract.py",
    "test_concurrency.py",
    "test_stream.py",
    "test_properties.py",
    "test_golden.py",
    "test_retry.py",
    "test_idempotency.py",
    "test_timeout.py",
    "test_budget.py",
    "test_breaker.py",
    "test_openai_port.py",
}


def test_every_spec_file_lives_under_stages():
    on_disk = {p.name for p in (TESTS_DIR / "stages").glob("test_*.py")}
    missing = SPEC_TABLE_FILES - on_disk
    assert not missing, f"spec-table tests missing from tests/stages/: {sorted(missing)}"


def test_stages_contains_nothing_but_spec_files():
    """Supporting tests must not hide among the assignment."""
    on_disk = {p.name for p in (TESTS_DIR / "stages").glob("test_*.py")}
    extra = on_disk - SPEC_TABLE_FILES
    assert not extra, (
        f"{sorted(extra)} are under tests/stages/ but are not in the spec's table. "
        f"Supporting tests belong in tests/support/."
    )


def test_every_stage_file_is_mapped_to_a_stage():
    """An unmapped file never shows up in the grouped summary."""
    on_disk = {p.name for p in (TESTS_DIR / "stages").glob("test_*.py")}
    unmapped = on_disk - set(STAGE_OF_FILE)
    assert not unmapped, (
        f"{sorted(unmapped)} have no entry in plugin_stages.STAGE_OF_FILE, so their "
        f"failures would be reported as uncategorised"
    )


def test_stage_map_has_no_phantom_entries():
    on_disk = {p.name for p in (TESTS_DIR / "stages").glob("test_*.py")}
    phantom = set(STAGE_OF_FILE) - on_disk
    assert not phantom, f"STAGE_OF_FILE names files that do not exist: {sorted(phantom)}"


def test_every_referenced_stage_has_a_goal_and_title():
    """The summary prints these; a missing one leaves a participant with no next step."""
    for stage in set(STAGE_OF_FILE.values()):
        assert stage in STAGE_GOALS, f"stage {stage} has no goal text"
        assert stage in STAGE_TITLES, f"stage {stage} has no title"


def test_support_files_are_not_marked_as_stages(pytestconfig):
    support = {p.name for p in (TESTS_DIR / "support").glob("test_*.py")}
    overlap = support & set(STAGE_OF_FILE)
    assert not overlap, f"support files mapped to a stage: {sorted(overlap)}"


def test_markers_are_declared_in_pytest_ini(pytestconfig):
    """`--strict-markers` is on, so an undeclared marker is an error."""
    declared = {line.split("(")[0].split(":")[0].strip()
                for line in pytestconfig.getini("markers")}
    for marker in ("stage", "support", "chaos", "latency"):
        assert marker in declared, f"marker {marker!r} is not declared in pytest.ini"


def test_stage_numbers_are_the_seven_homework_stages():
    """The grouping must use the README's numbering, not the build order.

    The build spec has its own stages (skeleton, mock, reference solution, cut
    the stubs, content, verification). A participant never sees those. If these
    ever drift apart again, someone reading "Stage 2 -- async and backpressure"
    will run pytest and be told stage 3 is failing.
    """
    stages = set(STAGE_OF_FILE.values())
    assert stages == {1, 2, 3, 4, 5, 6, 7}, f"expected stages 1-7, got {sorted(stages)}"

    expected = {
        1: {"test_extract.py"},
        2: {"test_concurrency.py"},
        3: {"test_stream.py"},
        4: {"test_properties.py", "test_golden.py"},
        5: {"test_retry.py", "test_idempotency.py", "test_timeout.py", "test_budget.py"},
        6: {"test_breaker.py"},
        7: {"test_openai_port.py"},
    }
    actual: dict[int, set[str]] = {}
    for name, stage in STAGE_OF_FILE.items():
        actual.setdefault(stage, set()).add(name)
    assert actual == expected, f"stage mapping drifted: {actual}"


def test_stage_six_ships_with_no_red_test():
    """Stage 6's work is wiring and a written report, which no test asserts.

    `test_breaker.py` is green at checkout by design. The README has to say so,
    or stage 6 looks finished before the participant has done any of it -- so
    this guards the fact the README depends on.
    """
    from tests.conftest import REPO_ROOT

    stage_six = [name for name, stage in STAGE_OF_FILE.items() if stage == 6]
    assert stage_six == ["test_breaker.py"], stage_six

    readme = (REPO_ROOT / "README.md").read_text().lower()
    assert "no red test" in readme or "no failing test" in readme, (
        "stage 6 has no red test, so the README must say so explicitly -- "
        "otherwise it reads as already done"
    )
