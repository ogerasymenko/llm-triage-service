import json

import pytest
from pydantic import ValidationError

from app.schema import TriageResult, triage_result_json_schema


def test_wire_schema_carries_no_numeric_bounds():
    """The provider rejects them, so they must not reach it."""
    blob = json.dumps(triage_result_json_schema())
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        assert key not in blob


def test_the_bound_still_exists_where_it_is_checked():
    """Stripping it from the wire must not weaken validation."""
    with pytest.raises(ValidationError):
        TriageResult(severity="high", component="auth", confidence=80)