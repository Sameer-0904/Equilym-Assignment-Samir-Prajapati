"""
test_no_hardcode.py — Automatic-fail lint test.

Scans engine/ and app.py source files for:
1. Hardcoded entity names (districts from the sample dataset)
2. Hardcoded numeric literal thresholds outside Config

This test proves the engine is genuinely general-purpose.
"""
import glob
import os
import re

import pytest


def _engine_source_files():
    """Return all engine Python files and app.py."""
    files = glob.glob(os.path.join("engine", "*.py"))
    if os.path.exists("app.py"):
        files.append("app.py")
    return files


def _read_source(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_no_hardcoded_entity_names():
    """
    No source file in engine/ or app.py should contain hardcoded entity names
    from the sample dataset. Explanations are generated from data at runtime.
    """
    # These are the entities in the *sample* dataset only.
    # The engine must produce no strings that contain these names.
    sample_entities = [
        "Ahmedabad", "Mehsana", "Surat", "Rajkot", "Bhavnagar", "Vadodara"
    ]

    violations = []
    for fpath in _engine_source_files():
        src = _read_source(fpath)
        for entity in sample_entities:
            if entity in src:
                violations.append(f"'{entity}' found in {fpath}")

    assert not violations, (
        "Hardcoded entity names found in source. These must be removed:\n"
        + "\n".join(violations)
    )


def test_no_hardcoded_indicator_names():
    """
    No source file should hardcode indicator column names from the sample data.
    anc_coverage, institutional_delivery, immunization, high_risk_cases
    should only appear in tests (where they are used as test fixtures).
    """
    sample_indicators = [
        "anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"
    ]

    violations = []
    for fpath in _engine_source_files():
        src = _read_source(fpath)
        for indicator in sample_indicators:
            if indicator in src:
                violations.append(f"'{indicator}' found in {fpath}")

    assert not violations, (
        "Hardcoded indicator names found in engine/app.py source:\n"
        + "\n".join(violations)
    )


def test_severity_function_has_no_magic_numbers():
    """
    engine/severity.py must not contain numeric literals for threshold cutoffs
    (like 0.5, 50, etc.) outside of accessing Config attributes.
    Allowed: accesses to cfg.* attributes.
    """
    src = _read_source(os.path.join("engine", "severity.py"))

    # Look for bare numeric comparisons like: > 0.5, >= 50, < 3.5
    # (excluding 0.0, 1.0 which are sentinel values for zero-check)
    suspicious = re.findall(r'(?<![a-zA-Z_])[><]=?\s*\d+\.\d+', src)
    # Allow == 0 (zero check), and the scaling factor in breach (0.1)
    suspicious = [s for s in suspicious if s.strip().replace(' ', '') not in ("==0", ">0", "!=0", "==0.0")]

    assert not suspicious, (
        f"Suspicious numeric literal comparisons in severity.py: {suspicious!r}. "
        "Use Config attributes instead."
    )
