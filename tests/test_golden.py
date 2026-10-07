"""
test_golden.py — Golden tests asserting VERIFIED numbers from the real sample dataset.

Dataset: data/sample.csv (company-provided, 12 rows, 6 districts × 2 months)
Months: 2026-07 and 2026-08
Districts: Ahmedabad, Surat, Vadodara, Rajkot, Mehsana, Bhavnagar
Indicators: anc_coverage (%), institutional_delivery (%), immunization (%), high_risk_cases (count)

All numbers below were verified by running calculations on the actual CSV.
"""
import pytest
import pandas as pd

from engine.config import Config
from engine.loader import load_csv
from engine.trends import detect_trends
from engine.outliers import detect_outliers
from engine.correlations import detect_correlations


@pytest.fixture
def loaded():
    df, report = load_csv("data/sample.csv")
    assert not report.is_fatal, f"Fatal load error: {report.fatal}"
    cfg = Config(
        outlier_method="robust_z",
        outlier_scope="pooled",
        polarity={"high_risk_cases": -1},
    )
    return df, cfg, report


# ── Trend tests ────────────────────────────────────────────────────────────────

def test_ahmedabad_anc_trend(loaded):
    """ANC coverage 85 → 69 = -18.8% (verified)"""
    df, cfg, _ = loaded
    trends = detect_trends(df, cfg)
    row = trends[
        (trends["entity"] == "Ahmedabad") &
        (trends["indicator"] == "anc_coverage")
    ]
    assert len(row) == 1, "Expected exactly one ANC trend row for this entity"
    pct = row.iloc[0]["pct_change"]
    assert abs(pct - (-18.8)) < 0.2, f"Expected ~-18.8%, got {pct:.1f}%"


def test_mehsana_anc_trend(loaded):
    """ANC coverage 84 → 42 = -50.0% (verified)"""
    df, cfg, _ = loaded
    trends = detect_trends(df, cfg)
    row = trends[
        (trends["entity"] == "Mehsana") &
        (trends["indicator"] == "anc_coverage")
    ]
    assert len(row) == 1
    pct = row.iloc[0]["pct_change"]
    assert abs(pct - (-50.0)) < 0.2, f"Expected ~-50.0%, got {pct:.1f}%"


def test_mehsana_highrisk_trend(loaded):
    """high_risk_cases 11 → 28 = +154.5% (verified)"""
    df, cfg, _ = loaded
    trends = detect_trends(df, cfg)
    row = trends[
        (trends["entity"] == "Mehsana") &
        (trends["indicator"] == "high_risk_cases")
    ]
    assert len(row) == 1
    pct = row.iloc[0]["pct_change"]
    assert abs(pct - 154.5) < 0.5, f"Expected ~+154.5%, got {pct:.1f}%"


def test_ahmedabad_highrisk_trend(loaded):
    """high_risk_cases 10 → 13 = +30.0% (verified extra finding)"""
    df, cfg, _ = loaded
    trends = detect_trends(df, cfg)
    row = trends[
        (trends["entity"] == "Ahmedabad") &
        (trends["indicator"] == "high_risk_cases")
    ]
    assert len(row) == 1
    pct = row.iloc[0]["pct_change"]
    assert abs(pct - 30.0) < 0.5, f"Expected ~+30.0%, got {pct:.1f}%"


# ── Correlation tests ──────────────────────────────────────────────────────────

def test_correlation_anc_highrisk(loaded):
    """ANC coverage vs high_risk_cases r ≈ -0.9335, p ≈ 9.16e-6 (verified)"""
    df, cfg, _ = loaded
    flagged, matrix = detect_correlations(df, cfg)
    r = matrix.loc["anc_coverage", "high_risk_cases"]
    assert abs(r - (-0.9335)) < 0.005, f"Expected r ≈ -0.9335, got {r:.4f}"


def test_correlation_delivery_immunization(loaded):
    """institutional_delivery vs immunization r ≈ +0.9786, p ≈ 3.40e-8 (verified)"""
    df, cfg, _ = loaded
    _, matrix = detect_correlations(df, cfg)
    r = matrix.loc["institutional_delivery", "immunization"]
    assert abs(r - 0.9786) < 0.005, f"Expected r ≈ +0.9786, got {r:.4f}"


def test_flagged_corr_pairs_count(loaded):
    """Exactly 2 pairs should be flagged at |r| >= 0.70 (verified)"""
    df, cfg, _ = loaded
    flagged, _ = detect_correlations(df, cfg)
    assert len(flagged) == 2, f"Expected 2 flagged pairs, got {len(flagged)}"


# ── Outlier tests ──────────────────────────────────────────────────────────────

def test_mehsana_anc_robust_z_value(loaded):
    """Mehsana ANC robust z ≈ -7.42 (verified, median=80.5, MAD=3.50)"""
    df, cfg, _ = loaded
    outliers = detect_outliers(df, cfg)
    row = outliers[
        (outliers["entity"] == "Mehsana") &
        (outliers["indicator"] == "anc_coverage")
    ]
    assert len(row) >= 1, "Mehsana ANC should be flagged as outlier"
    score = row.iloc[0]["score"]
    assert abs(score - (-7.42)) < 0.1, f"Expected robust_z ≈ -7.42, got {score:.2f}"
    assert row.iloc[0]["is_adverse"], "Should be adverse (lower is worse for ANC)"


def test_iqr_outliers_anc(loaded):
    """IQR pooled: lower fence ≈ 69.9 → Mehsana(42) and Ahmedabad(69) flagged"""
    df, cfg, _ = loaded
    iqr_cfg = Config(
        outlier_method="iqr",
        outlier_scope="pooled",
        iqr_k=1.5,
        polarity={"high_risk_cases": -1},
    )
    outliers = detect_outliers(df, iqr_cfg)
    anc_outliers = outliers[outliers["indicator"] == "anc_coverage"]["entity"].tolist()
    assert "Mehsana" in anc_outliers, f"Mehsana should be IQR outlier. Got: {anc_outliers}"
    assert "Ahmedabad" in anc_outliers, f"Ahmedabad(69) at fence 69.9 should be IQR outlier. Got: {anc_outliers}"


# ── Severity tests ─────────────────────────────────────────────────────────────

def test_severity_ahmedabad_anc_high(loaded):
    """-18.8% / 10% = ratio 1.88 ≥ 1.5 → High"""
    from engine.severity import severity_for_trend
    _, cfg, _ = loaded
    severity = severity_for_trend(-18.8, cfg, is_adverse=True)
    assert severity == "High", f"Expected High, got {severity}"


def test_severity_mehsana_anc_high(loaded):
    """-50.0% / 10% = ratio 5.0 ≥ 1.5 → High"""
    from engine.severity import severity_for_trend
    _, cfg, _ = loaded
    severity = severity_for_trend(-50.0, cfg, is_adverse=True)
    assert severity == "High", f"Expected High, got {severity}"


def test_severity_favourable_capped_low(loaded):
    """A favourable event (e.g. Bhavnagar high_risk drop) should be Low regardless of ratio"""
    from engine.severity import severity_for_trend
    _, cfg, _ = loaded
    # -11.8% change in high_risk_cases with polarity=-1 → quality is "improved" → not adverse
    severity = severity_for_trend(-11.8, cfg, is_adverse=False)
    assert severity == "Low", f"Favourable events must be capped at Low, got {severity}"

