"""
test_generality.py — Proves the engine works on a completely different dataset.

Uses a synthetic CSV with different entity names (Pune, Nagpur, Nashik)
and different indicator names (tb_cure_rate, malaria_cases).
No code changes should be needed for this to work.
"""
import pandas as pd
import pytest

from engine.config import Config
from engine.loader import load_csv, auto_detect_schema
from engine.trends import detect_trends
from engine.outliers import detect_outliers
from engine.correlations import detect_correlations
from engine.insights import build_insights


@pytest.fixture
def synthetic_path(tmp_path):
    """Write synthetic CSV to a temp file and return the path."""
    data = """period,zone,tb_cure_rate,malaria_cases
2024-01,Pune,82,120
2024-01,Nagpur,75,200
2024-01,Nashik,78,150
2024-02,Pune,70,110
2024-02,Nagpur,74,210
2024-02,Nashik,80,145
2024-03,Pune,68,140
2024-03,Nagpur,73,180
2024-03,Nashik,79,155
"""
    csv_file = tmp_path / "synthetic.csv"
    csv_file.write_text(data)
    return str(csv_file)


def test_auto_detect_schema_on_synthetic(synthetic_path):
    """Schema detection should find zone as entity and period as period."""
    df, report = load_csv(synthetic_path)
    assert not report.is_fatal
    assert report.entity_col == "zone"
    assert report.period_col == "period"
    assert "tb_cure_rate" in report.indicator_cols
    assert "malaria_cases" in report.indicator_cols


def test_trends_on_synthetic(synthetic_path):
    """Trends should be detected without any code changes."""
    df, report = load_csv(synthetic_path, entity_hint="zone", period_hint="period")
    cfg = Config(
        entity_col="zone", period_col="period",
        trend_threshold=5.0,
        polarity={"tb_cure_rate": 1, "malaria_cases": -1},
    )
    trends = detect_trends(df, cfg, entity_col="zone", period_col="period",
                           indicator_cols=["tb_cure_rate", "malaria_cases"])
    assert len(trends) > 0, "Should detect at least one trend on synthetic data"
    # Entity names in output should match synthetic data, not sample data
    entities = trends["entity"].unique().tolist()
    assert "Pune" in entities or "Nagpur" in entities or "Nashik" in entities


def test_outliers_on_synthetic(synthetic_path):
    """Outlier detection should work on synthetic data."""
    df, _ = load_csv(synthetic_path, entity_hint="zone", period_hint="period")
    cfg = Config(
        entity_col="zone", period_col="period",
        outlier_method="robust_z",
        robust_z_threshold=1.5,  # lower threshold to ensure some detection
    )
    outliers = detect_outliers(df, cfg, entity_col="zone", period_col="period",
                               indicator_cols=["tb_cure_rate", "malaria_cases"])
    # Should not crash; may or may not find outliers depending on threshold
    assert isinstance(outliers, pd.DataFrame)


def test_full_pipeline_on_synthetic(synthetic_path):
    """Full pipeline (load → trends → outliers → correlations → insights) on synthetic data."""
    df, report = load_csv(synthetic_path, entity_hint="zone", period_hint="period")
    assert not report.is_fatal

    cfg = Config(
        entity_col="zone", period_col="period",
        trend_threshold=5.0,
        outlier_method="robust_z",
        robust_z_threshold=1.5,
        corr_threshold=0.5,
        polarity={"tb_cure_rate": 1, "malaria_cases": -1},
    )

    trends = detect_trends(df, cfg, entity_col="zone", period_col="period",
                           indicator_cols=["tb_cure_rate", "malaria_cases"])
    outliers = detect_outliers(df, cfg, entity_col="zone", period_col="period",
                               indicator_cols=["tb_cure_rate", "malaria_cases"])
    flagged_corr, _ = detect_correlations(df, cfg, entity_col="zone", period_col="period",
                                           indicator_cols=["tb_cure_rate", "malaria_cases"])
    insights = build_insights(trends, outliers, flagged_corr, cfg)

    assert isinstance(insights, pd.DataFrame)
    required_cols = {"insight_id", "type", "indicator", "entity", "period",
                     "metric", "severity", "explanation"}
    assert required_cols.issubset(set(insights.columns)), (
        f"Missing columns: {required_cols - set(insights.columns)}"
    )

    # Verify insight_ids are in correct format
    if not insights.empty:
        for iid in insights["insight_id"]:
            assert iid.startswith("INS-"), f"Bad insight_id format: {iid}"
