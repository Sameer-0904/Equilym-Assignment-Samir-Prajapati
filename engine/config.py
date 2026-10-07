"""
config.py — Single source of truth for all thresholds and defaults.

IMPORTANT: No numeric literals for thresholds should appear anywhere else
in the codebase. All configurable values live here.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Trend ─────────────────────────────────────────────────────────────
    trend_threshold: float = 10.0
    """Minimum |%change| to flag a trend (1–50%). Default: 10%."""

    # ── Outlier ───────────────────────────────────────────────────────────
    outlier_method: str = "robust_z"
    """Method to use: 'iqr' | 'zscore' | 'robust_z'. Default: robust_z.
    Robust Z (MAD-based) is the default because plain z-score can mask outliers
    when the outlier itself inflates the standard deviation (masking effect).
    """
    iqr_k: float = 1.5
    """IQR fence multiplier (Q1 - k*IQR, Q3 + k*IQR). Default: 1.5."""
    z_threshold: float = 3.0
    """Plain z-score threshold. Default: 3.0."""
    robust_z_threshold: float = 3.5
    """MAD-based robust z threshold. Default: 3.5."""
    outlier_scope: str = "cross_section"
    """'cross_section' (per indicator per month) or 'pooled' (all rows)."""

    # ── Correlation ───────────────────────────────────────────────────────
    corr_threshold: float = 0.70
    """Minimum |r| to flag a correlation pair. Default: 0.70."""
    min_points_corr: int = 5
    """Refuse to compute correlation with fewer than this many observations.
    A correlation on 2 points is always exactly ±1 and carries no information.
    """

    # ── Severity bands ────────────────────────────────────────────────────
    severity_low_max: float = 1.25
    """ratio < severity_low_max → Low severity."""
    severity_med_max: float = 1.50
    """severity_low_max ≤ ratio < severity_med_max → Medium; else High."""

    # ── Threshold breach ──────────────────────────────────────────────────
    breach_enabled: bool = False
    breach_floors: dict = field(default_factory=dict)
    """Dict mapping indicator name → absolute floor value."""
    breach_ceilings: dict = field(default_factory=dict)
    """Dict mapping indicator name → absolute ceiling value."""

    # ── Polarity ──────────────────────────────────────────────────────────
    polarity: dict = field(default_factory=dict)
    """Dict mapping indicator name → +1 (higher is better) or -1 (lower is better).
    Unknown indicators default to +1.
    """

    # ── Column name overrides ─────────────────────────────────────────────
    entity_col: str = "district"
    """Name of the entity/district column. Auto-detected if not set."""
    period_col: str = "month"
    """Name of the period/month column. Auto-detected if not set."""

    # ── LLM summary (stand-out S6, off by default) ────────────────────────
    llm_enabled: bool = False
    gemini_api_key: str = ""

    def get_polarity(self, indicator: str) -> int:
        """Return +1 (higher=better) or -1 (lower=better) for an indicator."""
        return self.polarity.get(indicator, 1)

    def get_outlier_threshold(self) -> float:
        """Return the active outlier threshold value for the selected method."""
        return {
            "iqr": self.iqr_k,
            "zscore": self.z_threshold,
            "robust_z": self.robust_z_threshold,
        }[self.outlier_method]
