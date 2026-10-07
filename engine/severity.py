"""
severity.py — Severity computation for all insight types.

DESIGN PRINCIPLE:
    Severity is derived from a ratio (score / threshold), not from
    hardcoded conditions. This means:
    - When the user moves a slider (changing threshold), severity
      automatically updates for all insights.
    - There are NO magic numbers in this module — all cutoffs live in Config.
    - Favourable direction events (e.g. a good outlier) are capped at Low.
"""
from __future__ import annotations

from engine.config import Config


def compute_severity(
    score: float,
    threshold: float,
    cfg: Config,
    is_adverse: bool = True,
) -> str:
    """
    Compute Low / Medium / High severity from a normalised ratio.

    Parameters
    ----------
    score : the raw numeric score (pct_change, z-score, r, etc.)
    threshold : the user-configured significance threshold for this type
    cfg : Config containing severity_low_max and severity_med_max band cutoffs
    is_adverse : False = favourable event (capped at Low)

    Returns
    -------
    "Low", "Medium", or "High"
    """
    if threshold == 0:
        return "Low"  # Avoid division by zero

    ratio = abs(score) / threshold

    if not is_adverse:
        return "Low"  # Favourable events never escalate beyond Low

    if ratio < cfg.severity_low_max:
        return "Low"
    elif ratio < cfg.severity_med_max:
        return "Medium"
    else:
        return "High"


def severity_for_trend(pct_change: float, cfg: Config, is_adverse: bool = True) -> str:
    """Convenience wrapper for trend insights."""
    return compute_severity(pct_change, cfg.trend_threshold, cfg, is_adverse)


def severity_for_outlier(score: float, cfg: Config, is_adverse: bool = True) -> str:
    """Convenience wrapper for outlier insights."""
    threshold = cfg.get_outlier_threshold()
    return compute_severity(score, threshold, cfg, is_adverse)


def severity_for_correlation(r: float, cfg: Config) -> str:
    """Convenience wrapper for correlation insights (always treated as informational)."""
    return compute_severity(r, cfg.corr_threshold, cfg, is_adverse=True)


def severity_for_breach(
    value: float,
    bound: float,
    cfg: Config,
    is_adverse: bool = True,
) -> str:
    """
    Severity for threshold breach. Scales by how far the value is from the bound.
    """
    if bound == 0:
        return "Low"
    # Scale: deviation of 10% of bound = ratio of 1.0 (triggers at Low)
    ratio = abs(value - bound) / (abs(bound) * 0.1) if bound != 0 else 0
    if not is_adverse:
        return "Low"
    if ratio < cfg.severity_low_max:
        return "Low"
    elif ratio < cfg.severity_med_max:
        return "Medium"
    else:
        return "High"
