"""
correlations.py — Pearson and Spearman correlation detection.

IMPORTANT LIMITATION (stated in every output):
    Correlation computed across a small, potentially autocorrelated sample.
    Same entity appearing in multiple periods introduces non-independence.
    Treat any correlation finding as a hypothesis, not causal proof.

WHY NO PER-ENTITY CORRELATION:
    A correlation between two variables computed on only 2 points is
    always exactly ±1 and carries zero information. We use pooled
    cross-entity correlation and report n and p-value for transparency.
    Per-entity correlation is refused when n < cfg.min_points_corr.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from engine.config import Config

CORRELATION_CAVEAT = (
    "Correlation computed across {n} observations. With fewer than 30 independent "
    "observations, results are exploratory only. Same entity appearing in multiple "
    "periods introduces autocorrelation — treat as a hypothesis, not proof."
)

_CONFIDENCE_LABELS = [
    (10, "Low"),
    (30, "Moderate"),
    (float("inf"), "Adequate"),
]


def _confidence_label(n: int) -> str:
    for threshold, label in _CONFIDENCE_LABELS:
        if n < threshold:
            return label
    return "Adequate"


def _correlation_strength(r: float) -> str:
    abs_r = abs(r)
    if abs_r >= 0.90:
        return "very strong"
    if abs_r >= 0.70:
        return "strong"
    if abs_r >= 0.50:
        return "moderate"
    return "weak"


def detect_correlations(
    df: pd.DataFrame,
    cfg: Config,
    entity_col: Optional[str] = None,
    period_col: Optional[str] = None,
    indicator_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson and Spearman correlations across all indicator pairs.

    Parameters
    ----------
    df : validated tidy DataFrame
    cfg : Config object
    entity_col, period_col, indicator_cols : column names

    Returns
    -------
    (flagged_pairs_df, full_corr_matrix_df)

    flagged_pairs_df columns:
        indicator_a, indicator_b, pearson_r, spearman_r, p_value,
        n, confidence, period_range, strength, caveat

    full_corr_matrix_df:
        Pearson correlation matrix (indicator × indicator), same as df[indicators].corr()
    """
    entity_col = entity_col or cfg.entity_col
    period_col = period_col or cfg.period_col
    if indicator_cols is None:
        id_cols = {entity_col, period_col}
        indicator_cols = [c for c in df.columns if c not in id_cols
                          and pd.api.types.is_numeric_dtype(df[c])]

    if len(indicator_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Period range for reporting
    valid_periods = df[period_col].dropna()
    if len(valid_periods) > 0:
        period_range = f"{valid_periods.min()} .. {valid_periods.max()}"
    else:
        period_range = "unknown"

    # Full Pearson matrix (as required by spec 9.2)
    indicator_df = df[indicator_cols].dropna(how="all")
    corr_matrix = indicator_df.corr(method="pearson")

    # Pairwise flagged findings
    flagged_rows = []
    n_total = len(indicator_df.dropna())

    for i, col_a in enumerate(indicator_cols):
        for j, col_b in enumerate(indicator_cols):
            if j <= i:
                continue  # upper triangle only

            pair_df = indicator_df[[col_a, col_b]].dropna()
            n = len(pair_df)

            if n < cfg.min_points_corr:
                # Refuse — too few points for meaningful correlation
                continue

            pearson_r = corr_matrix.loc[col_a, col_b]
            if abs(pearson_r) < cfg.corr_threshold:
                continue  # below user threshold

            # Spearman
            spearman_r, _ = scipy_stats.spearmanr(pair_df[col_a], pair_df[col_b])

            # p-value (two-tailed)
            _, p_value = scipy_stats.pearsonr(pair_df[col_a], pair_df[col_b])

            confidence = _confidence_label(n)
            strength = _correlation_strength(pearson_r)
            caveat = CORRELATION_CAVEAT.format(n=n)

            flagged_rows.append({
                "indicator_a": col_a,
                "indicator_b": col_b,
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
                "p_value": p_value,
                "n": n,
                "confidence": confidence,
                "period_range": period_range,
                "strength": strength,
                "caveat": caveat,
            })

    flagged_df = pd.DataFrame(flagged_rows) if flagged_rows else pd.DataFrame(
        columns=[
            "indicator_a", "indicator_b", "pearson_r", "spearman_r",
            "p_value", "n", "confidence", "period_range", "strength", "caveat",
        ]
    )

    return flagged_df, corr_matrix
