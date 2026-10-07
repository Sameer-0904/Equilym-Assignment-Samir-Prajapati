"""
outliers.py — Outlier detection using IQR, Z-score, and Robust Z (MAD).

GENERALITY RULES:
- Works for any entity column, any period column, any numeric indicators.
- No hardcoded names or magic thresholds (all come from Config).

WHY ROBUST Z IS THE DEFAULT:
Plain z-score can miss outliers in small samples because the outlier itself
inflates the very standard deviation it is measured against (masking effect).
The MAD-based robust z avoids this by using the median and median absolute
deviation, which are resistant to extreme values.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from engine.config import Config

# ── Utility functions ──────────────────────────────────────────────────────────


def _robust_z(values: np.ndarray) -> np.ndarray:
    """
    Compute MAD-based robust z-scores.
    Formula: 0.6745 * (x - median) / MAD
    The 0.6745 scaling makes it equivalent to standard z at a normal distribution.
    """
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if mad == 0:
        # Fallback: all values are equal — no outliers
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - median) / mad


def _leave_one_out_z(values: np.ndarray, idx: int) -> float:
    """
    Compute z-score of values[idx] against the remaining n-1 values.
    Avoids masking: the suspect value doesn't inflate the reference stats.
    """
    mask = np.ones(len(values), dtype=bool)
    mask[idx] = False
    rest = values[mask]
    if len(rest) < 2:
        return np.nan
    mu = np.nanmean(rest)
    sd = np.nanstd(rest, ddof=1)
    if sd == 0:
        return np.nan
    return (values[idx] - mu) / sd


def _iqr_fences(values: np.ndarray, k: float) -> tuple[float, float]:
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


# ── Core detection ─────────────────────────────────────────────────────────────


def detect_outliers(
    df: pd.DataFrame,
    cfg: Config,
    entity_col: Optional[str] = None,
    period_col: Optional[str] = None,
    indicator_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Detect outliers in all indicators using the method specified in cfg.

    Primary scope: cross-section (compare entities within the same period).
    Optional scope: pooled (all rows for an indicator together).

    Parameters
    ----------
    df : validated tidy DataFrame
    cfg : Config object
    entity_col, period_col, indicator_cols : column names

    Returns
    -------
    DataFrame with columns:
        entity, indicator, period, value, method, score, threshold,
        is_above, is_adverse, polarity_value, scope, loo_z
    """
    entity_col = entity_col or cfg.entity_col
    period_col = period_col or cfg.period_col
    if indicator_cols is None:
        id_cols = {entity_col, period_col}
        indicator_cols = [c for c in df.columns if c not in id_cols
                          and pd.api.types.is_numeric_dtype(df[c])]

    threshold = cfg.get_outlier_threshold()
    scope = cfg.outlier_scope
    method = cfg.outlier_method

    rows = []

    if scope == "cross_section":
        # Group by period, detect outliers within each period's cross-section
        for period, period_df in df.groupby(period_col):
            for indicator in indicator_cols:
                _process_group(
                    period_df, indicator, entity_col, period,
                    method, threshold, cfg, scope, rows
                )
    else:  # pooled
        for indicator in indicator_cols:
            _process_group(
                df, indicator, entity_col, "all",
                method, threshold, cfg, "pooled", rows
            )

    if not rows:
        return pd.DataFrame(columns=[
            "entity", "indicator", "period", "value",
            "method", "score", "threshold", "is_above",
            "is_adverse", "polarity_value", "scope", "loo_z",
        ])

    return pd.DataFrame(rows)


def _process_group(
    group_df: pd.DataFrame,
    indicator: str,
    entity_col: str,
    period,
    method: str,
    threshold: float,
    cfg: Config,
    scope: str,
    rows: list,
) -> None:
    """Compute outlier scores for one group (period or pooled) and one indicator."""
    sub = group_df[[entity_col, indicator]].dropna(subset=[indicator])
    if len(sub) < 2:
        return  # need at least 2 points to compute outlier

    values = sub[indicator].values.astype(float)
    entities = sub[entity_col].values

    if method == "iqr":
        lower, upper = _iqr_fences(values, cfg.iqr_k)
        scores = np.where(
            values > upper, values - upper,
            np.where(values < lower, lower - values, 0.0)
        )
        # For IQR, score > 0 means outlier; we store signed deviation
        signed_scores = values - np.nanmean(values)  # for is_above determination
        flags = (values > upper) | (values < lower)

    elif method == "zscore":
        mu, sd = np.nanmean(values), np.nanstd(values, ddof=1)
        if sd == 0:
            return
        signed_scores = (values - mu) / sd
        flags = np.abs(signed_scores) >= threshold
        scores = signed_scores

    else:  # robust_z
        signed_scores = _robust_z(values)
        flags = np.abs(signed_scores) >= threshold
        scores = signed_scores

    polarity = cfg.get_polarity(indicator)

    for i, (entity, val, flag, score) in enumerate(
        zip(entities, values, flags, scores)
    ):
        if not flag:
            continue

        is_above = bool(val > np.nanmedian(values))

        # Compute leave-one-out z as bonus metadata
        loo = _leave_one_out_z(values, i)

        # Adverse: if polarity=+1 (higher=better) and value is LOW → bad
        #          if polarity=-1 (lower=better) and value is HIGH → bad
        if polarity == 1:
            is_adverse = not is_above
        else:
            is_adverse = is_above

        rows.append({
            "entity": entity,
            "indicator": indicator,
            "period": period,
            "value": val,
            "method": method,
            "score": float(score),
            "threshold": threshold,
            "is_above": is_above,
            "is_adverse": is_adverse,
            "polarity_value": polarity,
            "scope": scope,
            "loo_z": float(loo) if not np.isnan(loo) else None,
        })
