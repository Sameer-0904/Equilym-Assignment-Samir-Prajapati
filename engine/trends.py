"""
trends.py — Period-over-period trend detection.

GENERALITY RULES:
- Works for any entity column, any period column, any numeric indicators.
- No hardcoded district names, indicator names, or magic thresholds.
- All thresholds come from engine.config.Config.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from engine.config import Config

# ── Direction helpers ──────────────────────────────────────────────────────────


def _direction_words(pct_change: float, polarity: int) -> tuple[str, str]:
    """
    Return (raw_direction, quality_direction) for a percentage change.

    raw_direction: "rose" or "dropped"
    quality_direction: "improved" or "worsened" (polarity-aware)
    """
    raw = "rose" if pct_change > 0 else "dropped"
    if polarity == 1:
        quality = "improved" if pct_change > 0 else "worsened"
    else:  # lower is better
        quality = "worsened" if pct_change > 0 else "improved"
    return raw, quality


# ── Core computation ───────────────────────────────────────────────────────────


def detect_trends(
    df: pd.DataFrame,
    cfg: Config,
    entity_col: Optional[str] = None,
    period_col: Optional[str] = None,
    indicator_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Detect significant period-over-period changes in all indicator columns.

    For every (entity, consecutive_month_pair, indicator), compute pct_change.
    Returns only rows where |pct_change| >= cfg.trend_threshold.

    Edge case: previous == 0 → falls back to absolute change,
    sets `from_zero_baseline = True`.

    Parameters
    ----------
    df : validated tidy DataFrame
    cfg : Config with trend_threshold and polarity settings
    entity_col, period_col, indicator_cols : column names (defaults from cfg)

    Returns
    -------
    DataFrame with columns:
        entity, indicator, period_from, period_to,
        prev_value, curr_value, abs_change, pct_change,
        from_zero_baseline, direction, quality_direction, polarity_value
    """
    entity_col = entity_col or cfg.entity_col
    period_col = period_col or cfg.period_col
    if indicator_cols is None:
        id_cols = {entity_col, period_col}
        indicator_cols = [c for c in df.columns if c not in id_cols
                          and pd.api.types.is_numeric_dtype(df[c])]

    rows = []

    for entity, entity_df in df.groupby(entity_col):
        entity_df = entity_df.sort_values(period_col).reset_index(drop=True)
        periods = entity_df[period_col].tolist()

        for i in range(1, len(entity_df)):
            period_from = periods[i - 1]
            period_to = periods[i]

            for indicator in indicator_cols:
                prev = entity_df.loc[i - 1, indicator]
                curr = entity_df.loc[i, indicator]

                # Skip if either value is missing
                if pd.isna(prev) or pd.isna(curr):
                    continue

                from_zero = False

                if prev == 0:
                    # Cannot compute % change — use absolute
                    from_zero = True
                    pct_change = float(curr)  # absolute change as proxy
                else:
                    pct_change = (curr - prev) / abs(prev) * 100.0

                abs_pct = abs(pct_change)

                if abs_pct < cfg.trend_threshold:
                    continue  # not significant

                polarity = cfg.get_polarity(indicator)
                direction, quality = _direction_words(pct_change, polarity)

                rows.append({
                    "entity": entity,
                    "indicator": indicator,
                    "period_from": period_from,
                    "period_to": period_to,
                    "prev_value": prev,
                    "curr_value": curr,
                    "abs_change": curr - prev,
                    "pct_change": pct_change,
                    "from_zero_baseline": from_zero,
                    "direction": direction,
                    "quality_direction": quality,
                    "polarity_value": polarity,
                })

    if not rows:
        return pd.DataFrame(columns=[
            "entity", "indicator", "period_from", "period_to",
            "prev_value", "curr_value", "abs_change", "pct_change",
            "from_zero_baseline", "direction", "quality_direction", "polarity_value",
        ])

    return pd.DataFrame(rows)
