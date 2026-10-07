"""
insights.py — Assemble insight rows from raw findings, generate explanations,
              assign insight IDs, compute severity, and merge into stories.

GENERALITY RULES:
- No hardcoded entity names, indicator names, or numeric literals in templates.
- All numbers in explanation strings come from the data at runtime.
- Indicator labels are derived from column names + an overridable acronym map.
"""
from __future__ import annotations

import itertools
from typing import List, Optional

import pandas as pd

from engine.config import Config
from engine.severity import (
    severity_for_trend,
    severity_for_outlier,
    severity_for_correlation,
    severity_for_breach,
)

# ── Label formatting ───────────────────────────────────────────────────────────

# Acronyms to capitalise as-is (not title-cased). Extend as needed.
# This is a configuration, not narrative-specific hardcoding.
_ACRONYM_MAP: dict[str, str] = {
    "anc": "ANC",
    "tb": "TB",
    "hiv": "HIV",
    "ari": "ARI",
    "nmr": "NMR",
    "mmr": "MMR",
    "imr": "IMR",
}


def format_indicator_label(col_name: str) -> str:
    """Convert a snake_case column name to a human-readable label.
    e.g. 'my_indicator' -> 'My Indicator', acronyms in _ACRONYM_MAP are uppercased.
    """
    words = col_name.replace("_", " ").replace("-", " ").split()
    return " ".join(_ACRONYM_MAP.get(w.lower(), w.title()) for w in words)


# ── Explanation templates ──────────────────────────────────────────────────────

def _trend_explanation(row: dict, cfg: Config) -> str:
    label = format_indicator_label(row["indicator"])
    suffix = " (from zero baseline — absolute change shown)" if row.get("from_zero_baseline") else ""
    return (
        f"{label} in {row['entity']} {row['direction']} by "
        f"{abs(row['pct_change']):.1f}% compared to the previous period "
        f"({row['prev_value']} \u2192 {row['curr_value']}), "
        f"exceeding the {cfg.trend_threshold:.0f}% significant-change threshold. "
        f"This represents a {row['quality_direction']} in performance.{suffix}"
    )


def _outlier_explanation(row: dict, cfg: Config) -> str:
    label = format_indicator_label(row["indicator"])
    position = "above" if row["is_above"] else "below"
    scope_str = "period" if row["scope"] == "cross_section" else "overall"
    score_str = f"{abs(row['score']):.1f}\u03c3"  # σ symbol
    mean_label = "median" if cfg.outlier_method == "robust_z" else "mean"
    loo_note = ""
    if row.get("loo_z") is not None:
        loo_note = f" (leave-one-out z: {row['loo_z']:.1f}\u03c3)"
    return (
        f"{row['entity']}'s {label} of {row['value']:.1f} is {score_str} "
        f"{position} the {scope_str} {mean_label}{loo_note}, flagging for review."
    )


def _correlation_explanation(row: dict) -> str:
    label_a = format_indicator_label(row["indicator_a"])
    label_b = format_indicator_label(row["indicator_b"])
    sign = "negative" if row["pearson_r"] < 0 else "positive"
    return (
        f"{label_a} and {label_b} show a {row['strength']} {sign} correlation "
        f"(r={row['pearson_r']:.2f}, n={row['n']}). "
        f"{row['confidence']} confidence. "
        "Low sample size \u2014 interpret as a hypothesis only."
    )


def _breach_explanation(row: dict) -> str:
    label = format_indicator_label(row["indicator"])
    return (
        f"{row['entity']}'s {label} of {row['value']:.1f} is "
        f"{row['breach_direction']} the configured {row['bound_type']} "
        f"of {row['bound']:.1f}."
    )


# ── ID generator ───────────────────────────────────────────────────────────────

_id_counter = itertools.count(1)


def _next_id() -> str:
    return f"INS-{next(_id_counter):04d}"


def _reset_id_counter() -> None:
    global _id_counter
    _id_counter = itertools.count(1)


# ── Severity score for priority ranking ───────────────────────────────────────

_SEVERITY_SCORE = {"High": 3, "Medium": 2, "Low": 1}


# ── Main assembly function ─────────────────────────────────────────────────────

def build_insights(
    trends_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    cfg: Config,
    breaches_df: Optional[pd.DataFrame] = None,
    reset_ids: bool = True,
) -> pd.DataFrame:
    """
    Assemble all insights into a single DataFrame with all required fields.

    Parameters
    ----------
    trends_df : output of detect_trends()
    outliers_df : output of detect_outliers()
    correlations_df : output of detect_correlations()[0] (flagged pairs)
    cfg : Config object
    breaches_df : optional threshold breach findings
    reset_ids : if True, reset the global ID counter (use False for incremental)

    Returns
    -------
    DataFrame with columns matching spec 9.1:
        insight_id, type, indicator, entity, period, metric,
        prev_value, change, change_pct, severity, explanation,
        confidence, story_id
    """
    if reset_ids:
        _reset_id_counter()

    all_rows: List[dict] = []

    # ── Trend insights ─────────────────────────────────────────────────────
    for _, row in trends_df.iterrows():
        r = dict(row)
        is_adverse = (r["quality_direction"] == "worsened")
        severity = severity_for_trend(r["pct_change"], cfg, is_adverse)
        all_rows.append({
            "insight_id": _next_id(),
            "type": "trend",
            "indicator": r["indicator"],
            "entity": r["entity"],
            "period": str(r["period_to"]),
            "metric": r["curr_value"],
            "prev_value": r["prev_value"],
            "change": r["abs_change"],
            "change_pct": r["pct_change"],
            "severity": severity,
            "explanation": _trend_explanation(r, cfg),
            "confidence": "Low" if abs(r["pct_change"]) < cfg.trend_threshold * cfg.severity_low_max else "Moderate",
            "story_id": None,
        })

    # ── Outlier insights ───────────────────────────────────────────────────
    for _, row in outliers_df.iterrows():
        r = dict(row)
        severity = severity_for_outlier(r["score"], cfg, r["is_adverse"])
        all_rows.append({
            "insight_id": _next_id(),
            "type": "outlier",
            "indicator": r["indicator"],
            "entity": r["entity"],
            "period": str(r["period"]),
            "metric": r["value"],
            "prev_value": None,
            "change": r["score"],
            "change_pct": None,
            "severity": severity,
            "explanation": _outlier_explanation(r, cfg),
            "confidence": "Low",
            "story_id": None,
        })

    # ── Correlation insights ───────────────────────────────────────────────
    for _, row in correlations_df.iterrows():
        r = dict(row)
        severity = severity_for_correlation(r["pearson_r"], cfg)
        all_rows.append({
            "insight_id": _next_id(),
            "type": "correlation",
            "indicator": f"{r['indicator_a']}:{r['indicator_b']}",
            "entity": "all",
            "period": r["period_range"],
            "metric": r["pearson_r"],
            "prev_value": None,
            "change": r["pearson_r"],
            "change_pct": None,
            "severity": severity,
            "explanation": _correlation_explanation(r),
            "confidence": r["confidence"],
            "story_id": None,
        })

    # ── Threshold breach insights ──────────────────────────────────────────
    if breaches_df is not None and not breaches_df.empty:
        for _, row in breaches_df.iterrows():
            r = dict(row)
            is_adverse = r.get("is_adverse", True)
            severity = severity_for_breach(r["value"], r["bound"], cfg, is_adverse)
            all_rows.append({
                "insight_id": _next_id(),
                "type": "threshold_breach",
                "indicator": r["indicator"],
                "entity": r["entity"],
                "period": str(r["period"]),
                "metric": r["value"],
                "prev_value": None,
                "change": r["value"] - r["bound"],
                "change_pct": None,
                "severity": severity,
                "explanation": _breach_explanation(r),
                "confidence": "N/A",
                "story_id": None,
            })

    if not all_rows:
        return pd.DataFrame(columns=[
            "insight_id", "type", "indicator", "entity", "period",
            "metric", "prev_value", "change", "change_pct",
            "severity", "explanation", "confidence", "story_id",
        ])

    insights_df = pd.DataFrame(all_rows)

    # ── Story merging (S2 stand-out) ───────────────────────────────────────
    insights_df = _assign_stories(insights_df)

    # ── Priority ranking (S3 stand-out) ───────────────────────────────────
    # Stored as a separate operation (called separately in UI)

    return insights_df


# ── Story merging ──────────────────────────────────────────────────────────────

def _assign_stories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group related insights for the same entity and period into 'stories'.
    Only entities with >= 2 insights in the same period get a story_id.
    """
    df = df.copy()
    story_counter = itertools.count(1)

    # Group by entity and period (only non-correlation, non-"all" entities)
    entity_period_groups = df[
        (df["entity"] != "all") & (df["type"] != "correlation")
    ].groupby(["entity", "period"])

    for (entity, period), grp in entity_period_groups:
        if len(grp) >= 2:
            story_id = f"STR-{next(story_counter):04d}"
            df.loc[grp.index, "story_id"] = story_id

    return df


# ── Priority ranking ───────────────────────────────────────────────────────────

def compute_district_risk(insights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an aggregate risk score per entity.

    Returns a DataFrame sorted descending by risk_score:
        entity, risk_score, high_count, medium_count, low_count
    """
    if insights_df.empty:
        return pd.DataFrame(columns=["entity", "risk_score", "high_count", "medium_count", "low_count"])

    entity_data = (
        insights_df[insights_df["entity"] != "all"]
        .groupby("entity")["severity"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["High", "Medium", "Low"], fill_value=0)
    )
    entity_data["risk_score"] = (
        entity_data.get("High", 0) * _SEVERITY_SCORE["High"]
        + entity_data.get("Medium", 0) * _SEVERITY_SCORE["Medium"]
        + entity_data.get("Low", 0) * _SEVERITY_SCORE["Low"]
    )
    entity_data = entity_data.rename(columns={
        "High": "high_count", "Medium": "medium_count", "Low": "low_count"
    })
    entity_data = entity_data.sort_values("risk_score", ascending=False)
    entity_data.index.name = "entity"
    return entity_data.reset_index()


# ── Threshold breach detection (FR-5) ─────────────────────────────────────────

def detect_breaches(
    df: pd.DataFrame,
    cfg: Config,
    entity_col: Optional[str] = None,
    period_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Detect user-configured absolute floor or ceiling breaches.
    """
    if not cfg.breach_enabled:
        return pd.DataFrame(columns=[
            "entity", "indicator", "period", "value", "bound",
            "bound_type", "breach_direction", "is_adverse",
        ])

    entity_col = entity_col or cfg.entity_col
    period_col = period_col or cfg.period_col
    rows = []

    for _, r in df.iterrows():
        entity = r[entity_col]
        period = r[period_col]
        for ind, floor_val in cfg.breach_floors.items():
            if ind in r and pd.notna(r[ind]) and r[ind] < floor_val:
                polarity = cfg.get_polarity(ind)
                is_adverse = (polarity == 1)
                rows.append({
                    "entity": entity,
                    "indicator": ind,
                    "period": period,
                    "value": float(r[ind]),
                    "bound": float(floor_val),
                    "bound_type": "floor",
                    "breach_direction": "below",
                    "is_adverse": is_adverse,
                })
        for ind, ceil_val in cfg.breach_ceilings.items():
            if ind in r and pd.notna(r[ind]) and r[ind] > ceil_val:
                polarity = cfg.get_polarity(ind)
                is_adverse = (polarity == -1)
                rows.append({
                    "entity": entity,
                    "indicator": ind,
                    "period": period,
                    "value": float(r[ind]),
                    "bound": float(ceil_val),
                    "bound_type": "ceiling",
                    "breach_direction": "above",
                    "is_adverse": is_adverse,
                })

    if not rows:
        return pd.DataFrame(columns=[
            "entity", "indicator", "period", "value", "bound",
            "bound_type", "breach_direction", "is_adverse",
        ])

    return pd.DataFrame(rows)
