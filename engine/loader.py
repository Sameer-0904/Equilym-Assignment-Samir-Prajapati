"""
loader.py — CSV loading, validation, and schema auto-detection.

GENERALITY RULES:
- No district names, indicator names, or numeric thresholds are hardcoded here.
- All column detection is heuristic + overridable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Collects all validation findings without raising immediately."""
    fatal: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Feature flags derived from data quality
    trend_confidence_disabled: bool = False  # <3 months per district
    correlation_unstable: bool = False        # <10 districts
    entity_col: str = ""
    period_col: str = ""
    indicator_cols: List[str] = field(default_factory=list)

    @property
    def is_fatal(self) -> bool:
        return len(self.fatal) > 0

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_fatal(self, msg: str) -> None:
        self.fatal.append(msg)


# ── Auto-detection helpers ─────────────────────────────────────────────────────

_ENTITY_KEYWORDS = {"district", "region", "area", "entity", "location", "state", "city", "zone"}
_PERIOD_KEYWORDS = {"month", "date", "period", "year", "time", "week"}


def auto_detect_schema(
    df: pd.DataFrame,
    entity_hint: Optional[str] = None,
    period_hint: Optional[str] = None,
) -> Tuple[str, str, List[str]]:
    """
    Heuristically detect (entity_col, period_col, indicator_cols).

    Parameters
    ----------
    df : DataFrame with raw data
    entity_hint : user-supplied entity column override
    period_hint : user-supplied period column override

    Returns
    -------
    (entity_col, period_col, [indicator_cols...])
    """
    cols = list(df.columns)

    # ── Entity column ──────────────────────────────────────────────────────
    entity_col = _resolve_col(df, cols, entity_hint, _ENTITY_KEYWORDS, prefer_string=True)

    # ── Period column ──────────────────────────────────────────────────────
    period_col = _resolve_col(df, cols, period_hint, _PERIOD_KEYWORDS, prefer_string=True,
                              exclude={entity_col})

    # ── Indicator columns: everything numeric that is not an identifier ────
    id_cols = {entity_col, period_col}
    indicator_cols = [
        c for c in cols
        if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Also include string cols that look numeric after coercion
    for c in cols:
        if c not in id_cols and c not in indicator_cols:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.5:
                indicator_cols.append(c)

    return entity_col, period_col, indicator_cols


def _resolve_col(
    df: pd.DataFrame,
    cols: List[str],
    hint: Optional[str],
    keywords: set,
    prefer_string: bool = False,
    exclude: Optional[set] = None,
) -> str:
    exclude = exclude or set()

    # 1. User-supplied hint takes priority
    if hint and hint in cols:
        return hint

    # 2. Keyword match in column name
    for c in cols:
        if c in exclude:
            continue
        if c.lower() in keywords or any(kw in c.lower() for kw in keywords):
            return c

    # 3. Fallback: first string-type column not excluded
    if prefer_string:
        for c in cols:
            if c not in exclude and df[c].dtype == object:
                return c

    # 4. Last resort: first non-excluded column
    for c in cols:
        if c not in exclude:
            return c

    raise ValueError(f"Cannot auto-detect column from {cols!r}. Please specify manually.")


# ── Month parsing ──────────────────────────────────────────────────────────────

def parse_periods(series: pd.Series) -> pd.Series:
    """
    Parse a period column to pandas Period (monthly).
    Accepts: 'YYYY-MM', 'YYYY-MM-DD', any parseable date string.
    """
    # Try YYYY-MM format first
    parsed = pd.to_datetime(series, format="%Y-%m", errors="coerce")
    if parsed.isna().mean() > 0.5:
        # Try full date
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


# ── Main load function ─────────────────────────────────────────────────────────

def load_csv(
    source: Union[str, Path, IO],
    entity_hint: Optional[str] = None,
    period_hint: Optional[str] = None,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Load and validate a healthcare CSV.

    Parameters
    ----------
    source : file path, Path object, or file-like object (for Streamlit uploads)
    entity_hint : override auto-detected entity column name
    period_hint : override auto-detected period column name

    Returns
    -------
    (clean_df, ValidationReport)

    Notes
    -----
    - Fatal errors populate ValidationReport.fatal; the returned df may be empty/partial.
    - Non-fatal issues populate ValidationReport.warnings.
    - Duplicate (entity, period) rows are dropped with a warning.
    """
    report = ValidationReport()

    # ── Load raw ───────────────────────────────────────────────────────────
    try:
        raw_df = pd.read_csv(source)
    except Exception as exc:
        report.add_fatal(f"Could not read CSV: {exc}")
        return pd.DataFrame(), report

    if raw_df.empty:
        report.add_fatal("The CSV file is empty.")
        return raw_df, report

    # ── Schema detection ───────────────────────────────────────────────────
    try:
        entity_col, period_col, indicator_cols = auto_detect_schema(
            raw_df, entity_hint, period_hint
        )
    except ValueError as exc:
        report.add_fatal(str(exc))
        return raw_df, report

    report.entity_col = entity_col
    report.period_col = period_col
    report.indicator_cols = indicator_cols

    if not indicator_cols:
        report.add_fatal(
            f"No numeric indicator columns found. Entity='{entity_col}', Period='{period_col}'."
        )
        return raw_df, report

    df = raw_df.copy()

    # ── Parse period column ────────────────────────────────────────────────
    df[period_col] = parse_periods(df[period_col])
    n_unparsed = df[period_col].isna().sum()
    if n_unparsed > 0:
        report.add_warning(
            f"{n_unparsed} rows have unparseable values in '{period_col}'. "
            "These rows will be included but may sort incorrectly."
        )

    # ── Coerce indicators to numeric ───────────────────────────────────────
    for col in indicator_cols:
        original_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_na = df[col].isna().sum() - original_na
        if new_na > 0:
            report.add_warning(
                f"'{col}': {new_na} non-numeric value(s) coerced to NaN."
            )

    # ── Duplicate (entity, period) rows ───────────────────────────────────
    dupes = df.duplicated(subset=[entity_col, period_col], keep=False)
    if dupes.any():
        n_dupes = dupes.sum()
        report.add_warning(
            f"{n_dupes} duplicate ({entity_col}, {period_col}) rows found. "
            "Keeping the first occurrence of each."
        )
        df = df.drop_duplicates(subset=[entity_col, period_col], keep="first")

    # ── Negative values ────────────────────────────────────────────────────
    for col in indicator_cols:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            report.add_warning(
                f"'{col}': {n_neg} negative value(s). Verify these are valid."
            )

    # ── Percentage > 100 ──────────────────────────────────────────────────
    for col in indicator_cols:
        if "rate" in col.lower() or "coverage" in col.lower() or "immuniz" in col.lower():
            n_over = (df[col] > 100).sum()
            if n_over > 0:
                report.add_warning(
                    f"'{col}' looks like a percentage but has {n_over} value(s) > 100."
                )

    # ── Missing months within a district ──────────────────────────────────
    _check_missing_months(df, entity_col, period_col, report)

    # ── Recommended-minimum warnings ──────────────────────────────────────
    months_per_entity = df.groupby(entity_col)[period_col].nunique()
    if (months_per_entity < 3).any():
        n_low = (months_per_entity < 3).sum()
        report.trend_confidence_disabled = True
        report.add_warning(
            f"{n_low} entity/entities have fewer than 3 months of data. "
            "Trend confidence labels are disabled."
        )

    n_entities = df[entity_col].nunique()
    if n_entities < 10:
        report.correlation_unstable = True
        report.add_warning(
            f"Only {n_entities} distinct entities found (fewer than 10). "
            "Correlation results are highly unstable — treat as exploratory only."
        )

    # ── Missing values summary ─────────────────────────────────────────────
    for col in indicator_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            report.add_warning(f"'{col}': {n_miss} missing value(s).")

    # ── Sort ───────────────────────────────────────────────────────────────
    df = df.sort_values([entity_col, period_col]).reset_index(drop=True)

    return df, report


def _check_missing_months(
    df: pd.DataFrame,
    entity_col: str,
    period_col: str,
    report: ValidationReport,
) -> None:
    """Warn if any entity is missing months that other entities have."""
    all_periods = set(df[period_col].dropna().unique())
    for entity, grp in df.groupby(entity_col):
        entity_periods = set(grp[period_col].dropna().unique())
        missing = all_periods - entity_periods
        if missing:
            missing_strs = sorted(str(p) for p in missing)
            report.add_warning(
                f"Entity '{entity}' is missing month(s): {', '.join(missing_strs)}."
            )
