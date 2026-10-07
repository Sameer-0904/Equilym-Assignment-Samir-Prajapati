"""
export.py — Export helpers for insights CSV, correlation matrix CSV,
            insights JSON, and HTML report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def export_insights_csv(insights_df: pd.DataFrame, path: Optional[Union[str, Path]] = None) -> bytes:
    """
    Export insights to CSV per spec 9.1.
    If path is None, returns bytes (for Streamlit download_button).
    """
    csv_bytes = insights_df.to_csv(index=False).encode("utf-8")
    if path:
        Path(path).write_bytes(csv_bytes)
    return csv_bytes


def export_corr_matrix_csv(corr_matrix: pd.DataFrame, path: Optional[Union[str, Path]] = None) -> bytes:
    """
    Export full Pearson correlation matrix per spec 9.2.
    """
    csv_bytes = corr_matrix.to_csv().encode("utf-8")
    if path:
        Path(path).write_bytes(csv_bytes)
    return csv_bytes


def export_insights_json(insights_df: pd.DataFrame, path: Optional[Union[str, Path]] = None) -> bytes:
    """
    Export insights to JSON per spec 9.3.
    """
    records = insights_df.to_dict(orient="records")
    # Make period strings serializable
    for r in records:
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = str(v)
            elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
                r[k] = None
    json_bytes = json.dumps(records, indent=2, default=str).encode("utf-8")
    if path:
        Path(path).write_bytes(json_bytes)
    return json_bytes


def export_html_report(
    insights_df: pd.DataFrame,
    top_n: int = 10,
    path: Optional[Union[str, Path]] = None,
) -> bytes:
    """
    Export top-N insights as a simple, shareable HTML report (S10 stand-out).
    """
    severity_rank = {"High": 0, "Medium": 1, "Low": 2}
    if not insights_df.empty and "severity" in insights_df.columns:
        top = insights_df.sort_values(
            by="severity",
            key=lambda s: s.map(lambda x: severity_rank.get(x, 99)),
        ).head(top_n)
    else:
        top = insights_df.head(top_n)

    _SEVERITY_COLOURS = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}

    rows_html = ""
    for _, row in top.iterrows():
        colour = _SEVERITY_COLOURS.get(row.get("severity", "Low"), "#888")
        rows_html += f"""
        <tr>
            <td><strong>{row.get('insight_id', '')}</strong></td>
            <td><span style="background:{colour};color:white;padding:2px 8px;border-radius:4px;">
                {row.get('severity', '')}
            </span></td>
            <td>{row.get('type', '').replace('_', ' ').title()}</td>
            <td>{row.get('entity', '')}</td>
            <td>{row.get('period', '')}</td>
            <td>{row.get('explanation', '')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Auto-Analytics Engine — Insight Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2rem; color: #333; }}
  h1 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  th {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #ddd; }}
  tr:hover {{ background: #f5f5f5; }}
  .footer {{ margin-top: 2rem; font-size: 0.8em; color: #888; }}
</style>
</head>
<body>
<h1>🏥 Auto-Analytics Engine — Top {top_n} Insights</h1>
<p>Generated automatically. Numbers sourced from data, not hardcoded.</p>
<table>
  <thead>
    <tr><th>ID</th><th>Severity</th><th>Type</th><th>Entity</th><th>Period</th><th>Explanation</th></tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<p class="footer">
  ⚠️ Correlation findings have low confidence due to small sample size.
  Treat all findings as hypotheses requiring domain-expert review.
</p>
</body>
</html>"""

    html_bytes = html.encode("utf-8")
    if path:
        Path(path).write_bytes(html_bytes)
    return html_bytes
