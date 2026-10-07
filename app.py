"""
app.py — Auto-Analytics Engine Streamlit Web Application.

A professional, decision-support dashboard for automated insight generation
across district healthcare performance datasets.

Zero hardcoded entity names, indicator names, or magic threshold cutoffs.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.config import Config
from engine.correlations import CORRELATION_CAVEAT, detect_correlations
from engine.export import (
    export_corr_matrix_csv,
    export_html_report,
    export_insights_csv,
    export_insights_json,
)
from engine.insights import (
    build_insights,
    compute_district_risk,
    detect_breaches,
    format_indicator_label,
)
from engine.loader import ValidationReport, load_csv
from engine.outliers import detect_outliers
from engine.trends import detect_trends

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="District Health Auto-Analytics Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Date & Period Display Formatter ──────────────────────────────────────────
def format_period_display(p) -> str:
    """Format dates/periods cleanly as YYYY-MM or standard date string."""
    if isinstance(p, (pd.Timestamp, np.datetime64)) or hasattr(p, "strftime"):
        try:
            return pd.to_datetime(p).strftime("%Y-%m")
        except Exception:
            pass
    s = str(p)
    return s.replace(" 00:00:00", "")


# ── Executive Custom Styling (CSS) ────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.85rem;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }
    
    .badge-high {
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
        font-size: 0.78rem;
        border: 1px solid #fca5a5;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #b45309;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
        font-size: 0.78rem;
        border: 1px solid #fcd34d;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
        font-size: 0.78rem;
        border: 1px solid #86efac;
    }
    
    .story-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    .story-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #1e293b;
        margin-bottom: 0.35rem;
    }
    .story-body {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.45;
    }
    
    .stat-caveat {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #92400e;
        margin: 0.75rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data Loading Helper with Cache ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data_cached(
    source_bytes: Optional[bytes],
    source_path_str: Optional[str],
    entity_hint: Optional[str] = None,
    period_hint: Optional[str] = None,
) -> Tuple[pd.DataFrame, ValidationReport]:
    if source_bytes is not None:
        buf = io.BytesIO(source_bytes)
        return load_csv(buf, entity_hint=entity_hint, period_hint=period_hint)
    elif source_path_str:
        return load_csv(source_path_str, entity_hint=entity_hint, period_hint=period_hint)
    return pd.DataFrame(), ValidationReport()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <div>
            <h1>🏥 District Health Auto-Analytics Engine</h1>
            <p>Automated insight generation, statistical anomaly discovery & decision support</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Quick Guide Expander ──────────────────────────────────────────────────────
with st.expander("ℹ️ How to Use This Engine (User Guide)", expanded=False):
    st.markdown(
        """
        - **Data Ingestion:** By default, the application runs on the bundled sample dataset. You can upload any district healthcare CSV via the sidebar.
        - **Interactive Filters:** Multi-select districts, months, or indicators in the sidebar to dynamically focus your analysis.
        - **Sensitivity Calibration:** Tune trend percentage thresholds, outlier detection methods (Robust Z, IQR, or Z-Score), and correlation sensitivity $|r|$.
        - **Five Diagnostic Views:**
          1. **Executive Overview & Stories:** See the prioritised District Risk Ranking (*Where to intervene first*) and compound event stories.
          2. **Discovered Insights:** Browse the complete table of findings with data-driven severity badges and text search.
          3. **Visual Analytics:** Diagnostic charts including severity breakdowns, correlation heatmap, trajectory lines, and Month-over-Month change matrices.
          4. **Data & Validation Audit:** Review data quality checks, schema auto-detection, and profile statistics.
          5. **Export Reports:** Download official `insights.csv`, `correlation_matrix.csv`, `insights.json`, and shareable HTML reports.
        """
    )


# ── SIDEBAR CONTROLS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Engine Control Panel")

    # 1. Data Source Selection
    st.markdown("#### 📁 Data Source")
    data_source_mode = st.radio(
        "Choose dataset:",
        ["Bundled Sample Dataset", "Upload Custom CSV"],
        index=0,
        label_visibility="collapsed",
    )

    raw_bytes: Optional[bytes] = None
    file_path: Optional[str] = None

    if data_source_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload tidy CSV (month, district, indicators...)",
            type=["csv"],
            help="Upload a CSV with an entity column, period column, and numeric indicators.",
        )
        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
        else:
            st.info("Please upload a CSV file to begin analysis.")
    else:
        sample_path = Path(__file__).resolve().parent / "data" / "sample.csv"
        if not sample_path.exists():
            sample_path = Path("data/sample.csv")
        if sample_path.exists():
            file_path = str(sample_path)
        else:
            st.error("Sample file not found at data/sample.csv")

    # Initial probe to determine column names for overrides
    temp_df, temp_report = (
        load_data_cached(raw_bytes, file_path) if (raw_bytes or file_path) else (pd.DataFrame(), ValidationReport())
    )

    detected_entity_col = temp_report.entity_col or "district"
    detected_period_col = temp_report.period_col or "month"

    all_cols = list(temp_df.columns) if not temp_df.empty else []

    with st.expander("🛠️ Schema Mapping Overrides", expanded=False):
        chosen_entity_col = st.selectbox(
            "Entity / District Column",
            options=all_cols if all_cols else [detected_entity_col],
            index=all_cols.index(detected_entity_col) if detected_entity_col in all_cols else 0,
            help="Column identifying districts or regions",
        )
        chosen_period_col = st.selectbox(
            "Period / Month Column",
            options=all_cols if all_cols else [detected_period_col],
            index=all_cols.index(detected_period_col) if detected_period_col in all_cols else 0,
            help="Column identifying time periods",
        )

    # Load with selected schema hints
    df, val_report = (
        load_data_cached(
            raw_bytes,
            file_path,
            entity_hint=chosen_entity_col,
            period_hint=chosen_period_col,
        )
        if (raw_bytes or file_path)
        else (pd.DataFrame(), ValidationReport())
    )

    if df.empty or val_report.is_fatal:
        if val_report.fatal:
            for err in val_report.fatal:
                st.error(f"❌ {err}")
        st.stop()

    entity_col = val_report.entity_col
    period_col = val_report.period_col
    indicator_cols = val_report.indicator_cols

    # 2. Live Global Filters (FR-1)
    st.markdown("---")
    st.markdown("#### 🔍 Live Filters")

    unique_entities = sorted(list(df[entity_col].dropna().unique()))
    selected_entities = st.multiselect(
        f"Filter {format_indicator_label(entity_col)}s:",
        options=unique_entities,
        default=unique_entities,
        help="Filter specific entities across all views and computations",
    )

    unique_periods = sorted(list(df[period_col].dropna().unique()))
    selected_periods = st.multiselect(
        f"Filter {format_indicator_label(period_col)}s:",
        options=unique_periods,
        default=unique_periods,
        format_func=format_period_display,
        help="Select time periods to include in evaluation",
    )

    selected_indicators = st.multiselect(
        "Filter Indicators:",
        options=indicator_cols,
        default=indicator_cols,
        format_func=format_indicator_label,
        help="Select indicators for trend, outlier, and correlation scrutiny",
    )

    # 3. Detection Parameters & Thresholds
    st.markdown("---")
    st.markdown("#### 📊 Trend Sensitivity")
    trend_threshold = st.slider(
        "Period-over-Period Flag Threshold (%)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="Minimum percentage change between consecutive periods to trigger a trend insight",
    )

    st.markdown("---")
    st.markdown("#### 🔎 Outlier Detection")
    outlier_method_choice = st.radio(
        "Methodology:",
        ["Robust Z (MAD - Recommended)", "IQR Fences", "Standard Z-Score"],
        index=0,
        help=(
            "Robust Z uses Median Absolute Deviation, preventing outliers from inflating "
            "the standard deviation (avoids masking)."
        ),
    )
    method_key_map = {
        "Robust Z (MAD - Recommended)": "robust_z",
        "IQR Fences": "iqr",
        "Standard Z-Score": "zscore",
    }
    outlier_method = method_key_map[outlier_method_choice]

    col_out1, col_out2 = st.columns(2)
    with col_out1:
        if outlier_method == "robust_z":
            robust_z_thresh = st.number_input(
                "Robust Z (|score| ≥)",
                min_value=1.0,
                max_value=10.0,
                value=3.5,
                step=0.25,
            )
            z_thresh = 3.0
            iqr_k = 1.5
        elif outlier_method == "iqr":
            iqr_k = st.number_input(
                "IQR Multiplier (k)",
                min_value=0.5,
                max_value=4.0,
                value=1.5,
                step=0.25,
            )
            z_thresh = 3.0
            robust_z_thresh = 3.5
        else:
            z_thresh = st.number_input(
                "Z-Score (|z| ≥)",
                min_value=1.0,
                max_value=6.0,
                value=3.0,
                step=0.25,
            )
            iqr_k = 1.5
            robust_z_thresh = 3.5

    with col_out2:
        outlier_scope = st.selectbox(
            "Evaluation Scope:",
            ["cross_section", "pooled"],
            index=0,
            format_func=lambda s: "Within Period" if s == "cross_section" else "Pooled Across All",
            help="Evaluate relative to peer districts in the same period, or across all observations",
        )

    st.markdown("---")
    st.markdown("#### 🔗 Correlation Settings")
    corr_threshold = st.slider(
        "Minimum |r| to Flag:",
        min_value=0.30,
        max_value=0.99,
        value=0.70,
        step=0.05,
        help="Flag indicator pairs exceeding this correlation coefficient",
    )
    min_points_corr = st.number_input(
        "Minimum Data Points (n):",
        min_value=3,
        max_value=100,
        value=5,
        step=1,
        help="Correlations with fewer than this many observations are refused",
    )

    st.markdown("---")
    st.markdown("#### ⚠️ Severity Bands")
    col_sev1, col_sev2 = st.columns(2)
    with col_sev1:
        severity_low_max = st.number_input(
            "Low/Medium Ratio Cutoff",
            min_value=1.00,
            max_value=2.00,
            value=1.25,
            step=0.05,
            help="Ratio (score / threshold) < this is Low",
        )
    with col_sev2:
        severity_med_max = st.number_input(
            "Medium/High Ratio Cutoff",
            min_value=1.05,
            max_value=3.00,
            value=1.50,
            step=0.05,
            help="Ratio (score / threshold) ≥ this is High",
        )

    # 4. Polarity Configuration (S4 Stand-out)
    st.markdown("---")
    with st.expander("🎯 Indicator Polarity (Good vs Adverse)", expanded=False):
        st.caption(
            "Define whether a higher value signifies an improvement (+1) or deterioration (-1). "
            "Adverse shifts are escalated to High severity; favourable ones are capped at Low."
        )
        polarity_map: Dict[str, int] = {}
        for ind in indicator_cols:
            ind_lower = ind.lower()
            default_adverse = any(
                w in ind_lower
                for w in ["risk", "case", "death", "mortality", "infect", "drop", "incident", "malaria", "deficit"]
            )
            choice = st.radio(
                f"{format_indicator_label(ind)}:",
                ["Higher is Better (+1)", "Lower is Better (-1)"],
                index=1 if default_adverse else 0,
                key=f"pol_{ind}",
                horizontal=True,
            )
            polarity_map[ind] = 1 if "Higher" in choice else -1

    # 5. Threshold Breach Configuration (FR-5)
    with st.expander("🛡️ Absolute Threshold Breaches", expanded=False):
        breach_enabled = st.checkbox("Enable Floor/Ceiling Breach Checks", value=False)
        breach_floors: Dict[str, float] = {}
        breach_ceilings: Dict[str, float] = {}
        if breach_enabled:
            for ind in indicator_cols:
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    floor_val = st.text_input(f"{format_indicator_label(ind)} Floor", value="", key=f"fl_{ind}")
                    if floor_val.strip():
                        try:
                            breach_floors[ind] = float(floor_val)
                        except ValueError:
                            pass
                with col_b2:
                    ceil_val = st.text_input(f"{format_indicator_label(ind)} Ceiling", value="", key=f"cl_{ind}")
                    if ceil_val.strip():
                        try:
                            breach_ceilings[ind] = float(ceil_val)
                        except ValueError:
                            pass


# ── Filter Application ────────────────────────────────────────────────────────
filtered_df = df.copy()

if selected_entities:
    filtered_df = filtered_df[filtered_df[entity_col].isin(selected_entities)]

if selected_periods:
    filtered_df = filtered_df[filtered_df[period_col].isin(selected_periods)]

active_indicator_cols = [c for c in indicator_cols if c in selected_indicators]

if filtered_df.empty or not active_indicator_cols:
    st.warning("⚠️ No records match the selected filters. Please select at least one entity, period, and indicator in the sidebar.")
    st.stop()


# ── Assemble Config Object (Single Source of Truth) ───────────────────────────
cfg = Config(
    trend_threshold=float(trend_threshold),
    outlier_method=outlier_method,
    iqr_k=float(iqr_k),
    z_threshold=float(z_thresh),
    robust_z_threshold=float(robust_z_thresh),
    outlier_scope=outlier_scope,
    corr_threshold=float(corr_threshold),
    min_points_corr=int(min_points_corr),
    severity_low_max=float(severity_low_max),
    severity_med_max=float(severity_med_max),
    breach_enabled=breach_enabled,
    breach_floors=breach_floors,
    breach_ceilings=breach_ceilings,
    polarity=polarity_map,
    entity_col=entity_col,
    period_col=period_col,
)


# ── Run Engine Pipeline ───────────────────────────────────────────────────────
trends_df = detect_trends(
    filtered_df,
    cfg,
    entity_col=entity_col,
    period_col=period_col,
    indicator_cols=active_indicator_cols,
)

outliers_df = detect_outliers(
    filtered_df,
    cfg,
    entity_col=entity_col,
    period_col=period_col,
    indicator_cols=active_indicator_cols,
)

flagged_corr_df, full_corr_matrix = detect_correlations(
    filtered_df,
    cfg,
    entity_col=entity_col,
    period_col=period_col,
    indicator_cols=active_indicator_cols,
)

breaches_df = (
    detect_breaches(filtered_df, cfg, entity_col=entity_col, period_col=period_col)
    if breach_enabled
    else pd.DataFrame()
)

insights_df = build_insights(
    trends_df=trends_df,
    outliers_df=outliers_df,
    correlations_df=flagged_corr_df,
    cfg=cfg,
    breaches_df=breaches_df,
)

district_risk_df = compute_district_risk(insights_df)


# ── Executive KPI Summary Cards ───────────────────────────────────────────────
total_insights = len(insights_df)
high_sev_count = (insights_df["severity"] == "High").sum() if total_insights > 0 else 0
med_sev_count = (insights_df["severity"] == "Medium").sum() if total_insights > 0 else 0
low_sev_count = (insights_df["severity"] == "Low").sum() if total_insights > 0 else 0
trend_count = len(trends_df)
outlier_count = len(outliers_df)
corr_count = len(flagged_corr_df)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Insights</div>
            <div class="metric-value">{total_insights}</div>
            <div class="metric-sub">{len(selected_entities)} {format_indicator_label(entity_col)}s monitored</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #ef4444;">
            <div class="metric-label" style="color: #ef4444;">Critical / High Severity</div>
            <div class="metric-value" style="color: #b91c1c;">{high_sev_count}</div>
            <div class="metric-sub">Immediate intervention required</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Significant Trends</div>
            <div class="metric-value">{trend_count}</div>
            <div class="metric-sub">≥ {trend_threshold:.0f}% shift detected</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Statistical Outliers</div>
            <div class="metric-value">{outlier_count}</div>
            <div class="metric-sub">{outlier_method_choice.split()[0]} method</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Correlations (|r| ≥ {corr_threshold:.2f})</div>
            <div class="metric-value">{corr_count}</div>
            <div class="metric-sub">Across indicator pairs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Executive Overview & Stories",
        "💡 Discovered Insights",
        "📈 Visual Analytics",
        "📋 Data & Validation Audit",
        "📥 Export Reports",
    ]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE OVERVIEW & STORIES (S2, S3 Stand-outs)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🎯 Executive Decision Support")
    st.caption(
        "Synthesised intelligence combining multi-indicator anomaly grouping (Insight Stories) "
        "and prioritised district risk ranking to answer: *Where do we intervene first?*"
    )

    col_risk, col_stories = st.columns([1, 1.25])

    with col_risk:
        st.markdown("#### 🚨 Priority Action Ranking (Risk Index)")
        if not district_risk_df.empty:
            fig_risk = px.bar(
                district_risk_df,
                x="risk_score",
                y="entity",
                orientation="h",
                text="risk_score",
                color="risk_score",
                color_continuous_scale=["#fef3c7", "#f97316", "#ef4444"],
                labels={"risk_score": "Composite Risk Score", "entity": format_indicator_label(entity_col)},
            )
            fig_risk.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=320,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Compact table
            st.dataframe(
                district_risk_df.rename(
                    columns={
                        "entity": format_indicator_label(entity_col),
                        "risk_score": "Risk Index",
                        "high_count": "High (×3)",
                        "medium_count": "Medium (×2)",
                        "low_count": "Low (×1)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No priority risk scores — no significant findings detected under current thresholds.")

    with col_stories:
        st.markdown("#### 📑 Compound Insight Stories (De-duplicated Findings)")
        st.caption("Multiple anomalies occurring in the same district and period consolidated into coherent event stories.")

        story_groups = insights_df[insights_df["story_id"].notna()]
        if not story_groups.empty:
            unique_stories = story_groups["story_id"].unique()
            for s_id in unique_stories:
                sub = story_groups[story_groups["story_id"] == s_id]
                first_row = sub.iloc[0]
                entity_name = first_row["entity"]
                period_name = format_period_display(first_row["period"])

                # Build summary parts
                parts = []
                for _, r in sub.iterrows():
                    ind_fmt = format_indicator_label(r["indicator"])
                    if r["type"] == "trend":
                        chg = r["change_pct"]
                        arrow = "rose" if chg > 0 else "fell"
                        parts.append(f"{ind_fmt} {arrow} {abs(chg):.1f}%")
                    elif r["type"] == "outlier":
                        parts.append(f"{ind_fmt} outlier ({r['metric']:.1f})")
                    elif r["type"] == "threshold_breach":
                        parts.append(f"{ind_fmt} breach")

                summary_text = ", and ".join(parts)
                has_high = (sub["severity"] == "High").any()
                badge_class = "badge-high" if has_high else "badge-medium"
                badge_label = "HIGH PRIORITY" if has_high else "MODERATE"

                st.markdown(
                    f"""
                    <div class="story-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="story-title">📍 {entity_name} ({period_name})</div>
                            <span class="{badge_class}">{badge_label}</span>
                        </div>
                        <div class="story-body">
                            <strong>Story {s_id}:</strong> {summary_text}.<br/>
                            <span style="color: #64748b; font-size: 0.8rem;">
                                Includes {len(sub)} correlated events: {', '.join(sub['insight_id'].tolist())}.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No multi-anomaly compound stories detected. Individual findings are listed in the Discovered Insights tab.")

    st.markdown("---")
    st.markdown("#### 🔴 Urgent Items (High Severity Action Queue)")
    high_insights = insights_df[insights_df["severity"] == "High"]
    if not high_insights.empty:
        for _, row in high_insights.iterrows():
            formatted_p = format_period_display(row["period"])
            st.markdown(
                f"""
                <div style="background: #fff; border: 1px solid #fecaca; border-left: 4px solid #ef4444; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #b91c1c;">{row['insight_id']}</strong> • 
                        <span style="font-weight: 600; color: #1e293b;">{row['entity']} ({formatted_p}):</span> 
                        <span style="color: #334155;">{row['explanation']}</span>
                    </div>
                    <span class="badge-high">High</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ No High Severity issues flagged under the current configuration.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: ALL DISCOVERED INSIGHTS (FR-6, S5)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 💡 All Discovered Insights")
    st.caption(
        "Structured, human-readable insights with data-derived severity, dynamic templates, "
        "and sample-size confidence labelling."
    )

    if insights_df.empty:
        st.info("No insights found matching current filters and thresholds.")
    else:
        # Table filters
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            sev_filter = st.multiselect(
                "Filter by Severity:",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )
        with f_col2:
            available_types = sorted(list(insights_df["type"].unique()))
            type_filter = st.multiselect(
                "Filter by Type:",
                options=available_types,
                default=available_types,
                format_func=lambda t: t.replace("_", " ").title(),
            )
        with f_col3:
            search_query = st.text_input("🔍 Search Explanation / District:", placeholder="Type to filter...")

        filtered_insights = insights_df[
            insights_df["severity"].isin(sev_filter) & insights_df["type"].isin(type_filter)
        ]

        if search_query.strip():
            query = search_query.strip().lower()
            filtered_insights = filtered_insights[
                filtered_insights["explanation"].str.lower().str.contains(query, na=False)
                | filtered_insights["entity"].str.lower().str.contains(query, na=False)
                | filtered_insights["indicator"].str.lower().str.contains(query, na=False)
            ]

        st.markdown(f"**Showing {len(filtered_insights)} of {len(insights_df)} insights**")

        # Display table with formatting
        display_df = filtered_insights.copy()
        display_df["indicator"] = display_df["indicator"].apply(format_indicator_label)
        display_df["period"] = display_df["period"].apply(format_period_display)
        display_df["type"] = display_df["type"].str.replace("_", " ").str.title()

        def style_severity(val):
            if val == "High":
                return "background-color: #fee2e2; color: #b91c1c; font-weight: bold;"
            elif val == "Medium":
                return "background-color: #fef3c7; color: #b45309; font-weight: bold;"
            elif val == "Low":
                return "background-color: #dcfce7; color: #15803d; font-weight: bold;"
            return ""

        styled_table = display_df[
            [
                "insight_id",
                "severity",
                "type",
                "entity",
                "period",
                "indicator",
                "metric",
                "confidence",
                "explanation",
                "story_id",
            ]
        ].style.map(style_severity, subset=["severity"])

        st.dataframe(styled_table, use_container_width=True, hide_index=True)

        if "correlation" in type_filter:
            st.markdown(
                f"""
                <div class="stat-caveat">
                    <strong>⚠️ Statistical Integrity Caveat:</strong> {CORRELATION_CAVEAT.format(n=len(filtered_df))}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: INTERACTIVE VISUAL ANALYTICS (FR-7, S8)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Visual Analytics & Diagnostic Charts")

    v_row1_col1, v_row1_col2 = st.columns([1, 1.2])

    with v_row1_col1:
        st.markdown("#### Severity Breakdown by Insight Type")
        if not insights_df.empty:
            sev_summary = (
                insights_df.groupby(["type", "severity"])
                .size()
                .reset_index(name="count")
            )
            fig_bar = px.bar(
                sev_summary,
                x="type",
                y="count",
                color="severity",
                barmode="stack",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                labels={"type": "Insight Type", "count": "Number of Insights", "severity": "Severity"},
            )
            fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=340)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No insights to plot.")

    with v_row1_col2:
        st.markdown("#### Full Pearson Correlation Heatmap")
        if not full_corr_matrix.empty and len(full_corr_matrix.columns) >= 2:
            labels = [format_indicator_label(c) for c in full_corr_matrix.columns]
            fig_heat = px.imshow(
                full_corr_matrix,
                x=labels,
                y=labels,
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                text_auto=".2f",
                aspect="auto",
            )
            fig_heat.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=340)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Need at least 2 numeric indicators to construct a correlation heatmap.")

    st.markdown("---")

    # Chart C: Entity Longitudinal Trends with Outlier & Trend overlays
    st.markdown("#### Longitudinal Entity Trajectory & Anomaly Inspection")
    c_col1, c_col2 = st.columns([1.5, 3])

    with c_col1:
        chosen_chart_indicator = st.selectbox(
            "Select Indicator to Inspect:",
            options=active_indicator_cols,
            format_func=format_indicator_label,
            key="chart_indicator",
        )
        st.caption(
            "Solid lines display actual trajectories. Flagged points (anomalies or large shifts) "
            "are highlighted for quick visual verification."
        )

    with c_col2:
        if chosen_chart_indicator:
            fig_line = go.Figure()
            for entity_val in selected_entities:
                sub_ent = filtered_df[filtered_df[entity_col] == entity_val].sort_values(period_col)
                if not sub_ent.empty:
                    fig_line.add_trace(
                        go.Scatter(
                            x=[format_period_display(p) for p in sub_ent[period_col]],
                            y=sub_ent[chosen_chart_indicator],
                            mode="lines+markers",
                            name=str(entity_val),
                            line=dict(width=2),
                            marker=dict(size=6),
                        )
                    )

            # Highlight outliers
            ind_outliers = outliers_df[outliers_df["indicator"] == chosen_chart_indicator]
            if not ind_outliers.empty:
                fig_line.add_trace(
                    go.Scatter(
                        x=[format_period_display(p) for p in ind_outliers["period"]],
                        y=ind_outliers["value"],
                        mode="markers",
                        name="Outlier Flagged",
                        marker=dict(
                            symbol="star",
                            size=14,
                            color="#ef4444",
                            line=dict(width=1, color="#7f1d1d"),
                        ),
                    )
                )

            fig_line.update_layout(
                title=f"{format_indicator_label(chosen_chart_indicator)} across Periods",
                xaxis_title=format_indicator_label(period_col),
                yaxis_title="Observed Value",
                margin=dict(l=20, r=20, t=40, b=20),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_line, use_container_width=True)

    # Chart D: Month-over-Month % Change Heatmap (S8 Stand-out)
    st.markdown("---")
    st.markdown("#### 🗺️ Month-over-Month % Change Matrix (Entity × Indicator)")
    st.caption("Cross-district heatmap highlighting direction and intensity of shifts.")

    if not trends_df.empty:
        # Pivot trends for the latest available period pair
        latest_period_to = trends_df["period_to"].max()
        trends_sub = trends_df[trends_df["period_to"] == latest_period_to]

        if not trends_sub.empty:
            pivot_pct = trends_sub.pivot(
                index="entity",
                columns="indicator",
                values="pct_change",
            ).fillna(0.0)

            pivot_pct.columns = [format_indicator_label(c) for c in pivot_pct.columns]

            fig_mom = px.imshow(
                pivot_pct,
                color_continuous_scale="Spectral_r",
                text_auto=".1f",
                aspect="auto",
                labels=dict(color="% Change"),
            )
            fig_mom.update_layout(
                title=f"MoM % Shift towards {format_period_display(latest_period_to)}",
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
            )
            st.plotly_chart(fig_mom, use_container_width=True)
        else:
            st.info("No trends detected for the most recent period pair.")
    else:
        st.info("No significant trends detected under current % threshold to build the MoM matrix.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: DATA & VALIDATION AUDIT (FR-1, 5.3, S9)
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📋 Dataset Profile & Data Quality Audit")
    st.caption("Verification of data integrity, schema consistency, and statistical viability.")

    # Validation banner
    if val_report.warnings:
        with st.expander(f"⚠️ Data Quality Notices & Recommendations ({len(val_report.warnings)})", expanded=True):
            for w in val_report.warnings:
                st.warning(f"• {w}")

    # Summary metrics
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.metric("Total Records", f"{len(df):,}")
    with d_col2:
        st.metric("Unique Entities", f"{df[entity_col].nunique()}")
    with d_col3:
        st.metric("Evaluated Periods", f"{df[period_col].nunique()}")
    with d_col4:
        st.metric("Numeric Indicators", f"{len(indicator_cols)}")

    st.markdown("#### Raw Data Preview")
    preview_df = df.copy()
    preview_df[period_col] = preview_df[period_col].apply(format_period_display)
    st.dataframe(preview_df, use_container_width=True)

    # Missing values profile
    missing_counts = df[indicator_cols].isna().sum()
    if missing_counts.sum() > 0:
        st.markdown("#### Missing Values per Indicator")
        miss_df = pd.DataFrame({
            "Indicator": [format_indicator_label(c) for c in indicator_cols],
            "Missing Count": missing_counts.values,
        })
        st.dataframe(miss_df[miss_df["Missing Count"] > 0], hide_index=True)
    else:
        st.success("✅ Clean dataset: Zero missing (NaN) values detected across all indicator columns.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: EXPORT & REPORTS (FR-4, FR-7, S10)
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📥 Official Deliverable Exports & Reports")
    st.caption("One-click exports compliant with assignment technical specifications.")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("#### 📄 Data Deliverables")
        # 1. Insights CSV (spec 9.1)
        csv_bytes = export_insights_csv(insights_df)
        st.download_button(
            label="⬇️ Download Insights CSV (Spec 9.1)",
            data=csv_bytes,
            file_name="insights.csv",
            mime="text/csv",
            help="Download structured insights table with severity, metrics, and explanations",
        )

        # 2. Correlation Matrix CSV (spec 9.2)
        corr_bytes = export_corr_matrix_csv(full_corr_matrix)
        st.download_button(
            label="⬇️ Download Correlation Matrix CSV (Spec 9.2)",
            data=corr_bytes,
            file_name="correlation_matrix.csv",
            mime="text/csv",
            help="Download full Pearson correlation matrix across all numeric indicators",
        )

        # 3. Insights JSON (spec 9.3)
        json_bytes = export_insights_json(insights_df)
        st.download_button(
            label="⬇️ Download Insights JSON (Spec 9.3)",
            data=json_bytes,
            file_name="insights.json",
            mime="application/json",
            help="Download insights in JSON format for automated downstream ingestion",
        )

    with exp_col2:
        st.markdown("#### 📑 Executive Briefing Document")
        # 4. HTML Report (S10 Stand-out)
        html_bytes = export_html_report(insights_df, top_n=15)
        st.download_button(
            label="⬇️ Download Shareable HTML Report (S10)",
            data=html_bytes,
            file_name="executive_health_report.html",
            mime="text/html",
            help="Self-contained, styled HTML report of top critical findings",
        )

        # Copyable Markdown Summary
        st.markdown("#### 📋 Executive Text Briefing (Copy-Ready)")
        summary_md_lines = [
            f"**Auto-Analytics Executive Briefing — {len(insights_df)} Total Insights**",
            f"- **High Severity Items:** {high_sev_count}",
            f"- **Significant Trends:** {trend_count}",
            f"- **Outliers:** {outlier_count}",
            "",
            "**Top Action Items:**",
        ]
        for _, r in insights_df.head(5).iterrows():
            summary_md_lines.append(
                f"- [{r['severity']}] **{r['entity']} ({format_period_display(r['period'])})**: {r['explanation']}"
            )

        summary_md = "\n".join(summary_md_lines)
        st.text_area("Markdown Summary", value=summary_md, height=130)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 1rem 0;">
        Equilym Health Analytics Engine • Built with Python, Streamlit, Pandas & Plotly • 
        Fully Generalised Architecture (Zero Hardcoding)
    </div>
    """,
    unsafe_allow_html=True,
)
