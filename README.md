# 🏥 Auto-Analytics Engine for District Healthcare Performance

**Equilym Assignment 4 — Automated Insight Generation**  
**Author:** Samir Prajapati  
**Stack:** Python 3.11 · Streamlit · Pandas · NumPy · SciPy · Plotly · Pytest

---

## 📌 Executive Summary

The **Auto-Analytics Engine** transforms raw district-level health monitoring data into structured, actionable intelligence for state and district healthcare leadership. 

Without hardcoded district names, indicator logic, or arbitrary magic cutoffs, the engine automatically extracts:
1. **Longitudinal Trends:** Period-over-period shifts with polarity awareness (distinguishing deterioration from progress).
2. **Cross-Sectional Outliers:** Severe statistical anomalies detected using masking-resistant methodologies.
3. **Indicator Correlations:** Pooled associations with transparency on sample size ($n$), $p$-values, and Spearman comparisons.
4. **Insight Stories & District Priority Ranking:** De-duplicated multi-indicator narratives that answer: *"Which district requires immediate administrative intervention?"*

---

## 🚀 Quick Start

### 1. Local Setup (Virtual Environment)
```bash
# Clone and enter directory
cd "Equilym Assignment - Samir"

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

### 3. Run Automated Test Suite
```bash
pytest tests/ -v
```
All 19 automated tests (Golden tests, Generality tests on synthetic datasets, and No-hardcoding source linter) will execute.

---

## 🏗️ Architecture & Folder Structure

```
├── app.py                    # Streamlit Dashboard (Executive UI & Interactive Visuals)
├── requirements.txt          # Core dependencies (pandas, streamlit, plotly, scipy, etc.)
├── requirements-dev.txt      # Testing suite dependencies (pytest, pytest-cov)
├── README.md                 # Complete system documentation
├── data/
│   └── sample.csv            # Official company benchmark dataset
├── engine/                   # Decoupled Analytical Core
│   ├── __init__.py
│   ├── config.py             # Config Dataclass: Single source of truth for all thresholds
│   ├── loader.py             # Heuristic schema auto-detection & data validation
│   ├── trends.py             # Period-over-period % change & polarity wording
│   ├── outliers.py           # Robust Z (MAD), IQR, Standard Z, Leave-One-Out Z
│   ├── correlations.py       # Pearson, Spearman, sample-size confidence & p-values
│   ├── severity.py           # Normalised ratio-based severity classification
│   ├── insights.py           # Row assembly, templates, compound stories & risk index
│   └── export.py             # Deliverable exports (CSV, JSON, HTML)
├── tests/                    # Comprehensive Verification
│   ├── __init__.py
│   ├── conftest.py           # Benchmark & synthetic dataset fixtures
│   ├── test_golden.py        # Verified mathematical golden assertions
│   ├── test_generality.py    # Zero-hardcode proof with completely novel datasets
│   └── test_no_hardcode.py   # AST/source lint test asserting zero hardcoded names

```

---

## 🔬 Statistical Honesty & Key Methodological Decisions

### 1. Why Robust Z (MAD) is the Default for Outliers
In small healthcare cohorts (e.g. 6 districts), a severe outlier inflates the sample standard deviation, creating a **masking effect** where naive standard $z$-scores fail to flag genuine anomalies.
* In our benchmark dataset, a naive $z$-score on pooled ANC coverage yields only $-2.95\sigma$, missing a strict $3\sigma$ threshold.
* **Robust Z** uses the Median Absolute Deviation ($0.6745 \times \frac{x - \text{median}}{\text{MAD}}$), yielding $-7.4\sigma$, correctly identifying the crisis without masking.

### 2. Refusal of Per-District 2-Point Correlations
A correlation computed on only 2 timepoints per district is mathematically guaranteed to equal $\pm 1.0$ regardless of data relationships, conveying zero statistical meaning. The engine refuses correlation on $n < 5$ and performs cross-district pooled correlation with explicit $n$ and $p$-values.

### 3. Mandatory Small-Sample Correlation Caveat
> **⚠️ Statistical Integrity Caveat:** *Correlation computed across small cohorts ($n < 30$). With repeated observations per district, observations are not independent (autocorrelation). All correlation findings must be interpreted strictly as exploratory hypotheses requiring domain verification.*

### 4. Dynamic Severity Calibration (Zero Magic Cutoffs)
Severity is strictly data-driven via normalised threshold ratios:
$$\text{Ratio} = \frac{|\text{Score}|}{\text{User-Configured Threshold}}$$
* $\text{Ratio} < 1.25 \implies \text{Low}$
* $1.25 \le \text{Ratio} < 1.50 \implies \text{Medium}$
* $\text{Ratio} \ge 1.50 \implies \text{High}$
Favourable events (e.g. positive immunisation outliers) are capped at Low severity. Moving any threshold slider in the UI dynamically recalibrates all severity bands across the dataset.

---

## 📊 Export Deliverables

From **Tab 5 (Export Reports)**, health officials can export:
* **`insights.csv` (Spec 9.1):** All structured insights with IDs (`INS-XXXX`), severity, metrics, and templated explanations.
* **`correlation_matrix.csv` (Spec 9.2):** Full Pearson correlation matrix across all numeric indicators.
* **`insights.json` (Spec 9.3):** Programmatically ingestible JSON representation.
* **`executive_health_report.html` (Stand-out S10):** Self-contained, styled HTML report of critical priority findings.
