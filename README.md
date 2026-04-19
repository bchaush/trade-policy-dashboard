# Trade Policy Shocks & Semiconductor Markets

**Live App → [trade-policy-dashboard-stbpl4rkrwsfgjfxgtrigx.streamlit.app](https://trade-policy-dashboard-stbpl4rkrwsfgjfxgtrigx.streamlit.app)**

An end-to-end event study analyzing how USTR Section 301 tariff announcements affect equity returns of semiconductor-related firms. Combines a classical financial event study framework with LLM-based policy text classification.

---

## Key Findings

| Direction | Avg CAR | N |
|-----------|---------|---|
| Relief | **+1.47%** | 228 |
| Tightening | +0.07% | 24 |
| Mixed | **−4.03%** | 6 |

- **Relief announcements** consistently generate positive cumulative abnormal returns across all firms and model specifications
- **Information clarity dominates direction** — markets penalize ambiguity (mixed events) more than restrictive policy (tightening)
- **Sector heterogeneity** — AMD (β=1.96) and NVIDIA (β=1.91) show highest market sensitivity; Samsung (β=0.22) the lowest
- **LLM pipeline** (Gemini 2.0 Flash) successfully classifies all 43 policy documents by severity, surprise, and direction

---

## Dashboard

The Streamlit dashboard has 4 interactive tabs:

| Tab | Contents |
|-----|----------|
| **Event Explorer** | AR line chart by direction, CAR box plots, summary metrics |
| **Firm Comparison** | Avg CAR per firm, AR through event window, beta coefficients |
| **Key Findings** | 4 main findings, summary stats table, methodology |
| **Prediction Model** | Logistic regression predicting next-day AR direction, confusion matrix, feature coefficients, CV accuracy |

Sidebar filters let you slice by firm, event direction, document type, and year range.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | `yfinance`, `pandas`, `numpy` |
| Event Study | OLS market model (`numpy.linalg.lstsq`) |
| LLM Classification | Gemini 2.0 Flash API (`google-generativeai`) |
| Prediction Model | `scikit-learn` LogisticRegression |
| Dashboard | `streamlit`, `plotly` |
| Deployment | Streamlit Community Cloud |

---

## Methodology

**Event Study Framework**
- Estimation window: −120 to −20 trading days (pre-event)
- Event window: −5 to +5 trading days
- Market model: `Rᵢ,ₜ = αᵢ + βᵢ·Rₘ,ₜ + εᵢ,ₜ` (OLS per firm per event)
- Abnormal return: `ARᵢ,ₜ = Rᵢ,ₜ − (α̂ᵢ + β̂ᵢ·Rₘ,ₜ)`
- CAR: sum of ARs over event window
- Benchmark: S&P 500 (^GSPC)

**LLM Classification**
- Each USTR Section 301 document classified by Gemini 2.0 Flash
- Output: severity (0–3), surprise (0–1), direction (relief/tightening/mixed/unknown), rationale
- 43 events classified, results cached to CSV

**Prediction Model**
- Target: was next-day AR (day +1) positive?
- Features: firm beta, alpha, same-day AR, prior-day AR, firm identity, event direction, doc type
- 5-fold CV accuracy: ~50–55% (marginally above baseline, consistent with semi-strong market efficiency)

---

## Data

| File | Description |
|------|-------------|
| `event_study_results.csv` | 258 rows — CAR, AR by day, alpha, beta per firm per event |
| `events_llm_classified.csv` | 43 events with Gemini severity/surprise/direction scores |
| `events_llm_scored.csv` | Raw event data with manual labels and cleaned text |

**Firms:** TSMC, NVIDIA, AMD, Apple, Google, Samsung  
**Events:** 43 USTR Section 301 actions (2018–2020)  
**Benchmark:** S&P 500

---

## Run Locally

```bash
git clone https://github.com/bchaush/trade-policy-dashboard
cd trade-policy-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## Academic Context

Built as part of BUS244 — Brandeis International Business School, December 2025.  
Research team: Malcolm Hsu, Bora Chaush, Bhanu Immanni. Supervisor: Professor Shekhar.

The LLM classification pipeline was rebuilt using the Gemini API after the original Ollama 3.2 implementation failed due to an unsupported `response_format` parameter — producing the first working end-to-end version of this pipeline.
