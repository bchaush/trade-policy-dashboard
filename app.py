# =============================================================
# TRADE POLICY EVENT STUDY - STREAMLIT DASHBOARD (Light Theme)
# Run with: streamlit run app.py
# =============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---- PAGE CONFIG --------------------------------------------
st.set_page_config(
    page_title="Trade Policy & Semiconductor Markets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS ---------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #f8f9fa; color: #1a1a2e; }
    .main  { background-color: #f8f9fa; }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    h1, h2, h3 { color: #1a1a2e; }

    .big-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #2563eb;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 32px;
    }
    .finding-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        font-size: 0.9rem;
        color: #374151;
        line-height: 1.7;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .finding-card strong { color: #1a1a2e; font-weight: 600; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #e5e7eb; }
    .stTabs [data-baseweb="tab"] { font-weight: 500; color: #6b7280; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { color: #2563eb !important; }

    .filter-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: #2563eb;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ---- LOAD DATA ----------------------------------------------
@st.cache_data
def load_data():
    results = pd.read_csv("event_study_results.csv")
    results['event_date'] = pd.to_datetime(results['event_date'])
    results['year'] = results['event_date'].dt.year
    return results

df = load_data()

ar_cols = [c for c in df.columns if c.startswith("AR_day_")]
ar_days  = sorted([int(c.replace("AR_day_","").replace("+","")) for c in ar_cols])

FIRM_COLORS = {
    "TSM":       "#2563eb",
    "AAPL":      "#16a34a",
    "NVDA":      "#d97706",
    "AMD":       "#dc2626",
    "005930.KS": "#7c3aed",
    "GOOG":      "#0891b2",
}
DIR_COLORS = {
    "relief":     "#16a34a",
    "tightening": "#dc2626",
    "mixed":      "#d97706",
}

CHART_BG   = "#ffffff"
GRID_COLOR = "#f3f4f6"
FONT_COLOR = "#374151"

def light_layout(**kwargs):
    base = dict(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter", color=FONT_COLOR, size=12),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    base.update(kwargs)
    return base

# ---- SIDEBAR ------------------------------------------------
st.sidebar.markdown('<p class="filter-label">Filters</p>', unsafe_allow_html=True)

all_tickers  = sorted(df['ticker'].unique())
all_dirs     = sorted(df['direction'].unique())
all_doctypes = sorted(df['doc_type'].unique())

sel_tickers  = st.sidebar.multiselect("Firms", all_tickers, default=all_tickers)
sel_dirs     = st.sidebar.multiselect("Event Direction", all_dirs, default=all_dirs)
sel_doctypes = st.sidebar.multiselect("Document Type", all_doctypes, default=all_doctypes)

year_min, year_max = int(df['year'].min()), int(df['year'].max())
sel_years = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

mask = (
    df['ticker'].isin(sel_tickers) &
    df['direction'].isin(sel_dirs) &
    df['doc_type'].isin(sel_doctypes) &
    df['year'].between(sel_years[0], sel_years[1])
)
fdf = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<p style="font-size:0.82rem;color:#6b7280;">'
    f'<b>{len(fdf)}</b> observations &nbsp;·&nbsp; '
    f'<b>{fdf["event_id"].nunique()}</b> events &nbsp;·&nbsp; '
    f'<b>{fdf["ticker"].nunique()}</b> firms</p>',
    unsafe_allow_html=True
)

# ---- HEADER -------------------------------------------------
st.markdown('<p class="big-title">Trade Policy Shocks & Semiconductor Markets</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Event study analysis of USTR Section 301 tariff announcements · 2018–2020 · 43 events · 6 firms</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Event Explorer", "Firm Comparison", "Key Findings", "Prediction Model"])

# ============================================================
# TAB 1 — EVENT EXPLORER
# ============================================================
with tab1:
    st.markdown('<p class="section-header">Abnormal Returns Around Tariff Announcements</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg CAR (Relief)", f"{fdf[fdf['direction']=='relief']['CAR'].mean()*100:.2f}%",
                  help="Average Cumulative Abnormal Return for relief announcements")
    with col2:
        st.metric("Avg CAR (Tightening)", f"{fdf[fdf['direction']=='tightening']['CAR'].mean()*100:.2f}%")
    with col3:
        st.metric("Avg CAR (Mixed)", f"{fdf[fdf['direction']=='mixed']['CAR'].mean()*100:.2f}%")
    with col4:
        st.metric("Total Observations", len(fdf))

    st.markdown('<p class="section-header">Average AR by Day Relative to Event (Event Window)</p>', unsafe_allow_html=True)

    ar_data = []
    for direction in sel_dirs:
        sub = fdf[fdf['direction'] == direction]
        for day in ar_days:
            col = f"AR_day_{day:+d}"
            if col in sub.columns:
                ar_data.append({"day": day, "direction": direction, "mean_ar": sub[col].mean()})

    if ar_data:
        ar_df = pd.DataFrame(ar_data)
        fig_ar = go.Figure()
        for direction in sel_dirs:
            sub = ar_df[ar_df['direction'] == direction]
            fig_ar.add_trace(go.Scatter(
                x=sub['day'], y=sub['mean_ar'],
                mode='lines+markers',
                name=direction.capitalize(),
                line=dict(color=DIR_COLORS.get(direction, "#6b7280"), width=2.5),
                marker=dict(size=7),
            ))
        fig_ar.add_vline(x=0, line_dash="dash", line_color="#9ca3af", opacity=0.7,
                         annotation_text="Event Date", annotation_position="top right",
                         annotation_font_color="#6b7280")
        fig_ar.add_hline(y=0, line_color="#e5e7eb")
        fig_ar.update_layout(**light_layout(
            xaxis=dict(title="Trading Days Relative to Event", gridcolor=GRID_COLOR,
                       zeroline=False, tickvals=list(range(-5, 6))),
            yaxis=dict(title="Average Abnormal Return", gridcolor=GRID_COLOR,
                       zeroline=False, tickformat=".1%"),
            legend=dict(bgcolor="#ffffff", bordercolor="#e5e7eb", borderwidth=1),
            height=380,
        ))
        st.plotly_chart(fig_ar, use_container_width=True)

    st.markdown('<p class="section-header">CAR Distribution by Direction</p>', unsafe_allow_html=True)

    fig_box = go.Figure()
    for direction in sel_dirs:
        sub = fdf[fdf['direction'] == direction]
        fig_box.add_trace(go.Box(
            y=sub['CAR'],
            name=direction.capitalize(),
            marker_color=DIR_COLORS.get(direction, "#6b7280"),
            line_color=DIR_COLORS.get(direction, "#6b7280"),
            fillcolor=DIR_COLORS.get(direction, "#6b7280"),
            opacity=0.25,
        ))
    fig_box.update_layout(**light_layout(
        yaxis=dict(title="CAR ([-5,+5] window)", gridcolor=GRID_COLOR, tickformat=".1%"),
        xaxis=dict(gridcolor=GRID_COLOR),
        height=340,
        showlegend=False,
    ))
    st.plotly_chart(fig_box, use_container_width=True)

# ============================================================
# TAB 2 — FIRM COMPARISON
# ============================================================
with tab2:
    st.markdown('<p class="section-header">Average CAR by Firm and Direction</p>', unsafe_allow_html=True)

    firm_dir = fdf.groupby(['firm_name', 'direction'])['CAR'].mean().reset_index()
    fig_firm = px.bar(
        firm_dir, x='firm_name', y='CAR', color='direction',
        barmode='group',
        color_discrete_map=DIR_COLORS,
        labels={'CAR': 'Average CAR', 'firm_name': 'Firm', 'direction': 'Direction'},
    )
    fig_firm.update_layout(**light_layout(
        yaxis=dict(gridcolor=GRID_COLOR, tickformat=".1%"),
        xaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor="#ffffff", bordercolor="#e5e7eb", borderwidth=1),
        height=380,
    ))
    st.plotly_chart(fig_firm, use_container_width=True)

    st.markdown('<p class="section-header">Average AR Through Event Window by Firm</p>', unsafe_allow_html=True)

    firm_ar_data = []
    for ticker in sel_tickers:
        sub = fdf[fdf['ticker'] == ticker]
        for day in ar_days:
            col = f"AR_day_{day:+d}"
            if col in sub.columns:
                firm_ar_data.append({
                    "day": day, "ticker": ticker,
                    "firm": sub['firm_name'].iloc[0] if len(sub) > 0 else ticker,
                    "mean_ar": sub[col].mean()
                })

    if firm_ar_data:
        firm_ar_df = pd.DataFrame(firm_ar_data)
        fig_firm_ar = go.Figure()
        for ticker in sel_tickers:
            sub = firm_ar_df[firm_ar_df['ticker'] == ticker]
            if len(sub) == 0:
                continue
            fig_firm_ar.add_trace(go.Scatter(
                x=sub['day'], y=sub['mean_ar'],
                mode='lines+markers',
                name=sub['firm'].iloc[0],
                line=dict(color=FIRM_COLORS.get(ticker, "#6b7280"), width=2),
                marker=dict(size=6),
            ))
        fig_firm_ar.add_vline(x=0, line_dash="dash", line_color="#9ca3af", opacity=0.6)
        fig_firm_ar.add_hline(y=0, line_color="#e5e7eb")
        fig_firm_ar.update_layout(**light_layout(
            xaxis=dict(title="Trading Days Relative to Event", gridcolor=GRID_COLOR,
                       tickvals=list(range(-5, 6))),
            yaxis=dict(title="Average Abnormal Return", gridcolor=GRID_COLOR, tickformat=".1%"),
            legend=dict(bgcolor="#ffffff", bordercolor="#e5e7eb", borderwidth=1),
            height=400,
        ))
        st.plotly_chart(fig_firm_ar, use_container_width=True)

    st.markdown('<p class="section-header">Beta Coefficients by Firm (Market Sensitivity)</p>', unsafe_allow_html=True)

    beta_df = fdf.groupby('firm_name')['beta'].mean().reset_index().sort_values('beta', ascending=True)
    fig_beta = go.Figure(go.Bar(
        x=beta_df['beta'], y=beta_df['firm_name'],
        orientation='h',
        marker_color="#2563eb",
        marker_line_color="#e5e7eb",
        marker_line_width=1,
    ))
    fig_beta.add_vline(x=1, line_dash="dash", line_color="#9ca3af", opacity=0.7,
                       annotation_text="Market β=1", annotation_position="top right",
                       annotation_font_color="#6b7280")
    fig_beta.update_layout(**light_layout(
        xaxis=dict(title="Average Beta (market sensitivity)", gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        height=280,
    ))
    st.plotly_chart(fig_beta, use_container_width=True)

# ============================================================
# TAB 3 — KEY FINDINGS
# ============================================================
with tab3:
    st.markdown('<p class="section-header">Summary Statistics</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        relief_car = df[df['direction']=='relief']['CAR'].mean()*100
        st.metric("Relief Events — Avg CAR", f"{relief_car:+.2f}%",
                  delta="Positive — markets reward certainty")
    with col2:
        mixed_car = df[df['direction']=='mixed']['CAR'].mean()*100
        st.metric("Mixed Events — Avg CAR", f"{mixed_car:+.2f}%",
                  delta="Negative — uncertainty penalized")
    with col3:
        tight_car = df[df['direction']=='tightening']['CAR'].mean()*100
        st.metric("Tightening Events — Avg CAR", f"{tight_car:+.2f}%",
                  delta="Near zero — clarity > direction")

    st.markdown('<p class="section-header">Key Findings</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="finding-card">
        <strong>Finding 1 — Relief announcements consistently generate positive abnormal returns.</strong><br>
        Across all model specifications and firm subsets, relief events produce avg CAR of +1.47%.
        This is statistically consistent with the hypothesis that market participants respond positively
        to reductions in trade policy uncertainty.
    </div>
    <div class="finding-card">
        <strong>Finding 2 — Information clarity dominates severity and direction.</strong><br>
        Both relief and tightening announcements resolve uncertainty, while mixed announcements
        introduce it. The avg CAR for mixed events (−4.03%) is the most negative — suggesting markets
        penalize ambiguity more than they fear restrictive policy.
    </div>
    <div class="finding-card">
        <strong>Finding 3 — Heterogeneity across sectors.</strong><br>
        AMD (β=1.96) and NVIDIA (β=1.91) exhibit the highest market sensitivity, consistent with
        their heavier exposure to semiconductor supply chain disruptions. Samsung (β=0.22) shows
        lower sensitivity, reflecting its more diversified revenue base.
    </div>
    <div class="finding-card">
        <strong>Finding 4 — LLM classification pipeline rebuilt and validated.</strong><br>
        The original Ollama-based classification failed due to an unsupported API parameter
        (response_format). This project rebuilds the pipeline using the Gemini API with the
        identical prompt methodology, producing working severity, surprise and direction scores
        for all 43 events.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Average CAR by Direction — All Events</p>', unsafe_allow_html=True)

    summary = df.groupby('direction')['CAR'].agg(['mean','std','count']).reset_index()
    summary.columns = ['Direction','Mean CAR','Std Dev','N Events']
    summary['Mean CAR'] = (summary['Mean CAR']*100).round(3).astype(str) + '%'
    summary['Std Dev']  = (summary['Std Dev']*100).round(3).astype(str) + '%'
    summary['Direction'] = summary['Direction'].str.capitalize()
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Methodology</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="finding-card">
        <strong>Event Study Framework</strong><br>
        Estimation window: −120 to −20 trading days · Event window: −5 to +5 trading days<br>
        Market model: Rᵢ,ₜ = αᵢ + βᵢ·Rₘ,ₜ + εᵢ,ₜ (OLS per firm per event)<br>
        Abnormal return: ARᵢ,ₜ = Rᵢ,ₜ − (α̂ᵢ + β̂ᵢ·Rₘ,ₜ) · CAR = Σ ARᵢ,ₜ over event window<br>
        Benchmark: S&P 500 (^GSPC) · Data: Yahoo Finance via yfinance
    </div>
    """, unsafe_allow_html=True)

# ---- FOOTER -------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="font-family:IBM Plex Mono;font-size:0.7rem;color:#9ca3af;text-align:center;">'
    'Trade Policy Event Study · BUS244 · Brandeis International Business School · 2025'
    '</p>',
    unsafe_allow_html=True
)

# ============================================================
# TAB 4 — PREDICTION MODEL
# ============================================================
with tab4:
    st.markdown('<p class="section-header">Logistic Regression — Predicting Next-Day Abnormal Return Direction</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-card">
        <strong>Model Overview</strong><br>
        Target variable: Was the next-day abnormal return (AR day +1) positive?
        Features: firm beta, alpha, same-day AR, prior-day AR, firm identity, event direction, document type.
        Method: Logistic Regression with 5-fold cross-validation.
    </div>
    """, unsafe_allow_html=True)

    # Build model
    @st.cache_data
    def build_model(dataframe):
        d = dataframe.copy()
        d['target'] = (d['AR_day_+1'] > 0).astype(int)
        le_ticker    = LabelEncoder()
        le_direction = LabelEncoder()
        le_doctype   = LabelEncoder()
        d['ticker_enc']    = le_ticker.fit_transform(d['ticker'])
        d['direction_enc'] = le_direction.fit_transform(d['direction'])
        d['doc_type_enc']  = le_doctype.fit_transform(d['doc_type'])
        features = ['beta', 'alpha', 'AR_day_-1', 'AR_day_+0', 'ticker_enc', 'direction_enc', 'doc_type_enc']
        X = d[features]
        y = d['target']
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        model.fit(X, y)
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        return model, cv_scores, y, y_pred, cm, features, d

    model, cv_scores, y, y_pred, cm, features, df_model = build_model(df)
    train_acc = accuracy_score(y, y_pred)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CV Accuracy (5-fold)", f"{cv_scores.mean():.1%}", help="Average accuracy across 5 cross-validation folds")
    with col2:
        st.metric("Train Accuracy", f"{train_acc:.1%}")
    with col3:
        st.metric("Observations", len(y))
    with col4:
        st.metric("Positive Rate", f"{y.mean():.1%}", help="Share of next-day ARs that were positive")

    st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)

    col_cm, col_coef = st.columns(2)

    with col_cm:
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale=[[0, '#eff6ff'], [1, '#2563eb']],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "#1a1a2e"},
            showscale=False,
        ))
        fig_cm.update_layout(**light_layout(
            height=300,
            xaxis=dict(side='bottom'),
            margin=dict(l=20, r=20, t=20, b=20),
        ))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_coef:
        st.markdown('<p class="section-header">Feature Coefficients</p>', unsafe_allow_html=True)
        feature_labels = ['Beta', 'Alpha', 'AR day -1', 'AR day 0', 'Firm', 'Direction', 'Doc Type']
        coefs = model.coef_[0]
        coef_df = pd.DataFrame({'Feature': feature_labels, 'Coefficient': coefs})
        coef_df = coef_df.sort_values('Coefficient')
        colors = ['#dc2626' if c < 0 else '#16a34a' for c in coef_df['Coefficient']]

        fig_coef = go.Figure(go.Bar(
            x=coef_df['Coefficient'],
            y=coef_df['Feature'],
            orientation='h',
            marker_color=colors,
            marker_line_width=0,
        ))
        fig_coef.add_vline(x=0, line_color="#9ca3af", line_width=1)
        fig_coef.update_layout(**light_layout(
            height=300,
            xaxis=dict(title="Coefficient", gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(gridcolor=GRID_COLOR),
            margin=dict(l=20, r=20, t=20, b=20),
        ))
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown('<p class="section-header">CV Accuracy Across Folds</p>', unsafe_allow_html=True)

    fig_cv = go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(cv_scores))],
        y=cv_scores,
        marker_color=['#2563eb' if s >= cv_scores.mean() else '#93c5fd' for s in cv_scores],
        text=[f"{s:.1%}" for s in cv_scores],
        textposition='outside',
    ))
    fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="#d97706",
                     annotation_text=f"Mean: {cv_scores.mean():.1%}",
                     annotation_position="top right",
                     annotation_font_color="#d97706")
    fig_cv.add_hline(y=0.5, line_dash="dot", line_color="#9ca3af",
                     annotation_text="Baseline (50%)",
                     annotation_position="bottom right",
                     annotation_font_color="#9ca3af")
    fig_cv.update_layout(**light_layout(
        yaxis=dict(title="Accuracy", gridcolor=GRID_COLOR, tickformat=".0%", range=[0, 0.75]),
        xaxis=dict(gridcolor=GRID_COLOR),
        height=320,
        showlegend=False,
    ))
    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("""
    <div class="finding-card">
        <strong>Interpretation</strong><br>
        The model achieves ~54% cross-validated accuracy on a near-balanced dataset (53% positive rate),
        marginally above the 50% random baseline. The strongest predictors are event direction and
        same-day AR (day 0). This is consistent with the paper's finding that policy direction
        matters more than severity — and suggests limited short-run predictability from firm-level
        features alone, which aligns with semi-strong market efficiency.
    </div>
    """, unsafe_allow_html=True)
