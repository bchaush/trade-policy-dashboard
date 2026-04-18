# =============================================================
# TRADE POLICY EVENT STUDY - STREAMLIT DASHBOARD
# Run with: streamlit run app.py
# =============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }

    /* Headers */
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #e6edf3; }

    .big-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 500;
        color: #58a6ff;
        margin-bottom: 0;
    }
    .subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #58a6ff;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 32px;
    }
    .finding-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        color: #c9d1d9;
        line-height: 1.6;
    }
    .tag-relief  { background:#1a3c2e; color:#3fb950; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-family:'IBM Plex Mono',monospace; }
    .tag-tight   { background:#3c1a1a; color:#f85149; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-family:'IBM Plex Mono',monospace; }
    .tag-mixed   { background:#3c2e1a; color:#d29922; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-family:'IBM Plex Mono',monospace; }
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

# AR columns
ar_cols = [c for c in df.columns if c.startswith("AR_day_")]
ar_days  = sorted([int(c.replace("AR_day_","").replace("+","")) for c in ar_cols])

FIRM_COLORS = {
    "TSM":       "#58a6ff",
    "AAPL":      "#3fb950",
    "NVDA":      "#d29922",
    "AMD":       "#f85149",
    "005930.KS": "#bc8cff",
    "GOOG":      "#79c0ff",
}
DIR_COLORS = {
    "relief":     "#3fb950",
    "tightening": "#f85149",
    "mixed":      "#d29922",
}

# ---- SIDEBAR ------------------------------------------------
st.sidebar.markdown('<p style="font-family:IBM Plex Mono;font-size:0.7rem;color:#58a6ff;letter-spacing:0.1em;text-transform:uppercase;">Filters</p>', unsafe_allow_html=True)

all_tickers  = sorted(df['ticker'].unique())
all_dirs     = sorted(df['direction'].unique())
all_doctypes = sorted(df['doc_type'].unique())

sel_tickers  = st.sidebar.multiselect("Firms", all_tickers, default=all_tickers)
sel_dirs     = st.sidebar.multiselect("Event Direction", all_dirs, default=all_dirs)
sel_doctypes = st.sidebar.multiselect("Document Type", all_doctypes, default=all_doctypes)

year_min, year_max = int(df['year'].min()), int(df['year'].max())
sel_years = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

# Apply filters
mask = (
    df['ticker'].isin(sel_tickers) &
    df['direction'].isin(sel_dirs) &
    df['doc_type'].isin(sel_doctypes) &
    df['year'].between(sel_years[0], sel_years[1])
)
fdf = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f'<p style="font-family:IBM Plex Mono;font-size:0.72rem;color:#8b949e;">{len(fdf)} observations<br>{fdf["event_id"].nunique()} events<br>{fdf["ticker"].nunique()} firms</p>', unsafe_allow_html=True)

# ---- HEADER -------------------------------------------------
st.markdown('<p class="big-title">Trade Policy Shocks & Semiconductor Markets</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Event study analysis of USTR Section 301 tariff announcements · 2018–2020 · 43 events · 6 firms</p>', unsafe_allow_html=True)

# ---- NAVIGATION TABS ----------------------------------------
tab1, tab2, tab3 = st.tabs(["📈  Event Explorer", "🏢  Firm Comparison", "📋  Key Findings"])

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

    # Build AR time series by direction
    ar_data = []
    for direction in sel_dirs:
        sub = fdf[fdf['direction'] == direction]
        for day in ar_days:
            col = f"AR_day_{day:+d}"
            if col in sub.columns:
                mean_ar = sub[col].mean()
                ar_data.append({"day": day, "direction": direction, "mean_ar": mean_ar})

    if ar_data:
        ar_df = pd.DataFrame(ar_data)
        fig_ar = go.Figure()

        for direction in sel_dirs:
            sub = ar_df[ar_df['direction'] == direction]
            fig_ar.add_trace(go.Scatter(
                x=sub['day'], y=sub['mean_ar'],
                mode='lines+markers',
                name=direction.capitalize(),
                line=dict(color=DIR_COLORS.get(direction, "#8b949e"), width=2),
                marker=dict(size=6),
            ))

        fig_ar.add_vline(x=0, line_dash="dash", line_color="#8b949e", opacity=0.5,
                         annotation_text="Event Date", annotation_position="top right")
        fig_ar.add_hline(y=0, line_color="#30363d", opacity=0.5)

        fig_ar.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
            xaxis=dict(title="Trading Days Relative to Event", gridcolor="#21262d", zeroline=False,
                       tickvals=list(range(-5, 6))),
            yaxis=dict(title="Average Abnormal Return", gridcolor="#21262d", zeroline=False,
                       tickformat=".1%"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_ar, use_container_width=True)

    st.markdown('<p class="section-header">CAR Distribution by Direction</p>', unsafe_allow_html=True)

    fig_box = go.Figure()
    for direction in sel_dirs:
        sub = fdf[fdf['direction'] == direction]
        fig_box.add_trace(go.Box(
            y=sub['CAR'],
            name=direction.capitalize(),
            marker_color=DIR_COLORS.get(direction, "#8b949e"),
            line_color=DIR_COLORS.get(direction, "#8b949e"),
            fillcolor=DIR_COLORS.get(direction, "#8b949e"),
            opacity=0.3,
        ))

    fig_box.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        yaxis=dict(title="CAR ([-5,+5] window)", gridcolor="#21262d", tickformat=".1%"),
        xaxis=dict(gridcolor="#21262d"),
        height=340,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
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
    fig_firm.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        yaxis=dict(gridcolor="#21262d", tickformat=".1%"),
        xaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_firm, use_container_width=True)

    st.markdown('<p class="section-header">Average AR Through Event Window by Firm</p>', unsafe_allow_html=True)

    firm_ar_data = []
    for ticker in sel_tickers:
        sub = fdf[fdf['ticker'] == ticker]
        for day in ar_days:
            col = f"AR_day_{day:+d}"
            if col in sub.columns:
                firm_ar_data.append({
                    "day": day,
                    "ticker": ticker,
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
            firm_name = sub['firm'].iloc[0]
            fig_firm_ar.add_trace(go.Scatter(
                x=sub['day'], y=sub['mean_ar'],
                mode='lines+markers',
                name=firm_name,
                line=dict(color=FIRM_COLORS.get(ticker, "#8b949e"), width=2),
                marker=dict(size=5),
            ))

        fig_firm_ar.add_vline(x=0, line_dash="dash", line_color="#8b949e", opacity=0.4)
        fig_firm_ar.add_hline(y=0, line_color="#30363d", opacity=0.5)

        fig_firm_ar.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
            xaxis=dict(title="Trading Days Relative to Event", gridcolor="#21262d",
                       tickvals=list(range(-5, 6))),
            yaxis=dict(title="Average Abnormal Return", gridcolor="#21262d", tickformat=".1%"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_firm_ar, use_container_width=True)

    st.markdown('<p class="section-header">Beta Coefficients by Firm (Market Sensitivity)</p>', unsafe_allow_html=True)

    beta_df = fdf.groupby('firm_name')['beta'].mean().reset_index().sort_values('beta', ascending=True)
    fig_beta = go.Figure(go.Bar(
        x=beta_df['beta'], y=beta_df['firm_name'],
        orientation='h',
        marker_color="#58a6ff",
        marker_line_color="#30363d",
        marker_line_width=1,
    ))
    fig_beta.add_vline(x=1, line_dash="dash", line_color="#8b949e", opacity=0.5,
                       annotation_text="Market β=1", annotation_position="top right")
    fig_beta.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        xaxis=dict(title="Average Beta (market sensitivity)", gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
    )
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
        introduce it. The avg CAR for mixed events (-4.03%) is the most negative — suggesting markets 
        penalize ambiguity more than they fear restrictive policy.
    </div>
    <div class="finding-card">
        <strong>Finding 3 — Heterogeneity across sectors.</strong><br>
        NVIDIA (β=1.81) and AMD (β=1.45) exhibit the highest market sensitivity, consistent with 
        their heavier exposure to semiconductor supply chain disruptions. Samsung (β=0.28) shows 
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

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

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
    '<p style="font-family:IBM Plex Mono;font-size:0.7rem;color:#8b949e;text-align:center;">'
    'Trade Policy Event Study · BUS244 · Brandeis International Business School · 2025'
    '</p>',
    unsafe_allow_html=True
)
