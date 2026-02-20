import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from anthropic import Anthropic

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataWise — AI EDA Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --border: #2a2a38;
    --accent: #7c6af7;
    --accent2: #f7716a;
    --accent3: #4fd1c5;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --success: #48bb78;
    --warning: #f6ad55;
    --danger: #fc8181;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Typography */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(124,106,247,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #a78bfa 50%, #4fd1c5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
    line-height: 1.1;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Insight cards */
.insight-card {
    background: var(--surface2);
    border-left: 4px solid var(--accent);
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text);
}
.insight-card.warning { border-left-color: var(--warning); }
.insight-card.danger  { border-left-color: var(--danger); }
.insight-card.success { border-left-color: var(--success); }
.insight-card.info    { border-left-color: var(--accent3); }

.insight-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

/* Quality score ring */
.quality-ring {
    text-align: center;
    padding: 1.5rem;
    background: var(--surface2);
    border-radius: 16px;
    border: 1px solid var(--border);
}
.quality-score {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
}
.quality-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* Tags */
.tag {
    display: inline-block;
    background: rgba(124,106,247,0.15);
    border: 1px solid rgba(124,106,247,0.3);
    color: #a78bfa;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin: 0.2rem;
}
.tag.warning {
    background: rgba(246,173,85,0.15);
    border-color: rgba(246,173,85,0.3);
    color: #f6ad55;
}
.tag.danger {
    background: rgba(252,129,129,0.15);
    border-color: rgba(252,129,129,0.3);
    color: #fc8181;
}
.tag.success {
    background: rgba(72,187,120,0.15);
    border-color: rgba(72,187,120,0.3);
    color: #68d391;
}

/* Column cards */
.col-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.col-card:hover { border-color: var(--accent); }
.col-name {
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    color: var(--accent3);
    font-size: 0.9rem;
}
.col-type {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
}

/* Chat */
.chat-msg-user {
    background: rgba(124,106,247,0.12);
    border: 1px solid rgba(124,106,247,0.25);
    border-radius: 12px 12px 4px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    text-align: right;
}
.chat-msg-ai {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px 12px 12px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* Streamlit overrides */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
[data-testid="stExpander"] {
    background: var(--surface2);
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stDataFrame { border: 1px solid var(--border); border-radius: 8px; }
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: white !important; }

div[data-testid="stFileUploader"] {
    background: var(--surface2);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1rem;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(uploaded_file, encoding="latin1")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    elif name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    return None

def classify_columns(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    # Try to parse potential datetime strings
    for col in categorical:
        try:
            parsed = pd.to_datetime(df[col], format='mixed')
            if parsed.notna().sum() > 0.8 * len(df):
                datetime_cols.append(col)
                categorical.remove(col)
        except Exception:
            pass
    boolean = df.select_dtypes(include=["bool"]).columns.tolist()
    return numeric, categorical, datetime_cols, boolean

def compute_quality_score(df):
    score = 100
    missing_pct = df.isnull().mean().mean() * 100
    dup_pct = df.duplicated().sum() / len(df) * 100
    score -= min(missing_pct * 1.5, 40)
    score -= min(dup_pct * 2, 20)
    # High cardinality categoricals
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() / len(df) > 0.9:
            score -= 5
    return max(int(score), 0)

def quality_color(score):
    if score >= 80: return "#48bb78"
    if score >= 55: return "#f6ad55"
    return "#fc8181"

def quality_label(score):
    if score >= 80: return "Excellent"
    if score >= 55: return "Fair"
    return "Needs Work"

def plotly_theme():
    """Returns parameters safe for chart creation (px.histogram, px.scatter, etc.)"""
    return dict(
        template="plotly_dark",
    )

def plotly_layout_theme():
    """Returns parameters for fig.update_layout()"""
    return dict(
        paper_bgcolor="#111118",
        plot_bgcolor="#111118",
        font=dict(family="DM Sans", color="#e8e8f0"),
        margin=dict(l=40, r=20, t=40, b=40),
    )

def get_df_summary(df):
    numeric, categorical, datetime_cols, boolean = classify_columns(df)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": numeric,
        "categorical_cols": categorical,
        "datetime_cols": datetime_cols,
        "missing_values": missing[missing > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "quality_score": compute_quality_score(df),
        "sample_values": {col: df[col].dropna().head(3).tolist() for col in df.columns[:8]},
    }
    for col in numeric[:5]:
        summary[f"stats_{col}"] = df[col].describe().round(3).to_dict()
    return summary

def ask_claude(client, messages, system_prompt):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
    background:linear-gradient(135deg,#7c6af7,#4fd1c5);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;margin-bottom:0.2rem'>
    🔬 DataWise
    </div>
    <div style='color:#6b6b80;font-size:0.78rem;margin-bottom:1.5rem;font-family:DM Mono,monospace'>
    AI EDA Studio v1.0
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Claude API Key**")
    api_key_input = st.text_input(
        "API Key", type="password", label_visibility="collapsed",
        placeholder="sk-ant-...",
        value=st.session_state.api_key,
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
    st.caption("Get yours at [console.anthropic.com](https://console.anthropic.com)")

    st.divider()

    st.markdown("**Upload Dataset**")
    uploaded = st.file_uploader(
        "file", label_visibility="collapsed",
        type=["csv", "xlsx", "xls", "parquet", "json", "tsv"],
    )
    if uploaded:
        df = load_data(uploaded)
        if df is not None:
            st.session_state.df = df
            st.session_state.filename = uploaded.name
            st.session_state.chat_history = []
            st.success(f"✓ {uploaded.name}")

    st.markdown("**Or try a sample**")
    sample = st.selectbox(
        "sample", label_visibility="collapsed",
        options=["— pick one —", "Titanic", "Iris", "Tips", "Gapminder"],
    )
    if sample != "— pick one —":
        import plotly.express as px_data
        sample_map = {
            "Titanic": px_data.data.tips,  # placeholder, override below
            "Iris": px_data.data.iris,
            "Tips": px_data.data.tips,
            "Gapminder": px_data.data.gapminder().query("year == 2007"),
        }
        if sample == "Titanic":
            try:
                st.session_state.df = pd.read_csv(
                    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                )
            except Exception:
                st.session_state.df = px_data.data.tips()
        else:
            st.session_state.df = sample_map[sample]
        st.session_state.filename = f"{sample} (sample)"
        st.session_state.chat_history = []
        st.success(f"✓ {sample} loaded")

    st.divider()

    st.markdown("**Context** *(helps Claude)*")
    goal = st.selectbox("Your goal", [
        "Explore freely",
        "Predict a target variable",
        "Understand patterns",
        "Find anomalies",
        "Prepare for a competition",
    ])
    domain = st.selectbox("Domain", [
        "General", "Healthcare", "Finance", "Environment",
        "HR / People", "Retail / E-commerce", "Education", "Other",
    ])
    level = st.selectbox("Experience level", [
        "Total beginner", "Some Python", "Practiced data scientist",
    ])

    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem;color:#6b6b80;font-family:DM Mono,monospace;line-height:1.6'>
    Open source · MIT License<br>
    Built for WiDS 2026 🌸<br>
    <a href='https://github.com' style='color:#7c6af7'>★ Star on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='hero-title'>DataWise</div>
  <p class='hero-sub'>Upload any dataset → get deep EDA + feature engineering insights, narrated by AI in plain English</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.df is None:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;color:#6b6b80'>
      <div style='font-size:4rem;margin-bottom:1rem'>📂</div>
      <div style='font-family:Syne,sans-serif;font-size:1.3rem;color:#e8e8f0;margin-bottom:0.5rem'>
        Upload a dataset to begin
      </div>
      <div style='font-size:0.9rem'>CSV, Excel, Parquet, JSON, TSV — any tabular data works</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df
numeric, categorical, datetime_cols, boolean = classify_columns(df)
quality_score = compute_quality_score(df)
client = Anthropic(api_key=st.session_state.api_key) if st.session_state.api_key else None

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "🔍 EDA — Simple", "📊 EDA — Advanced", "⚙️ Feature Engineering", "💬 Ask Claude"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab1:
    col_score, col_meta = st.columns([1, 3])

    with col_score:
        color = quality_color(quality_score)
        label = quality_label(quality_score)
        st.markdown(f"""
        <div class='quality-ring'>
          <div class='quality-score' style='color:{color}'>{quality_score}</div>
          <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:{color};margin-top:0.3rem'>{label}</div>
          <div class='quality-label'>Data Quality Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col_meta:
        st.markdown(f"""
        <div class='metric-grid'>
          <div class='metric-card'>
            <div class='metric-value'>{df.shape[0]:,}</div>
            <div class='metric-label'>Rows</div>
          </div>
          <div class='metric-card'>
            <div class='metric-value'>{df.shape[1]}</div>
            <div class='metric-label'>Columns</div>
          </div>
          <div class='metric-card'>
            <div class='metric-value'>{len(numeric)}</div>
            <div class='metric-label'>Numeric</div>
          </div>
          <div class='metric-card'>
            <div class='metric-value'>{len(categorical)}</div>
            <div class='metric-label'>Categorical</div>
          </div>
          <div class='metric-card'>
            <div class='metric-value'>{df.isnull().sum().sum():,}</div>
            <div class='metric-label'>Missing Cells</div>
          </div>
          <div class='metric-card'>
            <div class='metric-value'>{df.duplicated().sum():,}</div>
            <div class='metric-label'>Duplicates</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Column inventory
    st.markdown("<div class='section-header'>📋 Column Inventory</div>", unsafe_allow_html=True)
    inv_cols = st.columns(3)
    all_cols_sorted = (
        [(c, "numeric") for c in numeric] +
        [(c, "categorical") for c in categorical] +
        [(c, "datetime") for c in datetime_cols] +
        [(c, "boolean") for c in boolean]
    )
    type_colors = {
        "numeric": "#7c6af7", "categorical": "#4fd1c5",
        "datetime": "#f7716a", "boolean": "#f6ad55"
    }
    for i, (col_name, col_type) in enumerate(all_cols_sorted):
        missing_n = df[col_name].isnull().sum()
        missing_p = missing_n / len(df) * 100
        miss_tag = ""
        if missing_p > 30:
            miss_tag = f"<span class='tag danger'>{missing_p:.0f}% missing</span>"
        elif missing_p > 5:
            miss_tag = f"<span class='tag warning'>{missing_p:.0f}% missing</span>"
        inv_cols[i % 3].markdown(f"""
        <div class='col-card'>
          <div class='col-name'>{col_name}</div>
          <div class='col-type' style='color:{type_colors[col_type]}'>{col_type}</div>
          {miss_tag}
        </div>
        """, unsafe_allow_html=True)

    # Sample data
    st.markdown("<div class='section-header'>👁️ Sample Rows</div>", unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    # AI First Impression
    st.markdown("<div class='section-header'>🤖 Claude's First Impression</div>", unsafe_allow_html=True)
    if client:
        if st.button("✨ Generate First Impression", key="first_imp"):
            summary = get_df_summary(df)
            with st.spinner("Claude is reading your dataset..."):
                prompt = f"""
You are a friendly, expert data scientist mentoring a {level} with a goal to "{goal}" in the {domain} domain.

Dataset summary:
- Shape: {summary['shape'][0]} rows × {summary['shape'][1]} columns
- Numeric columns: {summary['numeric_cols']}
- Categorical columns: {summary['categorical_cols']}
- Datetime columns: {summary['datetime_cols']}
- Missing values: {summary['missing_values']}
- Duplicates: {summary['duplicates']}
- Quality score: {summary['quality_score']}/100
- Sample values: {json.dumps(summary['sample_values'], default=str)}

Write a warm, encouraging "first impression" of this dataset in 3 short paragraphs:
1. What this dataset appears to be about and what story it might tell
2. The most interesting or important things you notice at first glance
3. What you'd recommend exploring first, given their goal

Use plain English. Avoid jargon. Be specific to THIS dataset, not generic. Be encouraging.
"""
                response = ask_claude(client, [{"role": "user", "content": prompt}],
                    "You are DataWise, an AI data science mentor. Be warm, specific, and educational.")
                st.session_state.first_impression = response
        if "first_impression" in st.session_state:
            st.markdown(f"<div class='insight-card'>{st.session_state.first_impression}</div>",
                unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='insight-card warning'>
        Add your Claude API key in the sidebar to unlock AI-powered explanations and insights.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — SIMPLE EDA
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📈 Distributions — Numeric Columns</div>", unsafe_allow_html=True)

    if numeric:
        sel_num = st.selectbox("Select numeric column", numeric, key="sel_num")
        col_data = df[sel_num].dropna()

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=sel_num, nbins=40, title=f"Distribution of {sel_num}",
                color_discrete_sequence=["#7c6af7"], **plotly_theme())
            fig.update_layout(showlegend=False, **plotly_layout_theme())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(df, y=sel_num, title=f"Box Plot — {sel_num}",
                color_discrete_sequence=["#4fd1c5"], **plotly_theme())
            fig2.update_layout(**plotly_layout_theme())
            st.plotly_chart(fig2, use_container_width=True)

        # Stats
        stats = col_data.describe()
        skew = col_data.skew()
        kurt = col_data.kurt()
        st.markdown(f"""
        <div class='metric-grid'>
          <div class='metric-card'><div class='metric-value'>{stats['mean']:.3g}</div><div class='metric-label'>Mean</div></div>
          <div class='metric-card'><div class='metric-value'>{stats['50%']:.3g}</div><div class='metric-label'>Median</div></div>
          <div class='metric-card'><div class='metric-value'>{stats['std']:.3g}</div><div class='metric-label'>Std Dev</div></div>
          <div class='metric-card'><div class='metric-value'>{skew:.2f}</div><div class='metric-label'>Skewness</div></div>
          <div class='metric-card'><div class='metric-value'>{stats['min']:.3g}</div><div class='metric-label'>Min</div></div>
          <div class='metric-card'><div class='metric-value'>{stats['max']:.3g}</div><div class='metric-label'>Max</div></div>
        </div>
        """, unsafe_allow_html=True)

        if client and st.button(f"🤖 Explain this column", key="explain_num"):
            with st.spinner("Analyzing..."):
                prompt = f"""
The user is exploring the column "{sel_num}" in their dataset.
Stats: mean={stats['mean']:.3g}, median={stats['50%']:.3g}, std={stats['std']:.3g}, 
skewness={skew:.2f}, min={stats['min']:.3g}, max={stats['max']:.3g}, 
missing={df[sel_num].isnull().sum()} out of {len(df)} rows.

Explain in plain English for a {level}:
1. WHAT: What does this distribution tell us?
2. WHY IT MATTERS: Why should they care about skewness, outliers, or the spread?
3. ACTION: What should they consider doing with this column before modeling?

Be specific, warm, and use an analogy if helpful. 3-4 sentences max per point.
"""
                response = ask_claude(client, [{"role": "user", "content": prompt}],
                    "You are DataWise, an AI data science mentor. Be concise, warm, and educational.")
                st.markdown(f"<div class='insight-card'>{response}</div>", unsafe_allow_html=True)
    else:
        st.info("No numeric columns found.")

    st.divider()
    st.markdown("<div class='section-header'>🏷️ Distributions — Categorical Columns</div>", unsafe_allow_html=True)

    if categorical:
        sel_cat = st.selectbox("Select categorical column", categorical, key="sel_cat")
        val_counts = df[sel_cat].value_counts().head(20)

        fig3 = px.bar(x=val_counts.index.astype(str), y=val_counts.values,
            title=f"Value Counts — {sel_cat}",
            labels={"x": sel_cat, "y": "Count"},
            color_discrete_sequence=["#f7716a"], **plotly_theme())
        fig3.update_layout(**plotly_layout_theme())
        st.plotly_chart(fig3, use_container_width=True)

        n_unique = df[sel_cat].nunique()
        cardinality_pct = n_unique / len(df) * 100
        st.markdown(f"""
        <div class='metric-grid'>
          <div class='metric-card'><div class='metric-value'>{n_unique}</div><div class='metric-label'>Unique Values</div></div>
          <div class='metric-card'><div class='metric-value'>{cardinality_pct:.1f}%</div><div class='metric-label'>Cardinality</div></div>
          <div class='metric-card'><div class='metric-value'>{df[sel_cat].isnull().sum()}</div><div class='metric-label'>Missing</div></div>
          <div class='metric-card'><div class='metric-value'>{val_counts.iloc[0] if len(val_counts) > 0 else 0}</div><div class='metric-label'>Top Freq</div></div>
        </div>
        """, unsafe_allow_html=True)

        if client and st.button(f"🤖 Explain this column", key="explain_cat"):
            with st.spinner("Analyzing..."):
                prompt = f"""
The user is exploring the categorical column "{sel_cat}".
It has {n_unique} unique values out of {len(df)} rows ({cardinality_pct:.1f}% cardinality).
Top 5 values: {val_counts.head(5).to_dict()}
Missing: {df[sel_cat].isnull().sum()} rows.

Explain for a {level}:
1. WHAT: What does this column's distribution tell us?
2. WATCH OUT: Are there any red flags (high cardinality, rare categories, imbalance)?
3. ACTION: How should they handle this column for analysis or modeling?

Be specific to this data. 3-4 sentences per point. Plain English.
"""
                response = ask_claude(client, [{"role": "user", "content": prompt}],
                    "You are DataWise, an AI data science mentor.")
                st.markdown(f"<div class='insight-card'>{response}</div>", unsafe_allow_html=True)
    else:
        st.info("No categorical columns found.")

    st.divider()
    st.markdown("<div class='section-header'>❓ Missing Values Map</div>", unsafe_allow_html=True)

    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if len(missing_data) > 0:
        fig_miss = px.bar(
            x=missing_data.index, y=(missing_data / len(df) * 100),
            title="Missing Values by Column (%)",
            labels={"x": "Column", "y": "Missing (%)"},
            color=(missing_data / len(df) * 100),
            color_continuous_scale=["#48bb78", "#f6ad55", "#fc8181"],
            **plotly_theme()
        )
        fig_miss.update_layout(coloraxis_showscale=False, **plotly_layout_theme())
        st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.markdown("<div class='insight-card success'>🎉 No missing values found in this dataset!</div>",
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — ADVANCED EDA
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🔗 Correlation Matrix</div>", unsafe_allow_html=True)

    if len(numeric) >= 2:
        corr_df = df[numeric].corr()
        fig_corr = px.imshow(
            corr_df, text_auto=".2f", aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale=["#f7716a", "#111118", "#7c6af7"],
            color_continuous_midpoint=0,
            **plotly_theme()
        )
        fig_corr.update_layout(height=max(400, len(numeric) * 50), **plotly_layout_theme())
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlations
        corr_pairs = []
        for i in range(len(corr_df.columns)):
            for j in range(i+1, len(corr_df.columns)):
                corr_pairs.append({
                    "col1": corr_df.columns[i],
                    "col2": corr_df.columns[j],
                    "correlation": corr_df.iloc[i, j]
                })
        corr_pairs_df = pd.DataFrame(corr_pairs).sort_values("correlation", key=abs, ascending=False)

        st.markdown("**Top Correlated Pairs**")
        for _, row in corr_pairs_df.head(5).iterrows():
            c = row["correlation"]
            tag_type = "danger" if abs(c) > 0.8 else "warning" if abs(c) > 0.5 else ""
            direction = "positively" if c > 0 else "negatively"
            st.markdown(f"""
            <div class='col-card'>
              <span class='col-name'>{row['col1']}</span>
              <span style='color:#6b6b80'> ↔ </span>
              <span class='col-name'>{row['col2']}</span>
              <span class='tag {tag_type}' style='float:right'>{c:.3f}</span>
              <div style='font-size:0.8rem;color:#6b6b80;margin-top:0.3rem'>
                {direction.capitalize()} correlated
              </div>
            </div>
            """, unsafe_allow_html=True)

        if client and st.button("🤖 Explain correlation matrix", key="explain_corr"):
            with st.spinner("Analyzing..."):
                top_pairs = corr_pairs_df.head(5)[["col1", "col2", "correlation"]].to_dict("records")
                prompt = f"""
The user has a correlation matrix for their dataset.
Top 5 correlated pairs: {top_pairs}
Number of numeric columns: {len(numeric)}
User goal: {goal}

Explain:
1. WHAT is correlation and what the matrix shows (for a {level})
2. The most important patterns you see in these top pairs
3. Which correlations might cause problems (multicollinearity) and what to do
4. Which correlations might be useful for their goal: "{goal}"

Plain English, 3-4 sentences each. Use a simple analogy for multicollinearity.
"""
                response = ask_claude(client, [{"role": "user", "content": prompt}],
                    "You are DataWise, an AI data science mentor.")
                st.markdown(f"<div class='insight-card'>{response}</div>", unsafe_allow_html=True)
    else:
        st.info("Need at least 2 numeric columns for correlation analysis.")

    st.divider()
    st.markdown("<div class='section-header'>🔍 Bivariate Analysis</div>", unsafe_allow_html=True)

    if len(numeric) >= 2:
        biv_c1, biv_c2 = st.columns(2)
        with biv_c1:
            x_col = st.selectbox("X axis", numeric, key="biv_x")
        with biv_c2:
            y_col = st.selectbox("Y axis", [c for c in numeric if c != x_col], key="biv_y")
        color_col = st.selectbox("Color by (optional)", ["None"] + categorical, key="biv_color")

        fig_scatter = px.scatter(
            df, x=x_col, y=y_col,
            color=None if color_col == "None" else color_col,
            title=f"{x_col} vs {y_col}",
            opacity=0.65, trendline="ols",
            **plotly_theme()
        )
        fig_scatter.update_layout(**plotly_layout_theme())
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-header'>📦 Outlier Detection</div>", unsafe_allow_html=True)

    if numeric:
        outlier_results = []
        for col in numeric:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_outliers > 0:
                outlier_results.append({
                    "column": col, "outliers": n_outliers,
                    "pct": n_outliers / len(df) * 100,
                    "lower_bound": round(lower, 3), "upper_bound": round(upper, 3)
                })

        if outlier_results:
            for r in sorted(outlier_results, key=lambda x: -x["pct"]):
                severity = "danger" if r["pct"] > 10 else "warning" if r["pct"] > 3 else ""
                st.markdown(f"""
                <div class='col-card'>
                  <span class='col-name'>{r['column']}</span>
                  <span class='tag {severity}' style='float:right'>{r['pct']:.1f}% outliers</span>
                  <div style='font-size:0.8rem;color:#6b6b80;margin-top:0.3rem'>
                    {r['outliers']} values outside [{r['lower_bound']}, {r['upper_bound']}]
                  </div>
                </div>
                """, unsafe_allow_html=True)

            if client and st.button("🤖 Explain outliers", key="explain_outliers"):
                with st.spinner("Analyzing..."):
                    prompt = f"""
Outlier analysis for a {level} with goal: "{goal}" in {domain} domain.
Outliers found: {outlier_results}

Explain:
1. WHAT are outliers and how they were detected (IQR method - use a simple analogy)
2. WHY they matter for their specific goal
3. OPTIONS: What are the 3 ways to handle outliers (remove, cap, keep) and when to use each?
4. RECOMMENDATION: Given these specific columns and domain, what would you suggest?

Plain English, educational, warm tone.
"""
                    response = ask_claude(client, [{"role": "user", "content": prompt}],
                        "You are DataWise, an AI data science mentor.")
                    st.markdown(f"<div class='insight-card'>{response}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='insight-card success'>✓ No significant outliers detected using the IQR method.</div>",
                unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='section-header'>📊 Target Variable Analysis</div>", unsafe_allow_html=True)
    target = st.selectbox("Select your target variable", ["None"] + df.columns.tolist(), key="target_sel")
    if target != "None":
        if target in numeric:
            fig_target = px.histogram(df, x=target, nbins=40, title=f"Target Distribution: {target}",
                color_discrete_sequence=["#4fd1c5"], **plotly_theme())
            fig_target.update_layout(**plotly_layout_theme())
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            vc = df[target].value_counts()
            fig_target = px.pie(values=vc.values, names=vc.index.astype(str),
                title=f"Target Distribution: {target}",
                color_discrete_sequence=px.colors.qualitative.Set3, **plotly_theme())
            fig_target.update_layout(**plotly_layout_theme())
            st.plotly_chart(fig_target, use_container_width=True)
            # Class imbalance warning
            if len(vc) >= 2:
                ratio = vc.iloc[0] / vc.iloc[-1]
                if ratio > 5:
                    st.markdown(f"""
                    <div class='insight-card warning'>
                    <div class='insight-title'>⚠️ Class Imbalance Detected</div>
                    Your target variable is imbalanced — the most common class is {ratio:.1f}× more frequent than the rarest.
                    This can make models biased toward the majority class. Techniques like SMOTE, class weights, or resampling can help.
                    </div>""", unsafe_allow_html=True)

        if client and st.button("🤖 Analyze target variable", key="explain_target"):
            with st.spinner("Analyzing..."):
                target_stats = df[target].describe().to_dict() if target in numeric else df[target].value_counts().head(5).to_dict()
                prompt = f"""
Target variable analysis: "{target}"
Stats/distribution: {target_stats}
Is numeric: {target in numeric}
User goal: {goal}, domain: {domain}, level: {level}

Explain:
1. WHAT kind of problem this is (regression vs classification) and what that means
2. What the target distribution tells us and whether it looks healthy
3. What to watch out for (imbalance, skewness, leakage risk)
4. How the other columns in the dataset relate to predicting this target

Practical, encouraging, plain English for a {level}.
"""
                response = ask_claude(client, [{"role": "user", "content": prompt}],
                    "You are DataWise, an AI data science mentor.")
                st.markdown(f"<div class='insight-card'>{response}</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>⚙️ Feature Engineering Recommendations</div>",
        unsafe_allow_html=True)

    fe_target = st.selectbox("Target variable (optional, improves recommendations)",
        ["None"] + df.columns.tolist(), key="fe_target")

    if client and st.button("✨ Generate Feature Engineering Plan", key="fe_plan"):
        with st.spinner("Claude is building your feature engineering roadmap..."):
            # Auto-detect issues
            skewed = []
            for col in numeric:
                if abs(df[col].skew()) > 1:
                    skewed.append({"col": col, "skew": round(df[col].skew(), 2)})

            high_card = []
            for col in categorical:
                if df[col].nunique() > 20:
                    high_card.append({"col": col, "n_unique": df[col].nunique()})

            date_cols = datetime_cols

            corr_info = ""
            if len(numeric) >= 2 and fe_target != "None" and fe_target in numeric:
                target_corr = df[numeric].corr()[fe_target].drop(fe_target).sort_values(key=abs, ascending=False).head(5)
                corr_info = str(target_corr.to_dict())

            prompt = f"""
You are advising a {level} on feature engineering for their dataset.
Goal: {goal} | Domain: {domain}
Target variable: {fe_target}

Dataset overview:
- {len(df)} rows × {len(df.columns)} columns
- Numeric columns: {numeric}
- Categorical columns: {categorical}
- Datetime columns: {date_cols}
- Skewed columns (|skew| > 1): {skewed}
- High cardinality categoricals (>20 unique): {high_card}
- Missing values: {df.isnull().sum()[df.isnull().sum() > 0].to_dict()}
- Top correlations with target: {corr_info}

Create a SPECIFIC, ACTIONABLE feature engineering plan in these sections:

## 🧹 1. Data Cleaning Steps
List specific steps with the exact column names

## 🔄 2. Encoding Categorical Variables  
For each categorical column, recommend the best encoding method and why

## 📐 3. Scaling & Transformations
Which numeric columns need scaling/transformation and which method to use

## 🔨 4. New Features to Create
Suggest 3-5 NEW features that could be derived from existing columns. Be specific.

## 🚫 5. Columns to Drop or Flag
Any columns that look like IDs, leakage risks, or are otherwise problematic

## 🗓️ 6. Datetime Features (if applicable)
How to extract useful features from date columns

For each recommendation: explain WHAT to do, WHY it helps, and HOW to do it in plain English.
Make it educational and specific to THIS dataset, not generic advice.
"""
            response = ask_claude(client, [{"role": "user", "content": prompt}],
                "You are DataWise, an AI data science mentor. Be specific, actionable, and educational.")
            st.session_state.fe_plan = response

    if "fe_plan" in st.session_state:
        st.markdown(f"<div class='insight-card info' style='font-size:0.92rem'>{st.session_state.fe_plan}</div>",
            unsafe_allow_html=True)

    st.divider()

    # Auto-detected issues panel
    st.markdown("<div class='section-header'>🔍 Auto-Detected Issues</div>", unsafe_allow_html=True)

    issues_found = False

    # Skewness
    skewed_cols = [(col, df[col].skew()) for col in numeric if abs(df[col].skew()) > 1]
    if skewed_cols:
        issues_found = True
        st.markdown("<div class='insight-title' style='padding:0.5rem 0'>📐 Skewed Distributions</div>",
            unsafe_allow_html=True)
        for col, skew_val in sorted(skewed_cols, key=lambda x: -abs(x[1])):
            direction = "right (positive)" if skew_val > 0 else "left (negative)"
            fix = "log transform or square root" if skew_val > 0 else "square or exponential transform"
            st.markdown(f"""
            <div class='col-card'>
              <span class='col-name'>{col}</span>
              <span class='tag warning' style='float:right'>skew={skew_val:.2f}</span>
              <div style='font-size:0.82rem;color:#6b6b80;margin-top:0.4rem'>
                Skewed {direction} → consider a {fix}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # High cardinality
    high_card_cols = [(col, df[col].nunique()) for col in categorical if df[col].nunique() > 20]
    if high_card_cols:
        issues_found = True
        st.markdown("<div class='insight-title' style='padding:0.5rem 0'>🔢 High Cardinality Categoricals</div>",
            unsafe_allow_html=True)
        for col, n in sorted(high_card_cols, key=lambda x: -x[1]):
            id_risk = n / len(df) > 0.8
            rec = "likely an ID column — consider dropping" if id_risk else "use target encoding or frequency encoding instead of one-hot"
            st.markdown(f"""
            <div class='col-card'>
              <span class='col-name'>{col}</span>
              <span class='tag {"danger" if id_risk else "warning"}' style='float:right'>{n} unique</span>
              <div style='font-size:0.82rem;color:#6b6b80;margin-top:0.4rem'>{rec}</div>
            </div>
            """, unsafe_allow_html=True)

    # Missing patterns
    heavy_missing = [(col, df[col].isnull().mean() * 100) for col in df.columns if df[col].isnull().mean() > 0.1]
    if heavy_missing:
        issues_found = True
        st.markdown("<div class='insight-title' style='padding:0.5rem 0'>❓ Columns with Significant Missing Data</div>",
            unsafe_allow_html=True)
        for col, pct in sorted(heavy_missing, key=lambda x: -x[1]):
            rec = "consider dropping" if pct > 50 else "impute with median/mode or add a binary 'was_missing' flag"
            st.markdown(f"""
            <div class='col-card'>
              <span class='col-name'>{col}</span>
              <span class='tag {"danger" if pct > 50 else "warning"}' style='float:right'>{pct:.1f}% missing</span>
              <div style='font-size:0.82rem;color:#6b6b80;margin-top:0.4rem'>{rec}</div>
            </div>
            """, unsafe_allow_html=True)

    if not issues_found:
        st.markdown("""
        <div class='insight-card success'>
        ✅ No critical issues auto-detected! Your dataset looks clean. Use the AI plan above
        for proactive feature engineering recommendations.
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='section-header'>💡 Interactive Feature Creator</div>", unsafe_allow_html=True)
    st.caption("Describe a feature you want to create and Claude will write the pandas code for it.")

    feat_desc = st.text_input("Describe your feature idea",
        placeholder='e.g. "create age groups from the age column" or "interaction between price and quantity"',
        key="feat_desc")
    if client and feat_desc and st.button("🤖 Generate code", key="gen_feat_code"):
        with st.spinner("Writing feature engineering code..."):
            prompt = f"""
The user wants to create this feature: "{feat_desc}"
Dataset columns: {df.columns.tolist()}
Column types: {df.dtypes.astype(str).to_dict()}
Sample values: { {c: df[c].dropna().head(3).tolist() for c in df.columns[:6]} }

Write clean pandas code to create this new feature. Include:
1. The actual pandas code (assume the dataframe is called `df`)
2. A brief explanation of what the code does
3. Why this feature might be useful

Format:
```python
# your code here
```
Then explain it in plain English.
"""
            response = ask_claude(client, [{"role": "user", "content": prompt}],
                "You are DataWise. Write clean, working pandas code with clear explanations.")
            st.markdown(f"<div class='insight-card info'>{response}</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 5 — ASK CLAUDE
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>💬 Ask Claude Anything About Your Data</div>",
        unsafe_allow_html=True)

    if not client:
        st.markdown("""
        <div class='insight-card warning'>
        Add your Claude API key in the sidebar to enable the AI chat assistant.
        </div>""", unsafe_allow_html=True)
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-msg-user'>🧑 {msg['content']}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-msg-ai'>🤖 {msg['content']}</div>",
                    unsafe_allow_html=True)

        # Suggested questions
        if not st.session_state.chat_history:
            st.markdown("**Suggested questions to get started:**")
            suggestions = [
                "What's the most important thing to fix in my dataset?",
                "Which columns are most likely to be useful for prediction?",
                "Explain what standard deviation means using my data as an example",
                "What kind of machine learning problem is this?",
                "How should I handle the missing values in my dataset?",
            ]
            for q in suggestions:
                if st.button(q, key=f"sugg_{q[:20]}"):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    st.rerun()

        # Input
        user_input = st.chat_input("Ask anything about your dataset...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            summary = get_df_summary(df)
            system = f"""You are DataWise, a friendly AI data science mentor.
The user is a {level} with goal: "{goal}" in the {domain} domain.

Dataset context:
- Shape: {summary['shape']}
- Columns: {summary['columns']}
- Numeric: {summary['numeric_cols']}
- Categorical: {summary['categorical_cols']}
- Missing: {summary['missing_values']}
- Quality score: {summary['quality_score']}/100
- Sample values: {json.dumps(summary['sample_values'], default=str)}

Answer questions about this specific dataset. Be warm, educational, and use the WHAT/WHY/ACTION framework.
Avoid jargon. Use analogies when helpful. Always relate answers back to their data."""

            with st.spinner("Claude is thinking..."):
                response = ask_claude(client, st.session_state.chat_history, system)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

        if st.session_state.chat_history and st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
