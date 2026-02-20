# 🔬 DataWise — AI-Powered EDA Studio

> Upload any dataset. Understand everything. No code required.

DataWise is an open-source, AI-powered exploratory data analysis tool that turns raw datasets into deep understanding — narrated in plain English by Claude AI. Built for learners, datathon participants, and anyone who wants to truly understand their data before modeling.

**Launched at WiDS (Women in Data Science) 2026 Datathon 🌸**

---

## ✨ Features

### 🏠 Overview
- Instant data quality score (0–100)
- Column inventory with type detection
- Missing value summary
- Claude's "first impression" — what your data is about, in plain English

### 🔍 Simple EDA
- Distribution plots (histogram + box plot) for every numeric column
- Value counts for categorical columns
- Missing values heatmap
- Claude explains every chart: *what you're seeing → why it matters → what to do*

### 📊 Advanced EDA
- Correlation matrix with top pairs highlighted
- Bivariate scatter plots with trend lines
- Outlier detection (IQR method) with severity flagging
- Target variable analysis with class imbalance detection

### ⚙️ Feature Engineering
- Auto-detects: skewed distributions, high cardinality, missing patterns
- AI-generated full feature engineering roadmap specific to YOUR data
- Interactive feature creator — describe a feature in plain English, get pandas code
- Encoding, scaling, and transformation recommendations

### 💬 Ask Claude
- Full conversational AI assistant with context about your dataset
- Suggested questions for beginners
- Persistent chat history per session

---

## 🚀 Quick Start

### Local
```bash
git clone https://github.com/yourusername/datawise
cd datawise
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Deploy!

Get a free Claude API key at [console.anthropic.com](https://console.anthropic.com)

---

## 📂 Supported File Formats

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| Parquet | `.parquet` |
| JSON | `.json` |
| TSV | `.tsv` |

---

## 🧠 The Teaching Philosophy

Every insight in DataWise follows this pattern:

```
WHAT you're seeing
  → WHY it matters for your goal
    → WHAT to do about it
      → WHAT concept you're learning
```

This isn't just another EDA tool. It's a **data science mentor** — built for the person who just Googled "what is a CSV" and the practicing analyst who wants faster insights.

---

## 🏗️ Architecture

```
User Upload (CSV/Excel/Parquet/JSON/TSV)
    +
Context (goal, domain, experience level)
    ↓
DataWise Engine
├── Data Profiler (Pandas + NumPy + SciPy)
├── Visualization Layer (Plotly)
├── Issue Detector (outliers, skew, cardinality, missing patterns)
└── Claude AI Narrator (Anthropic API)
    ↓
Outputs: Dashboard + AI Insights + Feature Engineering Plan + Code
```

---

## 🤝 Contributing

Pull requests welcome! Areas we'd love help with:

- [ ] Time series analysis module
- [ ] Geospatial column detection & mapping
- [ ] PDF report export
- [ ] Model recommendation engine
- [ ] Kaggle dataset URL import
- [ ] More sample datasets

---

## 📜 License

MIT License — free to use, fork, modify, and distribute.

---

## 🌸 About

Built by [Anu](https://linkedin.com/in/anu) — GenAI Lead, WiDS Ambassador, ODSC Austin Founder.

Inspired by the belief that data science education should be accessible to everyone, regardless of coding background.

Built with ❤️ using Streamlit + Claude API.

---

*If this tool helped you, consider giving it a ⭐ on GitHub and sharing it with your community.*
