"""
streamlit_app.py
================
Interactive dashboard for the Payroll Anomaly Detection system.

Pages:
    1. Overview      — KPIs, anomaly distribution, trends
    2. Employee Check — Enter a record, get prediction + SHAP explanation
    3. Bulk Scan     — Upload CSV, scan all records, download flagged ones
    4. Model Compare — Side-by-side model performance comparison
    5. Data Explorer — Browse and filter the full payroll dataset

Run:
    cd app
    streamlit run streamlit_app.py

Author: Lyna Medina Gassouma
Project: Intelligent Anomaly Detection in Global Payroll Data — ADP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ADP Payroll Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Style ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>

/* White text for general elements */
p, h1, h2, h3, h4, label, .stMarkdown, 
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Keep input fields readable — dark text on light background */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    color: #111111 !important;
    background-color: #ffffff !important;
}
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d26;
        border-right: 1px solid #2d3748;
        color: #ffffff !important;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2433 0%, #252d3d 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4fc3f7;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #ffffff;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Anomaly badge */
    .badge-anomaly {
        background: #ff4d4d22;
        border: 1px solid #ff4d4d;
        color: #ff4d4d;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-normal {
        background: #00c85322;
        border: 1px solid #00c853;
        color: #00c853;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Risk badges */
    .risk-high   { color: #ff4d4d; font-weight: 700; font-size: 1.1rem; }
    .risk-medium { color: #ffa726; font-weight: 700; font-size: 1.1rem; }
    .risk-low    { color: #00c853; font-weight: 700; font-size: 1.1rem; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4fc3f7;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 8px;
        margin: 16px 0 12px 0;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

API_URL = "http://127.0.0.1:8000"

COUNTRIES  = ["AU", "BR", "CA", "DE", "FR", "IN", "JP", "TN", "UK", "US"]
CURRENCIES = {"AU": "AUD", "BR": "BRL", "CA": "CAD", "DE": "EUR",
            "FR": "EUR", "IN": "INR", "JP": "JPY", "TN": "TND",
            "UK": "GBP", "US": "USD"}
ROLES = ["Data Scientist", "DevOps Engineer", "Director", "Finance Analyst",
         "HR Manager", "Intern", "Junior Engineer", "Payroll Specialist",
         "Project Manager", "Sales Manager", "Senior Engineer", "Support Specialist"]
DEPARTMENTS = ["Engineering", "Finance", "HR", "IT", "Operations", "Sales"]

MODEL_RESULTS = pd.DataFrame({
    "Model":     ["Random Forest", "Gradient Boosting", "ANN (MLP)",
                  "Isolation Forest", "XGBoost", "Autoencoder"],
    "Type":      ["Supervised", "Supervised", "Supervised",
                  "Unsupervised", "Supervised", "Unsupervised"],
    "Precision": [1.0000, 1.0000, 1.0000, 0.9611, 1.0000, 0.9948],
    "Recall":    [0.7118, 0.7118, 0.7118, 0.1465, 0.7118, 0.5670],
    "F1":        [0.8317, 0.8317, 0.8317, 0.2542, 0.8317, 0.7223],
    "AUC":       [0.8567, 0.8541, 0.8528, 0.8161, 0.8540, 0.8163],
})

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load and cache the main payroll dataset and predictions."""
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(base, "..")

    payroll_path     = os.path.join(root, "data", "synthetic_global_payroll.csv")
    predictions_path = os.path.join(root, "data", "processed", "payroll_predictions.csv")

    df = pd.read_csv(payroll_path, parse_dates=["pay_date"])
    try:
        preds = pd.read_csv(predictions_path)
    except:
        preds = None
    return df, preds

df, preds = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 ADP Anomaly Detection")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Overview", "👤 Employee Check", "📂 Bulk Scan",
        "🤖 Model Comparison", "📋 Data Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**API Status**")
    try:
        r = requests.get(f"{API_URL}/", timeout=2)
        if r.status_code == 200:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Error")
    except:
        st.warning("🟡 API Offline — Start with: `uvicorn main:app`")
    st.markdown("---")
    st.caption("Payroll Anomaly Detection v1.0\nADP GlobalView — PFE 2026")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.markdown("# 📊 Payroll Analytics Overview")
    st.markdown("Global payroll anomaly detection across **500,000 records** — 10 countries, 12 roles, 8 anomaly types.")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total       = len(df)
    anomalies   = df["is_anomaly"].sum()
    anomaly_rate = df["is_anomaly"].mean() * 100
    avg_salary  = df["base_salary"].mean()

    with col1:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{total:,}</p>
            <p class="metric-label">Total Employees</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value" style="color:#ff4d4d">{anomalies:,}</p>
            <p class="metric-label">Anomalies Detected</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value" style="color:#ffa726">{anomaly_rate:.2f}%</p>
            <p class="metric-label">Anomaly Rate</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value" style="color:#00c853">${avg_salary:,.0f}</p>
            <p class="metric-label">Avg Base Salary</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Row 2: Country bar + Anomaly type pie
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Anomalies by Country</p>', unsafe_allow_html=True)
        country_data = df[df["is_anomaly"] == 1].groupby("country").size().reset_index(name="count")
        fig = px.bar(country_data, x="count", y="country", orientation="h",
                     color="count", color_continuous_scale="Blues",
                     template="plotly_dark")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=0, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Anomaly Type Breakdown</p>', unsafe_allow_html=True)
        type_data = df[df["anomaly_type"] != "none"]["anomaly_type"].value_counts().reset_index()
        type_data.columns = ["type", "count"]
        fig = px.pie(type_data, values="count", names="type",
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     template="plotly_dark", hole=0.3)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Time trend + Salary by role
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Anomaly Trend — Jan to June 2023</p>', unsafe_allow_html=True)
        if "pay_month" in df.columns:
            trend = df[df["is_anomaly"] == 1].groupby("pay_month").size().reset_index(name="anomalies")
            fig = px.line(trend, x="pay_month", y="anomalies",
                          markers=True, template="plotly_dark",
                          color_discrete_sequence=["#4fc3f7"])
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280,
                              xaxis_title="Month", yaxis_title="Anomalies")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Average Salary by Role</p>', unsafe_allow_html=True)
        role_data = df.groupby("role")["base_salary"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(role_data, x="base_salary", y="role", orientation="h",
                     color="base_salary", color_continuous_scale="Teal",
                     template="plotly_dark")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=0, b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EMPLOYEE CHECK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👤 Employee Check":
    st.markdown("# 👤 Employee Anomaly Check")
    st.markdown("Enter a payroll record to get an AI prediction with SHAP explanation.")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<p class="section-header">Employee Details</p>', unsafe_allow_html=True)

        emp_id   = st.text_input("Employee ID", value="EMP00001")
        country  = st.selectbox("Country", COUNTRIES, index=9)  # US default
        currency = st.text_input("Currency", value=CURRENCIES.get(country, "USD"), disabled=True)
        role     = st.selectbox("Role", ROLES, index=10)  # Senior Engineer
        dept     = st.selectbox("Department", DEPARTMENTS, index=0)
        years_exp = st.slider("Years of Experience", 0.0, 30.0, 5.0, 0.5)

        st.markdown('<p class="section-header">Salary Details</p>', unsafe_allow_html=True)
        base_salary    = st.number_input("Base Salary", min_value=0.0, value=95000.0, step=1000.0)
        bonus          = st.number_input("Bonus", min_value=0.0, value=8000.0, step=500.0)
        tax            = st.number_input("Tax Deducted", min_value=0.0, value=0.0, step=500.0)
        social_sec     = st.number_input("Social Security", min_value=0.0, value=0.0, step=500.0)
        net_pay        = st.number_input("Net Pay", min_value=0.0, value=103000.0, step=1000.0)

        # Auto-update currency when country changes
        currency_val = CURRENCIES.get(country, "USD")

        analyze = st.button("🔍 Analyze Record", type="primary", use_container_width=True)
        # explain = st.button("💡 Full SHAP Explanation", use_container_width=True)
        explain = st.button("💡 Full SHAP Explanation", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<p class="section-header">Analysis Result</p>', unsafe_allow_html=True)

        if analyze or explain:
            payload = {
                "employee_id":      emp_id,
                "country":          country,
                "currency":         currency_val,
                "role":             role,
                "department":       dept,
                "years_experience": years_exp,
                "base_salary":      base_salary,
                "bonus":            bonus,
                "tax":              tax,
                "social_security":  social_sec,
                "net_pay":          net_pay
            }

            endpoint = "/explain" if explain else "/predict"

            try:
                response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
                result   = response.json()

                # Status badge
                if result["is_anomaly"]:
                    st.markdown('<span class="badge-anomaly">⚠️ ANOMALY DETECTED</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-normal">✅ NORMAL RECORD</span>', unsafe_allow_html=True)

                st.markdown("")

                # Risk level
                risk = result["risk_level"]
                risk_class = f"risk-{risk.lower()}"
                st.markdown(f'<p class="{risk_class}">Risk Level: {risk}</p>', unsafe_allow_html=True)

                # Anomaly score gauge
                score = result["anomaly_score"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    title={"text": "Anomaly Score", "font": {"color": "white"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "white"},
                        "bar":  {"color": "#ff4d4d" if score > 0.5 else "#00c853"},
                        "steps": [
                            {"range": [0, 30],  "color": "#1a2a1a"},
                            {"range": [30, 60], "color": "#2a2a1a"},
                            {"range": [60, 100],"color": "#2a1a1a"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 2}, "value": 50}
                    },
                    number={"suffix": "%", "font": {"color": "white"}}
                ))
                fig.update_layout(
                    height=250, paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Message
                # st.info(result["message"])
                st.info(result.get("message", f"Anomaly score: {result['anomaly_score']:.1%}"))

                # SHAP explanation
                if explain and "feature_scores" in result:
                    st.markdown('<p class="section-header">SHAP Feature Contributions</p>', unsafe_allow_html=True)
                    shap_df = pd.DataFrame(
                        list(result["feature_scores"].items()),
                        columns=["Feature", "SHAP Value"]
                    ).sort_values("SHAP Value", key=abs, ascending=True)

                    colors = ["#ff4d4d" if v > 0 else "#4fc3f7" for v in shap_df["SHAP Value"]]
                    fig = go.Figure(go.Bar(
                        x=shap_df["SHAP Value"],
                        y=shap_df["Feature"],
                        orientation="h",
                        marker_color=colors
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis_title="SHAP Contribution (red=anomaly, blue=normal)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Anomaly indicators
                    if result.get("anomaly_indicators"):
                        st.markdown('<p class="section-header">Triggered Rules</p>', unsafe_allow_html=True)
                        for indicator in result["anomaly_indicators"]:
                            st.error(f"⚠️ {indicator}")

                    # Recommendation
                    st.markdown('<p class="section-header">Recommendation</p>', unsafe_allow_html=True)
                    st.warning(result["recommendation"])

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure uvicorn is running:\n```\ncd api && python -m uvicorn main:app --reload --port 8000\n```")
            except Exception as e:
                st.error(f"❌ Error: {e}")

        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#8892a4;">
                <h3>👈 Fill in the form and click</h3>
                <p>• <b>Analyze Record</b> — quick prediction + risk score</p>
                <p>• <b>Full SHAP Explanation</b> — prediction + feature contributions + triggered rules</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BULK SCAN
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📂 Bulk Scan":
    st.markdown("# 📂 Bulk Payroll Scan")
    st.markdown("Upload a CSV file of payroll records to scan all of them at once.")
    st.markdown("---")

    st.markdown("### Required CSV columns:")
    st.code("employee_id, country, currency, role, department, years_experience, base_salary, bonus, tax, social_security, net_pay")

    uploaded = st.file_uploader("Upload payroll CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_upload):,} records")
        st.dataframe(df_upload.head(), use_container_width=True)

        if st.button("🚀 Run Anomaly Detection on All Records", type="primary"):
            results = []
            progress = st.progress(0)
            status   = st.empty()

            for i, row in df_upload.iterrows():
                try:
                    payload = {
                        "employee_id":      str(row.get("employee_id", f"EMP{i}")),
                        "country":          str(row.get("country", "US")),
                        "currency":         str(row.get("currency", "USD")),
                        "role":             str(row.get("role", "Junior Engineer")),
                        "department":       str(row.get("department", "Engineering")),
                        "years_experience": float(row.get("years_experience", 5)),
                        "base_salary":      float(row.get("base_salary", 50000)),
                        "bonus":            float(row.get("bonus", 0)),
                        "tax":              float(row.get("tax", 0)),
                        "social_security":  float(row.get("social_security", 0)),
                        "net_pay":          float(row.get("net_pay", 50000))
                    }
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    res = r.json()
                    results.append({
                        "employee_id":  payload["employee_id"],
                        "is_anomaly":   res["is_anomaly"],
                        "anomaly_score": res["anomaly_score"],
                        "risk_level":   res["risk_level"],
                        "message":      res["message"]
                    })
                except Exception as e:
                    results.append({"employee_id": str(row.get("employee_id", i)),
                                   "is_anomaly": None, "anomaly_score": None,
                                   "risk_level": "Error", "message": str(e)})

                progress.progress((i + 1) / len(df_upload))
                status.text(f"Scanning record {i+1} of {len(df_upload)}...")

            results_df = pd.DataFrame(results)
            flagged    = results_df[results_df["is_anomaly"] == True]

            st.success(f"✅ Scan complete — {len(flagged)} anomalies found out of {len(results_df)} records")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Scanned", len(results_df))
                st.metric("Anomalies Found", len(flagged))
            with col2:
                st.metric("High Risk", len(results_df[results_df["risk_level"] == "High"]))
                st.metric("Medium Risk", len(results_df[results_df["risk_level"] == "Medium"]))

            st.markdown("### Flagged Records")
            st.dataframe(flagged, use_container_width=True)

            csv = flagged.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Flagged Records CSV",
                csv, "flagged_anomalies.csv", "text/csv",
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Performance Comparison")
    st.markdown("6 models evaluated on 99,378 test records (20% holdout from 500k dataset).")
    st.markdown("---")

    # Metrics table
    st.markdown('<p class="section-header">All Models — Metrics</p>', unsafe_allow_html=True)
    st.dataframe(
        MODEL_RESULTS.style
        .background_gradient(subset=["Precision", "Recall", "F1", "AUC"], cmap="Greens")
        .format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}", "AUC": "{:.4f}"}),
        use_container_width=True, height=280
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">F1 Score by Model</p>', unsafe_allow_html=True)
        fig = px.bar(MODEL_RESULTS, x="Model", y="F1",
                     color="Type", barmode="group",
                     color_discrete_map={"Supervised": "#4fc3f7", "Unsupervised": "#ff9800"},
                     template="plotly_dark")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Precision vs Recall</p>', unsafe_allow_html=True)
        fig = px.scatter(MODEL_RESULTS, x="Recall", y="Precision",
                         text="Model", size="AUC", color="Type",
                         color_discrete_map={"Supervised": "#4fc3f7", "Unsupervised": "#ff9800"},
                         template="plotly_dark")
        fig.update_traces(textposition="top center")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown('<p class="section-header">Model Radar — Multi-Metric View</p>', unsafe_allow_html=True)
    categories = ["Precision", "Recall", "F1", "AUC"]
    fig = go.Figure()
    colors = ["#4fc3f7", "#00e676", "#ff9800", "#ff4d4d", "#ce93d8", "#80cbc4"]
    for i, row in MODEL_RESULTS.iterrows():
        vals = [row["Precision"], row["Recall"], row["F1"], row["AUC"]]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["Model"],
            line_color=colors[i], opacity=0.7
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_dark", height=450,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key insights
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**Best Supervised Model**\nRandom Forest / XGBoost\nPrecision: 1.0 | F1: 0.83")
    with col2:
        st.info("**Best Unsupervised Model**\nAutoencoder\nPrecision: 0.99 | F1: 0.72")
    with col3:
        st.warning("**Production Recommendation**\nRandom Forest for labeled data\nAutoencoder for unlabeled streams")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Data Explorer":
    st.markdown("# 📋 Payroll Data Explorer")
    st.markdown("Browse, filter, and analyze the full 500,000-record payroll dataset.")
    st.markdown("---")

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_countries = st.multiselect("Country", df["country"].unique(), default=list(df["country"].unique()))
    with col2:
        selected_roles = st.multiselect("Role", df["role"].unique(), default=list(df["role"].unique()))
    with col3:
        anomaly_filter = st.selectbox("Record Type", ["All", "Anomalies Only", "Normal Only"])
    with col4:
        anomaly_type_filter = st.selectbox("Anomaly Type", ["All"] + list(df["anomaly_type"].unique()))

    # Apply filters
    filtered = df[
        df["country"].isin(selected_countries) &
        df["role"].isin(selected_roles)
    ]
    if anomaly_filter == "Anomalies Only":
        filtered = filtered[filtered["is_anomaly"] == 1]
    elif anomaly_filter == "Normal Only":
        filtered = filtered[filtered["is_anomaly"] == 0]
    if anomaly_type_filter != "All":
        filtered = filtered[filtered["anomaly_type"] == anomaly_type_filter]

    st.markdown(f"**Showing {len(filtered):,} records**")
    st.dataframe(filtered.head(1000), use_container_width=True, height=400)

    # Download filtered
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Data",
        csv, "filtered_payroll.csv", "text/csv"
    )