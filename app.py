import time
import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from pptx import Presentation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="EduNest - EdTech Analytics Dashboard", layout="wide")
start_time = time.time()

# --------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
    body { background-color: #1e222e; color: #eee; }
    .stButton>button { border-radius:12px; box-shadow:2px 2px 5px rgba(0,0,0,0.5); }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- LOGO, TITLE, CAPTION ----------
logo_path = "logo.png"
col_logo, col_title = st.columns([1, 7])
with col_logo:
    st.image(logo_path, width=100)
with col_title:
    st.title("EduNest Survey Insights Dashboard")
    st.caption("EduNest • Unlocking the Power of Learner Analytics")
st.markdown("---")

# ----------- DATA LOADING & QUALITY ------------
BASE_PATH = Path(__file__).parent
DATA_FILE = BASE_PATH / "data" / "EduTech_Survey_Synthetic_Enhanced.csv"

if not DATA_FILE.exists():
    st.error(
        f"❌ Data file not found at:\n{DATA_FILE}\n\n"
        "- Ensure there is a folder named `data` next to `app.py`.\n"
        "- The file must be named exactly "
        "`EduTech_Survey_Synthetic_Enhanced.csv`."
    )
    st.stop()

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data(DATA_FILE)

# Data quality & automated alerts
n_missing = int(df.isnull().sum().sum())
numeric = df.select_dtypes(include=[np.number])
n_outliers = int(((numeric - numeric.mean()).abs() / numeric.std() > 4).sum().sum())
sub_rate = 100 * df['Paid Subscription'].value_counts(normalize=True).get('Yes', 0)
threshold_sub = 30

if n_missing or n_outliers:
    st.warning(f"⚠️ Data Quality: {n_missing} missing, {n_outliers} outliers detected.")
else:
    st.success("✅ Data Quality: Clean")

if sub_rate < threshold_sub:
    st.error(f"🚨 Subscription rate is only {sub_rate:.1f}% (< {threshold_sub}%)!")
else:
    st.info(f"Subscription rate: {sub_rate:.1f}%")

# ------------ SIDEBAR FILTERS -------------
with st.sidebar.expander("🧲 Filter Data", expanded=False):
    st.write("Drill down by demographics:")
    sel_age = st.multiselect("Age", sorted(df['Age'].unique()), default=sorted(df['Age'].unique()))
    sel_gen = st.multiselect("Gender", sorted(df['Gender'].unique()), default=sorted(df['Gender'].unique()))
    sel_inc = st.multiselect("Monthly Income", sorted(df['Monthly Income'].unique()), default=sorted(df['Monthly Income'].unique()))
    df = df[df['Age'].isin(sel_age) & df['Gender'].isin(sel_gen) & df['Monthly Income'].isin(sel_inc)]

# ----------------- TABS -----------------
tabs = st.tabs([
    "Executive Summary",
    "Data Visualisation",
    "Time-Series Analysis",
    "Classification",
    "Clustering",
    "Association Rules",
    "Regression",
    "Download PPT",
    "Feedback"
])

# ===== 1. EXECUTIVE SUMMARY =====
with tabs[0]:
    st.subheader("📋 Executive Summary")
    st.metric("Subscription Rate", f"{sub_rate:.1f}%")
    st.metric("Avg Satisfaction", f"{df['Satisfaction Level'].mean():.2f}/5")
    st.metric("Avg Engagement Time", f"{df['Engagement Time (mins)'].mean():.1f} min")
    retention_rate = 100 * df['Retention 7 days'].value_counts(normalize=True).get('Yes', 0)
    st.metric("7-Day Retention", f"{retention_rate:.1f}%")

# ===== 2. DATA VISUALISATION =====
with tabs[1]:
    st.header("📊 Data Visualisation")
    st.info("Use the sidebar filters and theme toggle below.")
    theme = st.radio("Chart Theme", ["Light", "Dark"], horizontal=True)
    tpl = "plotly_white" if theme == "Light" else "plotly_dark"

    with st.expander("📖 How to Use"):
        st.write("""
        - Sidebar filters apply globally.  
        - Hover over charts for details.  
        - Download filtered data or the PPT summary from the respective tabs.
        """)

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Subscription Rate", f"{sub_rate:.1f}%")
    k2.metric("Avg Satisfaction", f"{df['Satisfaction Level'].mean():.2f}/5")
    k3.metric("Avg Comfort", f"{df['App Comfort Level'].mean():.2f}/5")
    k4.metric("7-Day Retention", f"{retention_rate:.1f}%")

    # First row of charts
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="Age", color="Age", title="Age Distribution", template=tpl), use_container_width=True)
    c2.plotly_chart(px.pie(df, names="Gender", title="Gender Distribution", template=tpl), use_container_width=True)

    # Second row
    c1.plotly_chart(px.histogram(df, x="Monthly Income", color="Paid Subscription", barmode="group",
                                 title="Income vs Subscription", template=tpl), use_container_width=True)
    dev_counts = df['Device'].value_counts().reset_index(name='Count').rename(columns={'index':'Device'})
    c2.plotly_chart(px.bar(dev_counts, x="Device", y="Count", title="Preferred Devices", template=tpl), use_container_width=True)

    # Third row
    st.plotly_chart(px.pie(df, names="Internet Quality", title="Internet Quality", template=tpl), use_container_width=True)
    st.plotly_chart(px.bar(df, x="Age", color="Paid Subscription", title="Subscription by Age", template=tpl), use_container_width=True)

    # Funnel
    funnel = go.Figure(go.Funnel(
        y=["Ad Exposed", "Signed Up", "Subscribed"],
        x=[len(df), df['Signed Up'].value_counts().get('Yes', 0), df['Paid Subscription'].value_counts().get('Yes', 0)]
    ))
    st.plotly_chart(funnel, use_container_width=True)

    # Map
    country_counts = df['Country'].value_counts().reset_index(name='Count').rename(columns={'index':'Country'})
    st.plotly_chart(px.choropleth(
        country_counts,
        locations='Country',
        locationmode='country names',
        color='Count',
        title="Users by Country",
        template=tpl
    ), use_container_width=True)

    # Correlation
    num_df = df.select_dtypes(include=[np.number]).copy()
    corr = num_df.corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                              title="Correlation Heatmap", template=tpl), use_container_width=True)

    st.markdown("---")
    st.download_button("Download Filtered Data (CSV)", df.to_csv(index=False), "filtered_data.csv", "text/csv")

# ===== 3. TIME-SERIES ANALYSIS =====
with tabs[2]:
    st.header("⏱️ Time-Series Analysis")
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        ts = df.set_index('Timestamp').resample('D').agg({
            'Signed Up': lambda x: (x=="Yes").sum(),
            'Paid Subscription': lambda x: (x=="Yes").sum()
        }).rename(columns={'Signed Up':'Daily Signups','Paid Subscription':'Daily Subscriptions'})
        st.plotly_chart(px.line(ts, template=tpl), use_container_width=True)
    else:
        st.warning("No `Timestamp` column found for time-series analysis.")

# ===== 4. CLASSIFICATION =====
with tabs[3]:
    st.header("🧩 Classification Models")
    st.info("Predict Paid Subscription. Upload new data to predict.")
    # … same classification code as before …

# ===== 5. CLUSTERING =====
with tabs[4]:
    st.header("👥 K-Means Clustering")
    # … same clustering code as before …

# ===== 6. ASSOCIATION RULES =====
with tabs[5]:
    st.header("🔗 Association Rule Mining")
    # … same association rules code as before …

# ===== 7. REGRESSION =====
with tabs[6]:
    st.header("📈 Regression Insights")
    # … same regression code as before …

# ===== 8. DOWNLOAD PPT SUMMARY =====
with tabs[7]:
    st.header("📥 Download PPT Summary")
    bullets = [
        f"Subscription Rate: {sub_rate:.1f}%",
        f"Avg Satisfaction: {df['Satisfaction Level'].mean():.2f}/5",
        f"Avg Engagement Time: {df['Engagement Time (mins)'].mean():.1f} mins",
        f"7-Day Retention: {retention_rate:.1f}%"
    ]
    if st.button("Generate & Download PPT"):
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "EduNest Executive Summary"
        tf = slide.shapes.placeholders[1].text_frame
        for b in bullets:
            p = tf.add_paragraph(); p.text = b; p.level = 1
        buf = io.BytesIO(); prs.save(buf); buf.seek(0)
        st.download_button("Download PPT", buf, "EduNest_Summary.pptx",
                           "application/vnd.openxmlformats-officedocument.presentationml.presentation")

# ===== 9. FEEDBACK =====
with tabs[8]:
    st.header("💬 Feedback")
    fb = st.text_area("Enter your comments:")
    if st.button("Submit Feedback"):
        st.success("Thank you!")
        webhook = st.secrets.get("slack_webhook_url")
        if webhook and fb:
            requests.post(webhook, json={"text": fb})

# ---- FINAL TAGLINE & PERFORMANCE ----
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#777; font-size:14px;'>"
    "MBA Data Analytics for Insights & Decision Making Project • Team Subhayu | "
    "Unlocking end-to-end learner insights"
    "</div>",
    unsafe_allow_html=True
)
st.caption(f"Dashboard generated in {time.time() - start_time:.2f}s")
