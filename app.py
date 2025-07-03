
import time
import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
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
st.set_page_config(page_title="EduNest Enhanced Dashboard", layout="wide")
start_time = time.time()

# --------------- CUSTOM CSS -------------------
st.markdown("""
<style>
body { background-color: #1e222e; color: #eee; }
.stButton>button { border-radius:12px; box-shadow:2px 2px 5px rgba(0,0,0,0.5); }
</style>
""", unsafe_allow_html=True)

# ---------- LOGO, TITLE, CAPTION ----------
logo_path = "logo.png"
col1, col2 = st.columns([1,8])
with col1:
    st.image(logo_path, width=100)
with col2:
    st.title("EduNest Enhanced Analytics Dashboard")
    st.caption("EduNest • Unlocking the Power of Learner Analytics")
st.markdown("---")

# ----------- DATA LOADING & QUALITY ------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data("data/EduTech_Survey_Synthetic_Enhanced.csv")

# Data quality & alerts
n_missing = df.isnull().sum().sum()
num = df.select_dtypes(include=[np.number])
n_outliers = int(((num - num.mean()).abs() / num.std() > 4).sum().sum())
sub_rate = 100 * df['Paid Subscription'].value_counts(normalize=True).get('Yes', 0)
threshold_sub = 30

if n_missing or n_outliers:
    st.warning(f"⚠️ Data Quality issues: {n_missing} missing, {n_outliers} outliers")
else:
    st.success("✅ Data Quality: Clean")

if sub_rate < threshold_sub:
    st.error(f"🚨 Subscription rate low: {sub_rate:.1f}%")
else:
    st.info(f"Subscription rate: {sub_rate:.1f}%")

# ------------ SIDEBAR FILTERS -------------
with st.sidebar.expander("Filters"):
    sel_age = st.multiselect("Age", sorted(df['Age'].unique()), default=sorted(df['Age'].unique()))
    sel_gen = st.multiselect("Gender", sorted(df['Gender'].unique()), default=sorted(df['Gender'].unique()))
    sel_inc = st.multiselect("Income", sorted(df['Monthly Income'].unique()), default=sorted(df['Monthly Income'].unique()))
    df = df[df['Age'].isin(sel_age) & df['Gender'].isin(sel_gen) & df['Monthly Income'].isin(sel_inc)]

# ----------------- TABS -----------------
tabs = st.tabs([
    "Summary",
    "Visualisation",
    "Time-Series",
    "Classification",
    "Clustering",
    "Association Rules",
    "Regression",
    "Download PPT",
    "Feedback"
])

# 1. SUMMARY
with tabs[0]:
    st.header("Executive Summary")
    st.metric("Subscription Rate", f"{sub_rate:.1f}%")
    st.metric("Avg Satisfaction", f"{df['Satisfaction Level'].mean():.2f}/5")
    st.metric("Avg Engagement Time", f"{df['Engagement Time (mins)'].mean():.1f} min")
    st.metric("7-Day Retention", f"{100*df['Retention 7 days'].value_counts(normalize=True).get('Yes',0):.1f}%")

# 2. VISUALISATION
with tabs[1]:
    st.header("Data Visualisation")
    theme = st.radio("Theme", ["Light","Dark"], horizontal=True)
    tpl = "plotly_white" if theme=="Light" else "plotly_dark"
    st.plotly_chart(px.histogram(df, x="Ad Views Count", nbins=10, title="Ad Views Distribution", template=tpl), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="Engagement Time (mins)", nbins=20, title="Engagement Time (mins)", template=tpl), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="Session Count (1st week)", nbins=10, title="Session Count (1st week)", template=tpl), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="Click Through Rate", nbins=20, title="Click Through Rate", template=tpl), use_container_width=True)
    st.plotly_chart(px.pie(df, names="Retention 7 days", title="7-Day Retention Rate", template=tpl), use_container_width=True)

# 3. TIME-SERIES
with tabs[2]:
    st.header("Time-Series Analysis")
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        ts = df.set_index('Timestamp').resample('D').agg({
            'Ad Views Count':'sum',
            'Session Count (1st week)':'sum',
            'Paid Subscription': lambda x:(x=="Yes").sum()
        })
        st.plotly_chart(px.line(ts, template=tpl), use_container_width=True)
    else:
        st.warning("No timestamp column found.")

# 4. CLASSIFICATION
with tabs[3]:
    st.header("Classification")
    # ... classification code ...

# 5. CLUSTERING
with tabs[4]:
    st.header("Clustering")
    # ... clustering code ...

# 6. ASSOCIATION RULES
with tabs[5]:
    st.header("Association Rule Mining")
    # ... association rules code ...

# 7. REGRESSION
with tabs[6]:
    st.header("Regression Insights")
    # ... regression code ...

# 8. DOWNLOAD PPT
with tabs[7]:
    st.header("Download PPT Summary")
    if st.button("Generate PPT"):
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "EduNest Summary"
        tf = slide.shapes.placeholders[1].text_frame
        for metric, val in [
            ("Subscription Rate", f"{sub_rate:.1f}%"),
            ("Avg Satisfaction", f"{df['Satisfaction Level'].mean():.2f}/5"),
            ("Avg Engagement Time", f"{df['Engagement Time (mins)'].mean():.1f} min"),
            ("Retention 7 days", f"{100*df['Retention 7 days'].value_counts(normalize=True).get('Yes',0):.1f}%")
        ]:
            p = tf.add_paragraph()
            p.text = f"{metric}: {val}"
            p.level = 1
        buf = io.BytesIO()
        prs.save(buf)
        buf.seek(0)
        st.download_button("Download PPT", buf, "EduNest_Summary.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation")

# 9. FEEDBACK
with tabs[8]:
    st.header("Feedback")
    fb = st.text_area("Comments:")
    if st.button("Submit"):
        st.success("Thanks for your feedback!")
        webhook = st.secrets.get("slack_webhook_url")
        if webhook and fb:
            requests.post(webhook, json={"text": fb})

# FINAL TAGLINE
st.markdown("---")
st.markdown("<div style='text-align:center; color:#777;'>MBA Data Analytics for Insights & Decision Making Project • Team Subhayu | Unlocking end-to-end learner insights</div>", unsafe_allow_html=True)
st.caption(f"Generated in {time.time()-start_time:.2f}s")
