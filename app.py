import time
import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
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
primary = BASE_PATH / "data" / "EduTech_Survey_Synthetic_Enhanced.csv"
fallback = BASE_PATH / "EduTech_Survey_Synthetic_Enhanced.csv"
if primary.exists():
    DATA_FILE = primary
elif fallback.exists():
    DATA_FILE = fallback
else:
    st.error(
        f"❌ Data file not found.\nLooked for:\n  • {primary}\n  • {fallback}\n\n"
        "Please place `EduTech_Survey_Synthetic_Enhanced.csv` in a `data/` folder or next to `app.py`."
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
    theme = st.radio("Chart Theme", ["Light", "Dark"], horizontal=True)
    tpl = "plotly_white" if theme == "Light" else "plotly_dark"
    with st.expander("📖 How to Use"):
        st.write("""
        - Sidebar filters apply globally.
        - Hover for more details.
        - Download CSV or PPT from their respective tabs.
        """)
    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Subscription Rate", f"{sub_rate:.1f}%")
    k2.metric("Avg Satisfaction", f"{df['Satisfaction Level'].mean():.2f}/5")
    k3.metric("Avg Comfort", f"{df['App Comfort Level'].mean():.2f}/5")
    k4.metric("7-Day Retention", f"{retention_rate:.1f}%")
    # Charts grid
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="Age", color="Age", title="Age Distribution", template=tpl), use_container_width=True)
    c2.plotly_chart(px.pie(df, names="Gender", title="Gender Distribution", template=tpl), use_container_width=True)
    c1.plotly_chart(px.histogram(df, x="Monthly Income", color="Paid Subscription", barmode="group", title="Income vs Subscription", template=tpl), use_container_width=True)
    dev_counts = df['Device'].value_counts().reset_index(name='Count').rename(columns={'index':'Device'})
    c2.plotly_chart(px.bar(dev_counts, x="Device", y="Count", title="Preferred Devices", template=tpl), use_container_width=True)
    st.plotly_chart(px.pie(df, names="Internet Quality", title="Internet Quality", template=tpl), use_container_width=True)
    st.plotly_chart(px.bar(df, x="Age", color="Paid Subscription", title="Subscription by Age", template=tpl), use_container_width=True)
    # Funnel
    funnel = go.Figure(go.Funnel(y=["Ad Exposed","Signed Up","Subscribed"], x=[len(df), df['Signed Up'].value_counts().get('Yes',0), df['Paid Subscription'].value_counts().get('Yes',0)]))
    st.plotly_chart(funnel, use_container_width=True)
    # Map
    country_counts = df['Country'].value_counts().reset_index(name='Count').rename(columns={'index':'Country'})
    st.plotly_chart(px.choropleth(country_counts, locations='Country', locationmode='country names', color='Count', title="Users by Country", template=tpl), use_container_width=True)
    # Correlation
    corr = df.select_dtypes(include=[np.number]).corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap", template=tpl), use_container_width=True)
    st.markdown("---")
    st.download_button("Download Filtered Data (CSV)", df.to_csv(index=False), "filtered_data.csv", "text/csv")

# ===== 3. TIME-SERIES ANALYSIS =====
with tabs[2]:
    st.header("⏱️ Time-Series Analysis")
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        ts = df.set_index('Timestamp').resample('D').agg({
            'Signed Up': lambda x:(x=="Yes").sum(),
            'Paid Subscription': lambda x:(x=="Yes").sum()
        }).rename(columns={'Signed Up':'Daily Signups','Paid Subscription':'Daily Subscriptions'})
        st.plotly_chart(px.line(ts, template=tpl), use_container_width=True)
    else:
        st.warning("No `Timestamp` column found for time-series analysis.")

# ===== 4. CLASSIFICATION =====
with tabs[3]:
    st.header("🧩 Classification Models")
    st.info("Predict Paid Subscription. Upload new data to predict.")
    clf = df.copy().fillna("Unknown")
    drops = ["Country","City","Non-Subscription Reasons","Features Used Most","Other Apps Used","Motivation","Learning Goals","Preferred Subjects","Preferred App Features","Selection Factors","Learning Challenges"]
    clf.drop(columns=[c for c in drops if c in clf], inplace=True)
    le_map = {}
    for col in clf.select_dtypes(include='object'):
        if col != "Paid Subscription":
            le = LabelEncoder()
            clf[col] = le.fit_transform(clf[col].astype(str))
            le_map[col] = le
    clf["Paid Subscription"] = (clf["Paid Subscription"]=="Yes").astype(int)
    X = clf.drop("Paid Subscription", axis=1)
    y = clf["Paid Subscription"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=60, random_state=0),
        "GBRT": GradientBoostingClassifier(n_estimators=60, random_state=0)
    }
    results = {}
    probs = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        prob = model.predict_proba(X_test_s)[:,1]
        results[name] = {
            "Train Acc": accuracy_score(y_train, model.predict(X_train_s)),
            "Test Acc": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred)
        }
        probs[name] = prob
    st.dataframe(pd.DataFrame(results).T.round(3))
    sel = st.selectbox("Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[sel].predict(X_test_s))
    st.plotly_chart(px.imshow(cm, text_auto=True, labels=dict(x="Pred",y="True"), title=f"{sel} Confusion Matrix", template=tpl), use_container_width=True)
    roc_fig = go.Figure()
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    st.plotly_chart(roc_fig, use_container_width=True)
    up = st.file_uploader("Upload CSV to Predict", type="csv", key="pred")
    if up:
        pdf = pd.read_csv(up)
        for col, le in le_map.items():
            if col in pdf:
                pdf[col] = le.transform(pdf[col].astype(str))
        p_s = scaler.transform(pdf[X.columns])
        pdf["Predicted Subscription"] = np.where(models[sel].predict(p_s)==1, "Yes", "No")
        st.dataframe(pdf.head())
        st.download_button("Download Predictions", pdf.to_csv(index=False), "predictions.csv", "text/csv")

# ===== 5. CLUSTERING =====
with tabs[4]:
    st.header("👥 K-Means Clustering")
    st.info("Segment customers and view cluster centers.")
    cdf = df.copy()
    cols = ['Age','Gender','Monthly Income','Occupation','Courses Taken Last Year','Device']
    for c in cols:
        cdf[c] = LabelEncoder().fit_transform(cdf[c].astype(str))
    k = st.slider("Select number of clusters", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(cdf[cols])
    cdf['Cluster'] = km.labels_
    inertias = [KMeans(n_clusters=i, random_state=0, n_init=10).fit(cdf[cols]).inertia_ for i in range(2,11)]
    st.plotly_chart(px.line(x=list(range(2,11)), y=inertias, title="Elbow Chart", markers=True, template=tpl), use_container_width=True)
    centers = pd.DataFrame(km.cluster_centers_, columns=cols)
    st.dataframe(centers.round(2))
    st.download_button("Download Clustered Data", cdf.to_csv(index=False), "clustered.csv", "text/csv")

# ===== 6. ASSOCIATION RULES =====
with tabs[5]:
    st.header("🔗 Association Rule Mining")
    st.info("Discover top-10 rules from multi-select columns.")
    multi_cols = ['Motivation','Learning Goals','Preferred Subjects','Preferred App Features','Selection Factors','Learning Challenges','Non-Subscription Reasons','Features Used Most','Other Apps Used']
    sel_cols = st.multiselect("Columns", multi_cols, default=multi_cols[:2])
    sup = st.slider("Min Support", 0.01, 0.2, 0.05)
    conf = st.slider("Min Confidence", 0.2, 1.0, 0.6)
    if len(sel_cols) >= 2:
        transactions = []
        for _, row in df[sel_cols].iterrows():
            items = set()
            for col in sel_cols:
                for v in str(row[col]).split(','):
                    vv = v.strip()
                    if vv and vv.lower() != 'none':
                        items.add(f"{col}:{vv}")
            transactions.append(list(items))
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder().fit(transactions)
        df_te = pd.DataFrame(te.transform(transactions), columns=te.columns_)
        freq = apriori(df_te, min_support=sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=conf).sort_values('confidence', ascending=False).head(10)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ", ".join(x))
        rules['consequents'] = rules['consequents'].apply(lambda x: ", ".join(x))
        st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])
    else:
        st.warning("Select at least 2 columns.")

# ===== 7. REGRESSION =====
with tabs[6]:
    st.header("📈 Regression Insights")
    st.info("Run regression models on Spend or Satisfaction.")
    targets = {"Spend":"Willingness to Spend","Satisfaction":"Satisfaction Level"}
    choice = st.selectbox("Target", list(targets.keys()))
    ycol = targets[choice]
    if ycol == "Willingness to Spend":
        y = df[ycol].map({'<10':1,'10-20':2,'20-50':3,'50-100':4,'>100':5})
    else:
        y = df[ycol]
    X = df[['Age','Gender','Education','Monthly Income','Occupation','Device','Internet Quality','Courses Taken Last Year','App Comfort Level']].copy()
    for c in X.select_dtypes(include='object'):
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    sc = StandardScaler()
    X_train_s, X_test_s = sc.fit_transform(X_train), sc.transform(X_test)
    regmods = {
        "Linear":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(alpha=0.01),
        "Tree":DecisionTreeRegressor(max_depth=6, random_state=0)
    }
    out, preds = {}, {}
    for name, m in regmods.items():
        m.fit(X_train_s, y_train)
        p = m.predict(X_test_s)
        out[name] = {"Train R2":m.score(X_train_s, y_train),"Test R2":m.score(X_test_s, y_test)}
        preds[name] = p
    st.dataframe(pd.DataFrame(out).T.round(3))
    for name, p in preds.items():
        st.plotly_chart(px.scatter(x=y_test, y=p, labels={'x':'Actual','y':'Predicted'}, title=f"{name} Actual vs Predicted", template=tpl), use_container_width=True)
    imp = regmods["Tree"].feature_importances_
    st.plotly_chart(px.bar(x=X.columns, y=imp, title="Tree Feature Importances", template=tpl), use_container_width=True)

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
        st.download_button("Download PPT", buf, "EduNest_Summary.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation")

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
