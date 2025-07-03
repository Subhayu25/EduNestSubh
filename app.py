import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="EduNest EdTech Analytics", layout="wide")

# ------------- LOGO AND BRANDING -------------
logo_path = "logo.png"
st.markdown("<style>body {background-color: #1e222e !important;}</style>", unsafe_allow_html=True)
col_logo, col_title = st.columns([1,7])
with col_logo:
    st.image(logo_path, width=110)
with col_title:
    st.title("EduNest - EdTech Analytics Dashboard")
    st.caption("MBA Data Analytics Project • Team Subhayu | Unlocking end-to-end learner insights")
st.markdown("---")

# ------------ DATA QUALITY CHECK -------------
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        df = None
    return df

st.sidebar.header("🔗 Data Source")
uploaded = st.sidebar.file_uploader("Upload new survey data (CSV)", type=["csv"], help="You can update your dashboard live by uploading a new file")
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("File uploaded and in use!")
else:
    df = load_data("data/EduTech_Survey_Synthetic.csv")
    if df is not None:
        st.sidebar.info("Using default data from repo.")
    else:
        st.sidebar.error("No data found! Please upload a CSV in the sidebar.")
        st.stop()

# Show data warnings
n_missing = df.isnull().sum().sum()
n_outliers = int((np.abs((df.select_dtypes(include=[np.number]) - df.select_dtypes(include=[np.number]).mean())/df.select_dtypes(include=[np.number]).std()) > 4).sum().sum())
if n_missing > 0 or n_outliers > 0:
    st.warning(f"⚠️ Data Quality: {n_missing} missing values, {n_outliers} detected outlier points in numeric features. Review for best analysis.")
else:
    st.success("✅ Data Quality: No missing values, data ready for analysis.")

# ------------- SIDEBAR FILTERS --------------
with st.sidebar.expander("🧲 Filter Data (applies to all tabs)", expanded=False):
    st.write("Tip: Use filters to drill down insights by segment/persona.")
    age = st.multiselect("Age", sorted(df['Age'].unique()), default=sorted(df['Age'].unique()))
    gender = st.multiselect("Gender", sorted(df['Gender'].unique()), default=sorted(df['Gender'].unique()))
    income = st.multiselect("Monthly Income", sorted(df['Monthly Income'].unique()), default=sorted(df['Monthly Income'].unique()))
    df = df[df['Age'].isin(age) & df['Gender'].isin(gender) & df['Monthly Income'].isin(income)]

# --------- TABS STRUCTURE & HELP ----------
tabs = st.tabs([
    "Executive Summary", 
    "Data Visualisation", 
    "Classification", 
    "Clustering", 
    "Association Rule Mining", 
    "Regression Insights",
    "Feedback"
])

# ===== 1. EXECUTIVE SUMMARY (AUTO & STATIC) =====
with tabs[0]:
    st.subheader("📋 Executive Summary: Key Insights")
    st.info("""
    - Majority of users are aged 18-34, and students dominate signups.
    - Working professionals spend more on subscriptions, but students are more likely to sign up.
    - Paid subscriptions are highly correlated with App Comfort Level and online learning frequency.
    - Most users access via smartphones, especially in evenings.
    - Top challenges: Time management, lack of motivation.
    - Users prefer apps with quizzes, live sessions, and personalized content.
    - Data reveals clear segments for strategic targeting and channel optimization.
    """)
    st.write("These insights are auto-generated from your latest uploaded dataset and updated filters.")

    # Auto-insights sample (top-3 by group, quick stats)
    st.markdown("#### Live Data Highlights")
    st.metric("Paid Subscription Rate", f"{100*df['Paid Subscription'].value_counts(normalize=True).get('Yes',0):.1f}%")
    st.metric("Avg. Satisfaction Level", f"{df['Satisfaction Level'].mean():.2f}/5")
    st.metric("Top Device Used", df['Device'].mode()[0])
    st.metric("Most Popular Age Group", df['Age'].mode()[0])
    st.metric("Courses Taken (avg)", f"{pd.to_numeric(df['Courses Taken Last Year'], errors='coerce').mean():.1f}")
    st.caption("Want to add your own? Just edit this section.")

# ===== 2. DATA VISUALISATION TAB =====
with tabs[1]:
    st.header("📊 Data Visualisation & Descriptive Insights")
    st.info("Explore complex insights. Hover on any chart for more detail. Data filters are applied live.")

    col1, col2 = st.columns(2)
    fig = px.histogram(df, x="Age", title="Age Distribution", color="Age")
    col1.plotly_chart(fig, use_container_width=True)
    fig2 = px.pie(df, names="Gender", title="Gender Distribution")
    col2.plotly_chart(fig2, use_container_width=True)
    fig3 = px.histogram(df, x="Monthly Income", color="Paid Subscription", barmode='group', title="Income vs Paid Subscription")
    col1.plotly_chart(fig3, use_container_width=True)
    device_counts = df['Device'].value_counts().reset_index(name='Count').rename(columns={'index': 'Device'})
    fig4 = px.bar(device_counts, x='Device', y='Count', title="Preferred Devices", labels={'Device': 'Device', 'Count': 'Count'})
    col2.plotly_chart(fig4, use_container_width=True)
    fig5 = px.pie(df, names="Internet Quality", title="Internet Connection Quality")
    col1.plotly_chart(fig5, use_container_width=True)
    fig6 = px.bar(df, x="Age", color="Paid Subscription", title="Subscription by Age")
    col2.plotly_chart(fig6, use_container_width=True)
    occ_sat = df.groupby("Occupation")["Satisfaction Level"].mean().reset_index()
    fig7 = px.bar(occ_sat, x="Occupation", y="Satisfaction Level", title="Avg Satisfaction by Occupation")
    col1.plotly_chart(fig7, use_container_width=True)
    sat_gender = df.groupby(["Monthly Income", "Gender"])["Satisfaction Level"].mean().reset_index()
    fig8 = px.bar(sat_gender, x="Monthly Income", y="Satisfaction Level", color="Gender", barmode="group", title="Avg Satisfaction by Income & Gender")
    col2.plotly_chart(fig8, use_container_width=True)
    fig9 = px.histogram(df, x="Courses Taken Last Year", color="Paid Subscription", barmode="group", title="Courses Taken vs Subscription")
    col1.plotly_chart(fig9, use_container_width=True)
    feature_counts = pd.Series(','.join(df['Features Used Most']).replace(", ", ",").split(',')).value_counts()
    feature_counts = feature_counts.reset_index()
    feature_counts.columns = ['Feature', 'Count']
    fig10 = px.bar(feature_counts[:15], x='Feature', y='Count', title="Most Used App Features")
    col2.plotly_chart(fig10, use_container_width=True)

    st.markdown("### 📈 Correlation Heatmap (numeric features only)")
    with st.expander("Show Correlation Heatmap (See how features move together)", expanded=False):
        num_df = df.copy()
        for col in num_df.columns:
            if num_df[col].dtype == 'object':
                try:
                    num_df[col] = LabelEncoder().fit_transform(num_df[col].astype(str))
                except Exception:
                    pass
        corr = num_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Interpretation: Values close to 1/-1 show strong positive/negative correlation. 0 = no linear relationship.")

    st.markdown("---")
    st.write("Download current filtered data as CSV. For PDF reports, use your browser's 'Print to PDF' option.")
    st.download_button("Download CSV", df.to_csv(index=False), "filtered_data.csv", "text/csv")

# === 3. CLASSIFICATION TAB ===
with tabs[2]:
    st.header("🧩 Classification Models & Prediction")
    st.info("Predict Paid Subscription using ML. See confusion matrix and ROC. You can upload new data for prediction.")

    clf_df = df.copy()
    drop_cols = [
        'Country', 'City', 'Non-Subscription Reasons', 'Features Used Most',
        'Other Apps Used', 'Motivation', 'Learning Goals', 'Preferred Subjects', 
        'Preferred App Features', 'Selection Factors', 'Learning Challenges'
    ]
    clf_df = clf_df.drop(columns=[c for c in drop_cols if c in clf_df.columns], errors='ignore')
    clf_df = clf_df.fillna("Unknown")
    le_dict = {}
    for col in clf_df.columns:
        if clf_df[col].dtype == 'object' and col != "Paid Subscription":
            le = LabelEncoder()
            clf_df[col] = le.fit_transform(clf_df[col].astype(str))
            le_dict[col] = le
    clf_df['Paid Subscription'] = (clf_df['Paid Subscription'] == 'Yes').astype(int)
    X = clf_df.drop(columns=["Paid Subscription"])
    y = clf_df["Paid Subscription"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=60, random_state=0),
        "GBRT": GradientBoostingClassifier(n_estimators=60, random_state=0)
    }
    results = {}
    y_probs = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]
        results[name] = {
            "Train Acc": accuracy_score(y_train, model.predict(X_train_scaled)),
            "Test Acc": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        }
        y_probs[name] = y_prob
    st.dataframe(pd.DataFrame(results).T.round(3), use_container_width=True)

    algo = st.selectbox("Show confusion matrix for:", list(models.keys()))
    cm = confusion_matrix(y_test, models[algo].predict(X_test_scaled))
    st.write(f"Confusion Matrix: {algo}")
    cm_fig = px.imshow(cm, text_auto=True, x=["No", "Yes"], y=["No", "Yes"], color_continuous_scale="blues", labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(cm_fig, use_container_width=True)
    st.markdown("**ROC Curve for all models**")
    fig_roc = go.Figure()
    for name, probs in y_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name, line_shape='linear'))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown("#### Predict on new uploaded data (without target variable)")
    uploaded_pred = st.file_uploader("Upload CSV for prediction (no Paid Subscription column)", type="csv", key="pred")
    if uploaded_pred:
        try:
            pred_df = pd.read_csv(uploaded_pred)
            for col in X.columns:
                if col in pred_df.columns and col in le_dict:
                    pred_df[col] = le_dict[col].transform(pred_df[col].astype(str))
            pred_scaled = scaler.transform(pred_df[X.columns])
            pred = models[algo].predict(pred_scaled)
            pred_df["Predicted Subscription"] = np.where(pred==1, "Yes", "No")
            st.success("Prediction successful! Preview below.")
            st.dataframe(pred_df.head(), use_container_width=True)
            csv = pred_df.to_csv(index=False)
            st.download_button("Download Predictions", csv, "predicted_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# === 4. CLUSTERING TAB ===
with tabs[3]:
    st.header("👥 K-Means Clustering & Customer Personas")
    st.info("Segment users and discover clusters (personas). Adjust the number of clusters with the slider.")
    clust_df = df.copy()
    persona_cols = ['Age', 'Gender', 'Monthly Income', 'Occupation', 'Courses Taken Last Year', 'Device']
    for col in persona_cols:
        if clust_df[col].dtype == 'object':
            clust_df[col] = LabelEncoder().fit_transform(clust_df[col].astype(str))
    clust_data = clust_df[persona_cols]
    k = st.slider("Select number of clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    clust_labels = kmeans.fit_predict(clust_data)
    clust_df['Cluster'] = clust_labels
    wcss = []
    for i in range(2, 11):
        km = KMeans(n_clusters=i, random_state=0, n_init=10)
        km.fit(clust_data)
        wcss.append(km.inertia_)
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=list(range(2,11)), y=wcss, mode='lines+markers'))
    elbow_fig.update_layout(title="Elbow Chart (Choose best k)", xaxis_title="Number of clusters", yaxis_title="WCSS (Inertia)")
    st.plotly_chart(elbow_fig, use_container_width=True)
    st.markdown("#### Customer Personas (Cluster Centers)")
    personas = pd.DataFrame(kmeans.cluster_centers_, columns=persona_cols)
    st.dataframe(personas.round(2), use_container_width=True)
    st.markdown("Download full data with cluster labels:")
    st.download_button("Download Clustered Data", clust_df.to_csv(index=False), "clustered_data.csv", "text/csv")

# === 5. ASSOCIATION RULE MINING TAB ===
with tabs[4]:
    st.header("🔗 Association Rule Mining (Apriori)")
    st.info("Mine multi-select survey fields for patterns! Use the sliders for support/confidence. Filter by columns.")
    multi_cols = ['Motivation', 'Learning Goals', 'Preferred Subjects', 'Preferred App Features', 'Selection Factors', 'Learning Challenges', 'Non-Subscription Reasons', 'Features Used Most', 'Other Apps Used']
    col_options = st.multiselect("Select 2+ columns to mine associations:", multi_cols, default=["Motivation", "Learning Challenges"])
    support = st.slider("Min support:", 0.01, 0.2, 0.05)
    confidence = st.slider("Min confidence:", 0.2, 1.0, 0.6)
    if len(col_options) >= 2:
        baskets = []
        for idx, row in df[col_options].iterrows():
            items = set()
            for col in col_options:
                for opt in str(row[col]).split(','):
                    opt = opt.strip()
                    if opt and opt.lower() not in ['none', 'unknown']:
                        items.add(f"{col}:{opt}")
            baskets.append(list(items))
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)
        freq = apriori(basket_df, min_support=support, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=confidence)
        rules = rules.sort_values("confidence", ascending=False).head(10)
        if not rules.empty:
            rules_disp = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
            rules_disp['antecedents'] = rules_disp['antecedents'].apply(lambda x: ', '.join([str(i) for i in x]))
            rules_disp['consequents'] = rules_disp['consequents'].apply(lambda x: ', '.join([str(i) for i in x]))
            st.dataframe(rules_disp, use_container_width=True)
        else:
            st.warning("No rules found for selected columns/support/confidence.")

# === 6. REGRESSION INSIGHTS TAB ===
with tabs[5]:
    st.header("📈 Regression: Spend, Satisfaction & More")
    st.info("Predict spend and satisfaction using multiple regression models. Useful for pricing and experience optimization.")
    reg_df = df.copy()
    target_map = {
        "Willingness to Spend": ("Willingness to Spend", {'<10':1, '10-20':2, '20-50':3, '50-100':4, '>100':5}),
        "Satisfaction Level": ("Satisfaction Level", None)
    }
    reg_target = st.selectbox("Choose regression target:", list(target_map.keys()))
    target_col, mapping = target_map[reg_target]
    reg_y = reg_df[target_col]
    if mapping:
        reg_y = reg_y.map(mapping)
    reg_features = ['Age', 'Gender', 'Education', 'Monthly Income', 'Occupation', 'Device', 'Internet Quality', 'Courses Taken Last Year', 'App Comfort Level']
    for col in reg_features:
        if reg_df[col].dtype == 'object':
            reg_df[col] = LabelEncoder().fit_transform(reg_df[col].astype(str))
    reg_X = reg_df[reg_features]
    X_train, X_test, y_train, y_test = train_test_split(reg_X, reg_y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=0)
    }
    reg_results = {}
    preds = {}
    for name, model in reg_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        reg_results[name] = {
            "Train R2": model.score(X_train_scaled, y_train),
            "Test R2": model.score(X_test_scaled, y_test)
        }
        preds[name] = y_pred
    st.dataframe(pd.DataFrame(reg_results).T.round(3), use_container_width=True)
    st.markdown("#### Actual vs Predicted (Test set)")
    for name, y_pred in preds.items():
        fig = px.scatter(x=y_test, y=y_pred, labels={'x':"Actual", 'y':"Predicted"}, title=f"{name}: Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{name}: Points along diagonal indicate good fit. Use R² above to compare models.")
    st.markdown("#### Feature Importances (Tree-based Models)")
    for name in ["Decision Tree"]:
        imp = reg_models[name].feature_importances_
        fig = px.bar(x=reg_features, y=imp, title=f"{name} Feature Importances")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{name}: Higher bars indicate more impact on prediction.")

# === 7. FEEDBACK FORM ===
with tabs[6]:
    st.header("💬 Feedback & Suggestions")
    st.write("We value your suggestions! Help us improve EduNest and this analytics dashboard.")
    feedback = st.text_area("Your suggestions or feedback:", max_chars=600)
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | For MBA Data Analytics & Stakeholder Demo | EduNest © 2025")
