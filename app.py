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
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="EduTech Survey Dashboard", layout="wide")

# --- App Title ---
st.title("📊 EduTech Survey Insights Dashboard")
st.markdown(
    """
    <style>
    .reportview-container { background: #1e222e; }
    .sidebar .sidebar-content { background: #1e222e; }
    .css-1v0mbdj {background-color: #23262f;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        df = None
    return df

# --- Upload or use repo data ---
st.sidebar.header("🔗 Data Source")
uploaded = st.sidebar.file_uploader("Upload new survey data (CSV)", type=["csv"])
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

def get_categorical_columns(df):
    return [col for col in df.columns if df[col].dtype == 'object']

# --- Sidebar Filters ---
with st.sidebar.expander("🧲 Filter Data", expanded=False):
    age = st.multiselect("Age", sorted(df['Age'].unique()), default=sorted(df['Age'].unique()))
    gender = st.multiselect("Gender", sorted(df['Gender'].unique()), default=sorted(df['Gender'].unique()))
    income = st.multiselect("Monthly Income", sorted(df['Monthly Income'].unique()), default=sorted(df['Monthly Income'].unique()))
    df = df[df['Age'].isin(age) & df['Gender'].isin(gender) & df['Monthly Income'].isin(income)]

# --- Tabs ---
tabs = st.tabs([
    "Data Visualisation", 
    "Classification", 
    "Clustering", 
    "Association Rule Mining", 
    "Regression Insights"
])

# ========== 1. Data Visualisation ==========
with tabs[0]:
    st.header("Data Visualisation & Descriptive Insights")
    st.markdown("10+ insights, automatically generated below. Use filters in the sidebar.")

    col1, col2 = st.columns(2)
    # 1. Age Distribution
    fig = px.histogram(df, x="Age", title="Age Distribution", color="Age")
    col1.plotly_chart(fig, use_container_width=True)

    # 2. Gender Distribution
    fig2 = px.pie(df, names="Gender", title="Gender Distribution")
    col2.plotly_chart(fig2, use_container_width=True)

    # 3. Income vs Subscription
    fig3 = px.histogram(df, x="Monthly Income", color="Paid Subscription", barmode='group',
                        title="Income vs Paid Subscription")
    col1.plotly_chart(fig3, use_container_width=True)

    # 4. Device Usage (fixed)
    device_counts = df['Device'].value_counts().reset_index(name='Count').rename(columns={'index': 'Device'})
    fig4 = px.bar(
        device_counts,
        x='Device', y='Count',
        title="Preferred Devices",
        labels={'Device': 'Device', 'Count': 'Count'}
    )
    col2.plotly_chart(fig4, use_container_width=True)

    # 5. Internet Quality
    fig5 = px.pie(df, names="Internet Quality", title="Internet Connection Quality")
    col1.plotly_chart(fig5, use_container_width=True)

    # 6. Subscription by Age
    fig6 = px.bar(df, x="Age", color="Paid Subscription", title="Subscription by Age")
    col2.plotly_chart(fig6, use_container_width=True)

    # 7. Average Satisfaction by Occupation
    occ_sat = df.groupby("Occupation")["Satisfaction Level"].mean().reset_index()
    fig7 = px.bar(occ_sat, x="Occupation", y="Satisfaction Level", title="Avg Satisfaction by Occupation")
    col1.plotly_chart(fig7, use_container_width=True)

    # 8. Clustered Satisfaction by Income & Gender
    sat_gender = df.groupby(["Monthly Income", "Gender"])["Satisfaction Level"].mean().reset_index()
    fig8 = px.bar(sat_gender, x="Monthly Income", y="Satisfaction Level", color="Gender", barmode="group",
                  title="Avg Satisfaction by Income & Gender")
    col2.plotly_chart(fig8, use_container_width=True)

    # 9. Courses Taken vs Subscription
    fig9 = px.histogram(df, x="Courses Taken Last Year", color="Paid Subscription", barmode="group",
                        title="Courses Taken vs Subscription")
    col1.plotly_chart(fig9, use_container_width=True)

    # 10. Most Used App Features
    feature_counts = pd.Series(','.join(df['Features Used Most']).replace(", ", ",").split(',')).value_counts()
    feature_counts = feature_counts.reset_index()
    feature_counts.columns = ['Feature', 'Count']
    fig10 = px.bar(feature_counts[:15], x='Feature', y='Count', title="Most Used App Features")
    col2.plotly_chart(fig10, use_container_width=True)

    st.markdown("---")
    st.write("Download current filtered data:")
    st.download_button("Download CSV", df.to_csv(index=False), "filtered_data.csv", "text/csv")

# ========== 2. Classification ==========
with tabs[1]:
    st.header("Classification Models & Prediction")
    st.markdown("Apply KNN, Decision Tree, Random Forest, Gradient Boosting for predicting Paid Subscription.")

    clf_df = df.copy()
    label_cols = ["Paid Subscription"]
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

    results_df = pd.DataFrame(results).T.round(3)
    st.dataframe(results_df, use_container_width=True)

    algo = st.selectbox("Show confusion matrix for:", list(models.keys()))
    cm = confusion_matrix(y_test, models[algo].predict(X_test_scaled))
    st.write(f"Confusion Matrix: {algo}")
    cm_fig = px.imshow(cm, text_auto=True, x=["No", "Yes"], y=["No", "Yes"], 
        color_continuous_scale="blues", labels=dict(x="Predicted", y="Actual"))
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
    uploaded_pred = st.file_uploader("Upload CSV file for prediction (same structure, no Paid Subscription column)", type="csv", key="pred")
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

# ========== 3. Clustering ==========
with tabs[2]:
    st.header("K-Means Clustering & Customer Personas")
    st.markdown("Segment customers by clustering; adjust cluster count and see customer personas.")

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

# ========== 4. Association Rule Mining ==========
with tabs[3]:
    st.header("Association Rule Mining (Apriori)")
    st.markdown("Find patterns in multi-select questions. Set support/confidence, pick columns, see top 10 rules.")

    multi_cols = ['Motivation', 'Learning Goals', 'Preferred Subjects', 'Preferred App Features',
                  'Selection Factors', 'Learning Challenges', 'Non-Subscription Reasons',
                  'Features Used Most', 'Other Apps Used']
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

# ========== 5. Regression Insights ==========
with tabs[4]:
    st.header("Regression: Spend, Satisfaction & More")
    st.markdown("Apply Linear, Ridge, Lasso, Decision Tree Regression for actionable insights.")

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
    reg_features = ['Age', 'Gender', 'Education', 'Monthly Income', 'Occupation', 'Device',
                    'Internet Quality', 'Courses Taken Last Year', 'App Comfort Level']
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
        fig = px.scatter(x=y_test, y=y_pred, labels={'x':"Actual", 'y':"Predicted"},
                         title=f"{name}: Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{name}: Points along diagonal indicate good fit. Use R² above to compare models.")

    st.markdown("#### Feature Importances (Tree-based Models)")
    for name in ["Decision Tree"]:
        imp = reg_models[name].feature_importances_
        fig = px.bar(x=reg_features, y=imp, title=f"{name} Feature Importances")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{name}: Higher bars indicate more impact on prediction.")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | For MBA Data Analytics & Stakeholder Demo")
