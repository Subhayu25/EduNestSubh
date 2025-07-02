# EduTech Streamlit Dashboard

A modern, multi-tab analytics dashboard for EdTech survey data, ready to deploy on Streamlit Cloud.

## How to use

1. Clone or download this repo.
2. Add/update your `EduTech_Survey_Synthetic.csv` file in the `data/` folder.
3. Run locally:
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
4. Or deploy on [Streamlit Cloud](https://share.streamlit.io/):
    - Push this repo to GitHub.
    - On Streamlit Cloud, connect your GitHub repo and launch the app.

### Features
- Data visualization (10+ insights, filters, download)
- Classification (KNN, DT, RF, GBRT): metrics table, confusion matrix, ROC, predict/upload/download
- Clustering (K-means): elbow chart, persona table, download
- Association rule mining (apriori): top 10, filtering
- Regression (Linear, Ridge, Lasso, Decision Tree): 5–7 business insights

### Data structure
Update the dataset by replacing `data/EduTech_Survey_Synthetic.csv`.

---
Made for MBA/Business Analytics coursework and real stakeholder demos.