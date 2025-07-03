# EduNest Enhanced Analytics Dashboard

This repository contains a fully-featured Streamlit dashboard for EduNest, 
built on an **Enhanced Survey Dataset** that tracks the entire customer 
journey from ad exposure to retention.

## Files

- **app.py**: Streamlit application code.
- **data/EduTech_Survey_Synthetic_Enhanced.csv**: Synthetic survey dataset with additional journey columns.
- **requirements.txt**: List of Python dependencies.
- **README.md**: This file.

## Setup & Run Locally

1. Clone this repo.
2. Ensure Python 3.8+ is installed.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. On [Streamlit Cloud](https://share.streamlit.io/), connect your GitHub repo.
3. Set `root` as your app directory.
4. Deploy and enjoy!

## Dataset Details

The enhanced dataset adds:
- **Ad Views Count**  
- **Engagement Time (mins)** first week  
- **Session Count (1st week)**  
- **Click Through Rate**  
- **7-Day Retention**  

Use these new columns for richer 360° analytics and predictive modeling.