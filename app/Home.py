import streamlit as st

st.set_page_config(
    page_title="Tanzania Climate Predictions",
    page_icon="🌡️",
    layout="wide"
)

st.title("Tanzania Temperature Prediction Dashboard")
st.markdown("""
## Welcome to the Tanzania Climate Analysis Platform

This application provides comprehensive climate analysis and prediction tools:

### 📊 Exploratory Data Analysis
- Temperature distributions and patterns
- Seasonal trends
- Yearly analysis

### 🔬 Model Training
- Random Forest model
- Ridge Regression model
- Performance metrics

### 🎯 Temperature Predictions
- Interactive prediction interface
- Model selection
- Visual results

Navigate through the pages using the sidebar to explore different features.
""")