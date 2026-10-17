"""
Streamlit app - Electricity Load Forecasting (Ile-de-France)
============================================================

Three pages:
  - Forecast          : pick an instant of 2023 and get the 7-day forecast
                        curve with its uncertainty band, compared to reality.
  - History explorer  : real consumption and temperature over any period.
  - Model performance : scores on the test year against naive baselines,
                        errors by lead / day type / month, feature importance.

Run from the project root (after `python src/script.py`):
    streamlit run src/app.py
"""

import os
import sys

import streamlit as st

# Make script.py and common.py importable from the pages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Load forecasting - Ile-de-France",
    page_icon=":material/bolt:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page("app_pages/forecast.py", title="Forecast", icon=":material/insights:", default=True),
        st.Page("app_pages/history.py", title="History explorer", icon=":material/history:"),
        st.Page("app_pages/performance.py", title="Model performance", icon=":material/speed:"),
    ],
    position="top",
)

st.title("Electricity load forecasting - Ile-de-France")
st.caption("Data: RTE (consumption) + PVGIS (weather), 2021-2023, 30-minute step. "
           "Model: direct LightGBM with quantile bands, trained on 2021-2022, tested on 2023.")

page.run()
