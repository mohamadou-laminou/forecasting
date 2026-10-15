"""
Streamlit app - Electricity Load Forecasting (Ile-de-France)
============================================================

Two modes:
  1. History explorer : plot real consumption between two chosen dates/times.
  2. Forecast mode     : pick a date/time, run the models, and show the
                         recent consumption (solid) extended by the four
                         forecasts (dashed), plus a results table.

Run from the project root:
    streamlit run src/app.py

This app reuses the functions and trained models from script.py.
Make sure you have run `python src/script.py` at least once before.
"""

import os
import sys

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Allow importing script.py which sits in the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import script   # noqa: E402


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Load Forecasting - Ile-de-France", layout="wide")
st.title("Electricity Load Forecasting - Ile-de-France")
st.caption("Data: RTE (consumption) + PVGIS (weather), 2021-2023. Model: LightGBM.")


# ---------------------------------------------------------------------------
# Load models and data once (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_models_and_data():
    models, df = script.load_models()
    multistep = script.load_multistep_models()
    return models, df, multistep


@st.cache_resource(show_spinner=False)
def get_walk_forward_models(cutoff_str):
    """Retrain on every observation up to the chosen moment (cached per date)."""
    _, df_local = script.load_models()
    return script.walk_forward_train(pd.Timestamp(cutoff_str), df_local)


try:
    models, df, multistep_models = get_models_and_data()
except FileNotFoundError:
    st.error(
        "Trained models not found. Please run `python src/script.py` first "
        "to train and save the models in the results/ folder."
    )
    st.stop()

# Bounds of the available data
DATA_START = df.index.min().date()
DATA_END = df.index.max().date()


# ---------------------------------------------------------------------------
# Sidebar: choose the mode
# ---------------------------------------------------------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Choose what to do",
    ["History explorer", "Forecast"],
)
st.sidebar.info(f"Data available from {DATA_START} to {DATA_END}.")


# ===========================================================================
# MODE 1 : HISTORY EXPLORER
# ===========================================================================
if mode == "History explorer":
    st.subheader("Explore real consumption between two dates")
    st.write(
        "Pick a start and an end date/time to see how consumption changed "
        "over that period. This is the real measured data."
    )

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=DATA_START,
                                   min_value=DATA_START, max_value=DATA_END)
        start_time = st.time_input("Start time", value=pd.Timestamp("00:00").time())
    with c2:
        end_date = st.date_input("End date", value=pd.Timestamp("2021-12-31").date(),
                                 min_value=DATA_START, max_value=DATA_END)
        end_time = st.time_input("End time", value=pd.Timestamp("23:30").time())

    start = pd.Timestamp.combine(start_date, start_time)
    end = pd.Timestamp.combine(end_date, end_time)

    if start >= end:
        st.warning("The start must be before the end. Adjust your dates.")
    else:
        window = df.loc[start:end]

        if len(window) == 0:
            st.warning("No data in this range.")
        else:
            fig = go.Figure()
            # Consumption on the left axis
            fig.add_trace(go.Scatter(
                x=window.index, y=window["consumption"],
                mode="lines", name="Consumption",
                line=dict(width=1, color="steelblue"),
                yaxis="y1",
            ))
            # Temperature on the right axis
            fig.add_trace(go.Scatter(
                x=window.index, y=window["temperature"],
                mode="lines", name="Temperature",
                line=dict(width=1, color="orange"),
                yaxis="y2",
            ))
            fig.update_layout(
                height=470,
                xaxis_title="Time",
                yaxis=dict(title="Consumption (MW)", color="steelblue"),
                yaxis2=dict(title="Temperature (C)", color="orange",
                            overlaying="y", side="right"),
                title=f"Consumption vs temperature, {start} to {end}",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Consumption (blue, left axis) and temperature (orange, right "
                "axis). Notice how consumption rises when temperature drops - "
                "electric heating is the main driver in winter."
            )

            # A few quick stats
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Average", f"{window['consumption'].mean():,.0f} MW")
            k2.metric("Peak", f"{window['consumption'].max():,.0f} MW")
            k3.metric("Minimum", f"{window['consumption'].min():,.0f} MW")
            k4.metric("Avg temperature", f"{window['temperature'].mean():.1f} C")


# ===========================================================================
# MODE 2 : FORECAST
# ===========================================================================
else:
    st.subheader("Forecast consumption from a chosen moment")
    st.write(
        "Pick a date and time, then run the forecast. The model predicts "
        "consumption 30 minutes, 1 hour, 1 day and 7 days ahead, using only "
        "information available up to the chosen moment."
    )

    # The chosen instant must leave room for the recent past (lags need history)
    min_forecast_date = (df.index.min() + pd.Timedelta(days=8)).date()

    c1, c2 = st.columns(2)
    with c1:
        f_date = st.date_input("Date", value=pd.Timestamp("2023-06-27").date(),
                               min_value=min_forecast_date, max_value=DATA_END)
    with c2:
        # 30-minute steps only
        times = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
        f_time = st.selectbox("Time (30-min steps)", times,
                              index=times.index("08:30"))

    chosen = pd.Timestamp(f"{f_date} {f_time}")

    # --- Method options ---
    with st.expander("Forecast method (advanced)"):
        method = st.radio(
            "How should the continuous curve be built?",
            ["Direct multi-step (recommended)", "Recursive", "Compare both"],
            help=(
                "Direct multi-step: each point is predicted straight from the "
                "chosen moment, so errors do not accumulate. Recursive: the "
                "model feeds its own predictions back in, which drifts over "
                "time and exaggerates peaks."
            ),
        )
        use_walk_forward = st.checkbox(
            "Retrain on all data up to the chosen moment (walk-forward)",
            value=False,
            help=(
                "Methodologically cleaner: uses every observation available at "
                "the forecast time. Takes about 1-2 minutes. In our tests the "
                "accuracy gain was negligible, because two years already cover "
                "every seasonal regime."
            ),
        )

    run = st.button("Run forecast", type="primary")

    if run:
        if chosen not in df.index:
            st.error("This exact timestamp is not in the data. Try another one.")
        else:
            # Pick the models: either the pre-trained ones, or retrained
            # on everything available up to the chosen moment.
            if use_walk_forward:
                with st.spinner("Retraining on all data up to the chosen "
                                "moment (1-2 min)..."):
                    active_models, active_multistep = get_walk_forward_models(
                        str(chosen))
                st.success(
                    f"Models retrained on data up to {chosen} "
                    f"({len(df.loc[:chosen]):,} observations)."
                )
            else:
                active_models, active_multistep = models, multistep_models

            # Four reference forecasts (for the table)
            result = script.predict_from_datetime(chosen, df, active_models)

            # Continuous forecast curve(s)
            with st.spinner("Computing the forecast curve..."):
                curve_direct = curve_recursive = None
                if method in ("Direct multi-step (recommended)", "Compare both"):
                    curve_direct = script.multistep_forecast(
                        chosen, df, active_multistep)
                if method in ("Recursive", "Compare both"):
                    curve_recursive = script.recursive_forecast(
                        chosen, df, active_models, n_steps=336)

            # Recent real consumption before the chosen moment (2 days of context)
            history = df.loc[chosen - pd.Timedelta(days=2): chosen, "consumption"]

            # Real consumption over the forecast window (for comparison)
            ref_curve = curve_direct if curve_direct is not None else curve_recursive
            actual_future = df.loc[ref_curve.index, "consumption"]

            fig = go.Figure()

            # Solid blue : recent real consumption
            fig.add_trace(go.Scatter(
                x=history.index, y=history.values,
                mode="lines", name="Recent consumption (real)",
                line=dict(width=2, color="steelblue"),
            ))

            # Solid green : actual consumption during the forecast window
            fig.add_trace(go.Scatter(
                x=actual_future.index, y=actual_future.values,
                mode="lines", name="Actual (real, for comparison)",
                line=dict(width=1.5, color="mediumseagreen"),
                opacity=0.75,
            ))

            # Direct multi-step curve
            if curve_direct is not None:
                fig.add_trace(go.Scatter(
                    x=[chosen] + list(curve_direct.index),
                    y=[history.loc[chosen]] + list(curve_direct.values),
                    mode="lines", name="Forecast (direct multi-step)",
                    line=dict(width=2.5, color="firebrick"),
                ))

            # Recursive curve
            if curve_recursive is not None:
                fig.add_trace(go.Scatter(
                    x=[chosen] + list(curve_recursive.index),
                    y=[history.loc[chosen]] + list(curve_recursive.values),
                    mode="lines", name="Forecast (recursive)",
                    line=dict(width=1.8, color="darkorange", dash="dot"),
                ))

            # The four horizon markers
            fig.add_trace(go.Scatter(
                x=result["target_time"], y=result["predicted_MW"],
                mode="markers", name="Horizon markers (30min / 1h / 1d / 7d)",
                marker=dict(size=11, color="firebrick", symbol="circle"),
            ))

            fig.add_vline(x=chosen, line_dash="dot", line_color="grey")
            fig.update_layout(
                height=500,
                xaxis_title="Time",
                yaxis_title="Consumption (MW)",
                title=f"Forecast from {chosen} (next 7 days)",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- Accuracy of the curve(s) over the window ----
            def curve_mape(curve):
                real = df.loc[curve.index, "consumption"]
                ok = curve.notna() & real.notna()
                return float(np.mean(np.abs((real[ok] - curve[ok]) / real[ok])) * 100)

            cols = st.columns(2 if method == "Compare both" else 1)
            if curve_direct is not None:
                cols[0].metric("Curve error - direct multi-step",
                               f"{curve_mape(curve_direct):.2f}%")
            if curve_recursive is not None:
                idx = 1 if method == "Compare both" else 0
                cols[idx].metric("Curve error - recursive",
                                 f"{curve_mape(curve_recursive):.2f}%")

            st.caption(
                "Direct multi-step predicts every point straight from the chosen "
                "moment, so errors never accumulate - this tracks daily peaks "
                "much more closely. The recursive curve feeds its own output "
                "back in, which drifts and exaggerates peaks over time. The "
                "green line is the real consumption, shown so you can judge."
            )

            # ---- Results table (the four reference horizons) ----
            st.markdown("**Forecast details (reference horizons)**")
            table = result.copy()
            table = table.rename(columns={
                "horizon": "Horizon",
                "target_time": "Target time",
                "predicted_MW": "Predicted (MW)",
                "actual_MW": "Actual (MW)",
                "error_pct": "Error (%)",
            })
            st.dataframe(table, hide_index=True, use_container_width=True)

            st.caption(
                "The four horizons are the reference points from the dedicated "
                "models. The 'Actual' column shows the real values, available "
                "because we demonstrate on known data (2023)."
            )