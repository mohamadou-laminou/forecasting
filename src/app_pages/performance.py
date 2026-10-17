import calendar

import plotly.graph_objects as go
import streamlit as st

import common
import script

data, bundle = common.require_models()
report = bundle["report"]
h = report["horizons"]

st.subheader(f"How good is the model? (test year {script.TEST_YEAR}, never seen in training)")
st.write(
    "MAPE = mean absolute percentage error, lower is better. Each forecast is compared "
    "with simple baselines that need no model at all: if the model does not beat them, "
    "it is useless."
)

# ---- KPI row ---------------------------------------------------------------
with st.container(horizontal=True):
    for _, r in h.iterrows():
        best_naive = min(r["persistence_MAPE"], r["same_time_last_day_MAPE"], r["same_time_last_week_MAPE"])
        gain = (1 - r["model_MAPE"] / best_naive) * 100
        st.metric(f"MAPE in {r['horizon']}", f"{r['model_MAPE']:.2f} %",
                  delta=f"{gain:.0f} % better than the best baseline", delta_arrow="off",
                  border=True)

# ---- Main table ------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Scores by horizon**")
    st.dataframe(
        h.drop(columns=["lead"]),
        hide_index=True,
        column_config={
            "horizon": "Horizon",
            "model_MAPE": st.column_config.NumberColumn("Model MAPE (%)", format="%.2f"),
            "model_RMSE": st.column_config.NumberColumn("Model RMSE (MW)", format="%.0f"),
            "model_R2": st.column_config.NumberColumn("Model R2", format="%.3f"),
            "model_weather_error_MAPE": st.column_config.NumberColumn(
                "Model MAPE, realistic weather (%)", format="%.2f",
                help="With a simulated temperature-forecast error (about 1 C at day 1, 3 C at day 7)."),
            "persistence_MAPE": st.column_config.NumberColumn(
                "Baseline: value now (%)", format="%.2f"),
            "same_time_last_day_MAPE": st.column_config.NumberColumn(
                "Baseline: same time, last known day (%)", format="%.2f"),
            "same_time_last_week_MAPE": st.column_config.NumberColumn(
                "Baseline: same time, previous week (%)", format="%.2f"),
            "band_coverage_pct": st.column_config.NumberColumn(
                "Actual inside 80% band (%)", format="%.1f",
                help="About 80% is expected for a well-calibrated band."),
            "band_width_MW": st.column_config.NumberColumn("Mean band width (MW)", format="%.0f"),
        },
    )
    st.caption("'Perfect weather' uses the observed weather at the target time. In real "
               "operation a weather forecast is used: the 'realistic weather' column is the "
               "more honest estimate for long horizons.")

# ---- Error vs lead ----------------------------------------------------------
with st.container(border=True):
    st.markdown("**Error as a function of how far ahead we forecast**")
    bl = report["by_lead"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bl["hours"], y=bl["MAPE perfect weather"], mode="lines+markers",
                             name="Perfect weather", line=dict(color=common.FORECAST, width=2)))
    fig.add_trace(go.Scatter(x=bl["hours"], y=bl["MAPE weather error"], mode="lines+markers",
                             name="Realistic weather forecast error",
                             line=dict(color=common.TEMP, width=2, dash="dash")))
    common.base_layout(fig, height=360, y_title="MAPE (%)")
    fig.update_layout(xaxis_title="Lead (hours)", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Short horizons rely on the current consumption; beyond a day the weather "
               "forecast becomes the main source of error.")

left, right = st.columns(2)
horizon_names = list(script.HORIZONS)

with left, st.container(border=True):
    st.markdown("**Error by type of target day**")
    hz = st.segmented_control("Horizon", horizon_names, default="1 day", key="dt_horizon")
    dt = report["by_daytype"]
    dt = dt[dt["horizon"] == (hz or "1 day")].sort_values("MAPE")
    fig = go.Figure(go.Bar(x=dt["MAPE"], y=dt["day_type"], orientation="h",
                           marker_color=common.ACTUAL, text=dt["MAPE"].map("{:.2f} %".format),
                           customdata=dt["n_points"],
                           hovertemplate="%{y}: %{x:.2f} % (%{customdata} points)<extra></extra>"))
    common.base_layout(fig, height=320, y_title="")
    fig.update_layout(xaxis_title="MAPE (%)", hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    st.caption("Public holidays remain the hardest case: there are only ~11 per year to learn from.")

with right, st.container(border=True):
    st.markdown("**Error by month**")
    hz_m = st.segmented_control("Horizon", horizon_names, default="1 day", key="month_horizon")
    bm = report["by_month"]
    bm = bm[bm["horizon"] == (hz_m or "1 day")]
    fig = go.Figure(go.Bar(x=[calendar.month_abbr[m] for m in bm["month"]], y=bm["MAPE"],
                           marker_color=common.ACTUAL,
                           hovertemplate="%{x}: %{y:.2f} %<extra></extra>"))
    common.base_layout(fig, height=320, y_title="MAPE (%)")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    st.caption("August and the Christmas period are atypical (holidays, closed businesses).")

# ---- Feature importance ------------------------------------------------------
with st.container(border=True):
    st.markdown("**What drives the forecast? (share of total gain)**")
    imp = report["importance"]
    labels = {f"{a}-{b} steps": f"{a / 2:g} h - {b / 2:g} h ahead" for a, b in script.LEAD_BUCKETS}
    choice = st.segmented_control("Lead range", list(imp), default=list(imp)[-1],
                                  format_func=lambda k: labels.get(k, k), key="imp_bucket")
    top = imp[choice or list(imp)[-1]].head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=common.FORECAST,
                           hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
    common.base_layout(fig, height=400, y_title="")
    fig.update_layout(xaxis_title="Share of gain (%)", hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    st.caption("Prefix T_ = describes the target time. The models predict a ratio to the recent "
               "level, so the time of day (daily shape) and the weekday dominate; the weather acts "
               "mostly through the recent level and the temperature change up to the target. Gain "
               "is not causal: the 'realistic weather' test above shows how much the weather matters.")
