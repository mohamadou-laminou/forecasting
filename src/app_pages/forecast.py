import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
import script

data, bundle = common.require_models()
st.session_state.setdefault("forecast", None)

TEST_START = pd.Timestamp(f"{script.TEST_YEAR}-01-01")
DATA_END = data.index.max()

st.subheader("Forecast consumption from a chosen moment")
st.write(
    "Pick an instant: the model uses only what is known at that moment "
    "(consumption so far, calendar, weather at the target time) and forecasts "
    "every 30 minutes up to 7 days ahead. The real consumption is shown for comparison."
)

include_train = st.toggle(
    "Allow dates from the training years (2021-2022)",
    value=False,
    help="The model has seen those years during training, so errors there are "
         "optimistic. 2023 is the honest test year.",
)
min_date = (data.first_origin if include_train else TEST_START).date()

with st.form("forecast_form", border=True):
    c1, c2, c3 = st.columns([1, 1, 2])
    f_date = c1.date_input("Date", value=pd.Timestamp("2023-12-18").date(),
                           min_value=min_date, max_value=DATA_END.date())
    times = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    f_time = c2.selectbox("Time", times, index=times.index("08:00"))
    horizon_days = c3.segmented_control("Forecast length", [1, 3, 7], default=7,
                                        format_func=lambda d: f"{d} day" + ("s" if d > 1 else ""))

    c4, c5 = st.columns([2, 2])
    weather_mode = c4.segmented_control(
        "Weather at the target time",
        ["Perfect", "Realistic forecast error"],
        default="Perfect",
        help="Perfect: the observed weather is used, as if the weather forecast "
             "were exact. Realistic: a simulated temperature-forecast error is added "
             "(about 1 C at day 1, 3 C at day 7), closer to real operating conditions.",
    )
    walk_forward = c5.toggle(
        "Retrain on all data up to this moment (walk-forward)",
        value=False,
        help="Retrains every model on all observations available at the chosen "
             "moment, as in real operation. Takes a few minutes the first time.",
    )
    submitted = st.form_submit_button("Run forecast", type="primary", icon=":material/play_arrow:")

if submitted:
    origin = pd.Timestamp(f"{f_date} {f_time}")
    if origin not in data.index:
        st.error("This exact timestamp is not in the data.")
        st.stop()
    if walk_forward:
        with st.spinner(f"Retraining on all data up to {origin}..."):
            models = common.get_walk_forward_models(origin)
    else:
        models = bundle["models"]
    curve = script.forecast(
        origin, data, models,
        max_lead=(horizon_days or 7) * script.STEPS_PER_DAY,
        weather_error=(weather_mode == "Realistic forecast error"),
        seed=int(origin.value // 10**9),
    )
    st.session_state.forecast = {
        "origin": origin, "curve": curve, "walk_forward": walk_forward,
        "weather": weather_mode or "Perfect",
    }

result = st.session_state.forecast
if result is None:
    st.info("Choose a moment and press **Run forecast**.", icon=":material/touch_app:")
    st.stop()

origin, curve = result["origin"], result["curve"]
table = script.horizon_table(curve)

# ---- Key figures ----------------------------------------------------------
ape = (curve["p50"] - curve["actual"]).abs() / curve["actual"] * 100
inside = ((curve["actual"] >= curve["p10"]) & (curve["actual"] <= curve["p90"])).mean() * 100

with st.container(horizontal=True):
    for _, r in table.iterrows():
        st.metric(
            f"In {r['horizon']}",
            f"{r['predicted_MW']:,} MW",
            delta=f"{r['error_pct']:.1f} % error (actual {r['actual_MW']:,})",
            delta_color="off", delta_arrow="off", border=True,
        )
    st.metric("Curve error (MAPE)", f"{ape.mean():.2f} %", border=True,
              help="Mean absolute percentage error over the whole forecast curve.")
    st.metric("Actual inside the band", f"{inside:.0f} %", border=True,
              help="Share of points where the real value falls in the 10%-90% band "
                   "(about 80% is expected for a well-calibrated band).")

settings = [f"Origin **{origin:%A %d %B %Y, %H:%M}**", f"weather: **{result['weather'].lower()}**"]
if result["walk_forward"]:
    settings.append("models retrained up to the origin")
if origin < TEST_START:
    settings.append(":orange[training period - optimistic]")
st.caption(" · ".join(settings))

# ---- Chart ----------------------------------------------------------------
show_temp = st.toggle("Show temperature", value=False)
history = data.raw.loc[origin - pd.Timedelta(days=2): origin, "consumption"]
end = curve.index.max()

fig = go.Figure()
fig.add_trace(go.Scatter(x=curve.index, y=curve["p90"], mode="lines", line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=curve.index, y=curve["p10"], mode="lines", line=dict(width=0),
                         fill="tonexty", fillcolor=common.BAND, name="80% band (p10-p90)",
                         hoverinfo="skip"))
fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines",
                         name="Consumption before the origin",
                         line=dict(color=common.ACTUAL, width=2)))
fig.add_trace(go.Scatter(x=curve.index, y=curve["actual"], mode="lines",
                         name="Actual consumption (for comparison)",
                         line=dict(color=common.ACTUAL, width=1.5, dash="dot")))
fig.add_trace(go.Scatter(x=[origin] + list(curve.index), y=[history.iloc[-1]] + list(curve["p50"]),
                         mode="lines", name="Forecast",
                         line=dict(color=common.FORECAST, width=2.5)))
fig.add_trace(go.Scatter(x=table["target_time"], y=table["predicted_MW"], mode="markers",
                         name="Reference horizons", text=table["horizon"],
                         marker=dict(size=10, color=common.FORECAST, line=dict(color="white", width=1.5)),
                         hovertemplate="%{text}: %{y:,.0f} MW<extra></extra>"))
if show_temp:
    fig.add_trace(go.Scatter(x=curve.index, y=curve["temp_used"], mode="lines", yaxis="y2",
                             name="Temperature given to the model",
                             line=dict(color=common.TEMP, width=1.2)))
    if result["weather"] != "Perfect":
        fig.add_trace(go.Scatter(x=curve.index, y=curve["temp_observed"], mode="lines", yaxis="y2",
                                 name="Observed temperature",
                                 line=dict(color=common.TEMP, width=1, dash="dot")))
    fig.update_layout(yaxis2=dict(title="Temperature (C)", overlaying="y", side="right",
                                  showgrid=False, color=common.TEMP))

fig.add_vline(x=origin, line_dash="dash", line_color=common.MUTED)
common.shade_days_off(fig, data, history.index.min(), end)
common.base_layout(fig, height=520)
st.plotly_chart(fig, width="stretch")
st.caption("Grey background: weekends · purple background: public holidays. "
           "The band widens with the lead: uncertainty grows the further ahead we look.")

# ---- Details --------------------------------------------------------------
st.markdown("**Reference horizons**")
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "horizon": "Horizon",
        "target_time": st.column_config.DatetimeColumn("Target time", format="ddd DD MMM, HH:mm"),
        "predicted_MW": st.column_config.NumberColumn("Forecast (MW)", format="%d"),
        "low_MW": st.column_config.NumberColumn("Low, p10 (MW)", format="%d"),
        "high_MW": st.column_config.NumberColumn("High, p90 (MW)", format="%d"),
        "actual_MW": st.column_config.NumberColumn("Actual (MW)", format="%d"),
        "error_pct": st.column_config.NumberColumn("Error (%)", format="%.2f"),
    },
)

daily = curve.assign(day=curve.index.normalize()).groupby("day").agg(
    forecast_MW=("p50", "mean"), actual_MW=("actual", "mean"),
    peak_forecast_MW=("p50", "max"), peak_actual_MW=("actual", "max"))
daily["error_pct"] = (daily["forecast_MW"] - daily["actual_MW"]).abs() / daily["actual_MW"] * 100
with st.expander("Day by day summary", icon=":material/calendar_month:"):
    st.dataframe(
        daily.reset_index(),
        hide_index=True,
        column_config={
            "day": st.column_config.DateColumn("Day", format="ddd DD MMM"),
            "forecast_MW": st.column_config.NumberColumn("Mean forecast (MW)", format="%d"),
            "actual_MW": st.column_config.NumberColumn("Mean actual (MW)", format="%d"),
            "peak_forecast_MW": st.column_config.NumberColumn("Peak forecast (MW)", format="%d"),
            "peak_actual_MW": st.column_config.NumberColumn("Peak actual (MW)", format="%d"),
            "error_pct": st.column_config.NumberColumn("Error on the mean (%)", format="%.2f"),
        },
    )

with st.expander("How does the model work?", icon=":material/help:"):
    st.markdown(
        f"""
- **Direct forecasting**: every point of the curve is predicted straight from the
  origin, with the lead (30 min ... 7 days) as an input. Predictions are never fed
  back in, so errors do not accumulate.
- **Inputs about the target time**: time of day, weekday, season, public holidays,
  eves and days after holidays, bridge days, school holidays (zone C), Christmas,
  August, and the weather (temperature, temperature smoothed over ~2 days for the
  thermal inertia of buildings, heating / cooling degrees, sun, wind).
- **Inputs about the origin**: consumption now and just before, means over the
  last 2 h and 24 h, last known value at the same time of day and of week.
- **Uncertainty band**: two extra quantile models give the 10% and 90% bounds.
- {len(script.LEAD_BUCKETS)} model families, each handling a range of leads; trained on
  2021-2022 with the last quarter of 2022 used for early stopping.
"""
    )
