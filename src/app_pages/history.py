import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common

data, _ = common.require_models()
raw = data.raw
DATA_START, DATA_END = raw.index.min(), raw.index.max()


@st.cache_data(show_spinner=False)
def daily_table():
    """Daily means with the type of day, for the temperature-sensitivity chart."""
    d = raw.resample("D").mean()
    flags = data.descriptors.resample("D").first()
    d["day_type"] = "Weekday"
    d.loc[flags["is_weekend"] == 1, "day_type"] = "Weekend"
    d.loc[flags["is_public_holiday"] == 1, "day_type"] = "Public holiday"
    return d


st.subheader("Explore real consumption")

with st.container(border=True):
    c1, c2, c3 = st.columns([2, 1, 1])
    period = c1.date_input(
        "Period", value=(pd.Timestamp("2023-01-09").date(), pd.Timestamp("2023-01-22").date()),
        min_value=DATA_START.date(), max_value=DATA_END.date(),
    )
    resolution = c2.segmented_control("Resolution", ["30 min", "Hourly", "Daily"], default="30 min")
    show_temp = c3.toggle("Show temperature", value=True)

if not isinstance(period, tuple) or len(period) != 2:
    st.info("Pick an end date.", icon=":material/date_range:")
    st.stop()

start = pd.Timestamp(period[0])
end = pd.Timestamp(period[1]) + pd.Timedelta(hours=23, minutes=30)
window = raw.loc[start:end]
rule = {"Hourly": "h", "Daily": "D"}.get(resolution)
plot_df = window.resample(rule).mean() if rule else window

with st.container(horizontal=True):
    st.metric("Average", f"{window['consumption'].mean():,.0f} MW", border=True)
    st.metric("Peak", f"{window['consumption'].max():,.0f} MW", border=True,
              help=f"On {window['consumption'].idxmax():%d %b %Y %H:%M}")
    st.metric("Minimum", f"{window['consumption'].min():,.0f} MW", border=True)
    st.metric("Average temperature", f"{window['temperature'].mean():.1f} C", border=True)
    st.metric("Energy", f"{window['consumption'].sum() / 2 / 1000:,.0f} GWh", border=True,
              help="Total energy consumed over the period (MW x 0.5 h).")

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["consumption"], mode="lines",
                         name="Consumption", line=dict(color=common.ACTUAL, width=1.3)))
if show_temp:
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["temperature"], mode="lines",
                             name="Temperature", yaxis="y2", line=dict(color=common.TEMP, width=1)))
    fig.update_layout(yaxis2=dict(title="Temperature (C)", overlaying="y", side="right",
                                  showgrid=False, color=common.TEMP))
if (end - start).days <= 62:
    common.shade_days_off(fig, data, start, end)
common.base_layout(fig, height=460)
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)

with left, st.container(border=True):
    st.markdown("**Temperature sensitivity (whole dataset, daily means)**")
    d = daily_table()
    colors = {"Weekday": common.ACTUAL, "Weekend": common.MUTED, "Public holiday": "#8e44ad"}
    sc = go.Figure()
    for kind, color in colors.items():
        part = d[d["day_type"] == kind]
        sc.add_trace(go.Scatter(x=part["temperature"], y=part["consumption"], mode="markers",
                                name=kind, marker=dict(size=5, color=color, opacity=0.6),
                                text=part.index.strftime("%a %d %b %Y"),
                                hovertemplate="%{text}<br>%{x:.1f} C - %{y:,.0f} MW<extra></extra>"))
    common.base_layout(sc, height=380, y_title="Mean consumption (MW)")
    sc.update_layout(xaxis_title="Mean temperature (C)", hovermode="closest")
    st.plotly_chart(sc, width="stretch")
    st.caption("Below about 15 C, every degree less adds consumption: electric "
               "heating is the main driver. Weekends and holidays sit on a lower curve.")

with right, st.container(border=True):
    st.markdown("**Average daily profile over the selected period**")
    prof = window.assign(
        tod=window.index.strftime("%H:%M"),
        kind=["Weekend" if d >= 5 else "Weekday" for d in window.index.dayofweek],
    ).pivot_table(index="tod", columns="kind", values="consumption", aggfunc="mean")
    pf = go.Figure()
    for kind in prof.columns:
        pf.add_trace(go.Scatter(x=prof.index, y=prof[kind], mode="lines", name=kind,
                                line=dict(color=colors.get(kind), width=2)))
    common.base_layout(pf, height=380)
    pf.update_layout(xaxis_title="Time of day", xaxis=dict(nticks=12))
    st.plotly_chart(pf, width="stretch")
    st.caption("Morning ramp-up, midday plateau and evening peak; weekends start later and lower.")
