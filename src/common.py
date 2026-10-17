"""Shared helpers for the Streamlit pages: cached loading and chart styling."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import script

# Colors used consistently across pages
ACTUAL = "#2f6db5"        # real consumption
FORECAST = "#c0392b"      # central forecast
BAND = "rgba(192, 57, 43, 0.15)"
TEMP = "#e67e22"
MUTED = "#7f8c8d"


@st.cache_resource(show_spinner="Loading data and models...")
def get_data():
    """Dataset with all features (shared by every session)."""
    return script.prepare_dataset(script.load_data())


@st.cache_resource(show_spinner="Loading data and models...")
def get_bundle():
    return script.load_bundle()


@st.cache_resource(show_spinner=False, max_entries=3)
def get_walk_forward_models(cutoff):
    """Models retrained on every observation up to `cutoff` (a few per server)."""
    data, bundle = get_data(), get_bundle()
    models, _ = script.train_models(
        data, cutoff=cutoff, n_iterations=bundle["iterations"],
        band_margins=script.band_margins_of(bundle["models"]), verbose=False)
    return models


def require_models():
    """Stop the page with a clear message when training has not been run."""
    try:
        return get_data(), get_bundle()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(
            "Trained models not found or outdated. From the project root, run:\n\n"
            "`python src/script.py`\n\n"
            f"Details: {exc}",
            icon=":material/error:",
        )
        st.stop()


def base_layout(fig, height=480, y_title="Consumption (MW)"):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def shade_days_off(fig, data, start, end):
    """Grey background on weekends and public holidays between start and end."""
    days = data.descriptors.loc[start:end, ["is_weekend", "is_public_holiday"]]
    off = days[(days.sum(axis=1) > 0)].index.normalize().unique()
    for d in off:
        is_holiday = bool(data.descriptors.loc[d, "is_public_holiday"])
        fig.add_vrect(
            x0=max(d, pd.Timestamp(start)), x1=min(d + pd.Timedelta(days=1), pd.Timestamp(end)),
            fillcolor="rgba(142, 68, 173, 0.10)" if is_holiday else "rgba(127, 140, 141, 0.08)",
            line_width=0, layer="below",
        )
    return fig


def mape_fmt(x):
    return f"{x:.2f} %"


def new_figure():
    return go.Figure()
