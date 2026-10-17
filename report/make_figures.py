"""Figures of the internship report. Run from the project root after training:
    python report/make_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import script  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Validated categorical palette (slots 1-3) and neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})

data = script.prepare_dataset(script.load_data())
bundle = script.load_bundle()
raw = data.raw

# 0. Consumption patterns: daily profile (winter vs summer weekdays) and weekly profile
is_holiday = data.descriptors["is_public_holiday"] == 1
weekday = (raw.index.dayofweek < 5) & ~is_holiday
tod = raw.index.hour + raw.index.minute / 60
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1.35, 1]})
for name, months, color in [("Winter (Dec-Feb)", [12, 1, 2], BLUE), ("Summer (Jun-Aug)", [6, 7, 8], ORANGE)]:
    mask = weekday & raw.index.month.isin(months)
    p = raw.loc[mask, "consumption"].groupby(tod[mask]).mean() / 1000
    ax1.plot(p.index, p.values, color=color, lw=2, label=name)
ax1.set_xticks(range(0, 25, 4), [f"{h:02d}:00" for h in range(0, 25, 4)])
ax1.set_xlim(0, 23.5)
ax1.set_ylabel("Mean load (GW)")
ax1.set_title("Daily profile, working days", fontsize=9, color=INK)
ax1.legend(loc="lower right", fontsize=8)
daily = raw["consumption"].resample("D").mean()
daily_hol = is_holiday.resample("D").max()
d_ok = daily[~daily_hol.astype(bool)]
by_dow = d_ok.groupby(d_ok.index.dayofweek).mean()
rel = (by_dow / by_dow[:5].mean() - 1) * 100
ax2.bar(range(7), rel.values, color=BLUE, width=0.6)
ax2.axhline(0, color=INK2, lw=0.8)
ax2.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax2.set_ylabel("Daily load vs Mon-Fri mean (%)")
ax2.set_title("Weekly profile", fontsize=9, color=INK)
ax2.grid(axis="x", visible=False)
fig.tight_layout(w_pad=2)
fig.savefig(os.path.join(OUT, "fig0_profiles.png"))
print("weekly profile (% vs Mon-Fri):", rel.round(1).to_dict())

# 1. Temperature sensitivity (daily means)
d = raw.resample("D").mean()
flags = data.descriptors.resample("D").first()
fig, ax = plt.subplots(figsize=(3.3, 2.6))
groups = [("Weekday", (flags.is_weekend == 0) & (flags.is_public_holiday == 0), BLUE),
          ("Weekend", (flags.is_weekend == 1) & (flags.is_public_holiday == 0), ORANGE),
          ("Public holiday", flags.is_public_holiday == 1, AQUA)]
for name, mask, color in groups:
    part = d[mask]
    ax.scatter(part.temperature, part.consumption / 1000, s=8 if name != "Public holiday" else 16,
               color=color, alpha=0.65 if name != "Public holiday" else 1, label=name,
               edgecolors="white", linewidths=0.3)
ax.set_xlabel("Daily mean temperature (°C)")
ax.set_ylabel("Daily mean load (GW)")
ax.legend(loc="upper right", fontsize=8, handletextpad=0.2)
fig.savefig(os.path.join(OUT, "fig1_temperature.png"))

# 2. Monthly mean load by year (energy-saving plan from autumn 2022)
m = raw["consumption"].resample("MS").mean() / 1000
fig, ax = plt.subplots(figsize=(3.3, 2.6))
for year, color in zip((2021, 2022, 2023), (BLUE, ORANGE, AQUA)):
    s = m[m.index.year == year]
    ax.plot(s.index.month, s.values, color=color, lw=2, marker="o", ms=3.5, label=str(year))
ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("Monthly mean load (GW)")
ax.axvspan(9.5, 12.5, color=GRID, alpha=0.5, lw=0)
ax.text(11, ax.get_ylim()[0] + 0.08, "validation\nperiod", ha="center", va="bottom", fontsize=7, color=INK2)
ax.legend(loc="upper center", bbox_to_anchor=(0.47, 1.0), fontsize=8)
fig.savefig(os.path.join(OUT, "fig2_level_shift.png"))

# 3. Example: 7-day forecast from Monday 18 December 2023, 08:00
origin = pd.Timestamp("2023-12-18 08:00")
curve = script.forecast(origin, data, bundle["models"])
hist = raw.loc[origin - pd.Timedelta(days=1):origin, "consumption"] / 1000
fig, ax = plt.subplots(figsize=(6.8, 2.5))
ax.fill_between(curve.index, curve.p10 / 1000, curve.p90 / 1000, color=ORANGE, alpha=0.18, lw=0,
                label="80% band (p10-p90)")
ax.plot(hist.index, hist.values, color=BLUE, lw=2, label="Observed (before origin)")
ax.plot(curve.index, curve.actual / 1000, color=BLUE, lw=1.2, ls=(0, (2, 1.5)), label="Observed (after origin)")
ax.plot(curve.index, curve.p50 / 1000, color=ORANGE, lw=2, label="Forecast")
ax.axvline(origin, color=INK2, lw=1, ls="--")
for day in pd.date_range("2023-12-23", "2023-12-25"):
    ax.axvspan(day, day + pd.Timedelta(days=1), color=GRID, alpha=0.6, lw=0)
ax.text(pd.Timestamp("2023-12-24 12:00"), ax.get_ylim()[1] * 0.99, "weekend +\nChristmas", ha="center",
        va="top", fontsize=7, color=INK2)
ax.set_ylabel("Load (GW)")
ax.set_xlim(hist.index.min(), curve.index.max())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d"))
ax.legend(loc="lower left", ncol=4, fontsize=7.5, bbox_to_anchor=(0, -0.38))
fig.savefig(os.path.join(OUT, "fig3_forecast_example.png"))
ape = (curve.p50 - curve.actual).abs() / curve.actual * 100
inside = ((curve.actual >= curve.p10) & (curve.actual <= curve.p90)).mean() * 100
print(f"fig3: curve MAPE {ape.mean():.2f}%, inside band {inside:.0f}%")

# 4. Error vs lead
bl = bundle["report"]["by_lead"]
fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.plot(bl.hours / 24, bl["MAPE perfect weather"], color=BLUE, lw=2, marker="o", ms=3, label="Observed weather")
ax.plot(bl.hours / 24, bl["MAPE weather error"], color=ORANGE, lw=2, marker="o", ms=3,
        label="Simulated forecast error")
ax.axhline(7.98, color=INK2, lw=1, ls=":")
ax.text(0.1, 8.15, "best naive baseline at 7 days", fontsize=7, color=INK2)
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("MAPE on 2023 (%)")
ax.set_ylim(0, 9)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=8)
fig.savefig(os.path.join(OUT, "fig4_error_vs_lead.png"))

# 5. Feature importance, 3-7 days ahead
LABELS = {
    "T_hod_sin": "Time of day (sin)", "T_hod_cos": "Time of day (cos)",
    "cons_mean_24h": "Mean load, last 24 h", "cons_same_time_last_week": "Load, same time last week",
    "cons_same_time_last_day": "Load, same time last day", "cons_now": "Current load",
    "T_temp_day": "Target temp., 24 h mean", "T_temp": "Target temperature",
    "T_temp_smooth": "Target temp., smoothed", "T_hdd": "Target heating degrees",
    "T_dow": "Target weekday", "temp_smooth_now": "Smoothed temp. now", "temp_now": "Temperature now",
    "T_solar": "Target solar irradiance", "lead": "Lead time", "T_doy_sin": "Season (sin)",
    "T_doy_cos": "Season (cos)",
}
imp = bundle["report"]["importance"]["145-336 steps"].head(8).iloc[::-1]
imp.index = [LABELS.get(k, k) for k in imp.index]
fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
for y, v in enumerate(imp.values):
    ax.text(v + 0.5, y, f"{v:.0f}%", va="center", fontsize=7.5, color=INK)
ax.set_xlabel("Share of total gain (%)")
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(OUT, "fig5_importance.png"))

# Numbers quoted in the report
h = bundle["report"]["horizons"]
print(h.round(2).to_string(index=False))
dt = bundle["report"]["by_daytype"]
print(dt[dt.horizon.isin(["1 day", "7 days"])].round(2).to_string(index=False))
for k, v in bundle["report"]["importance"].items():
    print(k, v.head(4).round(1).to_dict())
