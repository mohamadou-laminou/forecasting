"""
Electricity Load Forecasting - Ile-de-France (2021-2023)
========================================================

This script:
  1. Loads the cleaned dataset (Excel file, 30-minute step)
  2. Builds two kinds of features:
       - descriptors of the TARGET instant (the moment we forecast):
         calendar (time of day, weekday, season), public holidays, holiday
         eves / days after, bridge days, school holidays, Christmas period,
         August, and weather (temperature, smoothed temperature, heating /
         cooling degrees, solar radiation, wind);
       - the STATE at the forecast origin (consumption now, recent means,
         last known value at the same time of day / week, temperature now).
  3. Splits the data chronologically:
       train      : 2021-01 -> 2022-09
       validation : 2022-10 -> 2022-12  (early stopping)
       test       : 2023                (never seen during training)
  4. Trains direct LightGBM models, one family per range of forecast leads
     (30 min ... 7 days). Each family has three models: a central forecast and
     the 10% / 90% quantiles, which give an uncertainty band (calibrated on the
     validation period). The models predict a ratio to the consumption level
     of the last 24 h, which makes them robust to level shifts such as the
     energy-saving plan of winter 2022-2023.
  5. Evaluates on 2023 against naive baselines, with perfect weather and
     with a simulated weather-forecast error, and saves everything in results/.

Key idea: the lead (how far ahead) is itself a feature, and every point of the
7-day curve is predicted DIRECTLY from the origin. No prediction is ever fed
back in, so errors do not accumulate.

About the weather:
  The dataset contains OBSERVED weather. Using it at the target instant is
  equivalent to a perfect weather forecast. In operations you would use a real
  weather forecast (e.g. Meteo-France / Open-Meteo), which is less accurate,
  so we also report the scores with a simulated forecast error whose size
  grows with the lead (about 1 C at day 1, 3 C at day 7).

Run from the project root:
    python src/script.py
"""

import os
import time
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates


# ---------------------------------------------------------------------------
# Configuration - paths are resolved relative to the PROJECT ROOT, so the
# script works whether you run it from the project folder or from src/.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "Ile_de_france_clean_dataset.xlsx")
SHEET_NAME = "donnees_idf_30min_lisse"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CLEAN_DATA_FILE = os.path.join(RESULTS_DIR, "data_clean.parquet")
BUNDLE_FILE = os.path.join(RESULTS_DIR, "forecast_bundle.joblib")

STEP = pd.Timedelta(minutes=30)
STEPS_PER_DAY = 48
STEPS_PER_WEEK = 336
MAX_LEAD = STEPS_PER_WEEK                 # 7 days at a 30-minute resolution

VALID_START = pd.Timestamp("2022-10-01")
TEST_YEAR = 2023

# Reference horizons, expressed in number of 30-minute steps
HORIZONS = {
    "30 min": 1,
    "1 hour": 2,
    "1 day": 48,
    "7 days": 336,
}

# One model family per range of leads (inclusive bounds, in steps).
# Short ranges near the origin, where the dynamics change fastest.
LEAD_BUCKETS = [(1, 2), (3, 12), (13, 48), (49, 144), (145, 336)]
LEADS_PER_ORIGIN = 4                      # leads sampled per origin when training

# The models predict target / reference level. Up to 6 h ahead the best
# reference is the current value; further out, the mean of the last 24 h.
SHORT_LEAD_MAX = 12


def level_column(bucket_index):
    return "cons_now" if LEAD_BUCKETS[bucket_index][1] <= SHORT_LEAD_MAX else "cons_mean_24h"

# Central forecast + bounds of the 80% uncertainty band
QUANTILES = {"p10": 0.10, "p90": 0.90}

LGB_PARAMS = dict(
    n_estimators=3000,                    # upper bound, early stopping decides
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 100

# Physical thresholds (deg C)
HEATING_BASE = 15.0                       # below this, electric heating starts
COOLING_BASE = 22.0                       # above this, air conditioning starts

# School-holiday zone for Ile-de-France
SCHOOL_ZONE = "C"


# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
def load_excel(path=DATA_PATH, sheet=SHEET_NAME):
    """Load the Excel file and build a clean, regular 30-minute index."""
    df = pd.read_excel(path, sheet_name=sheet)

    df = df.rename(columns={
        "Electric consumption (MW)": "consumption",
        "temperature (celsius)":     "temperature",
        "Gi (Solar Radiation W/m2)": "solar",
        "Windspeed (m/s)":           "wind",
    })
    df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[["consumption", "temperature", "solar", "wind"]]

    # The models rely on "n steps later = n x 30 min": enforce a regular grid
    full = pd.date_range(df.index.min(), df.index.max(), freq=STEP)
    if df.index.has_duplicates or len(full) != len(df):
        df = df[~df.index.duplicated()].reindex(full).interpolate(limit=4)
    return df


def load_data():
    """Load the cleaned data, from the fast parquet copy when it exists."""
    if os.path.exists(CLEAN_DATA_FILE):
        return pd.read_parquet(CLEAN_DATA_FILE)
    return load_excel()


# ===========================================================================
# 2. FEATURES DESCRIBING ONE INSTANT (used for the TARGET instant)
# ===========================================================================
def calendar_features(index):
    """Calendar and holiday flags for every timestamp of `index`."""
    out = pd.DataFrame(index=index)

    # Time of day including minutes (08:00 and 08:30 must differ)
    hod = index.hour + index.minute / 60
    out["hod_sin"] = np.sin(2 * np.pi * hod / 24)
    out["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    doy = index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["dow"] = index.dayofweek                          # 0 = Monday
    out["is_weekend"] = (index.dayofweek >= 5).astype(int)

    # Day-level flags, computed once per day, then mapped to timestamps
    days = pd.date_range(index.min().normalize(), index.max().normalize(), freq="D")
    public = pd.Series(
        [int(JoursFeries.is_bank_holiday(d.date(), zone="Métropole")) for d in days],
        index=days,
    )
    checker = SchoolHolidayDates()
    school, failures = [], 0
    for d in days:
        try:
            school.append(int(checker.is_holiday_for_zone(d.date(), SCHOOL_ZONE)))
        except Exception:
            school.append(0)
            failures += 1
    if failures:
        print(f"  WARNING: school holidays unknown for {failures} days (set to 0)")
    school = pd.Series(school, index=days)

    next_day = public.shift(-1, fill_value=0)
    prev_day = public.shift(1, fill_value=0)
    dow = days.dayofweek
    day_flags = pd.DataFrame({
        "is_public_holiday": public,
        "is_holiday_eve":    ((next_day == 1) & (public == 0)).astype(int),
        "is_holiday_after":  ((prev_day == 1) & (public == 0)).astype(int),
        # Monday before a Tuesday holiday / Friday after a Thursday holiday
        "is_bridge":         (((dow == 0) & (next_day == 1)) |
                              ((dow == 4) & (prev_day == 1))).astype(int),
        "is_school_holiday": school,
        "is_xmas_period":    (((days.month == 12) & (days.day >= 24)) |
                              ((days.month == 1) & (days.day <= 2))).astype(int),
        "is_august":         (days.month == 8).astype(int),
    }, index=days)

    out = out.join(day_flags.reindex(index.normalize()).set_axis(index))
    return out


def weather_features(df):
    """Weather descriptors. The smoothed temperature mimics the thermal inertia
    of buildings: a cold spell weighs more after several cold days."""
    out = pd.DataFrame(index=df.index)
    out["temp"] = df["temperature"]
    out["temp_day"] = df["temperature"].rolling(STEPS_PER_DAY, center=True, min_periods=1).mean()
    out["temp_smooth"] = df["temperature"].ewm(halflife=STEPS_PER_DAY).mean()
    out["hdd"] = np.maximum(HEATING_BASE - out["temp_smooth"], 0)   # heating degrees
    out["cdd"] = np.maximum(out["temp_day"] - COOLING_BASE, 0)      # cooling degrees
    out["solar"] = df["solar"]
    out["wind"] = df["wind"]
    return out


CALENDAR_COLS = [
    "hod_sin", "hod_cos", "doy_sin", "doy_cos", "dow", "is_weekend",
    "is_public_holiday", "is_holiday_eve", "is_holiday_after", "is_bridge",
    "is_school_holiday", "is_xmas_period", "is_august",
]
WEATHER_COLS = ["temp", "temp_day", "temp_smooth", "hdd", "cdd", "solar", "wind"]
TEMP_COLS = ["temp", "temp_day", "temp_smooth"]          # shifted by a weather error
TARGET_COLS = ["T_" + c for c in CALENDAR_COLS + WEATHER_COLS]
STATE_COLS = [
    "lead", "cons_now", "cons_prev", "cons_mean_2h", "cons_mean_24h",
    "cons_same_time_last_day", "cons_same_time_last_week",
    "temp_now", "temp_smooth_now",
]
FEATURE_COLUMNS = TARGET_COLS + STATE_COLS


# ===========================================================================
# 3. DATASET + DESIGN MATRIX
# ===========================================================================
@dataclass
class Dataset:
    """Everything needed to build model inputs, stored as aligned arrays."""
    raw: pd.DataFrame            # consumption, temperature, solar, wind
    descriptors: pd.DataFrame    # calendar + weather of every instant
    index: pd.DatetimeIndex
    cons: np.ndarray
    mean_2h: np.ndarray
    mean_24h: np.ndarray

    def position(self, timestamp):
        return self.index.get_loc(pd.Timestamp(timestamp))

    @property
    def first_origin(self):
        """Earliest origin with a full week of history behind it."""
        return self.index[STEPS_PER_WEEK]


def prepare_dataset(df):
    descriptors = pd.concat([calendar_features(df.index), weather_features(df)], axis=1)
    cons = df["consumption"].to_numpy(dtype=float)
    return Dataset(
        raw=df,
        descriptors=descriptors[CALENDAR_COLS + WEATHER_COLS].astype(float),
        index=df.index,
        cons=cons,
        # Means up to and including the origin (consumption at t is known at t)
        mean_2h=df["consumption"].rolling(4).mean().to_numpy(),
        mean_24h=df["consumption"].rolling(STEPS_PER_DAY).mean().to_numpy(),
    )


def design_matrix(data, origins, leads, temp_error=None):
    """Model inputs for pairs (origin position, lead in steps).

    At origin t, consumption up to and including t is known.
    `temp_error` (deg C, one value per row) shifts the target temperature to
    simulate an imperfect weather forecast.
    """
    origins = np.asarray(origins)
    leads = np.asarray(leads)
    targets = origins + leads

    D = data.descriptors.to_numpy()[targets]
    X = pd.DataFrame(D, columns=TARGET_COLS)

    if temp_error is not None:
        for col in TEMP_COLS:
            X["T_" + col] += temp_error
        X["T_hdd"] = np.maximum(HEATING_BASE - X["T_temp_smooth"], 0)
        X["T_cdd"] = np.maximum(X["T_temp_day"] - COOLING_BASE, 0)

    c = data.cons
    last_day_lag = STEPS_PER_DAY * np.ceil(leads / STEPS_PER_DAY).astype(int)
    X["lead"] = leads
    X["cons_now"] = c[origins]
    X["cons_prev"] = c[origins - 1]
    X["cons_mean_2h"] = data.mean_2h[origins]
    X["cons_mean_24h"] = data.mean_24h[origins]
    X["cons_same_time_last_day"] = c[targets - last_day_lag]      # always <= origin
    X["cons_same_time_last_week"] = c[targets - STEPS_PER_WEEK]
    X["temp_now"] = data.descriptors["temp"].to_numpy()[origins]
    X["temp_smooth_now"] = data.descriptors["temp_smooth"].to_numpy()[origins]
    return X[FEATURE_COLUMNS]


def weather_error_sigma(leads):
    """Typical std of a temperature forecast error (deg C) vs lead (steps):
    about 1.1 C at day 1, 2.9 C at day 7."""
    return 0.8 + 0.3 * np.asarray(leads) / STEPS_PER_DAY


def simulated_weather_error(leads, seed=None, correlated=True):
    """Random temperature errors for the given leads.

    correlated=True: a smooth error along one forecast curve (a real weather
    forecast is wrong for hours or days in a row, not point by point).
    """
    rng = np.random.default_rng(seed)
    leads = np.asarray(leads)
    if correlated:
        phi = 0.98                                   # memory of about 1 day
        z = np.empty(len(leads))
        z[0] = rng.standard_normal()
        for i in range(1, len(leads)):
            z[i] = phi * z[i - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal()
    else:
        z = rng.standard_normal(len(leads))
    return weather_error_sigma(leads) * z


def bucket_of(leads):
    """Index of the lead bucket handling each lead."""
    uppers = np.array([b for _, b in LEAD_BUCKETS])
    return np.searchsorted(uppers, np.asarray(leads))


def sample_pairs(data, origin_mask, bucket, seed):
    """Training pairs (origin, lead) for one lead bucket."""
    lo, hi = bucket
    n = len(data.index)
    origins = np.flatnonzero(origin_mask)
    origins = origins[(origins >= STEPS_PER_WEEK) & (origins + hi < n)]

    n_leads = hi - lo + 1
    if n_leads <= LEADS_PER_ORIGIN:
        leads = np.tile(np.arange(lo, hi + 1), len(origins))
        origins = np.repeat(origins, n_leads)
    else:
        rng = np.random.default_rng(seed)
        origins = np.repeat(origins, LEADS_PER_ORIGIN)
        leads = rng.integers(lo, hi + 1, size=len(origins))
    return origins, leads


# ===========================================================================
# 4. TRAINING
# ===========================================================================
def _fit(X, y, quantile=None, n_estimators=None, eval_set=None):
    params = dict(LGB_PARAMS)
    if quantile is not None:
        params.update(objective="quantile", alpha=quantile)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)] if eval_set else None
    model.fit(X, y, eval_set=eval_set, callbacks=callbacks)
    return model


BAND_LEVEL = 0.80                         # target coverage of the p10-p90 band


def train_models(data, cutoff=None, n_iterations=None, band_margins=None, verbose=True):
    """Train every (bucket, quantile) model.

    The models predict the RATIO  target / reference level (current value for
    short leads, mean of the last 24 h beyond, see `level_column`). A ratio adapts at once to a change of level (e.g. the energy-saving plan
    of winter 2022-2023 lowered consumption by 5-10%), while a model trained
    on raw MW would keep the old level in its trees.

    Default (cutoff=None): early stopping on the validation period chooses the
    number of trees, and the validation errors calibrate the band (conformal
    calibration). The models are then refit on all data before the test year.

    Walk-forward (cutoff given): refit on every observation whose target is
    before `cutoff`, reusing the tree counts and band margins found before.

    Returns (models, iterations) with models[bucket_index] holding the
    "p50", "p10", "p90" models and the "band_margin".
    """
    idx = data.index
    names = ["p50"] + list(QUANTILES)
    models, iterations = {}, {}

    for b, bucket in enumerate(LEAD_BUCKETS):
        origins, leads = sample_pairs(data, np.ones(len(idx), bool), bucket, seed=b)
        X = design_matrix(data, origins, leads)
        y = data.cons[origins + leads] / X[level_column(b)].to_numpy()
        t_origin = idx[origins]
        t_target = idx[origins + leads]

        if cutoff is None:
            fit_mask = t_target < VALID_START
            val_mask = (t_origin >= VALID_START) & (t_target.year < TEST_YEAR)
            final_mask = t_target.year < TEST_YEAR
        else:
            final_mask = t_target <= pd.Timestamp(cutoff)

        models[b], iterations[b], val_pred = {}, {}, {}
        for name in names:
            q = QUANTILES.get(name)
            start = time.time()
            if cutoff is None:
                probe = _fit(X[fit_mask], y[fit_mask], q,
                             eval_set=[(X[val_mask], y[val_mask])])
                n_iter = max(probe.best_iteration_, 50)
                val_pred[name] = probe.predict(X[val_mask])
            else:
                n_iter = n_iterations[b][name]
            models[b][name] = _fit(X[final_mask], y[final_mask], q, n_estimators=n_iter)
            iterations[b][name] = n_iter
            if verbose:
                print(f"  leads {bucket[0]:>3}-{bucket[1]:<3} {name}: "
                      f"{n_iter:4d} trees, {final_mask.sum():,} rows, {time.time() - start:5.1f}s")

        if cutoff is None:
            # Conformal calibration: how much must the band be widened so that
            # BAND_LEVEL of the validation points fall inside it?
            yv = y[val_mask]
            lo = np.minimum(val_pred["p10"], val_pred["p50"])
            hi = np.maximum(val_pred["p90"], val_pred["p50"])
            score = np.maximum(lo - yv, yv - hi)
            margin = float(np.quantile(score, BAND_LEVEL))
        else:
            margin = band_margins[b]
        models[b]["band_margin"] = margin
    return models, iterations


def band_margins_of(models):
    return {b: m["band_margin"] for b, m in models.items()}


def predict(models, X):
    """Central forecast and band (MW) for each row of X (any mix of leads)."""
    out = {name: np.empty(len(X)) for name in ["p10", "p50", "p90"]}
    buckets = bucket_of(X["lead"].to_numpy())
    for b in np.unique(buckets):
        rows = buckets == b
        for name in out:
            out[name][rows] = models[b][name].predict(X[rows])
        # Quantile models are trained separately: make sure p10 <= p50 <= p90,
        # then apply the calibrated margin
        out["p10"][rows] = np.minimum(out["p10"][rows], out["p50"][rows]) - models[b]["band_margin"]
        out["p90"][rows] = np.maximum(out["p90"][rows], out["p50"][rows]) + models[b]["band_margin"]
        level = X.loc[rows, level_column(b)].to_numpy()  # ratio -> MW
        for name in out:
            out[name][rows] *= level
    return out


# ===========================================================================
# 5. EVALUATION ON THE TEST YEAR
# ===========================================================================
def _scores(y, pred):
    return {
        "MAPE": float(np.mean(np.abs(y - pred) / y) * 100),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def test_pairs(data, lead, every=1):
    """All origins of the test year whose target is also in the test year."""
    idx = data.index
    origins = np.flatnonzero(idx.year == TEST_YEAR)[::every]
    origins = origins[origins + lead < len(idx)]
    origins = origins[idx[origins + lead].year == TEST_YEAR]
    return origins, np.full(len(origins), lead)


def evaluate(data, models, verbose=True):
    """Scores on the test year, stored in a dict used by the Streamlit app."""
    c = data.cons
    rows, by_daytype, by_month = [], [], []

    for name, lead in HORIZONS.items():
        origins, leads = test_pairs(data, lead)
        targets = origins + leads
        y = c[targets]
        X = design_matrix(data, origins, leads)
        pred = predict(models, X)

        err = simulated_weather_error(leads, seed=lead, correlated=False)
        pred_noisy = predict(models, design_matrix(data, origins, leads, temp_error=err))["p50"]

        last_day_lag = STEPS_PER_DAY * int(np.ceil(lead / STEPS_PER_DAY))
        row = {"horizon": name, "lead": lead}
        row.update({f"model_{k}": v for k, v in _scores(y, pred["p50"]).items()})
        row["model_weather_error_MAPE"] = _scores(y, pred_noisy)["MAPE"]
        row["persistence_MAPE"] = _scores(y, c[origins])["MAPE"]
        row["same_time_last_day_MAPE"] = _scores(y, c[targets - last_day_lag])["MAPE"]
        row["same_time_last_week_MAPE"] = _scores(y, c[targets - STEPS_PER_WEEK])["MAPE"]
        row["band_coverage_pct"] = float(np.mean((y >= pred["p10"]) & (y <= pred["p90"])) * 100)
        row["band_width_MW"] = float(np.mean(pred["p90"] - pred["p10"]))
        rows.append(row)

        # Error by type of target day (where are the hard cases?)
        ape = np.abs(y - pred["p50"]) / y * 100
        T = data.descriptors.iloc[targets]
        groups = {
            "Weekday": (T["is_weekend"] == 0) & (T["is_public_holiday"] == 0),
            "Weekend": T["is_weekend"] == 1,
            "Public holiday": T["is_public_holiday"] == 1,
            "Holiday eve / after / bridge": (T[["is_holiday_eve", "is_holiday_after",
                                                "is_bridge"]].sum(axis=1) > 0),
            "School holidays": T["is_school_holiday"] == 1,
            "Christmas period": T["is_xmas_period"] == 1,
        }
        for g, mask in groups.items():
            mask = mask.to_numpy()
            if mask.any():
                by_daytype.append({"horizon": name, "day_type": g,
                                   "MAPE": float(ape[mask].mean()), "n_points": int(mask.sum())})
        months = data.index[targets].month
        for m in range(1, 13):
            mask = months == m
            if mask.any():
                by_month.append({"horizon": name, "month": m, "MAPE": float(ape[mask].mean())})

    # Error as a function of the lead (subsampled origins to stay fast)
    by_lead = []
    for lead in sorted(set([1, 2, 4, 6, 8, 12, 18, 24, 36, 48] + list(range(72, MAX_LEAD + 1, 24)))):
        origins, leads = test_pairs(data, lead, every=3)
        y = c[origins + leads]
        p = predict(models, design_matrix(data, origins, leads))["p50"]
        err = simulated_weather_error(leads, seed=1000 + lead, correlated=False)
        pn = predict(models, design_matrix(data, origins, leads, temp_error=err))["p50"]
        by_lead.append({"lead": lead, "hours": lead / 2,
                        "MAPE perfect weather": float(np.mean(np.abs(y - p) / y) * 100),
                        "MAPE weather error": float(np.mean(np.abs(y - pn) / y) * 100)})

    # Feature importance (gain) of each central model
    importance = {}
    for b, bucket in enumerate(LEAD_BUCKETS):
        booster = models[b]["p50"].booster_
        gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        importance[f"{bucket[0]}-{bucket[1]} steps"] = (gain / gain.sum() * 100).sort_values(ascending=False)

    report = {
        "horizons": pd.DataFrame(rows),
        "by_daytype": pd.DataFrame(by_daytype),
        "by_month": pd.DataFrame(by_month),
        "by_lead": pd.DataFrame(by_lead),
        "importance": importance,
    }
    if verbose:
        cols = ["horizon", "model_MAPE", "model_weather_error_MAPE", "persistence_MAPE",
                "same_time_last_day_MAPE", "same_time_last_week_MAPE", "band_coverage_pct"]
        print(report["horizons"][cols].round(2).to_string(index=False))
    return report


# ===========================================================================
# 6. FORECASTING FROM A CHOSEN INSTANT
# ===========================================================================
def forecast(origin, data, models, max_lead=MAX_LEAD, weather_error=False, seed=None):
    """Forecast curve from `origin`, every 30 minutes up to `max_lead` steps.

    Returns a DataFrame indexed by target time with columns
    lead, p10, p50, p90, actual (the real value, known for historical dates)
    and temp_used (the target temperature given to the model).
    """
    pos = data.position(origin)
    if pos < STEPS_PER_WEEK:
        raise ValueError(f"Pick an origin after {data.first_origin}: "
                         "the models need one week of history.")
    leads = np.arange(1, max_lead + 1)
    leads = leads[pos + leads < len(data.index)]
    if len(leads) == 0:
        raise ValueError("No data after this instant: nothing to compare with.")

    origins = np.full(len(leads), pos)
    err = simulated_weather_error(leads, seed=seed) if weather_error else None
    X = design_matrix(data, origins, leads, temp_error=err)
    pred = predict(models, X)

    targets = pos + leads
    return pd.DataFrame({
        "lead": leads,
        "p10": pred["p10"],
        "p50": pred["p50"],
        "p90": pred["p90"],
        "actual": data.cons[targets],
        "temp_used": X["T_temp"].to_numpy(),
        "temp_observed": data.descriptors["temp"].to_numpy()[targets],
    }, index=data.index[targets])


def horizon_table(curve):
    """The reference horizons, read from a forecast curve."""
    rows = []
    for name, lead in HORIZONS.items():
        hit = curve[curve["lead"] == lead]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append({
            "horizon": name,
            "target_time": hit.index[0],
            "predicted_MW": round(float(r["p50"])),
            "low_MW": round(float(r["p10"])),
            "high_MW": round(float(r["p90"])),
            "actual_MW": round(float(r["actual"])),
            "error_pct": round(float(abs(r["p50"] - r["actual"]) / r["actual"] * 100), 2),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 7. SAVE / LOAD
# ===========================================================================
def save_bundle(models, iterations, report, path=BUNDLE_FILE):
    joblib.dump({
        "models": models,
        "iterations": iterations,
        "report": report,
        "features": FEATURE_COLUMNS,
        "lead_buckets": LEAD_BUCKETS,
        "trained_until": f"{TEST_YEAR - 1}-12-31",
    }, path)


def load_bundle(path=BUNDLE_FILE):
    bundle = joblib.load(path)
    if bundle["features"] != FEATURE_COLUMNS:
        raise RuntimeError("Saved models do not match the code: run `python src/script.py`.")
    return bundle


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    print("Loading data...")
    df = load_excel()
    df.to_parquet(CLEAN_DATA_FILE)
    print(f"  {len(df)} rows, from {df.index.min()} to {df.index.max()}")

    print("Building features...")
    data = prepare_dataset(df)

    print(f"\nTraining {len(LEAD_BUCKETS)} lead ranges x 3 models "
          f"(train < {VALID_START.date()}, validation until {TEST_YEAR - 1}-12-31):")
    models, iterations = train_models(data)

    print(f"\nEvaluation on {TEST_YEAR} (MAPE %, lower is better):")
    report = evaluate(data, models)

    save_bundle(models, iterations, report)
    print(f"\nModels and report saved in '{BUNDLE_FILE}' ({time.time() - t0:.0f}s total).")

    print("\nExample: forecast from 2023-06-27 08:30")
    print(horizon_table(forecast("2023-06-27 08:30", data, models)).to_string(index=False))


if __name__ == "__main__":
    main()
