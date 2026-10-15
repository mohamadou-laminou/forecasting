"""
Electricity Load Forecasting - Ile-de-France (2021-2023)
========================================================

This script:
  1. Loads the cleaned dataset (Excel file, 30-minute step)
  2. Builds features (calendar, cyclic encoding, lags, rolling means,
     public holidays, school holidays, extreme-weather flags)
  3. Splits the data chronologically (train: 2021-2022, test: 2023)
  4. Trains one LightGBM model per horizon (30 min, 1 h, 1 day, 7 days)
  5. Provides a `predict_from_datetime()` function: give it a date/time
     from 2023 and it returns the four forecasts next to the true values

Note on 2024:
  The dataset stops at 2023-12-31, so we cannot forecast a real 2024 date
  (the recent past needed by the short-term models does not exist for 2024).
  We therefore demonstrate on 2023, which is the held-out test year the
  models never saw during training. This is the rigorous approach.
"""

import os
import datetime as dt

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates


# ---------------------------------------------------------------------------
# Configuration - paths are resolved relative to the PROJECT ROOT, so the
# script works whether you run it from the project folder or from src/.
# ---------------------------------------------------------------------------
# This file lives in <project>/src/ ; the project root is one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "Ile_de_france_clean_dataset.xlsx")
SHEET_NAME = "donnees_idf_30min_lisse"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Horizons expressed in number of 30-minute steps
HORIZONS = {
    "30 min": 1,
    "1 hour": 2,
    "1 day": 48,
    "7 days": 336,
}

# Extreme-weather thresholds (deg C) for Ile-de-France
COLD_WAVE_THRESHOLD = 0.0     # below this = cold snap (heating peak)
HEAT_WAVE_THRESHOLD = 30.0    # above this = heat wave (cooling peak)

# School-holiday zone for Ile-de-France
SCHOOL_ZONE = "C"


# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
def load_data(path=DATA_PATH, sheet=SHEET_NAME):
    """Load the Excel file and build a clean datetime index."""
    df = pd.read_excel(path, sheet_name=sheet)

    # Rename columns to short names for convenience
    df = df.rename(columns={
        "Electric consumption (MW)": "consumption",
        "temperature (celsius)":     "temperature",
        "Gi (Solar Radiation W/m2)": "solar",
        "Windspeed (m/s)":           "wind",
    })

    # Build a single timestamp from date + time
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str)
    )

    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[["consumption", "temperature", "solar", "wind"]]
    return df


# ===========================================================================
# 2. FEATURE ENGINEERING
# ===========================================================================
def add_calendar_features(df):
    """Add basic calendar columns."""
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek          # 0 = Monday, 6 = Sunday
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_cyclic_features(df):
    """Encode cyclic time variables with sine/cosine so that, e.g.,
    hour 23 is close to hour 0."""
    def encode(series, period):
        return np.sin(2 * np.pi * series / period), np.cos(2 * np.pi * series / period)

    df["hour_sin"], df["hour_cos"] = encode(df["hour"], 24)
    df["month_sin"], df["month_cos"] = encode(df["month"], 12)
    df["dow_sin"], df["dow_cos"] = encode(df["dayofweek"], 7)
    return df


def add_holiday_features(df):
    """Add public-holiday and school-holiday flags (France, zone C)."""
    unique_dates = pd.Series(df.index.date).unique()

    # Public holidays
    public = {}
    for d in unique_dates:
        public[d] = int(JoursFeries.is_bank_holiday(d, zone="Métropole"))

    # School holidays (zone C = Paris / Ile-de-France)
    checker = SchoolHolidayDates()
    school = {}
    for d in unique_dates:
        try:
            school[d] = int(checker.is_holiday_for_zone(d, SCHOOL_ZONE))
        except Exception:
            school[d] = 0

    dates = pd.Series(df.index.date, index=df.index)
    df["is_public_holiday"] = dates.map(public).astype(int)
    df["is_school_holiday"] = dates.map(school).astype(int)
    return df


def add_extreme_weather_features(df):
    """Flag cold snaps and heat waves from the temperature column.
    These extreme episodes drive consumption up (heating / cooling)."""
    df["is_cold_wave"] = (df["temperature"] < COLD_WAVE_THRESHOLD).astype(int)
    df["is_heat_wave"] = (df["temperature"] > HEAT_WAVE_THRESHOLD).astype(int)
    return df


def add_lag_features(df):
    """Add past-consumption features. Step = 30 min, so:
    1 step = 30 min, 2 = 1 h, 48 = 24 h, 336 = 7 days."""
    df["lag_30min"] = df["consumption"].shift(1)
    df["lag_1h"]    = df["consumption"].shift(2)
    df["lag_24h"]   = df["consumption"].shift(48)
    df["lag_7d"]    = df["consumption"].shift(336)

    # Rolling means (shifted by 1 step to use past only -> no data leakage)
    df["roll_mean_2h"]  = df["consumption"].shift(1).rolling(4).mean()
    df["roll_mean_24h"] = df["consumption"].shift(1).rolling(48).mean()
    return df


def build_features(df):
    """Apply all feature-engineering steps and drop rows with missing lags."""
    df = add_calendar_features(df)
    df = add_cyclic_features(df)
    df = add_holiday_features(df)
    df = add_extreme_weather_features(df)
    df = add_lag_features(df)
    df = df.dropna()
    return df


# ===========================================================================
# 3. TRAIN / TEST SPLIT
# ===========================================================================
def chronological_split(df):
    """Train on 2021-2022, test on 2023 (no random split for time series)."""
    train = df[df.index.year < 2023]
    test = df[df.index.year == 2023]
    return train, test


# The columns given to the model as inputs (raw calendar columns replaced
# by their cyclic version are excluded).
FEATURE_COLUMNS = [
    "temperature", "solar", "wind",
    "is_weekend", "dayofyear",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_public_holiday", "is_school_holiday", "is_cold_wave", "is_heat_wave",
    "lag_30min", "lag_1h", "lag_24h", "lag_7d",
    "roll_mean_2h", "roll_mean_24h",
]


# ===========================================================================
# 4. MODEL TRAINING
# ===========================================================================
def evaluate(name, y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:22s} | MAPE = {mape:5.2f}%  | RMSE = {rmse:7.1f} MW  | R2 = {r2:.3f}")
    return {"model": name, "MAPE": mape, "RMSE": rmse, "R2": r2}


def train_one_horizon(train, test, n_steps, name):
    """Train a LightGBM model that predicts consumption n_steps ahead."""
    # Target = consumption shifted into the future
    y_train_full = train["consumption"].shift(-n_steps)
    y_test_full = test["consumption"].shift(-n_steps)

    ok_train = y_train_full.notna()
    ok_test = y_test_full.notna()

    X_train = train.loc[ok_train, FEATURE_COLUMNS]
    y_train = y_train_full[ok_train]
    X_test = test.loc[ok_test, FEATURE_COLUMNS]
    y_test = y_test_full[ok_test]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    score = evaluate(f"LightGBM {name}", y_test, pred)
    return model, score


def train_all_models(train, test):
    """Train one model per horizon and return them in a dict."""
    models = {}
    scores = []

    # Baseline: consumption 7 days earlier (same hour)
    base_pred = test["lag_7d"]
    scores.append(evaluate("Baseline (7 days)", test["consumption"], base_pred))

    print()
    for name, n_steps in HORIZONS.items():
        model, score = train_one_horizon(train, test, n_steps, name)
        models[name] = model
        scores.append(score)

    return models, pd.DataFrame(scores)


# ===========================================================================
# 5. PREDICTION FROM A CHOSEN DATE/TIME
# ===========================================================================
def predict_from_datetime(chosen, df_features, models, verbose=False):
    """
    Given a timestamp, return the four forecasts together with the true
    values (when available, so predictions can be checked).

    `chosen`      : a string like "2023-06-27 08:30" or a pandas Timestamp
    `df_features` : the full feature dataframe (output of build_features)
    `models`      : dict of trained models (output of train_all_models or
                    load_models)
    `verbose`     : if True, also print the result (used for the CLI demo)

    Returns a DataFrame with one row per horizon.
    """
    chosen = pd.Timestamp(chosen)

    if chosen not in df_features.index:
        raise ValueError(
            f"{chosen} is not in the dataset. "
            "Pick a timestamp on a 30-minute mark (e.g. 08:00 or 08:30) "
            "within 2021-2023."
        )

    # Feature row for the chosen instant
    x = df_features.loc[[chosen], FEATURE_COLUMNS]

    if verbose:
        print(f"\nForecast made from: {chosen}")
        print("-" * 55)

    results = []
    for name, n_steps in HORIZONS.items():
        target_time = chosen + pd.Timedelta(minutes=30 * n_steps)
        pred = models[name].predict(x)[0]

        # True value, if available in the data
        if target_time in df_features.index:
            true = df_features.loc[target_time, "consumption"]
            error = abs(pred - true) / true * 100
            actual_val = round(float(true))
            error_val = round(float(error), 1)
        else:
            actual_val = None
            error_val = None

        if verbose:
            true_str = f"{actual_val:8.0f} MW" if actual_val is not None else "     n/a"
            err_str = f"{error_val:5.1f}%" if error_val is not None else "  n/a"
            print(f"{name:8s} -> {target_time} | predicted: {pred:8.0f} MW | "
                  f"actual: {true_str} | error: {err_str}")

        results.append({
            "horizon": name,
            "target_time": target_time,
            "predicted_MW": round(float(pred)),
            "actual_MW": actual_val,
            "error_pct": error_val,
        })

    return pd.DataFrame(results)


# ===========================================================================
# 6. LOAD ALREADY-TRAINED MODELS (used by the Streamlit app)
# ===========================================================================
def load_models(results_dir=RESULTS_DIR):
    """Load the models and feature data saved by a previous training run.
    Returns (models_dict, df_features)."""
    models = {}
    for name in HORIZONS:
        safe = name.replace(" ", "_")
        path = os.path.join(results_dir, f"model_{safe}.pkl")
        models[name] = joblib.load(path)

    data_path = os.path.join(results_dir, "data_with_features.csv")
    df_features = pd.read_csv(data_path, index_col=0)
    df_features.index = pd.to_datetime(df_features.index)
    return models, df_features


# ===========================================================================
# 7bis. MULTI-STEP DIRECT MODELS (continuous curve without drift)
# ===========================================================================
# Grid of forecast steps: dense at the beginning (fast-changing), sparser
# further out. 336 steps = 7 days at a 30-minute resolution.
MULTISTEP_GRID = (
    list(range(1, 13))          # every 30 min for the first 6 hours
    + list(range(14, 49, 2))    # every 1 h up to 24 h
    + list(range(54, 169, 6))   # every 3 h up to 3.5 days
    + list(range(180, 337, 12)) # every 6 h up to 7 days
)


def train_multistep_models(train, grid=MULTISTEP_GRID, n_estimators=300):
    """Train one model per forecast step in the grid.

    Each model predicts directly from the present instant, so no prediction is
    ever fed back in: there is no error accumulation ('drift'), unlike the
    recursive approach.
    """
    models = {}
    for step in grid:
        y = train["consumption"].shift(-step)
        ok = y.notna()
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            verbose=-1,
        )
        model.fit(train.loc[ok, FEATURE_COLUMNS], y[ok])
        models[step] = model
    return models


def multistep_forecast(chosen, df_features, multistep_models, horizon_days=7):
    """Build a continuous forecast curve using the direct multi-step models.

    Each point is predicted directly from the chosen instant, then the curve is
    interpolated onto the 30-minute grid for a smooth line.

    Returns a pandas Series (index = future timestamps, values = predicted MW).
    """
    chosen = pd.Timestamp(chosen)
    x = df_features.loc[[chosen], FEATURE_COLUMNS]

    times, values = [], []
    for step, model in sorted(multistep_models.items()):
        target = chosen + pd.Timedelta(minutes=30 * step)
        if target in df_features.index:
            times.append(target)
            values.append(float(model.predict(x)[0]))

    if not times:
        return pd.Series(dtype=float)

    anchors = pd.Series(values, index=times)

    # Interpolate onto every 30-minute step for a smooth curve
    full_index = pd.date_range(
        chosen + pd.Timedelta(minutes=30),
        chosen + pd.Timedelta(days=horizon_days),
        freq="30min",
    )
    full_index = full_index[full_index.isin(df_features.index)]
    curve = (anchors.reindex(anchors.index.union(full_index))
                    .interpolate(method="time")
                    .reindex(full_index))
    return curve


def load_multistep_models(results_dir=RESULTS_DIR):
    """Load the multi-step models saved during training."""
    path = os.path.join(results_dir, "models_multistep.pkl")
    return joblib.load(path)


# ===========================================================================
# 7ter. WALK-FORWARD RETRAINING (expanding window)
# ===========================================================================
def walk_forward_train(cutoff, df_features, grid=MULTISTEP_GRID,
                       n_estimators=300):
    """Retrain the models using every observation up to `cutoff`.

    This mimics real operating conditions: at the moment of forecasting, all
    past data is available, not just the fixed 2021-2022 training set.

    Note: in our tests this brought almost no accuracy gain (two years already
    cover every seasonal regime), but it is methodologically cleaner.

    Returns (horizon_models, multistep_models).
    """
    cutoff = pd.Timestamp(cutoff)
    train = df_features.loc[:cutoff]

    horizon_models = {}
    for name, n_steps in HORIZONS.items():
        y = train["consumption"].shift(-n_steps)
        ok = y.notna()
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            random_state=42, verbose=-1,
        )
        model.fit(train.loc[ok, FEATURE_COLUMNS], y[ok])
        horizon_models[name] = model

    multistep_models = train_multistep_models(train, grid, n_estimators)
    return horizon_models, multistep_models


# ===========================================================================
# 7. RECURSIVE FORECAST (continuous curve, step by step)
# ===========================================================================
def recursive_forecast(chosen, df_features, models, n_steps=336):
    """
    Produce a CONTINUOUS forecast curve by predicting one 30-minute step at a
    time with the '30 min' model, feeding each prediction back in to predict
    the next step.

    Because the model consumes its own predictions, small errors accumulate
    over time ("drift"): the curve is very accurate at first and slowly
    departs from reality further out. This is expected and illustrates why
    long-range forecasting is hard.

    Weather / calendar features at each future step are taken from the data
    (we demonstrate on known 2023 dates). Only the consumption-based features
    (lags, rolling means) are rebuilt from the predictions so far.

    Returns a pandas Series (index = future timestamps, values = predicted MW).
    """
    model = models["30 min"]
    chosen = pd.Timestamp(chosen)

    # Known consumption up to and including the chosen moment
    known = df_features["consumption"].copy()

    preds = []
    times = []

    for i in range(1, n_steps + 1):
        target = chosen + pd.Timedelta(minutes=30 * i)
        if target not in df_features.index:
            break

        # Start from the real feature row at the target time (weather, calendar)
        x = df_features.loc[[target - pd.Timedelta(minutes=30)], FEATURE_COLUMNS].copy()

        # Rebuild the consumption-based features from what we know + predicted
        # so far. 'series' = real past values followed by our predictions.
        series = pd.concat([known.loc[:chosen], pd.Series(preds, index=times)])

        def val_at(delta_steps):
            t = target - pd.Timedelta(minutes=30 * delta_steps)
            return series.get(t, series.iloc[-1])

        # Overwrite the lag/rolling features with values consistent with the
        # recursive prediction (the previous instant is 1 step before target)
        x["lag_30min"] = series.iloc[-1]                       # 1 step back
        x["lag_1h"]    = val_at(2)
        x["lag_24h"]   = val_at(48)
        x["lag_7d"]    = val_at(336)
        last_values = series.iloc[-4:]
        x["roll_mean_2h"] = last_values.mean() if len(last_values) else series.iloc[-1]
        last_day = series.iloc[-48:]
        x["roll_mean_24h"] = last_day.mean() if len(last_day) else series.iloc[-1]

        p = float(model.predict(x)[0])
        preds.append(p)
        times.append(target)

    return pd.Series(preds, index=times)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows, from {df.index.min()} to {df.index.max()}")

    print("Building features...")
    df_feat = build_features(df)
    print(f"  {len(df_feat)} rows after feature engineering")

    print("Splitting (train: 2021-2022, test: 2023)...")
    train, test = chronological_split(df_feat)
    print(f"  train: {len(train)} rows | test: {len(test)} rows")

    print("\nTraining models:\n")
    models, scores = train_all_models(train, test)

    print("\nSummary:")
    print(scores.to_string(index=False))

    # Save everything
    for name, model in models.items():
        safe = name.replace(" ", "_")
        joblib.dump(model, os.path.join(RESULTS_DIR, f"model_{safe}.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(RESULTS_DIR, "feature_columns.pkl"))
    df_feat.to_csv(os.path.join(RESULTS_DIR, "data_with_features.csv"))
    print(f"\nModels and data saved in '{RESULTS_DIR}/'.")

    # ---- Multi-step models for the continuous forecast curve ----
    print(f"\nTraining {len(MULTISTEP_GRID)} multi-step models "
          "(continuous curve, no drift)... this takes about a minute.")
    multistep = train_multistep_models(train)
    joblib.dump(multistep, os.path.join(RESULTS_DIR, "models_multistep.pkl"))
    print("Multi-step models saved.")

    # ---- Example prediction ----
    print("\n" + "=" * 55)
    print("EXAMPLE: forecast from 2023-06-27 08:30")
    print("=" * 55)
    predict_from_datetime("2023-06-27 08:30", df_feat, models, verbose=True)

    return df_feat, models


if __name__ == "__main__":
    main()