# Electricity load forecasting - Ile-de-France

Forecasts the electricity consumption of Ile-de-France from 30 minutes up to 7 days
ahead, at a 30-minute resolution, with an uncertainty band. Data: RTE (consumption)
and PVGIS (weather), 2021-2023.

## Results on 2023 (test year, never seen during training)

MAPE (mean absolute percentage error, lower is better):

| Horizon | Model | Model, realistic weather forecast error | Best naive baseline | Actual inside 80% band |
|---|---|---|---|---|
| 30 min | **0.88 %** | 0.89 % | 2.31 % | 82.6 % |
| 1 hour | **1.07 %** | 1.08 % | 4.31 % | 81.9 % |
| 1 day | **2.80 %** | 3.06 % | 5.66 % | 79.9 % |
| 7 days | **4.49 %** | 6.93 % | 7.98 % | 81.1 % |

"Model" uses the observed weather at the target time, as a perfect weather forecast
would. The next column adds a simulated temperature-forecast error (about 1 C at day 1,
3 C at day 7), which is closer to real operation.

## Data

- **Consumption:** RTE éCO2mix, Île-de-France regional files, 2021-2023, 30-minute step
  (52,560 points). Published in French local time with 48 half-hours every day (RTE fills
  the missing spring hour and keeps the repeated autumn hour once).
- **Weather:** PVGIS (ERA5 / SARAH3), hourly, in UTC: converted to local time and
  interpolated linearly to 30 minutes. Temperature, solar irradiance, wind speed.
- **Calendar:** French public holidays and school holidays (zone C).
- Île-de-France was chosen over national data so that one regional temperature series
  matches the load. The regional files have no RTE day-ahead forecast, so the model is
  compared with naive baselines.

Main consumption patterns: winter working days peak at 19:00 (11.3 GW) and bottom out at
04:00 (7.8 GW); weekends are 10 % below the working days of the same week; public holidays
13 % below the same weekday; January is 1.85 times August; daily mean load and temperature
are correlated at r = -0.87.

## Method

- **Direct multi-horizon forecasting**: every point of the 7-day curve is predicted
  straight from the forecast origin, with the lead as an input. Nothing is fed back,
  so errors do not accumulate. Five LightGBM model families each cover a range of
  leads (30 min-1 h, 1.5-6 h, 6.5-24 h, 1-3 days, 3-7 days).
- **Features describing the target time**: time of day, weekday, season, public
  holidays, holiday eves and days after, bridge days, school holidays (zone C),
  Christmas period, August, temperature, temperature smoothed over ~2 days (thermal
  inertia of buildings), heating / cooling degrees, solar radiation, wind.
- **Features describing the origin**: current consumption, means over 2 h and 24 h,
  last known value at the same time of day and of week, current temperature.
- **Ratio target**: the models predict target / recent level, which makes them robust
  to level shifts such as the energy-saving plan of winter 2022-2023 (-5 to -10 %).
- **Uncertainty band**: 10% and 90% quantile models, calibrated on the validation
  period (conformal calibration) so that about 80% of real values fall inside.
- **Chronological split**: train 2021-01 to 2022-09, validation 2022-10 to 2022-12
  (early stopping, band calibration), test 2023.

## Usage

```bash
python -m venv venv
venv\Scripts\activate            # Windows (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt

python src/script.py             # trains the models (~6 min), writes results/
streamlit run src/app.py         # opens the app
```

The trained models are not versioned: run `python src/script.py` once after cloning.

## App

- **Forecast**: pick an instant of 2023, get the forecast curve with its band,
  compared with reality; perfect or realistic weather; optional walk-forward
  retraining on all data up to the chosen instant.
- **History explorer**: consumption and temperature over any period, temperature
  sensitivity, daily profiles.
- **Model performance**: scores against baselines, error by lead, by day type and by
  month, feature importance.

## Project structure

```
dataset/                 cleaned source data (Excel)
src/script.py            features, training, evaluation, forecasting
src/app.py               Streamlit entry point
src/common.py            shared app helpers (cached loading, chart styling)
src/app_pages/           the three app pages
report/make_figures.py   figures used in the internship report
results/                 generated: clean data, models and report (not versioned)
```

## Limits

- Demonstration on historical data: in operation, the weather at the target time would
  come from a real weather forecast (e.g. Meteo-France, Open-Meteo).
- Public holidays remain the hardest days (~11 per year to learn from).
- Walk-forward retraining with 25 % more data (up to June 2023) improved the July-December
  2023 MAPE by only 0.01 to 0.09 point.
