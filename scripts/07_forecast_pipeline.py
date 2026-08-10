"""Fetch Open-Meteo weather forecasts and create the next three AQI predictions."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_CSV = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LABEL_ENCODING_PATH = OUTPUTS_DIR / "label_encoding.json"
OUTPUT_JSON = OUTPUTS_DIR / "forecast_latest.json"
LATITUDE, LONGITUDE = 14.987471, 102.117965
TIMEZONE = "Asia/Bangkok"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FEATURES = [
    "pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure", "windspeed_10m", "winddirection_10m",
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_roll_mean3", "pm25_roll_mean7", "pm25_trend3",
    "temp_lag1", "temp_lag2", "temp_lag3", "temp_roll_mean3", "temp_roll_mean7",
    "humidity_lag1", "humidity_lag2", "humidity_lag3", "humidity_roll_mean3", "humidity_roll_mean7",
    "pressure_lag1", "pressure_trend3",
    "windspeed_lag1", "windspeed_lag2", "windspeed_lag3", "windspeed_roll_mean3", "windspeed_roll_mean7",
    "wind_from_burning_sector", "wind_direction_sin", "wind_direction_cos",
    "aqi_category_lag1", "day_of_year", "month", "month_sin", "month_cos", "stagnation_score",
]


def load_label_mapping() -> dict[str, int]:
    """Load the fixed mapping from text labels to ordinal integers."""
    with LABEL_ENCODING_PATH.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {label: int(value) for label, value in mapping.items()}


def circular_mean_degrees(values: pd.Series) -> float:
    """Return the circular mean of wind direction readings in degrees (0-360)."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    radians = np.deg2rad(clean.to_numpy())
    mean_angle = np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float((mean_angle + 360.0) % 360.0)


def load_history() -> pd.DataFrame:
    """Load recent chronological rows from cleaned data for lag and rolling calculations."""
    if not CLEANED_CSV.exists():
        raise FileNotFoundError("Run 01_data_cleaning.py first; it supplies the latest historical PM2.5.")
    history = pd.read_csv(CLEANED_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if len(history) < 7:
        raise ValueError("Need at least 7 historical rows to build lag/rolling PM2.5 features.")
    return history.tail(30).reset_index(drop=True)


def _zscore(series: pd.Series) -> pd.Series:
    """Return a zero-mean, unit-variance z-score series with a safe fallback for constant inputs."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def add_feature_columns(history: pd.DataFrame) -> pd.DataFrame:
    """Add PM2.5, weather, wind, and seasonal features using row-order chronology."""
    data = history.copy()

    for lag in [1, 2, 3]:
        data[f"pm25_lag{lag}"] = data["pm25"].shift(lag)
        data[f"temp_lag{lag}"] = data["temperature_2m"].shift(lag)
        data[f"humidity_lag{lag}"] = data["relative_humidity_2m"].shift(lag)
        data[f"windspeed_lag{lag}"] = data["windspeed_10m"].shift(lag)
    for window in [3, 7]:
        data[f"pm25_roll_mean{window}"] = data["pm25"].rolling(window=window, min_periods=1).mean()
        data[f"temp_roll_mean{window}"] = data["temperature_2m"].rolling(window=window, min_periods=1).mean()
        data[f"humidity_roll_mean{window}"] = data["relative_humidity_2m"].rolling(window=window, min_periods=1).mean()
        data[f"windspeed_roll_mean{window}"] = data["windspeed_10m"].rolling(window=window, min_periods=1).mean()

    data["pm25_trend3"] = data["pm25"] - data["pm25_lag3"]
    data["pressure_lag1"] = data["surface_pressure"].shift(1)
    data["pressure_trend3"] = data["surface_pressure"] - data["surface_pressure"].shift(3)
    data["aqi_category_lag1"] = data["aqi_category"].shift(1)
    data["wind_from_burning_sector"] = ((data["winddirection_10m"] >= 0) & (data["winddirection_10m"] <= 90)).astype(int)
    data["wind_direction_sin"] = np.sin(np.deg2rad(data["winddirection_10m"]))
    data["wind_direction_cos"] = np.cos(np.deg2rad(data["winddirection_10m"]))
    data["day_of_year"] = data["date"].dt.dayofyear
    data["month"] = data["date"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    low_wind = -_zscore(data["windspeed_roll_mean3"])
    high_pressure = _zscore(data["surface_pressure"])
    low_temp_trend = -_zscore(data["temperature_2m"] - data["temperature_2m"].shift(3))
    data["stagnation_score"] = (low_wind + high_pressure + low_temp_trend) / 3.0
    return data


def fetch_daily_weather() -> pd.DataFrame:
    """Fetch hourly forecast data and average it into the next three local days."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
        "forecast_days": 4,
        "timezone": TIMEZONE,
    }
    print("Requesting forecast weather from Open-Meteo FORECAST API...")
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=30)
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        weather = pd.DataFrame(hourly)
        if weather.empty:
            raise ValueError("Open-Meteo returned no hourly forecast values.")
        weather = weather.rename(columns={"wind_speed_10m": "windspeed_10m", "wind_direction_10m": "winddirection_10m"})
        weather["time"] = pd.to_datetime(weather["time"])
        weather["date"] = weather["time"].dt.date
        daily = weather.groupby("date", as_index=False).agg(
            temperature_2m=("temperature_2m", "mean"),
            relative_humidity_2m=("relative_humidity_2m", "mean"),
            surface_pressure=("surface_pressure", "mean"),
            windspeed_10m=("windspeed_10m", "mean"),
            winddirection_10m=("winddirection_10m", circular_mean_degrees),
        )
        tomorrow = datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=1)
        wanted_dates = [tomorrow + timedelta(days=offset) for offset in range(3)]
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily[daily["date"].dt.date.isin(wanted_dates)].sort_values("date").reset_index(drop=True)
        if len(daily) == 3:
            return daily
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"Forecast API unavailable or incomplete ({exc}); using recent historical weather as fallback.")

    history = pd.read_csv(CLEANED_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    latest = history.iloc[-1].copy()
    tomorrow = datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=1)
    fallback_rows = []
    for offset in range(3):
        target_date = tomorrow + timedelta(days=offset)
        fallback_rows.append({
            "date": pd.Timestamp(target_date),
            "temperature_2m": float(latest["temperature_2m"]),
            "relative_humidity_2m": float(latest["relative_humidity_2m"]),
            "surface_pressure": float(latest["surface_pressure"]),
            "windspeed_10m": float(latest["windspeed_10m"]),
            "winddirection_10m": float(latest["winddirection_10m"]),
        })
    return pd.DataFrame(fallback_rows).sort_values("date").reset_index(drop=True)


def _recent_window(history: pd.DataFrame, weather: pd.DataFrame, horizon: int, column: str) -> list[float]:
    """Build a causal sequence of available values for a given feature and forecast horizon."""
    values = history[column].tolist()
    for offset in range(horizon - 1):
        values.append(float(weather.iloc[offset][column]))
    return values


def build_feature_row(history: pd.DataFrame, weather: pd.DataFrame, forecast_day: pd.Series, horizon: int, persisted_pm25: float, label_map: dict[str, int]) -> pd.Series:
    """Construct one feature row using real weather forecasts for wind/temp/humidity, while PM2.5 remains a persistence approximation."""
    actual_latest = history.iloc[-1]
    recent_temp = _recent_window(history, weather, horizon, "temperature_2m")
    recent_humidity = _recent_window(history, weather, horizon, "relative_humidity_2m")
    recent_wind = _recent_window(history, weather, horizon, "windspeed_10m")
    recent_pressure = _recent_window(history, weather, horizon, "surface_pressure")

    base = {
        "pm25": float(actual_latest["pm25"] if horizon == 1 else persisted_pm25),
        "temperature_2m": float(forecast_day["temperature_2m"]),
        "relative_humidity_2m": float(forecast_day["relative_humidity_2m"]),
        "surface_pressure": float(forecast_day["surface_pressure"]),
        "windspeed_10m": float(forecast_day["windspeed_10m"]),
        "winddirection_10m": float(forecast_day["winddirection_10m"]),
        "day_of_year": int(forecast_day["date"].dayofyear),
        "month": int(forecast_day["date"].month),
        "month_sin": float(np.sin(2 * np.pi * forecast_day["date"].month / 12)),
        "month_cos": float(np.cos(2 * np.pi * forecast_day["date"].month / 12)),
        "pm25_lag1": float(actual_latest["pm25_lag1"] if pd.notna(actual_latest["pm25_lag1"]) else persisted_pm25),
        "pm25_lag2": float(actual_latest["pm25_lag2"] if pd.notna(actual_latest["pm25_lag2"]) else persisted_pm25),
        "pm25_lag3": float(actual_latest["pm25_lag3"] if pd.notna(actual_latest["pm25_lag3"]) else persisted_pm25),
        "pm25_roll_mean3": float(actual_latest["pm25_roll_mean3"] if pd.notna(actual_latest["pm25_roll_mean3"]) else persisted_pm25),
        "pm25_roll_mean7": float(actual_latest["pm25_roll_mean7"] if pd.notna(actual_latest["pm25_roll_mean7"]) else persisted_pm25),
        "pm25_trend3": float(actual_latest["pm25_trend3"] if pd.notna(actual_latest["pm25_trend3"]) else 0.0),
        "temp_lag1": float(recent_temp[-1]),
        "temp_lag2": float(recent_temp[-2]) if len(recent_temp) >= 2 else float(recent_temp[-1]),
        "temp_lag3": float(recent_temp[-3]) if len(recent_temp) >= 3 else float(recent_temp[-1]),
        "temp_roll_mean3": float(np.mean(recent_temp[-3:])),
        "temp_roll_mean7": float(np.mean(recent_temp[-7:])),
        "humidity_lag1": float(recent_humidity[-1]),
        "humidity_lag2": float(recent_humidity[-2]) if len(recent_humidity) >= 2 else float(recent_humidity[-1]),
        "humidity_lag3": float(recent_humidity[-3]) if len(recent_humidity) >= 3 else float(recent_humidity[-1]),
        "humidity_roll_mean3": float(np.mean(recent_humidity[-3:])),
        "humidity_roll_mean7": float(np.mean(recent_humidity[-7:])),
        "pressure_lag1": float(recent_pressure[-1]),
        "pressure_trend3": float(forecast_day["surface_pressure"] - recent_pressure[-3]) if len(recent_pressure) >= 3 else 0.0,
        "windspeed_lag1": float(recent_wind[-1]),
        "windspeed_lag2": float(recent_wind[-2]) if len(recent_wind) >= 2 else float(recent_wind[-1]),
        "windspeed_lag3": float(recent_wind[-3]) if len(recent_wind) >= 3 else float(recent_wind[-1]),
        "windspeed_roll_mean3": float(np.mean(recent_wind[-3:])),
        "windspeed_roll_mean7": float(np.mean(recent_wind[-7:])),
        "wind_from_burning_sector": int((0 <= float(forecast_day["winddirection_10m"]) <= 90)),
        "wind_direction_sin": float(np.sin(np.deg2rad(float(forecast_day["winddirection_10m"])))),
        "wind_direction_cos": float(np.cos(np.deg2rad(float(forecast_day["winddirection_10m"])))),
        "aqi_category_lag1": int(label_map.get(str(actual_latest["aqi_category"]), 0)),
    }

    if horizon == 1:
        print("Day +1 features: real historical PM2.5 + real forecast weather; all weather lags are causal and observed/forecasted values.")
    else:
        print(f"Day +{horizon} features: wind/temp/humidity are real forecast values; PM2.5-derived features remain persisted approximations.")
        base["pm25"] = persisted_pm25
        base["pm25_lag1"] = persisted_pm25
        base["pm25_lag2"] = persisted_pm25
        base["pm25_lag3"] = persisted_pm25
        base["pm25_roll_mean3"] = persisted_pm25
        base["pm25_roll_mean7"] = persisted_pm25
        base["pm25_trend3"] = 0.0

    low_wind = -_zscore(pd.Series([base["windspeed_roll_mean3"], float(history["windspeed_10m"].mean())]))
    high_pressure = _zscore(pd.Series([base["surface_pressure"], float(history["surface_pressure"].mean())]))
    low_temp_trend = -_zscore(pd.Series([base["temperature_2m"] - base["temp_lag3"], float((history["temperature_2m"] - history["temperature_2m"].shift(3)).mean())]))
    base["stagnation_score"] = float((low_wind.iloc[0] + high_pressure.iloc[0] + low_temp_trend.iloc[0]) / 3.0)
    return pd.Series(base)


def main() -> None:
    """Combine forecast weather with a persistence-based PM2.5 approximation and write AQI forecasts."""
    if not CLEANED_CSV.exists():
        raise FileNotFoundError("Run 01_data_cleaning.py first; it supplies the latest observed PM2.5.")

    history = load_history()
    history = add_feature_columns(history)
    latest_pm25 = float(history["pm25"].iloc[-1])
    latest_category = str(history["aqi_category"].iloc[-1])
    label_map = load_label_mapping()
    reverse_map = {value: label for label, value in label_map.items()}
    print(f"Using latest observed PM2.5 ({latest_pm25:.2f} µg/m³) as the persistence anchor.")
    print(f"Latest observed AQI category: {latest_category}")

    weather = fetch_daily_weather()
    output = {}
    for horizon in range(1, 4):
        model_path = MODELS_DIR / f"best_day{horizon}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}; run 05_train_model.py first.")

        forecast_day = weather.iloc[horizon - 1]
        feature_row = build_feature_row(history, weather, forecast_day, horizon, latest_pm25, label_map)
        feature_row = feature_row[FEATURES]

        model = joblib.load(model_path)
        prediction = int(model.predict(feature_row.to_frame().T)[0])

        confidence = "high (wind/temp/humidity forecasted; PM2.5 persistence remains low)" if horizon > 1 else "high"
        output[f"day{horizon}"] = {
            "date": str(forecast_day["date"].date()),
            "predicted_aqi_category": reverse_map.get(prediction, str(prediction)),
            "feature_confidence": confidence,
        }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved latest forecast: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
