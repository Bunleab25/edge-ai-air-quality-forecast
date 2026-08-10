"""Build horizon-specific future AQI labels from cleaned daily observations."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/model_ready/train_test_shifted.csv"


def _zscore(series: pd.Series) -> pd.Series:
    """Return a zero-mean, unit-variance z-score series with a safe fallback for constant inputs."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def main() -> None:
    """Add lag/rolling/trend/seasonal features and shift AQI labels one to three rows ahead."""
    print(f"Loading cleaned data: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError("Run 01_data_cleaning.py first.")

    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    print("Adding PM2.5-derived lag/rolling/trend features...")
    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm25_lag2"] = df["pm25"].shift(2)
    df["pm25_lag3"] = df["pm25"].shift(3)
    df["pm25_roll_mean3"] = df["pm25"].rolling(window=3, min_periods=1).mean()
    df["pm25_roll_mean7"] = df["pm25"].rolling(window=7, min_periods=1).mean()
    rolling_count_7 = df["pm25"].rolling(window=7, min_periods=1).count()
    incomplete_rolling_rows = int((rolling_count_7 < 7).sum())
    print(f"Rows with incomplete 7-day rolling PM2.5 windows: {incomplete_rolling_rows}")
    df["pm25_trend3"] = df["pm25"] - df["pm25_lag3"]

    print("Adding temperature, humidity, pressure, wind, and circular-direction features...")
    for lag in [1, 2, 3]:
        df[f"temp_lag{lag}"] = df["temperature_2m"].shift(lag)
        df[f"humidity_lag{lag}"] = df["relative_humidity_2m"].shift(lag)
        df[f"windspeed_lag{lag}"] = df["windspeed_10m"].shift(lag)
    for window in [3, 7]:
        df[f"temp_roll_mean{window}"] = df["temperature_2m"].rolling(window=window, min_periods=1).mean()
        df[f"humidity_roll_mean{window}"] = df["relative_humidity_2m"].rolling(window=window, min_periods=1).mean()
        df[f"windspeed_roll_mean{window}"] = df["windspeed_10m"].rolling(window=window, min_periods=1).mean()

    df["pressure_lag1"] = df["surface_pressure"].shift(1)
    df["pressure_trend3"] = df["surface_pressure"] - df["surface_pressure"].shift(3)

    # Default NE-to-E sector (0-90°) for a common biomass-burning transport direction in Thailand.
    # If local fire-location data become available, this sector should be validated and adjusted.
    df["wind_from_burning_sector"] = ((df["winddirection_10m"] >= 0) & (df["winddirection_10m"] <= 90)).astype(int)
    df["wind_direction_sin"] = np.sin(np.deg2rad(df["winddirection_10m"]))
    df["wind_direction_cos"] = np.cos(np.deg2rad(df["winddirection_10m"]))

    # Stagnation score is intentionally simple and uses z-scored daily conditions: weak wind,
    # elevated pressure, and a cooling/flat temperature trend indicate a more stagnant air mass.
    low_wind = -_zscore(df["windspeed_roll_mean3"])
    high_pressure = _zscore(df["surface_pressure"])
    low_temp_trend = -_zscore(df["temperature_2m"] - df["temperature_2m"].shift(3))
    df["stagnation_score"] = (low_wind + high_pressure + low_temp_trend) / 3.0

    label_map = {"Very Good": 0, "Good": 1, "Moderate": 2, "Unhealthy": 3}
    df["aqi_category_lag1"] = df["aqi_category"].shift(1).map(label_map)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    feature_columns = [
        "pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure", "windspeed_10m", "winddirection_10m",
        "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_roll_mean3", "pm25_roll_mean7", "pm25_trend3",
        "temp_lag1", "temp_lag2", "temp_lag3", "temp_roll_mean3", "temp_roll_mean7",
        "humidity_lag1", "humidity_lag2", "humidity_lag3", "humidity_roll_mean3", "humidity_roll_mean7",
        "pressure_lag1", "pressure_trend3",
        "windspeed_lag1", "windspeed_lag2", "windspeed_lag3", "windspeed_roll_mean3", "windspeed_roll_mean7",
        "wind_from_burning_sector", "wind_direction_sin", "wind_direction_cos",
        "aqi_category_lag1", "day_of_year", "month", "month_sin", "month_cos", "stagnation_score",
    ]
    print(f"Feature columns before dropping leading NaN rows: {feature_columns}")

    lead_nan_cols = [
        "pm25_lag1", "pm25_lag2", "pm25_lag3",
        "pm25_trend3",
        "temp_lag1", "temp_lag2", "temp_lag3",
        "humidity_lag1", "humidity_lag2", "humidity_lag3",
        "pressure_lag1", "pressure_trend3",
        "windspeed_lag1", "windspeed_lag2", "windspeed_lag3",
        "aqi_category_lag1",
    ]
    rows_before_drop = len(df)
    df = df.dropna(subset=lead_nan_cols).copy()
    print(f"Dropped {rows_before_drop - len(df)} leading rows with NaN lag/trend features.")

    for horizon in range(1, 4):
        df[f"label_day{horizon}"] = df["aqi_category"].shift(-horizon)

    original_rows = len(df)
    df = df.dropna(subset=["label_day1", "label_day2", "label_day3"]).copy()
    print(f"Dropped {original_rows - len(df)} final row(s) without all future targets.")

    final_feature_columns = feature_columns
    print(f"Final feature columns: {final_feature_columns}")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} model-ready rows to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
