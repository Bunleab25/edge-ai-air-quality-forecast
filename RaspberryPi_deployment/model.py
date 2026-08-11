"""
AeroSense RMUTI — Flask Inference API (45-feature, 3-day forecast)
====================================================================
Runs on the Raspberry Pi (edge server). Loads THREE trained RandomForest
models (rf_day1.pkl, rf_day2.pkl, rf_day3.pkl), each expecting the full
45-feature set reported by model.feature_names_in_:

  pm25, temperature_2m, relative_humidity_2m, surface_pressure,
  month_sin, month_cos, dow_sin, dow_cos,
  pm25_lag1..7, hum_lag1..7, temp_lag1..7,
  pm25_roll_mean3/7, hum_roll_mean3/7,
  pm25_roll_std3/7, pm25_roll_max3/7,
  pm25_ewma3/7,
  pm25_delta1, hum_delta1, press_delta1, temp_delta1,
  pm25_momentum_3d, pm25_momentum_7d


"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "."
MODEL_PATHS = {
    "day1": os.path.join(MODEL_DIR, "rf_day1.pkl"),
    "day2": os.path.join(MODEL_DIR, "rf_day2.pkl"),
    "day3": os.path.join(MODEL_DIR, "rf_day3.pkl"),
}

HISTORY_LOG_PATH = "live_sensor_log.csv"

# Need 7 full prior days for lag7/roll7/ewma7 — keep a comfortable buffer.
MAX_HISTORY = 21

RMUTI_LAT, RMUTI_LON = 14.987471, 102.117965


# ============================================================
# 1. Load the three trained models once at startup
# ============================================================
models = {}
for horizon, path in MODEL_PATHS.items():
    try:
        models[horizon] = joblib.load(path)
        n_feat = getattr(models[horizon], "n_features_in_", "unknown")
        print(f"Loaded {horizon} model from {path} (expects {n_feat} features)")
    except FileNotFoundError:
        print(f"ERROR: Cannot find {path}. Ensure it is in the same folder.")
        models[horizon] = None


# ============================================================
# 2. History buffer (CSV-backed) for lag/rolling/ewma features
# ============================================================
def load_history():
    try:
        df = pd.read_csv(HISTORY_LOG_PATH, parse_dates=["date"])
        df = df.sort_values("date").tail(MAX_HISTORY).reset_index(drop=True)
        print(f"Pre-loaded {len(df)} historical rows from {HISTORY_LOG_PATH}")
        return df
    except FileNotFoundError:
        print(f"WARNING: {HISTORY_LOG_PATH} not found. Starting with an empty buffer.")
        return pd.DataFrame(columns=["date", "pm25", "temperature_2m",
                                      "relative_humidity_2m", "surface_pressure"])
    except Exception as e:
        print(f"WARNING: could not read history CSV ({e}). Starting with an empty buffer.")
        return pd.DataFrame(columns=["date", "pm25", "temperature_2m",
                                      "relative_humidity_2m", "surface_pressure"])


history_df = load_history()


def append_and_save(row: dict):
    global history_df
    
    # Convert row date to datetime object for comparison
    row_date = pd.to_datetime(row["date"])
    
    # Check if today's date is already in the history buffer
    if not history_df.empty and row_date in history_df["date"].values:
        # Update the existing row for today with the latest sensor readings
        idx = history_df.index[history_df['date'] == row_date].tolist()[0]
        for key, val in row.items():
            history_df.at[idx, key] = val
    else:
        # It's a new day, append as a new row
        history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
        
    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df = history_df.sort_values("date").tail(MAX_HISTORY).reset_index(drop=True)
    
    try:
        history_df.to_csv(HISTORY_LOG_PATH, index=False)
    except Exception as e:
        print(f"WARNING: failed to persist history log ({e})")


# ============================================================
# 3. Fetch weather forecast from Open-Meteo for day+2/+3
# ============================================================
def fetch_forecast_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": RMUTI_LAT,
        "longitude": RMUTI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure",
        "forecast_days": 4,
        "timezone": "Asia/Bangkok",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    df = pd.DataFrame(hourly)
    df["date"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")

    daily = {}
    for date, group in df.groupby("date"):
        daily[date] = {
            "temperature_2m": group["temperature_2m"].mean(),
            "relative_humidity_2m": group["relative_humidity_2m"].mean(),
            "surface_pressure": group["surface_pressure"].mean(),
        }
    return daily


# ============================================================
# 4. Feature builder — reproduces the 45-feature schema
# ============================================================
def safe_lag(series: pd.Series, i: int):
    """Return the value i steps back from the end, or the earliest
    available value if history is shorter than i (cold-boot fallback)."""
    if len(series) > i:
        return float(series.iloc[-1 - i])
    if len(series) > 0:
        return float(series.iloc[0])
    return 0.0


def build_feature_row(df_with_target_row: pd.DataFrame):
    """
    df_with_target_row: history rows (ascending by date) with the target
    day's own values as the LAST row (real for day+1, forecasted/persisted
    for day+2/+3). Returns a dict of all 45 features for that last row.
    """
    s_pm = df_with_target_row["pm25"].astype(float).reset_index(drop=True)
    s_hum = df_with_target_row["relative_humidity_2m"].astype(float).reset_index(drop=True)
    s_temp = df_with_target_row["temperature_2m"].astype(float).reset_index(drop=True)
    s_press = df_with_target_row["surface_pressure"].astype(float).reset_index(drop=True)
    target_date = pd.to_datetime(df_with_target_row["date"].iloc[-1])

    row = {
        "pm25": float(s_pm.iloc[-1]),
        "temperature_2m": float(s_temp.iloc[-1]),
        "relative_humidity_2m": float(s_hum.iloc[-1]),
        "surface_pressure": float(s_press.iloc[-1]),
    }

    month = target_date.month
    dow = target_date.weekday()  # Monday=0
    row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * month / 12))
    row["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    row["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))

    for i in range(1, 8):
        row[f"pm25_lag{i}"] = safe_lag(s_pm, i)
        row[f"hum_lag{i}"] = safe_lag(s_hum, i)
        row[f"temp_lag{i}"] = safe_lag(s_temp, i)

    def window(series, w):
        return series.iloc[-w:] if len(series) >= w else series

    row["pm25_roll_mean3"] = float(window(s_pm, 3).mean())
    row["pm25_roll_mean7"] = float(window(s_pm, 7).mean())
    row["hum_roll_mean3"] = float(window(s_hum, 3).mean())
    row["hum_roll_mean7"] = float(window(s_hum, 7).mean())
    row["pm25_roll_std3"] = float(window(s_pm, 3).std(ddof=0))
    row["pm25_roll_std7"] = float(window(s_pm, 7).std(ddof=0))
    row["pm25_roll_max3"] = float(window(s_pm, 3).max())
    row["pm25_roll_max7"] = float(window(s_pm, 7).max())

    row["pm25_ewma3"] = float(s_pm.ewm(span=3, adjust=False).mean().iloc[-1])
    row["pm25_ewma7"] = float(s_pm.ewm(span=7, adjust=False).mean().iloc[-1])

    row["pm25_delta1"] = float(s_pm.iloc[-1] - safe_lag(s_pm, 1))
    row["hum_delta1"] = float(s_hum.iloc[-1] - safe_lag(s_hum, 1))
    row["press_delta1"] = float(s_press.iloc[-1] - safe_lag(s_press, 1))
    row["temp_delta1"] = float(s_temp.iloc[-1] - safe_lag(s_temp, 1))

    row["pm25_momentum_3d"] = float(s_pm.iloc[-1] - safe_lag(s_pm, 3))
    row["pm25_momentum_7d"] = float(s_pm.iloc[-1] - safe_lag(s_pm, 7))

    return row


# ============================================================
# ROUTES
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": {k: v is not None for k, v in models.items()},
        "history_rows": len(history_df),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON like:
    {
        "pm25": 42.3,
        "temperature": 28.5,
        "humidity": 65,
        "pressure": 995
    }
    Returns predicted AQI category for day+1, day+2, day+3.
    """
    if any(m is None for m in models.values()):
        return jsonify({"error": "One or more models failed to load on server."}), 500

    try:
        data = request.get_json()
        pm25 = float(data["pm25"])
        temp = float(data["temperature"])
        humidity = float(data["humidity"])
        pressure = float(data["pressure"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing input: {e}"}), 400

    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    # --- Append today's REAL observation to history, then build day+1 row ---
    append_and_save({
        "date": today_str,
        "pm25": pm25,
        "temperature_2m": temp,
        "relative_humidity_2m": humidity,
        "surface_pressure": pressure,
    })

    row_day1 = build_feature_row(history_df)

    # --- Pull live forecast for day+2/day+3 weather ---
    forecast_error = None
    forecast_by_date = {}
    try:
        forecast_by_date = fetch_forecast_weather()
    except requests.exceptions.RequestException as e:
        forecast_error = str(e)
        print(f"WARNING: forecast fetch failed ({e}); "
              f"day+2/+3 will reuse today's weather as fallback.")

    def forecast_or_fallback(date_str, key, fallback):
        if date_str in forecast_by_date:
            return forecast_by_date[date_str][key]
        return fallback

    date_day2 = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    date_day3 = (today + timedelta(days=3)).strftime("%Y-%m-%d")

    # Day+2 pseudo-row: real forecasted weather, PM2.5 persisted from today
    df_day2 = pd.concat([history_df, pd.DataFrame([{
        "date": date_day2,
        "pm25": pm25,  # persistence — true future PM2.5 is unknown
        "temperature_2m": forecast_or_fallback(date_day2, "temperature_2m", temp),
        "relative_humidity_2m": forecast_or_fallback(date_day2, "relative_humidity_2m", humidity),
        "surface_pressure": forecast_or_fallback(date_day2, "surface_pressure", pressure),
    }])], ignore_index=True)
    row_day2 = build_feature_row(df_day2)

    # Day+3 pseudo-row: builds on the day+2 pseudo-row
    df_day3 = pd.concat([df_day2, pd.DataFrame([{
        "date": date_day3,
        "pm25": pm25,  # persistence
        "temperature_2m": forecast_or_fallback(date_day3, "temperature_2m", temp),
        "relative_humidity_2m": forecast_or_fallback(date_day3, "relative_humidity_2m", humidity),
        "surface_pressure": forecast_or_fallback(date_day3, "surface_pressure", pressure),
    }])], ignore_index=True)
    row_day3 = build_feature_row(df_day3)

    confidence_notes = {
        "day1": "high",
        "day2": "mixed (weather=forecasted/real, pm25-derived=persisted/low)"
                if not forecast_error else "low (forecast unavailable, used fallback)",
        "day3": "mixed (weather=forecasted/real, pm25-derived=persisted/low)"
                if not forecast_error else "low (forecast unavailable, used fallback)",
    }

    # ------------------------------------------------------------
    # Run each horizon's model on its matching feature row, using
    # that model's own expected column order/names.
    # ------------------------------------------------------------
    results = {}
    for horizon, row in [("day1", row_day1), ("day2", row_day2), ("day3", row_day3)]:
        model = models[horizon]
        expected_cols = list(model.feature_names_in_)
        try:
            X = pd.DataFrame([row])[expected_cols]
        except KeyError as e:
            return jsonify({
                "error": f"Feature mismatch building {horizon} row: missing {e}. "
                         f"Model expects: {expected_cols}"
            }), 500

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        class_probs = dict(zip(model.classes_, proba.round(3)))
        results[horizon] = {
            "predicted_category": pred,
            "confidence": round(float(max(proba)), 3),
            "class_probabilities": class_probs,
            "feature_confidence": confidence_notes[horizon],
        }

    return jsonify({
        "forecast": results,
        "forecast_fetch_error": forecast_error,
        "input_received": {
            "pm25": pm25,
            "temperature": temp,
            "humidity": humidity,
            "pressure": pressure,
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
