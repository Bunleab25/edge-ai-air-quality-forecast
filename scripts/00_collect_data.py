"""Collect daily PM2.5 and weather inputs for the AeroSense RMUTI dataset."""
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data/raw/rmuti_dataset_2023_2026_final.csv"
METADATA_JSON = PROJECT_ROOT / "data/raw/dataset_metadata.json"
OPENAQ_SENSOR_ID = 1304112
OPENAQ_STATION_ID = 225602
OPENAQ_DAYS_URL = f"https://api.openaq.org/v3/sensors/{OPENAQ_SENSOR_ID}/days"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE, LONGITUDE = 14.987471, 102.117965
START_DATE = date(2023, 1, 1)
MAX_ARCHIVE_END_DATE = date(2026, 8, 8)
TIMEZONE = "Asia/Bangkok"


def pm25_to_category(pm25: float) -> str:
    """Map PM2.5 to the project's four-class Thai PCD simplification."""
    if pm25 <= 15:
        return "Very Good"
    if pm25 <= 25:
        return "Good"
    if pm25 <= 37.5:
        return "Moderate"
    return "Unhealthy"


def _result_date(result: dict) -> str:
    """Extract an ISO local date from an OpenAQ v3 result.

    OpenAQ v3 exposes daily period dates in a few shapes across API versions:
    - legacy: result["datetime"]["local"]
    - current: result["period"]["datetimeFrom"]["local"]
    - fallback: result["period"]["datetimeFrom"]["utc"]
    """
    timestamp = result.get("datetime")
    if isinstance(timestamp, dict):
        timestamp = timestamp.get("local") or timestamp.get("utc")

    if not timestamp:
        period = result.get("period") or {}
        period_start = period.get("datetimeFrom") or period.get("datetime_to") or {}
        if isinstance(period_start, dict):
            timestamp = period_start.get("local") or period_start.get("utc")

    if not timestamp:
        raise ValueError("OpenAQ result has no datetime field.")
    return str(timestamp)[:10]


def circular_mean_degrees(values: pd.Series) -> float:
    """Return the circular mean of wind direction readings in degrees (0-360)."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    radians = np.deg2rad(clean.to_numpy())
    mean_angle = np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float((mean_angle + 360.0) % 360.0)


def collect_pm25(api_key: str, end_date: date) -> pd.DataFrame:
    """Fetch OpenAQ daily PM2.5 observations in one request window per year."""
    headers = {"X-API-Key": api_key}
    rows = []
    for year in range(START_DATE.year, end_date.year + 1):
        chunk_start = max(START_DATE, date(year, 1, 1))
        chunk_end = min(end_date, date(year, 12, 31))
        page = 1
        print(f"Fetching OpenAQ daily PM2.5: {chunk_start} to {chunk_end}")
        while True:
            response = requests.get(
                OPENAQ_DAYS_URL,
                headers=headers,
                params={
                    "datetime_from": chunk_start.isoformat(),
                    "datetime_to": chunk_end.isoformat(),
                    "limit": 1000,
                    "page": page,
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            rows.extend({"date": _result_date(item), "pm25": item["value"]} for item in results)
            if len(results) < 1000:
                break
            page += 1
    pm25 = pd.DataFrame(rows)
    if pm25.empty:
        raise ValueError("OpenAQ returned no PM2.5 data for the requested sensor/date range.")
    pm25["pm25"] = pd.to_numeric(pm25["pm25"], errors="coerce")
    return pm25.groupby("date", as_index=False)["pm25"].mean()


def collect_weather(end_date: date) -> pd.DataFrame:
    """Fetch hourly Open-Meteo archive weather and aggregate it to daily means."""
    print(f"Fetching Open-Meteo archive weather: {START_DATE} to {end_date}")
    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": START_DATE.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
            "timezone": TIMEZONE,
        },
        timeout=90,
    )
    response.raise_for_status()
    weather = pd.DataFrame(response.json().get("hourly", {}))
    if weather.empty:
        raise ValueError("Open-Meteo Archive API returned no hourly weather data.")
    weather["date"] = pd.to_datetime(weather.pop("time")).dt.strftime("%Y-%m-%d")
    weather = weather.rename(columns={"wind_speed_10m": "windspeed_10m", "wind_direction_10m": "winddirection_10m"})
    daily = weather.groupby("date", as_index=False).agg(
        temperature_2m=("temperature_2m", "mean"),
        relative_humidity_2m=("relative_humidity_2m", "mean"),
        surface_pressure=("surface_pressure", "mean"),
        windspeed_10m=("windspeed_10m", "mean"),
        winddirection_10m=("winddirection_10m", circular_mean_degrees),
    )
    return daily


def main() -> None:
    """Collect, merge, label, and save the reproducible raw dataset."""
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAQ_API_KEY before running this script.")
    end_date = min(date.today(), MAX_ARCHIVE_END_DATE)
    print(f"Using collection end date: {end_date.isoformat()} (Open-Meteo archive range capped at {MAX_ARCHIVE_END_DATE.isoformat()})")
    pm25 = collect_pm25(api_key, end_date)
    weather = collect_weather(end_date)
    dataset = pm25.merge(weather, on="date", how="inner").sort_values("date")
    dataset["aqi_category"] = dataset["pm25"].map(pm25_to_category)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(RAW_CSV, index=False)
    metadata = {
        "generated_on": date.today().isoformat(),
        "collection_window": {"start": START_DATE.isoformat(), "end": end_date.isoformat()},
        "openaq": {"sensor_id": OPENAQ_SENSOR_ID, "station_id": OPENAQ_STATION_ID,
                    "daily_endpoint": OPENAQ_DAYS_URL, "station_distance_from_campus_km": 2.3},
        "open_meteo": {"archive_endpoint": OPEN_METEO_ARCHIVE_URL, "latitude": LATITUDE,
                        "longitude": LONGITUDE, "timezone": TIMEZONE,
                        "daily_wind_direction_method": "circular mean via atan2(mean(sin(rad)), mean(cos(rad)))"},
        "aqi_standard": {"name": "Thai PCD four-class simplification",
                         "breakpoints_ug_m3": {"Very Good": "0.0-15.0", "Good": "15.1-25.0",
                                                 "Moderate": "25.1-37.5", "Unhealthy": ">37.5"}},
        "rows": len(dataset),
        "weather_columns": ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                            "windspeed_10m", "winddirection_10m"],
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(dataset)} collected rows to: {RAW_CSV}")
    print(f"Saved dataset metadata to: {METADATA_JSON}")


if __name__ == "__main__":
    main()
