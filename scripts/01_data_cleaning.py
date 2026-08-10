"""Clean and validate the RMUTI daily air-quality dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data/raw/rmuti_dataset_2023_2026_final.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"
NUMERIC_COLUMNS = [
    "pm25",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "windspeed_10m",
    "winddirection_10m",
]


def pm25_to_category(pm25: float) -> str:
    """Return the project's four-class Thai PCD AQI category."""
    if pm25 <= 15:
        return "Very Good"
    if pm25 <= 25:
        return "Good"
    if pm25 <= 37.5:
        return "Moderate"
    return "Unhealthy"


def _format_gap_ranges(dates: pd.Index) -> str:
    """Convert a sorted date index into concise date gap ranges."""
    if len(dates) == 0:
        return "None"
    ranges = []
    start = prev = dates[0]
    for current in dates[1:]:
        if current != prev + pd.Timedelta(days=1):
            ranges.append(f"{start.date()} to {prev.date()}")
            start = current
        prev = current
    ranges.append(f"{start.date()} to {prev.date()}")
    return "; ".join(ranges)


def interpolate_circular_direction(series: pd.Series) -> pd.Series:
    """Interpolate compass bearings using sine/cosine components to avoid 0/360 discontinuity."""
    working = series.copy()
    radians = np.deg2rad(working)
    sin_values = np.sin(radians)
    cos_values = np.cos(radians)
    sin_interp = pd.Series(sin_values, index=working.index).interpolate(method="linear", limit_direction="both")
    cos_interp = pd.Series(cos_values, index=working.index).interpolate(method="linear", limit_direction="both")
    direction = np.mod(np.degrees(np.arctan2(sin_interp.to_numpy(), cos_interp.to_numpy())), 360.0)
    return pd.Series(direction, index=working.index)


def main() -> None:
    """Load, clean, and write the chronologically ordered dataset."""
    print(f"Loading raw data: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Place the source CSV at: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"date", *NUMERIC_COLUMNS, "aqi_category"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_dates = df["date"].isna().sum()
    if invalid_dates:
        print(f"Dropping {invalid_dates} row(s) with invalid dates.")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")

    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_dates = full_range.difference(pd.DatetimeIndex(df["date"]))
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Missing calendar dates: {len(missing_dates)}")
    if len(missing_dates):
        print(f"Missing calendar gap ranges: {_format_gap_ranges(missing_dates)}")

    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"
    df = df.reset_index()

    print("Missing values before handling:\n", df[NUMERIC_COLUMNS].isna().sum())

    long_gap_ranges = []
    if len(missing_dates):
        start = None
        prev = None
        for current in sorted(missing_dates):
            if start is None:
                start = prev = current
            elif current == prev + pd.Timedelta(days=1):
                prev = current
            else:
                if (prev - start).days + 1 >= 3:
                    long_gap_ranges.append((start, prev))
                start = prev = current
        if start is not None and (prev - start).days + 1 >= 3:
            long_gap_ranges.append((start, prev))

    if long_gap_ranges:
        print("Long-gap date ranges to drop after interpolation:")
        for start, end in long_gap_ranges:
            print(f"  - {start.date()} to {end.date()}")

    interpolated = df[NUMERIC_COLUMNS].copy()
    for column in ["pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure", "windspeed_10m"]:
        interpolated[column] = interpolated[column].interpolate(method="linear", limit=3, limit_direction="both")
    interpolated["winddirection_10m"] = interpolate_circular_direction(interpolated["winddirection_10m"])
    df[NUMERIC_COLUMNS] = interpolated

    for start, end in long_gap_ranges:
        df.loc[(df["date"] >= start) & (df["date"] <= end), NUMERIC_COLUMNS] = pd.NA
    df = df.dropna(subset=NUMERIC_COLUMNS).copy()

    outlier_summary = []
    for column in NUMERIC_COLUMNS:
        if column == "winddirection_10m":
            outlier_summary.append(f"{column}: skipped for IQR detection (circular compass direction).")
            continue
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((df[column] < lower) | (df[column] > upper)).sum()
        outlier_summary.append(f"{column}: {outlier_count} outliers (IQR bounds {lower:.3f} to {upper:.3f})")
    print("Outlier summary (IQR 1.5x, not dropped):")
    for line in outlier_summary:
        print(f"  - {line}")

    df["aqi_category"] = df["pm25"].map(pm25_to_category)
    df = df[["date", *NUMERIC_COLUMNS, "aqi_category"]].sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} cleaned rows to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
