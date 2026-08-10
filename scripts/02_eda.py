"""Create exploratory plots for the cleaned RMUTI dataset."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES = ["pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure"]
WEATHER_FEATURES = FEATURES[1:]


def main() -> None:
    """Generate correlation, category boxplot, and seasonal PM2.5 plots."""
    print(f"Loading cleaned data: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError("Run 01_data_cleaning.py first.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date")
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix: PM2.5 and Weather Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=160)
    plt.close()
    print("Saved correlation_matrix.png")

    category_order = ["Very Good", "Good", "Moderate", "Unhealthy"]
    for feature in WEATHER_FEATURES:
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=df, x="aqi_category", y=feature, order=category_order)
        plt.xticks(rotation=20, ha="right")
        plt.title(f"{feature} by AQI Category")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"boxplot_{feature}.png", dpi=160)
        plt.close()
    print("Saved weather-feature boxplots.")

    dry = df["date"].dt.month.isin([11, 12, 1, 2, 3, 4])
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["pm25"], color="#4c78a8", linewidth=1, label="Daily PM2.5")
    plt.fill_between(df["date"], 0, df["pm25"], where=dry, color="#e45756", alpha=0.18,
                     label="Dry season (Nov–Apr)")
    plt.fill_between(df["date"], 0, df["pm25"], where=~dry, color="#54a24b", alpha=0.12,
                     label="Wet season (May–Oct)")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.title("RMUTI PM2.5 Time Series and Seasonal Pattern")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pm25_time_series_seasons.png", dpi=160)
    plt.close()
    print("Saved pm25_time_series_seasons.png")


if __name__ == "__main__":
    main()
