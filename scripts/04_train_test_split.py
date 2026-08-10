"""Create a chronological train/test split for AQI forecasting."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data/model_ready/train_test_shifted.csv"
TRAIN_CSV = PROJECT_ROOT / "data/model_ready/train.csv"
TEST_CSV = PROJECT_ROOT / "data/model_ready/test.csv"
TEST_FRACTION = 0.20


def main() -> None:
    """Put the earliest 80% in train and latest 20% in test, without shuffling."""
    print(f"Loading model-ready data: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError("Run 03_feature_engineering.py first.")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("At least 10 rows are required for a useful time-based split.")
    split_index = int(len(df) * (1 - TEST_FRACTION))
    train, test = df.iloc[:split_index].copy(), df.iloc[split_index:].copy()
    for frame in (train, test):
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"Time-based split: {len(train)} train rows through {train['date'].iloc[-1]}; "
          f"{len(test)} test rows from {test['date'].iloc[0]}.")
    print(f"Saved: {TRAIN_CSV} and {TEST_CSV}")


if __name__ == "__main__":
    main()
