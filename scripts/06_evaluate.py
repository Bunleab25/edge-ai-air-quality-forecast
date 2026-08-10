"""Evaluate the three horizon-specific AQI gradient-boosting models."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = PROJECT_ROOT / "data/model_ready/test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUTS_DIR / "evaluation_report.json"
CM_DIR = OUTPUTS_DIR / "confusion_matrices"
LABEL_ENCODING_PATH = OUTPUTS_DIR / "label_encoding.json"
FEATURES = [
    "pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "pm25_lag1", "pm25_lag2", "pm25_lag3",
    "pm25_roll_mean3", "pm25_roll_mean7",
    "pm25_trend3",
    "pressure_lag1", "pressure_trend3",
    "aqi_category_lag1",
    "day_of_year", "month", "month_sin", "month_cos",
]
LABEL_ORDER = ["Very Good", "Good", "Moderate", "Unhealthy"]


def load_label_mapping() -> dict[str, int]:
    """Load the fixed 4-class ordinal mapping used in the model training script."""
    with LABEL_ENCODING_PATH.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {key: int(value) for key, value in mapping.items()}


def encode_feature_frame(frame: pd.DataFrame, label_map: dict[str, int]) -> pd.DataFrame:
    """Coerce the AQI lag category to the same ordinal encoding used by the trained models."""
    prepared = frame.copy()
    if "aqi_category_lag1" in prepared.columns:
        prepared["aqi_category_lag1"] = prepared["aqi_category_lag1"].map(label_map).fillna(0).astype(int)
    return prepared


def main() -> None:
    """Write metrics, feature importance, and one confusion matrix per horizon."""
    print(f"Loading test data: {TEST_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError("Run 04_train_test_split.py first.")
    test = pd.read_csv(TEST_CSV)
    label_map = load_label_mapping()
    reverse_map = {value: label for label, value in label_map.items()}
    test = encode_feature_frame(test, label_map)
    CM_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")
    results = {"test_rows": len(test), "horizons": {}}
    accuracies = []

    for horizon in range(1, 4):
        target = f"label_day{horizon}"
        model_path = MODELS_DIR / f"best_day{horizon}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}; run 05_train_model.py first.")
        model = joblib.load(model_path)

        actual = test[target].map(label_map).fillna(-1).astype(int)
        predictions = model.predict(test[FEATURES]).astype(int)
        accuracy = accuracy_score(actual, predictions)
        macro_f1 = f1_score(actual, predictions, average="macro", zero_division=0)
        accuracies.append(accuracy)

        present_labels = sorted(set(actual).union(set(predictions)))
        label_names = [reverse_map.get(label, str(label)) for label in present_labels]
        matrix = confusion_matrix(actual, predictions, labels=present_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names,
                    yticklabels=label_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"AQI Forecast Confusion Matrix — Day +{horizon}")
        plt.tight_layout()
        cm_path = CM_DIR / f"best_day{horizon}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=160)
        plt.close()

        importances = dict(zip(FEATURES, (float(value) for value in model.feature_importances_)))
        results["horizons"][f"day{horizon}"] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "feature_importances": importances,
            "confusion_matrix_file": str(cm_path.relative_to(PROJECT_ROOT)),
        }
        print(f"Day +{horizon}: accuracy={accuracy:.3f}, macro F1={macro_f1:.3f}")

    results["accuracy_degradation_day1_to_day3"] = float(accuracies[0] - accuracies[2])
    print(f"Accuracy degradation (day1 - day3): {accuracies[0] - accuracies[2]:.3f}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved evaluation report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
