"""Train and compare XGBoost and LightGBM models per AQI forecast horizon."""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "data/model_ready/train.csv"
TEST_CSV = PROJECT_ROOT / "data/model_ready/test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LABEL_ENCODING_PATH = OUTPUTS_DIR / "label_encoding.json"
COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.json"
RANDOM_STATE = 42
TARGET_LABELS = ["Very Good", "Good", "Moderate", "Unhealthy"]
LABEL_MAPPING = {label: index for index, label in enumerate(TARGET_LABELS)}


def _load_train_and_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load chronological train and test sets."""
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError("Run 04_train_test_split.py first.")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train = train.sort_values("date").reset_index(drop=True)
    test = test.sort_values("date").reset_index(drop=True)
    return train, test


def _encode_label_series(series: pd.Series) -> pd.Series:
    """Encode a categorical AQI label series using the fixed ordinal mapping."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return series.map(LABEL_MAPPING).fillna(-1).astype(int)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode AQI category lag and remove unneeded columns before modeling."""
    prepared = frame.copy()
    if "aqi_category_lag1" in prepared.columns and not pd.api.types.is_numeric_dtype(prepared["aqi_category_lag1"]):
        prepared["aqi_category_lag1"] = _encode_label_series(prepared["aqi_category_lag1"])
    for column in ["date", "aqi_category"]:
        if column in prepared.columns:
            prepared = prepared.drop(columns=[column])
    return prepared


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the model feature columns, excluding target and metadata columns."""
    excluded = {"date", "aqi_category", "label_day1", "label_day2", "label_day3"}
    return [column for column in frame.columns if column not in excluded]


def _save_label_mapping() -> None:
    """Persist the fixed mapping used to convert category names to ordinal integers."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_ENCODING_PATH.write_text(json.dumps(LABEL_MAPPING, indent=2), encoding="utf-8")
    print(f"Saved label encoding: {LABEL_ENCODING_PATH}")


def _save_feature_importance(path: Path, feature_names: list[str], importances: list[float], model_name: str) -> None:
    """Write model feature importances to JSON for inspection."""
    payload = {
        "model": model_name,
        "feature_names": feature_names,
        "importances": [float(value) for value in importances],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved feature importance: {path}")


def _fit_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    """Tune a time-series XGBoost classifier with a macro-F1 objective."""
    xgb_model = XGBClassifier(
        objective="multi:softmax",
        num_class=4,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=12,
        cv=tscv,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def _fit_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    """Tune a time-series LightGBM classifier with a macro-F1 objective."""
    lgbm_model = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=lgbm_model,
        param_distributions=param_distributions,
        n_iter=12,
        cv=tscv,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def main() -> None:
    """Train XGBoost and LightGBM per horizon, select the better model, and save artifacts."""
    start_time = time.perf_counter()
    _save_label_mapping()

    train, test = _load_train_and_test()
    train = _prepare_frame(train)
    test = _prepare_frame(test)

    feature_columns = _feature_columns(train)
    if not feature_columns:
        raise ValueError("No feature columns available for model training.")

    print(f"Using feature columns: {feature_columns}")
    print(f"Training rows: {len(train)} | test rows: {len(test)}")
    print("Chronological cross-validation only: TimeSeriesSplit(n_splits=5)")

    comparison = {}
    for horizon in range(1, 4):
        target_name = f"label_day{horizon}"
        if target_name not in train.columns:
            raise ValueError(f"Missing target column {target_name} in training data.")

        y_train = _encode_label_series(train[target_name])
        X_train = train[feature_columns].copy()
        X_test = test[feature_columns].copy()
        if target_name in test.columns:
            y_test = _encode_label_series(test[target_name])
            print(f"Test label distribution for {target_name}:\n{pd.Series(y_test).value_counts().sort_index()}")

        print(f"\n=== Horizon {horizon}: {target_name} ===")
        print("Training XGBoost...")
        xgb_search = _fit_xgb(X_train, y_train)
        xgb_best_score = float(xgb_search.best_score_)
        print(f"XGBoost best CV F1 (macro): {xgb_best_score:.4f}")

        print("Training LightGBM...")
        lgbm_search = _fit_lgbm(X_train, y_train)
        lgbm_best_score = float(lgbm_search.best_score_)
        print(f"LightGBM best CV F1 (macro): {lgbm_best_score:.4f}")

        selected_model_name = "xgboost" if xgb_best_score >= lgbm_best_score else "lightgbm"
        selected_model = xgb_search.best_estimator_ if selected_model_name == "xgboost" else lgbm_search.best_estimator_
        print(f"Selected model for {target_name}: {selected_model_name}")

        xgb_path = MODELS_DIR / f"xgb_day{horizon}.pkl"
        lgbm_path = MODELS_DIR / f"lgbm_day{horizon}.pkl"
        best_path = MODELS_DIR / f"best_day{horizon}.pkl"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(xgb_search.best_estimator_, xgb_path)
        joblib.dump(lgbm_search.best_estimator_, lgbm_path)
        joblib.dump(selected_model, best_path)
        print(f"Saved: {xgb_path}, {lgbm_path}, {best_path}")

        feature_importance_path = OUTPUTS_DIR / f"feature_importance_day{horizon}.json"
        feature_importance = selected_model.feature_importances_
        _save_feature_importance(feature_importance_path, feature_columns, feature_importance, selected_model_name)

        comparison[f"day{horizon}"] = {
            "xgboost_cv_f1": round(xgb_best_score, 6),
            "lightgbm_cv_f1": round(lgbm_best_score, 6),
            "selected": selected_model_name,
        }

    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved model comparison summary: {COMPARISON_PATH}")

    runtime_seconds = time.perf_counter() - start_time
    print(f"Total training runtime: {runtime_seconds:.2f} seconds for 2 models x 3 horizons (6 randomized searches).")


if __name__ == "__main__":
    main()
