# edge-ai-air-quality-forecast

AeroSense RMUTI is a machine-learning project for forecasting next-day, two-day, and three-day  PM2.5 AQI categories for the RMUTI campus in Korat. 

## Project overview

- Forecast horizons: Day +1, +2, +3
- Target classes: Very Good, Good, Moderate, Unhealthy
- Data sources:
  - OpenAQ PM2.5 observations
  - Open-Meteo historical weather data
- Modeling approach:
  - Strict chronological train/test split
  - TimeSeriesSplit cross-validation
  - Model comparison by macro F1 score

## Repository structure

```text
aerosense/
├── data/
│   ├── raw/
│   ├── processed/
│   └── model_ready/
├── scripts/
├── tests/
├── models/
├── outputs/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── .venv/
```

## Environment setup

This project requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your OpenAQ API key in the shell or in a local `.env` file:

```bash
export OPENAQ_API_KEY="your_api_key_here"
```

## Data pipeline

Run the scripts in order from the project root:

```bash
python scripts/00_collect_data.py
python scripts/01_data_cleaning.py
python scripts/02_eda.py
python scripts/03_feature_engineering.py
python scripts/04_train_test_split.py
python scripts/05_train_model.py
python scripts/06_evaluate.py
python scripts/07_forecast_pipeline.py
```

The pipeline:

1. Collects daily PM2.5 data and weather features
2. Cleans and merges the datasets
3. Creates lag and rolling features
4. Builds a chronological 80/20 split
5. Trains and compares XGBoost vs LightGBM per horizon
6. Saves the best model and evaluation metrics
7. Produces a forecast file and forecast summaries

## Validation

Run the test suite with:

```bash
pytest -q
```

## Notes

- The project intentionally avoids random train/test shuffling.
- The AQI label mapping follows the Thai PM2.5 category thresholds with the highest two tiers merged into a combined "Unhealthy" class.
- The forecasting step falls back gracefully if the external weather API is temporarily unavailable.

## License

This project is distributed under the MIT License.
