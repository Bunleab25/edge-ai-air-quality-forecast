# Edge AI Air-Quality Forecast System

AeroSense RMUTI is a campus-focused air-quality intelligence system designed for deployment on a Raspberry Pi 5 at the edge. It combines local sensor measurements, short-term weather context, and trained AQI forecasting models to estimate air-quality conditions for the next one to three days for the RMUTI campus in Nakhon Ratchasima, Thailand.

This is no longer just a research prototype. It is structured as an edge AI system with:

- local real-time inference at the device edge
- sensor-based input collection
- AQI forecasting for short horizons
- lightweight model serving on Raspberry Pi 5
- future-ready integration with ESP32-based environmental monitoring

## Scenario

The system models an air-quality monitoring scenario for RMUTI campus. It is intended for edge deployment where a small computing device runs inference close to the source of sensor data, reduces cloud dependency, and provides timely AQI information to local users or campus services.

## Edge AI architecture

```text
ESP32 / sensor node
        │
        ▼
Raspberry Pi 5 (edge device)
        │
        ├── local sensor ingestion
        ├── AQI feature preprocessing
        ├── ML inference model
        ├── short-term forecast logic
        └── local API / dashboard output
```

The edge service in this repo is implemented in [model.py](model.py). It accepts sensor readings, builds the required feature set, runs the trained model, and returns AQI predictions for the next three days.

## What this repo contains

- [scripts/00_collect_data.py](scripts/00_collect_data.py): data collection from OpenAQ and weather sources
- [scripts/01_data_cleaning.py](scripts/01_data_cleaning.py): data cleaning and labeling
- [scripts/03_feature_engineering.py](scripts/03_feature_engineering.py): lag, rolling, and time features
- [scripts/05_train_model.py](scripts/05_train_model.py): training and model comparison
- [scripts/06_evaluate.py](scripts/06_evaluate.py): evaluation metrics and reporting
- [scripts/07_forecast_pipeline.py](scripts/07_forecast_pipeline.py): forecast generation pipeline
- [model.py](model.py): Flask-based edge inference API for Raspberry Pi 5
- [tests/test_openaq_parser.py](tests/test_openaq_parser.py): validation for the data pipeline

## AI and deployment goal

This project is designed to support the following real-world edge scenario:

1. Read local environmental conditions from a sensor or input source
2. Build the required AQI feature context on-device
3. Run inference with a trained model on Raspberry Pi 5
4. Forecast air-quality category for Day +1, Day +2, and Day +3
5. Expose the result through an API or dashboard endpoint

## Quick start

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the training pipeline

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

### Run the edge inference service

```bash
python model.py
```

Then send a request to the local API:

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pm25": 42.3,
    "temperature": 28.5,
    "humidity": 65,
    "pressure": 995
  }'
```

## Hardware target

This repository is intended for a Raspberry Pi 5 edge deployment, with the possibility of pairing with an ESP32 sensor node. The current code already follows the structure of a local inference service intended to run close to the monitored environment.

## Project status

- Research and data pipeline: complete
- Training and evaluation pipeline: complete
- Edge inference prototype: complete
- Production-style deployment packaging: next step

## Notes

- The project intentionally avoids random train/test shuffling and uses chronological evaluation.
- AQI categories are mapped into a compact four-class scheme for improved stability.
- Forecast fallback logic is included so the system can continue operating even when external weather APIs are temporarily unavailable.

## License

This project is distributed under the MIT License.
