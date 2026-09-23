# 🍦 Ice Cream Revenue Predictor

A machine learning web app that predicts ice cream revenue from the day's temperature. It uses a scikit-learn linear regression model served through a FastAPI REST API, with a small web page on top.

**Live demo:** _add your Render link here_ · **API docs:** _add your Render link_/docs

Built during my AI & Generative AI internship at YBI Foundation, then extended from a Colab notebook into a deployed, tested web service.

## Features

- Trains a linear regression model on temperature vs revenue data (`train.py`)
- REST API with input validation (Pydantic): `POST /predict`
- Warns when a temperature is outside the range the model was trained on, instead of silently extrapolating
- Never returns negative revenue
- Web page with a slider and number input
- 12 automated tests (pytest)

## Tech stack

Python, FastAPI, Uvicorn, scikit-learn, pandas, joblib, Pydantic, pytest, HTML/CSS/JavaScript

## Project structure

```
├── data/sales.csv                          training data (20 rows)
├── train.py                                trains the model and saves it
├── model/model.joblib                      the trained model
├── main.py                                 FastAPI app
├── static/index.html                       web page
├── tests/test_api.py                       pytest tests
├── Icecream_Revenue_Prediction_vinit.ipynb original Colab notebook
├── requirements.txt / requirements-dev.txt
```

## Run locally

```
git clone https://github.com/vinitkumar7629/IceCream-Revenue-Prediction.git
cd IceCream-Revenue-Prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 for the web page or http://127.0.0.1:8000/docs for the interactive API docs.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web page |
| POST | `/predict` | Predict revenue for a temperature |
| GET | `/model-info` | Slope, intercept and training range |
| GET | `/health` | Health check |

Example request:

```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 35}'
```

Example response:

```json
{
  "temperature": 35.0,
  "predicted_revenue": 1001.54,
  "in_training_range": true,
  "note": null
}
```

Temperatures must be between -10 and 60 °C, otherwise the API returns a 422 validation error.

## Model and results

Simple linear regression on one feature (temperature), with an 80/20 train/test split (`random_state=42`).

| Metric (test set, 4 rows) | Value |
|---|---|
| R² | 0.9941 |
| MAE | 21.44 |
| Slope | 39.02 revenue per °C |
| Intercept | -364.14 |

## Limitations

- The dataset is small (20 rows) and illustrative, not real store sales, so the test set is only 4 rows and the metrics should be read with that in mind.
- One feature only. Real revenue also depends on the day of the week, holidays, location and more.
- The relationship is assumed to be linear. The model was trained on 14-46 °C, and predictions outside that range are flagged as unreliable.

## Tests

```
pip install -r requirements-dev.txt
pytest -v
```

## Future improvements

- Collect real sales data and add features (day of week, humidity, holidays)
- Compare against other models (polynomial regression, random forest) with cross-validation
- Show a prediction interval instead of a single number
- Containerize with Docker and add CI to run the tests on every push