from pathlib import Path

import joblib
import pandas as pd
from fastapi.responses import FileResponse
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
DATA_PATH = BASE_DIR / "data" / "sales.csv"

app = FastAPI(
    title="Ice Cream Revenue Predictor",
    description="Predicts ice cream revenue from temperature using linear regression.",
    version="1.0.0",
)

# Load the trained model once, when the server starts
model = joblib.load(MODEL_PATH)

# The temperature range the model was trained on (used to warn about extrapolation)
_data = pd.read_csv(DATA_PATH)
TRAIN_MIN = float(_data["Temperature"].min())
TRAIN_MAX = float(_data["Temperature"].max())


class PredictionRequest(BaseModel):
    temperature: float = Field(..., ge=-10, le=60, description="Temperature in °C")


class PredictionResponse(BaseModel):
    temperature: float
    predicted_revenue: float
    in_training_range: bool
    note: str | None = None


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model": "LinearRegression",
        "slope": round(float(model.coef_[0]), 2),
        "intercept": round(float(model.intercept_), 2),
        "training_range_celsius": [TRAIN_MIN, TRAIN_MAX],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = pd.DataFrame([[request.temperature]], columns=["Temperature"])
    raw_prediction = float(model.predict(features)[0])
    prediction = max(0.0, raw_prediction)  # revenue can't be negative

    in_range = TRAIN_MIN <= request.temperature <= TRAIN_MAX
    note = None
    if not in_range:
        note = (
            f"Temperature is outside the training range ({TRAIN_MIN:.0f}-{TRAIN_MAX:.0f} °C); "
            "this prediction is an extrapolation and may be unreliable."
        )
    elif raw_prediction < 0:
        note = "Model output was negative and was clipped to 0."

    return PredictionResponse(
        temperature=request.temperature,
        predicted_revenue=round(prediction, 2),
        in_training_range=in_range,
        note=note,
    )