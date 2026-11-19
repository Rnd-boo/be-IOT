
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

app = FastAPI()

# Load the trained model once
model = XGBRegressor()
model.load_model("solar_xgboost.json")


class SensorData(BaseModel):
    voltage: float
    current: float
    power: float
    panel_temperature: float | None = None
    light_lux: float
    air_temperature: float
    humidity: float
    recorded_at: str


@app.post("/predict")
def predict(data: SensorData):

    timestamp = pd.Timestamp(data.recorded_at)

    # Convert timestamp to WIB
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")

    local_time = timestamp.tz_convert("Asia/Jakarta")

    hour = local_time.hour + local_time.minute / 60

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Prepare features in the same order as training
    features = pd.DataFrame([{
        "voltage": data.voltage,
        "current": data.current,
        "power": data.power,
        "panel_temperature": data.panel_temperature,
        "light_lux": data.light_lux,
        "air_temperature": data.air_temperature,
        "humidity": data.humidity,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week": local_time.dayofweek
    }])

    prediction = float(model.predict(features)[0])

    return {
        "predicted_power": round(prediction, 2),
        "unit": "W",
        "forecast_minutes": 30
    }
