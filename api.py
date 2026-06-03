"""
FastAPI backend — Personalized Weather Intelligence System
Run: uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import json
from pathlib import Path

from modules.weather_api import get_weather, get_forecast
from modules.predictions import (
    predict_feels_like,
    predict_health_risk,
    predict_activities,
    recommend_activity_times,
)
from config import APP_NAME, VERSION

app = FastAPI(title="Weather Intelligence API", version=VERSION)
UI_FILE = Path(__file__).parent / "frontend" / "simple" / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Pydantic models ────────────────────────────────────────

class ProfileModel(BaseModel):
    name: str = "Guest"
    age: int = 25
    gender: str = "Prefer not to say"
    health_conditions: List[str] = ["None"]
    activity_preferences: List[str] = ["Walking"]
    heat_sensitivity: str = "Medium"
    cold_sensitivity: str = "Medium"


class PredictRequest(BaseModel):
    city: str
    profile: ProfileModel


# ── Routes ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"app": APP_NAME, "version": VERSION, "status": "running"}


@app.get("/ui", response_class=HTMLResponse)
def react_ui():
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="React UI file not found")
    return UI_FILE.read_text(encoding="utf-8")


@app.get("/api/info")
def get_info():
    return {"name": APP_NAME, "version": VERSION}


@app.get("/api/weather/{city}")
def weather_endpoint(city: str):
    weather, error = get_weather(city)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return weather


@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):
    weather, error = get_weather(req.city)
    if error:
        raise HTTPException(status_code=400, detail=error)

    profile = req.profile.model_dump()

    feels_like = predict_feels_like(weather, profile)
    risk_level, risk_explanation, confidence, risk_factors = predict_health_risk(weather, profile)
    activities = predict_activities(weather, profile)
    forecast, forecast_error = get_forecast(req.city)
    activity_times = recommend_activity_times(forecast or [], profile, weather)

    return {
        "weather": weather,
        "feels_like": round(float(feels_like), 1),
        "risk_level": risk_level,
        "risk_explanation": risk_explanation,
        "confidence": round(float(confidence), 3),
        "risk_factors": risk_factors,
        "activities": activities,
        "forecast": forecast or [],
        "forecast_error": forecast_error,
        "activity_times": activity_times,
    }


@app.get("/api/metrics")
def get_metrics_endpoint():
    try:
        with open("models/metrics.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run: python models/train_models.py"
        )
