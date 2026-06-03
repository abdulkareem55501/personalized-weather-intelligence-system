<<<<<<< HEAD
# Personalized Weather Intelligence System

Final Year University Project: a machine learning powered weather application that delivers personalized weather insights based on user profiles.

The project now runs as a FastAPI backend with a simple browser-based React frontend served directly from FastAPI. No Node, npm, or Vite is required.

## Main Features

- Real-time weather data from OpenWeatherMap
- Personalized feels-like temperature prediction
- Health risk assessment
- Activity suitability recommendations
- Forecast-based safer activity windows
- ML evaluation dashboard with comparison graphs and validation tables
- Model metrics exposed to the frontend

## Project Structure

```text
api.py                  FastAPI backend
config.py               App settings and environment loading
modules/weather_api.py  OpenWeatherMap integration
modules/predictions.py  ML prediction logic
models/                 Trained model files and training pipeline
frontend/simple/        Browser React UI served by FastAPI
```

## Backend Setup

Create a `.env` file in the project root:

```text
OPENWEATHER_API_KEY=your_openweathermap_api_key
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI backend:

```bash
uvicorn api:app --reload --port 8000
```

Open the API root:

```text
http://localhost:8000
```

Interactive API docs:

```text
http://localhost:8000/docs
```

## Frontend

Open the React app served by FastAPI:

```text
http://localhost:8000/ui
```

The page uses React from browser CDNs, so it needs internet access when the page first loads. It does not require Node or npm.

The landing page includes a secondary **View ML Evaluation** button. This opens the model evaluation dashboard with:

- Feels-like regression metrics
- Activity suitability classifier metrics
- Health-risk classifier metrics
- Algorithm comparison bar charts
- Feature-importance graph
- Health-risk precision/recall table

## Model Training

If model artifacts are missing, run:

```bash
python models/train_models.py
```

This uses the dataset in `data/` and writes model artifacts into `models/`.
=======
# personalized-weather-intelligence-system
Personalised Weather Intelligence System — FastAPI and React app combining real-time weather data with user health profiles and trained ML models. Random Forest Regressor (MAE 0.216), Random Forest Classifier (F1 0.968), MLP Neural Network (accuracy 0.992). Stack: Python, FastAPI, React, Scikit-learn, XGBoost, Pandas.
>>>>>>> 8184ab63f1179e7d80823070888d5279688f3d75
