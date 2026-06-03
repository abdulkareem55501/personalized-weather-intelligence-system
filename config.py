import os
from dotenv import load_dotenv

# Load .env file explicitly with full path
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# OpenWeatherMap API settings
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5"
AIR_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
UV_URL = "http://api.openweathermap.org/data/2.5/uvi"

# App settings
APP_NAME = "Personalized Weather Intelligence System"
VERSION = "1.0.0"

# Database
DB_NAME = "weather_app.db"

# ML Model paths
FEELS_LIKE_MODEL = "models/feels_like.pkl"
ACTIVITY_MODEL = "models/activity.pkl"
HEALTH_RISK_MODEL = "models/health_risk.pkl"
