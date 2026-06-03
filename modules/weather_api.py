import requests
from config import API_KEY, BASE_URL, AIR_URL


def get_weather(city):
    """Fetch current weather, AQI and UV index for a given city."""
    try:
        url = f"{BASE_URL}/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 401:
            return None, "Invalid API key. Please check your .env file."
        if response.status_code == 429:
            return None, "API rate limit reached. Please wait a moment and try again."
        if data.get("cod") != 200:
            return None, f"City '{city}' not found. Please check the spelling."

        weather = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": round(data["main"]["temp"], 1),
            "feels_like_api": round(data["main"]["feels_like"], 1),
            "humidity": data["main"]["humidity"],
            "wind_speed": round(data["wind"]["speed"], 1),
            "condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"].title(),
            "visibility": data.get("visibility", 0) // 1000,
            "pressure": data["main"]["pressure"],
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"],
        }

        # Air quality
        try:
            air_url = f"{AIR_URL}?lat={weather['lat']}&lon={weather['lon']}&appid={API_KEY}"
            air_data = requests.get(air_url).json()
            weather["aqi"] = air_data["list"][0]["main"]["aqi"]
            weather["aqi_label"] = get_aqi_label(weather["aqi"])
        except Exception:
            weather["aqi"] = 1
            weather["aqi_label"] = "Good"

        # UV Index
        try:
            uv_url = f"{BASE_URL}/uvi?lat={weather['lat']}&lon={weather['lon']}&appid={API_KEY}"
            uv_data = requests.get(uv_url).json()
            weather["uv_index"] = round(uv_data.get("value", 0), 1)
            weather["uv_label"] = get_uv_label(weather["uv_index"])
        except Exception:
            weather["uv_index"] = 0
            weather["uv_label"] = "Low"

        return weather, None

    except Exception as e:
        return None, f"Connection error: {str(e)}"


def get_forecast(city):
    """Fetch 5-day / 3-hour forecast data from OpenWeatherMap."""
    try:
        url = f"{BASE_URL}/forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code == 401:
            return None, "Invalid API key. Please check your .env file."
        if response.status_code == 429:
            return None, "API rate limit reached. Please wait a moment and try again."
        if data.get("cod") != "200":
            return None, f"Forecast for '{city}' not found. Please check the spelling."

        forecast = []
        for item in data.get("list", [])[:24]:
            main = item.get("main", {})
            wind = item.get("wind", {})
            weather = (item.get("weather") or [{}])[0]
            forecast.append({
                "datetime": item.get("dt_txt", ""),
                "temperature": round(main.get("temp", 0), 1),
                "feels_like_api": round(main.get("feels_like", main.get("temp", 0)), 1),
                "humidity": main.get("humidity", 0),
                "wind_speed": round(wind.get("speed", 0), 1),
                "condition": weather.get("main", "Unknown"),
                "description": weather.get("description", "Unknown").title(),
                "rain_probability": round(item.get("pop", 0) * 100),
            })

        return forecast, None

    except Exception as e:
        return None, f"Forecast connection error: {str(e)}"


def get_aqi_label(aqi):
    """Convert AQI number to human readable label."""
    return {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }.get(aqi, "Unknown")


def get_uv_label(uv):
    """Convert UV index to risk label."""
    if uv <= 2:
        return "Low"
    elif uv <= 5:
        return "Moderate"
    elif uv <= 7:
        return "High"
    elif uv <= 10:
        return "Very High"
    else:
        return "Extreme"


def get_weather_icon(condition):
    """Return emoji icon for weather condition."""
    icons = {
        "Clear": "☀️",
        "Clouds": "☁️",
        "Rain": "🌧️",
        "Drizzle": "🌦️",
        "Thunderstorm": "⛈️",
        "Snow": "❄️",
        "Mist": "🌫️",
        "Fog": "🌫️",
        "Haze": "🌫️",
        "Dust": "🌪️",
        "Sand": "🌪️",
        "Ash": "🌋",
        "Squall": "💨",
        "Tornado": "🌪️",
    }
    return icons.get(condition, "🌡️")
