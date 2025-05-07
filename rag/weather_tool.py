# rag/weather_tool.py
import requests

def get_weather_forecast(city: str) -> str:
    try:
        url = f"http://wttr.in/{city}?format=j1"  # JSON format
        res = requests.get(url)
        data = res.json()
        forecast = data['weather'][1]['hourly'][4]['weatherDesc'][0]['value']  # approx. midday tomorrow
        return forecast
    except Exception as e:
        return f"Error: {e}"
