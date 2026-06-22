import sys
import requests
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

COLUMNS = [
    "date",
    "temp",
    "tempmin",
    "tempmax",
    "humidity",
    "precip",
    "precipprob",
    "cloudcover",
    "windspeed",
]

def get_incoming_weather(API_KEY, BASE_URL, LOCATION, FORECAST_DAYS, TOTAL_DAYS):

    today    = date.today()
    end_date = today + timedelta(days=FORECAST_DAYS - 1)

    url = f"{BASE_URL}/{LOCATION}/{today.isoformat()}/{end_date.isoformat()}"

    params = {
        "key":         API_KEY,
        "unitGroup":   "metric",
        "include":     "days",
        "elements":    "datetime,temp,tempmin,tempmax,humidity,precip,precipprob,cloudcover,windspeed",
        "contentType": "json",
    }

    print(f"Fetching forecast for Schiphol ({LOCATION}) ...")
    response = requests.get(url, params=params, timeout=15)

    if response.status_code == 429:
        sys.exit("ERROR: Daily quota exceeded (1,000 records/day). Try again tomorrow.")
    if response.status_code == 401:
        sys.exit("ERROR: Invalid API key.")
    if not response.ok:
        sys.exit(f"ERROR: API returned {response.status_code}: {response.text[:300]}")

    days = response.json().get("days", [])
    if not days:
        sys.exit("ERROR: No forecast data returned.")

    print(f"  Received {len(days)} forecast days.")

    rows = []
    for day in days:
        rows.append({
            "date":       day.get("datetime"),
            "temp":       day.get("temp"),
            "tempmin":    day.get("tempmin"),
            "tempmax":    day.get("tempmax"),
            "humidity":   day.get("humidity"),
            "precip":     day.get("precip"),
            "precipprob": day.get("precipprob"),
            "cloudcover": day.get("cloudcover"),
            "windspeed":  day.get("windspeed"),
        })

    df = pd.DataFrame(rows, columns=COLUMNS)

    if len(df) < TOTAL_DAYS:
        last_date = pd.to_datetime(df["date"].iloc[-1])
        padding = [
            {"date": (last_date + timedelta(days=i)).strftime("%Y-%m-%d")}
            for i in range(1, TOTAL_DAYS - len(df) + 1)
        ]
        df = pd.concat([df, pd.DataFrame(padding)], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    return df
