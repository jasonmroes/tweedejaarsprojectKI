import io
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import xarray as xr
from dotenv import load_dotenv


START_DATE = "2022-01-01"
END_DATE = "2026-06-09"
DATASET = "daily-in-situ-meteorological-observations"
VERSION = "1.0"

SCHIPHOL_STATION = "0-20000-0-06240"

OUTPUT_FILE = "weather_amsterdam1.csv"


load_dotenv()

API_KEY = os.getenv("KNMI_API_KEY")

if not API_KEY:
    raise ValueError("KNMI_API_KEY not found in .env")

HEADERS = {
    "Authorization": API_KEY
}


def get_weather_for_date(date_str):
    filename = f"daily-observations-{date_str.replace('-', '')}.nc"

    metadata_url = (
        f"https://api.dataplatform.knmi.nl/open-data/v1/"
        f"datasets/{DATASET}/versions/{VERSION}/files/{filename}/url"
    )

    response = requests.get(metadata_url, headers=HEADERS)
    response.raise_for_status()

    download_url = response.json()["temporaryDownloadUrl"]

    file_response = requests.get(download_url)
    file_response.raise_for_status()

    ds = xr.open_dataset(io.BytesIO(file_response.content))

    try:
        s = ds.sel(station=SCHIPHOL_STATION).squeeze()

        return {
            "date": date_str,
            "temp_avg": float(s["TG"]),
            "temp_min": float(s["TN"]),
            "temp_max": float(s["TX"]),
            "rain_mm": float(s["RH"]),
            "sun_hours": float(s["SQ"]),
            "wind_avg": float(s["FG"]),
            "cloud": float(s["NG"]),
            "humidity": float(s["UG"]),
        }

    finally:
        ds.close()


def build_dataset(start_date, end_date):
    rows = []

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end - current).days + 1
    counter = 0

    while current <= end:
        counter += 1

        date_str = current.strftime("%Y-%m-%d")

        try:
            print(f"[{counter}/{total_days}] {date_str}")

            weather = get_weather_for_date(date_str)
            rows.append(weather)

        except Exception as e:
            print(f"Failed: {date_str} -> {e}")

        current += timedelta(days=1)

    return pd.DataFrame(rows)


df = build_dataset(START_DATE, END_DATE)

df.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Saved {len(df)} rows")
print(f"CSV: {OUTPUT_FILE}")