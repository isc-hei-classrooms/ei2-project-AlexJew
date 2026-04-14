"""MeteoSwiss data download and save utilities.

Run this file directly to download historical forecasts and measurements
month by month:
    uv run python utils/data_acquisition.py

To download serving/forecast data in a marimo notebook, use:
    from utils.data_acquisition import download_forecast, download_measurement
    from datetime import UTC, datetime, timedelta

    now = datetime.now(tz=UTC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    stop = start + timedelta(days=2)
    df_forecast = download_forecast(
        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        stop=stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
        site="Sion",
    )
    df_measurement = download_measurement(
        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        stop=stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
        site="Sion",
    )
"""

import certifi
from datetime import date, datetime
from pathlib import Path

import polars as pl
from influxdb_client.client.influxdb_client import InfluxDBClient

from utils.config import load_config

cfg = load_config()

# Key parameters
FORECASTS = [
    "PRED_DURSUN_ctrl",
    "PRED_GLOB_ctrl",
    "PRED_RELHUM_2M_ctrl",
    "PRED_T_2M_ctrl",
    "PRED_TOT_PREC_ctrl",
]

FORECAST_PREDICTIONS = [f"{i:02d}" for i in range(14, 34)]

MEASUREMENTS = [
    "Air temperature 2m above ground (current value)",
    "Global radiation (ten minutes mean)",
    "Precipitation (ten minutes total)",
    "Relative air humidity 2m above ground (current value)",
    "Sunshine duration (ten minutes total)"
]

TRAINING_START = date(2022, 9, 30) # Start period of the training
TRAINING_STOP = date(2025, 9, 30) # End period of the training
SITE = "Sion"  # MeteoSwiss station site

# Functions
def download_forecast(start: str, stop: str, site: str) -> pl.DataFrame:
    """Download MeteoSwiss 9 AM forecast data from InfluxDB.

    Downloads predictions 15-33 (9 AM + 15h to 9 AM + 33h), which cover
    midnight to 6 PM of the next day. Only keeps forecasts issued at 9 AM.

    Args:
        start: Flux-compatible start time (e.g. "2022-10-01T00:00:00Z").
        stop: Flux-compatible stop time (e.g. "2025-09-30T23:59:59Z").

    Returns
    -------
        Pivoted Polars DataFrame with one column per measurement.
    """
    org = cfg.influx.org
    bucket = cfg.influx.bucket
    token = cfg.influx.token
    client = InfluxDBClient(
        url=cfg.influx.url,
        token=token,
        org=org,
        ssl_ca_cert=certifi.where(),
        timeout=1000000,
    )

    measurement_set = ", ".join(f'"{m}"' for m in FORECASTS)
    prediction_set = ", ".join(f'"{p}"' for p in FORECAST_PREDICTIONS)
    query = f'''
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r.Site == "{site}")
  |> filter(fn: (r) => contains(value: r._measurement, set: [{measurement_set}]))
  |> filter(fn: (r) => r._field == "Value")
  |> filter(fn: (r) => contains(value: r.Prediction, set: [{prediction_set}]))
'''
    tables = client.query_api().query(org=org, query=query)
    records = []
    for table in tables:
        for record in table.records:
            records.append(
                {
                    "timestamp": record["_time"],
                    "measurement": record["_measurement"],
                    "value": record["_value"],
                    "prediction": int(record["Prediction"]),
                }
            )
    client.close()

    df = pl.DataFrame(records)
    if not df.is_empty():
        df = df.filter(
            (pl.col("timestamp").dt.hour() + 24 - pl.col("prediction")) % 24 == 9
        )
        df = df.drop("prediction").pivot(
            index="timestamp",
            on="measurement",
            values="value",
        ).sort("timestamp")
    return df

def download_measurement(start: str, stop: str, site: str) -> pl.DataFrame:
    """Download MeteoSwiss historical measurement data from InfluxDB.

    Args:
        start: Flux-compatible start time (e.g. "2022-10-01T00:00:00Z").
        stop: Flux-compatible stop time (e.g. "2025-09-30T23:59:59Z").

    Returns
    -------
        Pivoted Polars DataFrame with one column per measurement.
    """
    org = cfg.influx.org
    bucket = cfg.influx.bucket
    token = cfg.influx.token
    client = InfluxDBClient(
        url=cfg.influx.url,
        token=token,
        org=org,
        ssl_ca_cert=certifi.where(),
        timeout=1000000,
    )

    measurement_set = ", ".join(f'"{m}"' for m in MEASUREMENTS)
    query = f'''
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r.Site == "{site}")
  |> filter(fn: (r) => contains(value: r._measurement, set: [{measurement_set}]))
  |> filter(fn: (r) => r._field == "Value")
'''
    tables = client.query_api().query(org=org, query=query)
    records = []
    for table in tables:
        for record in table.records:
            records.append(
                {
                    "timestamp": record["_time"],
                    "measurement": record["_measurement"],
                    "value": record["_value"],
                }
            )
    client.close()

    df = pl.DataFrame(records)
    if not df.is_empty():
        df = df.pivot(
            index="timestamp",
            on="measurement",
            values="value",
        ).sort("timestamp")
    return df

def save_meteoswiss(df: pl.DataFrame, filename_prefix: str, city: str = None) -> Path:
    """Save a MeteoSwiss DataFrame to a timestamped CSV file.

    Args:
        df: Polars DataFrame to save.
        filename_prefix: Prefix for the output CSV file (e.g. "sion_weather").
        city: Optional city name to update in config.toml.

    Returns
    -------
        Path to the saved CSV file.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = data_dir / f"{filename_prefix}_{timestamp}.csv"

    df.write_csv(filename)
    print(f"Data saved to {filename}")

    if city:
        from utils.config import update_version
        update_version("raw_data.acquisition", city, timestamp)
        print(f"Updated config.toml with raw_data.acquisition.{city} = {timestamp}")

    return filename

if __name__ == "__main__":
    # Download forecasts
    forecast_frames: list[pl.DataFrame] = []
    current = TRAINING_START
    while current < TRAINING_STOP:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        end = min(next_month, TRAINING_STOP)

        chunk = download_forecast(
            start=f"{current.isoformat()}T00:00:00Z",
            stop=f"{end.isoformat()}T00:00:00Z",
            site=SITE,
        )
        print(f"Downloaded forecasts for {current.strftime('%B %Y')}")
        if not chunk.is_empty():
            forecast_frames.append(chunk)

        current = next_month

    df_forecast = pl.concat(forecast_frames).sort("timestamp")
    save_meteoswiss(df_forecast, filename_prefix=f"{SITE.lower()}_forecast", city=SITE.lower())

    # Download measurements
    measurement_frames: list[pl.DataFrame] = []
    current = TRAINING_START
    while current < TRAINING_STOP:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        end = min(next_month, TRAINING_STOP)

        chunk = download_measurement(
            start=f"{current.isoformat()}T00:00:00Z",
            stop=f"{end.isoformat()}T00:00:00Z",
            site=SITE,
        )
        print(f"Downloaded measurements for {current.strftime('%B %Y')}")
        if not chunk.is_empty():
            measurement_frames.append(chunk)

        current = next_month

    df_measurement = pl.concat(measurement_frames).sort("timestamp")
    save_meteoswiss(df_measurement, filename_prefix=f"{SITE.lower()}_measurement", city=SITE.lower())
