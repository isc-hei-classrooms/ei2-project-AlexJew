"""Shared MeteoSwiss data download function."""

import certifi
import os
from datetime import datetime
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from influxdb_client.client.influxdb_client import InfluxDBClient

load_dotenv()

MEASUREMENTS = [
    "Air temperature 2m above ground (current value)",
    "Global radiation (ten minutes mean)",
    "Precipitation (ten minutes total)",
    "Relative air humidity 2m above ground (current value)",
    "Sunshine duration (ten minutes total)",
    "PRED_DURSUN_ctrl",
    "PRED_GLOB_ctrl",
    "PRED_RELHUM_2M_ctrl",
    "PRED_T_2M_ctrl",
    "PRED_TOT_PREC_ctrl",
]


def download_meteoswiss(start: str, stop: str) -> pl.DataFrame:
    """Download MeteoSwiss data from InfluxDB.

    Args:
        start: Flux-compatible start time (e.g. "2022-10-01T00:00:00Z").
        stop: Flux-compatible stop time (e.g. "2025-09-30T23:59:59Z").

    Returns
    -------
        Pivoted Polars DataFrame with one column per measurement.
    """
    org = os.environ["INFLUXDB_ORG"]
    bucket = os.environ["INFLUXDB_BUCKET"]
    token = os.environ["INFLUXDB_TOKEN"]
    client = InfluxDBClient(
        url="https://timeseries.hevs.ch",
        token=token,
        org=org,
        ssl_ca_cert=certifi.where(),
        timeout=1000000,
    )

    measurement_set = ", ".join(f'"{m}"' for m in MEASUREMENTS)
    query = f'''
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r.Site == "Sion")
  |> filter(fn: (r) => contains(value: r._measurement, set: [{measurement_set}]))
  |> filter(fn: (r) => r._field == "Value")
  |> group(columns: ["_measurement", "_time"])
  |> last()
  |> group(columns: ["_measurement"])
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


def save_meteoswiss(df: pl.DataFrame, filename_prefix: str) -> Path:
    """Save a MeteoSwiss DataFrame to a timestamped CSV file.

    Args:
        df: Polars DataFrame to save.
        filename_prefix: Prefix for the output CSV file (e.g. "sion_weather").

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
    return filename
