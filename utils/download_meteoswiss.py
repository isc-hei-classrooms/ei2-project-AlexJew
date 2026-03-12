import certifi
import os
from datetime import datetime
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

load_dotenv()

if __name__ == "__main__":
    org = os.getenv("INFLUXDB_ORG")
    bucket = os.getenv("INFLUXDB_BUCKET")
    token = os.getenv("INFLUXDB_TOKEN")
    url = os.getenv("INFLUXDB_URL")
    client = InfluxDBClient(url="https://timeseries.hevs.ch", token=token, org=org,
                            ssl_ca_cert=certifi.where(), timeout=1000000)
    
    measurements = [
		"PRED_T_2M_ctrl",
        "Air temperature 2m above ground (current value)",
        "Atmospheric pressure at barometric altitude",
        "Global radiation (ten minutes mean)",
        "Gust peak (one second) (maximum)",
        "Precipitation (ten minutes total)",
        "PRED_DD_10M_ctrl",
        "PRED_DURSUN_ctrl",
        "PRED_FF_10M_ctrl",
        "PRED_GLOB_ctrl",
        "PRED_PS_ctrl",
        "PRED_RELHUM_2M_ctrl",
        "PRED_TOT_PREC_ctrl",
        "Relative air humidity 2m above ground (current value)",
        "Sunshine duration (ten minutes total)",
        "Wind Direction (ten minutes mean)",
        "Wind speed scalar (ten minutes mean)"
    ]
    measurement_set = ", ".join(f'"{measurement}"' for measurement in measurements)
    query = f'''
from(bucket: "{bucket}")
  |> range(start: 2022-10-01T00:00:00Z, stop: 2025-09-30T23:59:59Z)
  |> filter(fn: (r) => r.Site == "Sion")
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

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = data_dir / f"sion_weather_{timestamp}.csv"

    # Save to CSV (one column per measurement)
    df = pl.DataFrame(records)
    if not df.is_empty():
        # Add an index for duplicate timestamp-measurement pairs
        df = df.with_columns(
            pl.arange(0, pl.len()).over(["timestamp", "measurement"]).alias("dup_idx")
        )
        # Pivot to have one column per measurement-dup_idx combination
        df = df.pivot(
            index="timestamp",
            columns=["measurement", "dup_idx"],
            values="value",
            aggregate_function="first",  # handle any remaining duplicates
        ).sort("timestamp")
        # Flatten multi-level columns: PRED_T_2M_ctrl_0, PRED_T_2M_ctrl_1, etc.
        new_columns = [f"{col[0]}_{col[1]}" for col in df.columns]
        df = df.rename(dict(zip(df.columns, new_columns)))
    df.write_csv(filename)
    print(f"Data saved to {filename}")
