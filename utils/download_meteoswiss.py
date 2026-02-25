import certifi
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
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
    
    query = 'from(bucket:"' + bucket + '")\
    |> range(start: -3h, stop: now())\
    |> filter(fn: (r) => r["_measurement"] == "Air temperature 2m above ground (current value)")\
    |> filter(fn: (r) => r["Site"] == "Sion")'
    tables = client.query_api().query(org=org, query=query)
    times = []
    data = []
    for table in tables:
        for record in table.records:
            times.append(record["_time"])
            data.append(record["_value"])
    client.close()

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = data_dir / f"sion_temperature_{timestamp}.csv"

    # Save to CSV
    df = pd.DataFrame({"timestamp": times, "temperature": data})
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
