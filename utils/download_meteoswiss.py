import certifi
import os
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv

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
