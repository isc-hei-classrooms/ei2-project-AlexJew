"""Download MeteoSwiss historical weather data for training."""

from datetime import date

import polars as pl

from download_meteoswiss import download_meteoswiss, save_meteoswiss

START = date(2022, 10, 1)
STOP = date(2025, 9, 30)

if __name__ == "__main__":
    frames: list[pl.DataFrame] = []
    current = START
    while current < STOP:
        # Advance to the first day of the next month (or STOP)
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        end = min(next_month, STOP)

        chunk = download_meteoswiss(
            start=f"{current.isoformat()}T00:00:00Z",
            stop=f"{end.isoformat()}T00:00:00Z",
        )
        print(f"Downloaded {current.strftime('%B %Y')}")
        if not chunk.is_empty():
            frames.append(chunk)

        current = next_month

    df = pl.concat(frames).sort("timestamp")
    save_meteoswiss(df, filename_prefix="sion_weather")
