"""Download MeteoSwiss historical weather data for training."""

from download_meteoswiss import download_meteoswiss, save_meteoswiss

if __name__ == "__main__":
    df = download_meteoswiss(
        start="2022-10-01T00:00:00Z",
        stop="2025-09-30T23:59:59Z",
    )
    save_meteoswiss(df, filename_prefix="sion_weather")
