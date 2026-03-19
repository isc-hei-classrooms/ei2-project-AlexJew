"""Download MeteoSwiss historical weather data for training."""

from utils.download_meteoswiss import download_meteoswiss

if __name__ == "__main__":
    download_meteoswiss(
        start="2022-10-01T00:00:00Z",
        stop="2025-09-30T23:59:59Z",
        filename_prefix="sion_weather",
    )
