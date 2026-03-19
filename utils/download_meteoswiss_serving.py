"""Download MeteoSwiss forecast data for serving."""

from datetime import UTC, datetime, timedelta

from utils.download_meteoswiss import download_meteoswiss

if __name__ == "__main__":
    now = datetime.now(tz=UTC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    stop = start + timedelta(days=2)
    download_meteoswiss(
        start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        stop=stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
        filename_prefix="sion_forecast",
    )
