import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt

    return mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Energy Load & Production Forecasting

    This notebook analyzes data for energy load and production forecasting.

    ## Available Data

    | File | Description | Resolution |
    |------|-------------|------------|
    | `oiken_data.csv` | Electricity load (standardised) and solar production by area | 15-min |
    | `sion_weather_full.csv` | Weather measurements and forecasts from MeteoSwiss (Sion) | 10-min |

    ## OIKEN Data Variables
    - **standardised load [-]**: Net electricity consumption (standardised)
    - **standardised forecast load [-]**: Forecasted load
    - **Solar production [kWh]**: Central Valais, Sion, Sierre, Remote areas

    ## Weather Variables
    - **Current**: Temperature, pressure, global radiation, wind, precipitation, humidity
    - **Forecasts (PRED_*)**: 12-hour predictions for multiple weather variables

    ## Analysis Sections

    1. **Data Loading & Overview**
    2. **Time Series Visualization**
    3. **Correlation Analysis**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Data Loading

    Loading OIKEN and weather data with Polars...
    """)
    return


@app.cell(hide_code=True)
def _(pl):
    oiken_df = pl.read_csv(
        "data/oiken_data.csv",
        null_values=["#N/A"],
        schema_overrides={
            "central valais solar production [kWh]": pl.Float64,
            "sion area solar production [kWh]": pl.Float64,
            "sierre area production [kWh]": pl.Float64,
            "remote solar production [kWh]": pl.Float64,
        }
    )
    # Handle two different date formats: try 4-digit year first, then 2-digit
    oiken_df = oiken_df.with_columns(
        pl.col("timestamp")
        .str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False)
        .fill_null(
            pl.col("timestamp").str.strptime(pl.Datetime, "%d/%m/%y %H:%M", strict=False)
        )
        .alias("timestamp")
    )
    oiken_df
    return (oiken_df,)


@app.cell
def _(oiken_df):
    oiken_df.schema
    return


@app.cell
def _(pl):
    weather_df = pl.read_csv("data/sion_weather_full.csv", try_parse_dates=True)
    # Strip UTC timezone to match OIKEN data format (both naive datetimes)
    weather_df = weather_df.with_columns(
        pl.col("timestamp").dt.replace_time_zone(None)
    )
    weather_df
    return (weather_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Data Overview
    """)
    return


@app.cell(hide_code=True)
def _(mo, oiken_df, weather_df):
    mo.hstack([
        mo.md(f"""**OIKEN Data**
    - Rows: {oiken_df.height:,}
    - Columns: {oiken_df.width}
    - Date range: {oiken_df['timestamp'].min()} → {oiken_df['timestamp'].max()}"""),
        mo.md(f"""**Weather Data**
    - Rows: {weather_df.height:,}
    - Columns: {weather_df.width}
    - Date range: {weather_df['timestamp'].min()} → {weather_df['timestamp'].max()}"""),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Correlation Analysis

    Computing correlation between weather variables and load...
    """)
    return


@app.cell
def _(oiken_df, weather_df):
    # Select weather columns (polars sanitizes JSON column names)
    weather_resampled = weather_df.select(
        "timestamp",
        "Air temperature 2m above ground (current value)_0",
        "Global radiation (ten minutes mean)_0",
        "Relative air humidity 2m above ground (current value)_0",
    ).rename({
        "Air temperature 2m above ground (current value)_0": "temperature",
        "Global radiation (ten minutes mean)_0": "global_radiation",
        "Relative air humidity 2m above ground (current value)_0": "humidity",
    })

    # Merge datasets on timestamp (both now naive datetimes)
    merged_df = oiken_df.join(
        weather_resampled,
        on="timestamp",
        how="inner"
    ).select(
        "timestamp",
        "standardised load [-]",
        "temperature",
        "global_radiation",
        "humidity",
        "central valais solar production [kWh]",
    )

    merged_df
    return (merged_df,)


@app.cell
def _(merged_df, mo):
    mo.md(f"""
    **Merged Dataset**: {merged_df.height:,} rows (aligned timestamps)

    Variables for correlation:
    - Standardised load
    - Temperature
    - Global radiation
    - Humidity
    - Solar production
    """)
    return


@app.cell
def _(merged_df, pl):
    # Compute correlation matrix
    corr_matrix = merged_df.select(
        pl.col("standardised load [-]").cast(float),
        pl.col("temperature").cast(float),
        pl.col("global_radiation").cast(float),
        pl.col("humidity").cast(float),
        pl.col("central valais solar production [kWh]").cast(float),
    ).corr()

    corr_matrix
    return


if __name__ == "__main__":
    app.run()
