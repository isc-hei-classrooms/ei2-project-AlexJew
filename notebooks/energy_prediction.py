import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    # Library import
    import altair as alt
    import marimo as mo
    import polars as pl
    import numpy as np
    import datetime
    import sys
    import os

    from sklearn.feature_selection import mutual_info_regression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler as _StdScaler

    from pathlib import Path

    utils_path = Path("./utils")

    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))

    from feature_engineering import (
        add_temporal_features,
        add_holiday_features,
        add_working_day_flag,
        add_dst_feature,
        add_cyclical_features,
        add_lag_features,
        compute_poa_irradiance,
        estimate_solar_capacity,
        add_remote_yield_ratio,
    )

    import model_preparation
    import metrics
    import joblib
    import lightgbm as lgb

    # Global parameters
    SPLIT_DATE = datetime.datetime(2024, 10, 1)
    return (
        IsotonicRegression,
        SPLIT_DATE,
        add_cyclical_features,
        add_dst_feature,
        add_holiday_features,
        add_lag_features,
        add_remote_yield_ratio,
        add_temporal_features,
        add_working_day_flag,
        alt,
        compute_poa_irradiance,
        datetime,
        estimate_solar_capacity,
        joblib,
        lgb,
        metrics,
        mo,
        model_preparation,
        mutual_info_regression,
        np,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Energy net load forecasting

    This notebook analyzes data for net energy load forecasting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Data loading

    There are three data sources available

    | File | Description | Resolution |
    |------|-------------|------------|
    | `oiken_data.csv` | Electricity load (standardised) and solar production by area | 15-min |
    | `weather_station_forecast.csv` | Weather forecasts from MeteoSwiss | 1-h |
    | `weather_station_measurement.csv` | Historical weather measurements from MeteoSwiss | 10-min |

    OIKEN data variables
    - **standardised load [-]**: Net electricity consumption (standardised)
    - **standardised forecast load [-]**: Forecasted load
    - **Solar production [kWh]**: Central Valais, Sion, Sierre, Remote areas

    Weather variables
    - **Forecasts (PRED_*)**: 24-hour predictions for multiple weather variables taken from the prediction made at 9AM the day before
    - **Measurements**: Air temperature, global radiation, precipitation, relative humidity, sunshine duration
    """)
    return


@app.cell(hide_code=True)
def _(mo, pl):
    oiken_df = pl.read_csv(
        "data/oiken_data.csv",
        null_values=["#N/A"],
        schema_overrides={
            "central valais solar production [kWh]": pl.Float64,
            "sion area solar production [kWh]": pl.Float64,
            "sierre area production [kWh]": pl.Float64,
            "remote solar production [kWh]": pl.Float64,
        },
    )
    # Handle two date formats in the CSV:
    #   Early rows: "1/10/22 0:15"  (2-digit year)
    #   Later rows: "29/09/2025 23:00" (4-digit year)
    # Try 2-digit year first (%y maps 22->2022), then 4-digit year
    oiken_df = oiken_df.with_columns(
        pl.col("timestamp")
        .str.strptime(pl.Datetime, "%d/%m/%y %H:%M", strict=False)
        .fill_null(
            pl.col("timestamp")
            .str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False)
        )
        .alias("timestamp")
    )
    mo.accordion({"OIKEN raw data": oiken_df})
    return (oiken_df,)


@app.cell(hide_code=True)
def _(mo, pl):
    _forecast_files = {
        "sion": "data/sion_forecast_2026-03-24_18-31.csv",
        "evionnaz": "data/evionnaz_forecast_2026-04-10_10-12.csv",
        "evolene": "data/evolene_villa_forecast_2026-04-10_10-11.csv",
        "montana": "data/montana_forecast_2026-04-10_10-07.csv",
        "visp": "data/visp_forecast_2026-04-10_10-11.csv",
        "basel": "data/basel_forecast_2026-04-10_10-39.csv",
        "bern": "data/bern_forecast_2026-04-10_10-39.csv",
        "geneve": "data/geneve_forecast_2026-04-10_10-39.csv",
        "pully": "data/pully_forecast_2026-04-10_10-39.csv",
        "zurich": "data/zurich_forecast_2026-04-10_10-39.csv",
    }

    _raw_to_clean = {
        "PRED_T_2M_ctrl": "forecast_temperature",
        "PRED_GLOB_ctrl": "forecast_global_radiation",
        "PRED_TOT_PREC_ctrl": "forecast_precipitation",
        "PRED_RELHUM_2M_ctrl": "forecast_humidity",
        "PRED_DURSUN_ctrl": "forecast_sunshine_duration",
    }

    weather_df = None
    for _name, _path in _forecast_files.items():
        _df = pl.read_csv(_path, try_parse_dates=True)
        _df = _df.rename({k: f"{_name}_{v}" for k, v in _raw_to_clean.items()})
        if weather_df is None:
            weather_df = _df
        else:
            weather_df = weather_df.join(
                _df, on="timestamp", how="full", coalesce=True
            )

    weather_df = weather_df.sort("timestamp")
    mo.accordion({"Weather forecast data (all stations)": weather_df})
    return (weather_df,)


@app.cell(hide_code=True)
def _(mo, pl):
    _measurement_files = {
        "sion": "data/sion_measurement_2026-03-26_13-44.csv",
        "evionnaz": "data/evionnaz_measurement_2026-04-10_10-14.csv",
        "evolene": "data/evolene_villa_measurement_2026-04-10_10-14.csv",
        "montana": "data/montana_measurement_2026-04-10_10-09.csv",
        "visp": "data/visp_measurement_2026-04-10_10-13.csv",
        "basel": "data/basel_measurement_2026-04-10_10-41.csv",
        "bern": "data/bern_measurement_2026-04-10_10-41.csv",
        "geneve": "data/geneve_measurement_2026-04-10_10-41.csv",
        "pully": "data/pully_measurement_2026-04-10_10-41.csv",
        "zurich": "data/zurich_measurement_2026-04-10_10-41.csv",
    }

    _raw_to_clean = {
        "Air temperature 2m above ground (current value)": "measured_temperature",
        "Global radiation (ten minutes mean)": "measured_global_radiation",
        "Precipitation (ten minutes total)": "measured_precipitation",
        "Relative air humidity 2m above ground (current value)": "measured_humidity",
        "Sunshine duration (ten minutes total)": "measured_sunshine_duration",
    }

    measurement_df = None
    for _name, _path in _measurement_files.items():
        _df = pl.read_csv(_path, try_parse_dates=True).filter(
            pl.col("timestamp").dt.minute().is_in([0, 30])
        )
        _df = _df.rename({k: f"{_name}_{v}" for k, v in _raw_to_clean.items()})
        if measurement_df is None:
            measurement_df = _df
        else:
            measurement_df = measurement_df.join(
                _df, on="timestamp", how="full", coalesce=True
            )

    measurement_df = measurement_df.sort("timestamp")
    mo.accordion({"Weather measurement data (all stations)": measurement_df})
    return (measurement_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Data processing
    - Renaming of the columns
    - Filtering and merging of the two datasets
    - Forward fill of the missing values (appropriate since consecutive values are correlated)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Renaming of the columns
    The first part relates to the replacement of the column names by easier names written in _snake_case_
    """)
    return


@app.cell(hide_code=True)
def _(mo, oiken_df):
    oiken_renamed = oiken_df.rename(
        {
            "standardised load [-]": "load",
            "standardised forecast load [-]": "forecast_load",
            "central valais solar production [kWh]": "solar_central_valais",
            "sion area solar production [kWh]": "solar_sion",
            "sierre area production [kWh]": "solar_sierre",
            "remote solar production [kWh]": "solar_remote",
        }
    )
    mo.accordion({"OIKEN renamed": oiken_renamed})
    return (oiken_renamed,)


@app.cell(hide_code=True)
def _(mo, weather_df):
    weather_renamed = weather_df.select(
        [c for c in weather_df.columns if not c.endswith("_0")]
    )
    mo.accordion({"Weather forecast renamed": weather_renamed})
    return (weather_renamed,)


@app.cell(hide_code=True)
def _(measurement_df, mo):
    measurement_renamed = measurement_df
    mo.accordion({"Weather measurement renamed": measurement_renamed})
    return (measurement_renamed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Merging of the data sets
    The OIKEN timestamps (Swiss local time) are converted to UTC to match the weather forecast data. Both datasets then use naive UTC timestamps stored in the `utc_timestamp` column. A `local_timestamp` column (Europe/Zurich, naive) is added for display purposes.

    Important: The UTC offset depends on whether it is winter or summer ⚠️

    The expected offsets for Europe/Zurich are:
      - Winter (CET): UTC+1 → 1 hour offset
      - Summer (CEST): UTC+2 → 2 hours offset

    The switch happens on the last Sunday of March (→ CEST) and last Sunday of October (→ CET).
    Ambiguous timestamps during the autumn DST overlap are assigned to the earlier (CEST) occurrence.
    """)
    return


@app.cell(hide_code=True)
def _(measurement_renamed, mo, oiken_renamed, pl, weather_renamed):
    # Convert OIKEN timestamps from Swiss local time to naive UTC
    # non_existent="null" handles the spring DST gap (02:00-03:00 doesn't exist)
    oiken_utc = oiken_renamed.with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone(
            "Europe/Zurich", ambiguous="earliest", non_existent="null"
        )
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
    ).with_columns(pl.col("timestamp").forward_fill())

    # Strip timezone from weather timestamps (already UTC)
    weather_utc = weather_renamed.with_columns(
        pl.col("timestamp").dt.replace_time_zone(None)
    )
    measurement_utc = measurement_renamed.with_columns(
        pl.col("timestamp").dt.replace_time_zone(None)
    )

    # Merge datasets on timestamp (outer join), then rename to utc_timestamp
    merged_df = (
        oiken_utc.join(weather_utc, on="timestamp", how="full", coalesce=True)
        .join(measurement_utc, on="timestamp", how="full", coalesce=True)
        .sort("timestamp")
        .rename({"timestamp": "utc_timestamp"})
        .with_columns(
            pl.col("utc_timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Europe/Zurich")
            .dt.replace_time_zone(None)
            .alias("local_timestamp")
        )
        .select(
            "utc_timestamp",
            "local_timestamp",
            pl.exclude("utc_timestamp", "local_timestamp"),
        )
    )
    mo.accordion({"Merged dataset": merged_df})
    return (merged_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Cleaning of the data set
    The null values are cleaned and filled with through a forward fill. We look for null values, empty values and negative counts (where there shouldn't be any).

    Important: The forward fill drops all the rows for which contain a null cell for which there is no earlier data available ⚠️
    """)
    return


@app.cell(hide_code=True)
def _(merged_df, mo, pl):
    invalid_counts = pl.DataFrame({
        "column": merged_df.columns,
        "null_count": [merged_df[col].null_count() for col in merged_df.columns],
        "empty_count": [
            (merged_df[col] == "").sum()
            if merged_df[col].dtype == pl.Utf8 
            else 0 
            for col in merged_df.columns
        ],
        "negative_count": [
            (merged_df[col] < 0).sum()
            if merged_df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]
            else 0
            for col in merged_df.columns
        ],
        "min_value": [str(merged_df[col].min()) for col in merged_df.columns],
        "max_value": [str(merged_df[col].max()) for col in merged_df.columns],
    })

    mo.accordion({"Invalid value counts per column": invalid_counts})
    return


@app.cell(hide_code=True)
def _(merged_df, mo, pl):
    # Replace negative forecasted values by zero for all stations
    _clip_suffixes = [
        "forecast_global_radiation",
        "forecast_precipitation",
        "forecast_humidity",
        "forecast_sunshine_duration",
    ]
    df_clean = merged_df.with_columns(
        [
            pl.col(c).clip(lower_bound=0)
            for c in merged_df.columns
            if any(c.endswith(s) for s in _clip_suffixes)
        ]
    )

    # Handle missing values with time-series aware forward-fill
    df_clean = df_clean.fill_null(strategy="forward")

    # Drop any remaining nulls at the beginning of the dataset
    df_clean = df_clean.drop_nulls()

    # Show cleaning summary
    original_nulls = merged_df.null_count().row(0)
    remaining_nulls = df_clean.null_count().row(0)
    total_cleaned = sum(original_nulls) - sum(remaining_nulls)

    mo.accordion(
        {
            "Data cleaning applied": mo.vstack(
                [
                    mo.md(f"""
                **Cleaned**: {total_cleaned:,} values filled
                """),
                    df_clean,
                ]
            )
        }
    )
    return (df_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Data visualization

    Exploring the cleaned data across three categories: electrical load, weather forecasts, and solar production.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.1 Load data

    Time series of the standardised load, OIKEN forecast, and two potential naive models (day before and week before). The rolling mean absolute error is computed for each model, making it possible to explore the evolution of the error on different time intervals.

    Conclusion: The week before model performs badly compared to the day before model 💡
    """)
    return


@app.cell(hide_code=True)
def _(df_clean, mo):
    _min_date = df_clean["utc_timestamp"].min().date()
    _max_date = df_clean["utc_timestamp"].max().date()
    load_date_start = mo.ui.date(value=_min_date, label="Start date")
    load_mae_window = mo.ui.slider(
        start=1, stop=24 * 7, value=24, step=1, label="Rolling MAE window (hours)"
    )
    load_date_end = mo.ui.date(value=_max_date, label="End date")
    load_series_select = mo.ui.multiselect(
        options=["load", "OIKEN_forecast", "day_before", "week_before"],
        value=["load", "OIKEN_forecast"],
        label="Series",
    )
    mae_series_select = mo.ui.multiselect(
        options=["mae_OIKEN", "mae_day_before", "mae_week_before"],
        value=["mae_OIKEN", "mae_day_before", "mae_week_before"],
        label="Models",
    )
    return (
        load_date_end,
        load_date_start,
        load_mae_window,
        load_series_select,
        mae_series_select,
    )


@app.cell(hide_code=True)
def _(
    alt,
    df_clean,
    load_date_end,
    load_date_start,
    load_series_select,
    mo,
    pl,
):
    _selected = list(load_series_select.value)

    if not _selected:
        _output = mo.md("> Select at least one series to display.")
    else:
        _interval_min = (
            df_clean["utc_timestamp"][1] - df_clean["utc_timestamp"][0]
        ).total_seconds() / 60
        _rows_per_day = int(24 * 60 / _interval_min)

        _with_baselines = df_clean.sort("utc_timestamp").with_columns(
            [
                pl.col("load").shift(_rows_per_day).alias("day_before"),
                pl.col("load").shift(_rows_per_day * 7).alias("week_before"),
                pl.col("forecast_load").alias("OIKEN_forecast"),
            ]
        )

        _filtered = _with_baselines.filter(
            pl.col("utc_timestamp")
            .dt.date()
            .is_between(load_date_start.value, load_date_end.value)
        )

        _load_data = (
            _filtered.select("utc_timestamp", *_selected)
            .sample(n=min(5000, _filtered.height), seed=42)
            .sort("utc_timestamp")
        )

        _load_long = _load_data.unpivot(
            index="utc_timestamp",
            on=_selected,
            variable_name="series",
            value_name="value",
        )

        _output = (
            alt.Chart(_load_long)
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("utc_timestamp:T", title="Time"),
                y=alt.Y("value:Q", title="Load (standardised)"),
                color=alt.Color(
                    "series:N",
                    title="Series",
                    scale=alt.Scale(
                        domain=[
                            "load",
                            "OIKEN_forecast",
                            "day_before",
                            "week_before",
                        ],
                        range=["#4c78a8", "#f58518", "#54a24b", "#e45756"],
                    ),
                ),
                opacity=alt.value(0.7),
            )
            .properties(width="container", height=300, title="Load vs forecasts")
            .interactive()
        )

    mo.accordion(
        {
            "Load vs forecasts": mo.vstack(
                [
                    mo.hstack(
                        [load_series_select, load_date_start, load_date_end]
                    ),
                    _output,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(
    alt,
    df_clean,
    load_date_end,
    load_date_start,
    load_mae_window,
    mae_series_select,
    mo,
    pl,
):
    _selected_mae = list(mae_series_select.value)

    if not _selected_mae:
        _mae_output = mo.md("> Select at least one model to display.")
        _mae_md = mo.md("")
    else:
        _interval_min = (
            df_clean["utc_timestamp"][1] - df_clean["utc_timestamp"][0]
        ).total_seconds() / 60
        _rows_per_day = int(24 * 60 / _interval_min)
        _rows_per_hour = int(60 / _interval_min)

        _with_errors = (
            df_clean.sort("utc_timestamp")
            .with_columns(
                [
                    pl.col("load").shift(_rows_per_day).alias("day_before"),
                    pl.col("load").shift(_rows_per_day * 7).alias("week_before"),
                    pl.col("forecast_load").alias("OIKEN_forecast"),
                ]
            )
            .with_columns(
                [
                    (pl.col("OIKEN_forecast") - pl.col("load"))
                    .abs()
                    .alias("err_OIKEN"),
                    (pl.col("day_before") - pl.col("load"))
                    .abs()
                    .alias("err_day_before"),
                    (pl.col("week_before") - pl.col("load"))
                    .abs()
                    .alias("err_week_before"),
                ]
            )
            .with_columns(
                [
                    pl.col("err_OIKEN")
                    .rolling_mean(
                        window_size=load_mae_window.value * _rows_per_hour
                    )
                    .alias("mae_OIKEN"),
                    pl.col("err_day_before")
                    .rolling_mean(
                        window_size=load_mae_window.value * _rows_per_hour
                    )
                    .alias("mae_day_before"),
                    pl.col("err_week_before")
                    .rolling_mean(
                        window_size=load_mae_window.value * _rows_per_hour
                    )
                    .alias("mae_week_before"),
                ]
            )
        )

        _filtered = _with_errors.filter(
            pl.col("utc_timestamp")
            .dt.date()
            .is_between(load_date_start.value, load_date_end.value)
        )

        _sampled = (
            _filtered.select("utc_timestamp", *_selected_mae)
            .drop_nulls()
            .sample(n=min(5000, _filtered.height), seed=42)
            .sort("utc_timestamp")
        )

        _mae_long = _sampled.unpivot(
            index="utc_timestamp",
            on=_selected_mae,
            variable_name="model",
            value_name="MAE",
        )

        _mae_output = (
            alt.Chart(_mae_long)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("utc_timestamp:T", title="Time"),
                y=alt.Y("MAE:Q", title="Rolling MAE (standardised)"),
                color=alt.Color(
                    "model:N",
                    title="Model",
                    scale=alt.Scale(
                        domain=["mae_OIKEN", "mae_day_before", "mae_week_before"],
                        range=["#f58518", "#54a24b", "#e45756"],
                    ),
                ),
            )
            .properties(
                width="container",
                height=250,
                title=f"Forecast error – rolling MAE ({load_mae_window.value}h window)",
            )
            .interactive()
        )

    _avg_mae = _filtered.select(
        pl.col("err_OIKEN").mean().alias("OIKEN"),
        pl.col("err_day_before").mean().alias("Day before"),
        pl.col("err_week_before").mean().alias("Week before"),
    ).row(0)

    _mae_md = mo.md(
        f"""**Average MAE (standardised)** over selected period
        - OIKEN: `{_avg_mae[0]:.4f}` 
        - Day before: `{_avg_mae[1]:.4f}` 
        - Week before: `{_avg_mae[2]:.4f}`"""
    )

    mo.vstack(
        [
            mo.accordion(
                {
                    "Forecast error (rolling MAE)": mo.vstack(
                        [
                            _mae_md,
                            mo.hstack([mae_series_select, load_mae_window]),
                            _mae_output,
                        ]
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.2 Weather forecast data

    Time series of the weather forecast variables. Select a variable from the dropdown below.
    """)
    return


@app.cell(hide_code=True)
def _(df_clean, mo):
    _min_date = df_clean["utc_timestamp"].min().date()
    _max_date = df_clean["utc_timestamp"].max().date()
    weather_var_dropdown = mo.ui.dropdown(
        options={
            "Temperature": "forecast_temperature",
            "Global radiation": "forecast_global_radiation",
            "Precipitation": "forecast_precipitation",
            "Humidity": "forecast_humidity",
            "Sunshine duration": "forecast_sunshine_duration",
        },
        value="Temperature",
        label="Variable",
    )
    weather_station_select = mo.ui.multiselect(
        options=[
            "sion",
            "evionnaz",
            "evolene",
            "montana",
            "visp",
            "basel",
            "bern",
            "geneve",
            "pully",
            "zurich",
        ],
        value=["sion"],
        label="Stations",
    )
    weather_date_start = mo.ui.date(value=_min_date, label="Start date")
    weather_date_end = mo.ui.date(value=_max_date, label="End date")
    return (
        weather_date_end,
        weather_date_start,
        weather_station_select,
        weather_var_dropdown,
    )


@app.cell(hide_code=True)
def _(
    alt,
    df_clean,
    mo,
    pl,
    weather_date_end,
    weather_date_start,
    weather_station_select,
    weather_var_dropdown,
):
    _var = weather_var_dropdown.value
    _stations = list(weather_station_select.value)

    if not _stations:
        _output = mo.md("> Select at least one station to display.")
    else:
        _cols = [f"{s}_{_var}" for s in _stations]
        _filtered = df_clean.filter(
            pl.col("utc_timestamp")
            .dt.date()
            .is_between(weather_date_start.value, weather_date_end.value)
        )
        _weather_data = (
            _filtered.select("utc_timestamp", *_cols)
            .sample(n=min(5000, _filtered.height), seed=42)
            .sort("utc_timestamp")
        )

        _weather_long = _weather_data.unpivot(
            index="utc_timestamp",
            on=_cols,
            variable_name="station",
            value_name="value",
        )

        _output = (
            alt.Chart(_weather_long)
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("utc_timestamp:T", title="Time"),
                y=alt.Y("value:Q", title=_var),
                color=alt.Color("station:N", title="Station"),
            )
            .properties(width="container", height=300, title=_var)
            .interactive()
        )

    mo.accordion(
        {
            "Weather forecast data": mo.vstack(
                [
                    mo.hstack(
                        [
                            weather_var_dropdown,
                            weather_station_select,
                            weather_date_start,
                            weather_date_end,
                        ]
                    ),
                    _output,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.3 Solar production and irradiance

    Compare solar station output with radiation and sunshine forecasts. Use the multiselect widgets below to toggle series on/off. Solar production uses the left Y-axis (kWh), weather forecasts use the right Y-axis.

    Important: The solar sion production only starts in mid 2023 ⚠️

    Conclusion:

    - The global radiation forecast is a better predictor of solar production than the sunshine duration 💡
    - From the gross vs net production analysis, it stands out that the behaviors from all four solar production plants are aligned. This means that they are all probably providing net injection values (i.e. solar production from which self-consumption is deduced)
    - The solar production is mostly all included inside solar_remote
    """)
    return


@app.cell(hide_code=True)
def _(df_clean, mo):
    _min_date = df_clean["utc_timestamp"].min().date()
    _max_date = df_clean["utc_timestamp"].max().date()
    _stations = [
        "sion",
        "evionnaz",
        "evolene",
        "montana",
        "visp",
        "basel",
        "bern",
        "geneve",
        "pully",
        "zurich",
    ]

    solar_viz_select = mo.ui.multiselect(
        options=[
            "solar_central_valais",
            "solar_sion",
            "solar_sierre",
            "solar_remote",
        ],
        value=[
            "solar_central_valais",
            "solar_sion",
            "solar_sierre",
            "solar_remote",
        ],
        label="Solar stations",
    )
    weather_viz_select = mo.ui.multiselect(
        options=[
            f"{s}_forecast_{v}"
            for s in _stations
            for v in ["global_radiation", "sunshine_duration"]
        ],
        value=[],
        label="Weather forecasts",
    )
    load_viz_select = mo.ui.multiselect(
        options=["load", "forecast_load"],
        value=[],
        label="Load",
    )
    measurement_viz_select = mo.ui.multiselect(
        options=[
            f"{s}_measured_{v}"
            for s in _stations
            for v in ["global_radiation", "sunshine_duration"]
        ],
        value=[],
        label="Weather measurements",
    )
    solar_date_start = mo.ui.date(value=_min_date, label="Start date")
    solar_date_end = mo.ui.date(value=_max_date, label="End date")
    return (
        load_viz_select,
        measurement_viz_select,
        solar_date_end,
        solar_date_start,
        solar_viz_select,
        weather_viz_select,
    )


@app.cell(hide_code=True)
def _(alt, df_clean, mo, pl):
    _solar_cols = [
        "solar_central_valais",
        "solar_sion",
        "solar_sierre",
        "solar_remote",
    ]
    _daily = (
        df_clean.with_columns(pl.col("utc_timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg([pl.col(c).mean() for c in _solar_cols])
        .sort("date")
    )
    _daily_total_max = _daily.select(
        sum(pl.col(c).fill_null(0) for c in _solar_cols).max()
    ).item()
    _daily_total_max = (
        _daily_total_max if _daily_total_max and _daily_total_max > 0 else 1.0
    )
    _daily_norm = _daily.select(
        "date",
        *[
            (pl.col(c).fill_null(0) / _daily_total_max).alias(c)
            for c in _solar_cols
        ],
    )
    _daily_long = _daily_norm.unpivot(
        index="date",
        on=_solar_cols,
        variable_name="series",
        value_name="value",
    )
    _daily_output = (
        alt.Chart(_daily_long)
        .mark_area(opacity=0.5)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", stack=True, title="Normalised daily mean (0–max)"),
            color=alt.Color("series:N", title="Series"),
        )
        .properties(
            width="container", height=300, title="Daily average – solar production"
        )
        .interactive()
    )

    mo.accordion({"Solar production (daily average)": _daily_output})
    return


@app.cell(hide_code=True)
def _(
    alt,
    df_clean,
    load_viz_select,
    measurement_viz_select,
    mo,
    pl,
    solar_date_end,
    solar_date_start,
    solar_viz_select,
    weather_viz_select,
):
    _solar_cols = list(solar_viz_select.value)
    _weather_cols = list(weather_viz_select.value)
    _load_cols = list(load_viz_select.value)
    _measurement_cols = list(measurement_viz_select.value)

    if (
        not _solar_cols
        and not _weather_cols
        and not _load_cols
        and not _measurement_cols
    ):
        _output = mo.md("> Select at least one series to display.")
    else:
        # Compute global max from full dataset for stable normalization
        _all_cols = _solar_cols + _weather_cols + _load_cols + _measurement_cols
        _global_max = {c: df_clean[c].max() for c in _all_cols}

        _filtered = df_clean.filter(
            pl.col("utc_timestamp")
            .dt.date()
            .is_between(solar_date_start.value, solar_date_end.value)
        )
        _sampled = _filtered.sample(n=min(5000, _filtered.height), seed=42).sort(
            "utc_timestamp"
        )

        _layers = []

        if _solar_cols:
            # Normalise by the max of the total solar production (stacked sum peaks at 1)
            _solar_total_max = _sampled.select(
                sum(pl.col(c).fill_null(0) for c in _solar_cols).max()
            ).item()
            _solar_total_max = (
                _solar_total_max
                if _solar_total_max and _solar_total_max > 0
                else 1.0
            )
            _solar_norm = _sampled.select(
                "utc_timestamp",
                *[
                    (pl.col(c).fill_null(0) / _solar_total_max).alias(c)
                    for c in _solar_cols
                ],
            )
            _solar_long = _solar_norm.unpivot(
                index="utc_timestamp",
                on=_solar_cols,
                variable_name="series",
                value_name="value",
            )
            _solar_chart = (
                alt.Chart(_solar_long)
                .mark_area(opacity=0.5)
                .encode(
                    x=alt.X("utc_timestamp:T", title="Time"),
                    y=alt.Y("value:Q", stack=True, title="Normalised (0–max)"),
                    color=alt.Color("series:N", title="Series"),
                )
            )
            _layers.append(_solar_chart)

        if _weather_cols:
            _weather_norm = _sampled.select(
                "utc_timestamp",
                *[
                    (pl.col(c) / _global_max[c]).alias(c)
                    if _global_max[c] and _global_max[c] > 0
                    else pl.lit(0.0).alias(c)
                    for c in _weather_cols
                ],
            )
            _weather_long = _weather_norm.unpivot(
                index="utc_timestamp",
                on=_weather_cols,
                variable_name="series",
                value_name="value",
            )
            _weather_chart = (
                alt.Chart(_weather_long)
                .mark_line(opacity=0.7)
                .encode(
                    x=alt.X("utc_timestamp:T", title="Time"),
                    y=alt.Y("value:Q", title="Normalised (0–max)"),
                    color=alt.Color("series:N", title="Series"),
                )
            )
            _layers.append(_weather_chart)

        if _load_cols:
            # Shift by global min then divide by range so negative values map to [0, 1]
            _global_min = {c: df_clean[c].min() for c in _load_cols}
            _load_norm = _sampled.select(
                "utc_timestamp",
                *[
                    (
                        (pl.col(c) - _global_min[c])
                        / (_global_max[c] - _global_min[c])
                    ).alias(c)
                    if _global_max[c] is not None
                    and _global_min[c] is not None
                    and _global_max[c] != _global_min[c]
                    else pl.lit(0.0).alias(c)
                    for c in _load_cols
                ],
            )
            _load_long = _load_norm.unpivot(
                index="utc_timestamp",
                on=_load_cols,
                variable_name="series",
                value_name="value",
            )
            _load_chart = (
                alt.Chart(_load_long)
                .mark_line(opacity=0.7, strokeDash=[4, 2])
                .encode(
                    x=alt.X("utc_timestamp:T", title="Time"),
                    y=alt.Y("value:Q", title="Normalised (0–max)"),
                    color=alt.Color("series:N", title="Series"),
                )
            )
            _layers.append(_load_chart)

        if _measurement_cols:
            _meas_norm = _sampled.select(
                "utc_timestamp",
                *[
                    (pl.col(c) / _global_max[c]).alias(c)
                    if _global_max[c] and _global_max[c] > 0
                    else pl.lit(0.0).alias(c)
                    for c in _measurement_cols
                ],
            )
            _meas_long = _meas_norm.unpivot(
                index="utc_timestamp",
                on=_measurement_cols,
                variable_name="series",
                value_name="value",
            )
            _meas_chart = (
                alt.Chart(_meas_long)
                .mark_line(opacity=0.7, strokeDash=[2, 2])
                .encode(
                    x=alt.X("utc_timestamp:T", title="Time"),
                    y=alt.Y("value:Q", title="Normalised (0–max)"),
                    color=alt.Color("series:N", title="Series"),
                )
            )
            _layers.append(_meas_chart)

        _combined = (
            alt.layer(*_layers)
            .properties(
                width="container",
                height=300,
                title="Solar production & irradiance (quarter-hourly)",
            )
            .interactive()
        )

        _output = _combined

    mo.accordion(
        {
            "Solar production & irradiance (quarter-hourly)": mo.vstack(
                [
                    mo.hstack(
                        [
                            solar_viz_select,
                            weather_viz_select,
                            measurement_viz_select,
                            load_viz_select,
                        ]
                    ),
                    mo.hstack([solar_date_start, solar_date_end]),
                    _output,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(alt, df_clean, mo, pl):
    _solar_cols = [
        "solar_central_valais",
        "solar_sion",
        "solar_sierre",
        "solar_remote",
    ]

    # Filter to daylight hours (GHI > 50 W/m2)
    _daylight = df_clean.filter(pl.col("sion_forecast_global_radiation") > 50)

    # --- Panel A: Normalised production/GHI ratio by hour ---
    _daylight_h = _daylight.with_columns(
        pl.col("utc_timestamp").dt.hour().alias("_hour"),
        *[
            (pl.col(c) / pl.col("sion_forecast_global_radiation")).alias(
                f"_ratio_{c}"
            )
            for c in _solar_cols
        ],
    )
    _ratio_cols = [f"_ratio_{c}" for c in _solar_cols]

    # Normalise each station's ratio by its overall mean (shape comparison)
    _means = {rc: _daylight_h[rc].drop_nulls().mean() for rc in _ratio_cols}
    _daylight_h = _daylight_h.with_columns(
        *[
            (pl.col(rc) / _means[rc]).alias(rc)
            for rc in _ratio_cols
            if _means[rc] is not None and _means[rc] > 0
        ],
    )

    _hourly = (
        _daylight_h.group_by("_hour")
        .agg([pl.col(rc).mean() for rc in _ratio_cols])
        .sort("_hour")
    )

    _hourly_long = _hourly.unpivot(
        index="_hour",
        on=_ratio_cols,
        variable_name="station",
        value_name="normalised_ratio",
    ).with_columns(
        pl.col("station").str.replace("_ratio_", ""),
    )

    _ratio_chart = (
        alt.Chart(_hourly_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("_hour:O", title="Hour of day (UTC)"),
            y=alt.Y("normalised_ratio:Q", title="Relative efficiency (mean = 1)"),
            color=alt.Color("station:N", title="Station"),
        )
        .properties(
            width=400,
            height=300,
            title="Normalised production efficiency by hour of day",
        )
    )

    # --- Panel B: Min-max normalised daily profile (shape comparison) ---
    _hourly_prod = (
        _daylight.with_columns(
            pl.col("utc_timestamp").dt.hour().alias("_hour"),
        )
        .group_by("_hour")
        .agg([pl.col(c).mean() for c in _solar_cols])
        .sort("_hour")
    )

    # Min-max normalise each station to [0, 1] so shapes are directly comparable
    _hourly_prod = _hourly_prod.with_columns(
        *[
            (
                (pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())
            ).alias(c)
            for c in _solar_cols
        ],
    )
    _profile_long = _hourly_prod.unpivot(
        index="_hour",
        on=_solar_cols,
        variable_name="station",
        value_name="normalised_production",
    )

    _profile_chart = (
        alt.Chart(_profile_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("_hour:O", title="Hour of day (UTC)"),
            y=alt.Y(
                "normalised_production:Q", title="Normalised production (0–1)"
            ),
            color=alt.Color("station:N", title="Station"),
        )
        .properties(
            width=400,
            height=300,
            title="Daily production profile (min-max normalised)",
        )
    )

    # --- Panel C: Pearson correlation with GHI ---
    _corrs = []
    for _c in _solar_cols:
        _valid = _daylight.filter(pl.col(_c).is_not_null())
        _r = _valid.select(pl.corr(_c, "sion_forecast_global_radiation")).item()
        _corrs.append({"station": _c, "pearson_r": _r})

    _corr_df = pl.DataFrame(_corrs).with_columns(
        pl.when(pl.col("station") == "solar_remote")
        .then(pl.lit("Prosumer (net?)"))
        .otherwise(pl.lit("Utility-scale (gross)"))
        .alias("type"),
    )

    _corr_chart = (
        alt.Chart(_corr_df)
        .mark_bar()
        .encode(
            x=alt.X("station:N", title="Station", sort="-y"),
            y=alt.Y(
                "pearson_r:Q", title="Pearson r", scale=alt.Scale(domain=[0, 1])
            ),
            color=alt.Color("type:N", title="Type"),
        )
        .properties(
            width=350,
            height=300,
            title="Correlation with GHI (forecast_global_radiation)",
        )
    )

    mo.accordion(
        {
            "Gross vs. net production analysis": mo.vstack(
                [
                    mo.hstack([_ratio_chart, _profile_chart], gap=2),
                    _corr_chart,
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Feature engineering

    Engineered features capture temporal patterns and structural properties of the energy system:
    - **Daily patterns**: hour of day (morning peak, evening dip)
    - **Weekly patterns**: weekday vs weekend
    - **Seasonal patterns**: month, day of year
    - **Special days**: holidays, working days
    - **DST regime**: winter time (CET) vs summer time (CEST)
    - **Solar capacity**: estimated installed PV capacity over time

    ### Feature extraction strategy
    1. Basic temporal features (hour, day of week, month, etc.)
    2. Swiss holiday calendar (Valais region)
    3. Working day classification
    4. Winter/summer hour (DST)
    5. Cyclical encoding (sin/cos) for periodic features
    6. Estimated total solar capacity (production / irradiation ratio)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.1 Basic temporal features

    Based on the timestamps, it is possible to extract basic temporal features (hour, day of week, month). This results in ordinal features _hour_, _day_of_week_, _month_, and a binary feature _is_weekend_
    """)
    return


@app.cell(hide_code=True)
def _(add_temporal_features, df_clean, mo):
    df_with_temporal = add_temporal_features(df_clean)
    mo.accordion(
        {
            "Temporal features preview": df_with_temporal.select(
                "local_timestamp",
                "local_hour",
                "local_day_of_week",
                "local_month",
                "local_is_weekend",
            )
        }
    )
    return (df_with_temporal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.2 Swiss holiday calendar

    Focusing on the Valais region, it is possible to identify the holiday days. This results in a binary variable _is_holiday_

    **National holidays** (affect all of Switzerland):
    - 1 January: New Year's Day
    - 1 August: Swiss National Day
    - 25 December: Christmas Day
    - 26 December: St. Stephen's Day
    - Moveable: Easter Monday, Ascension, Whit Monday

    **Valais-specific holidays**:
    - 15 August: Assumption of Mary (Fête du 15 août)
    - 1 November: All Saints' Day (Toussaint)
    - Corpus Christi (Fête-Dieu)
    """)
    return


@app.cell(hide_code=True)
def _(add_holiday_features, df_with_temporal, mo, pl):
    df_with_holidays = add_holiday_features(df_with_temporal)

    holiday_examples = df_with_holidays.filter(
        pl.col("local_is_holiday") == True
    ).select("local_timestamp", "local_is_holiday", "local_is_weekend")
    mo.accordion({"Holiday examples": holiday_examples})
    return (df_with_holidays,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.3 Working day classification

    A "working day" is typically defined as:
    - NOT a weekend (Saturday/Sunday)
    - NOT a public holiday

    Working days have different consumption patterns than non-working days
    (industrial/commercial activity is higher).
    """)
    return


@app.cell(hide_code=True)
def _(add_working_day_flag, df_with_holidays, mo, pl):
    df_working_days = add_working_day_flag(df_with_holidays)

    calendar_summary = (
        df_working_days.group_by(pl.col("local_timestamp").dt.date().alias("date"))
        .agg(
            pl.col("local_is_weekend").max().alias("is_weekend_day"),
            pl.col("local_is_holiday").max().alias("is_holiday_day"),
            pl.col("local_is_working_day").max().alias("is_working_day"),
        )
        .select(
            pl.col("is_weekend_day").sum().alias("weekend_days"),
            pl.col("is_holiday_day").sum().alias("holidays"),
            pl.col("is_working_day").sum().alias("working_days"),
        )
    )
    mo.accordion({"Calendar summary": calendar_summary})
    return (df_working_days,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.4 Winter/summer hour (DST)

    Switzerland observes Central European Time (CET, UTC+1) in winter and Central European Summer Time (CEST, UTC+2) in summer. The clock change affects energy consumption patterns — notably lighting demand and heating/cooling schedules.

    The DST flag is derived from the offset between `local_timestamp` and `utc_timestamp`:
    - **Offset = 1 h** → CET (winter time)
    - **Offset = 2 h** → CEST (summer time)
    """)
    return


@app.cell(hide_code=True)
def _(add_dst_feature, df_working_days, mo, pl):
    df_with_dst = add_dst_feature(df_working_days)

    _dst_transitions = (
        df_with_dst.with_columns(
            pl.col("local_is_summer_time").shift(1).alias("_prev")
        )
        .filter(pl.col("local_is_summer_time") != pl.col("_prev"))
        .select("utc_timestamp", "local_timestamp", "local_is_summer_time")
    )
    mo.accordion({"DST transitions": _dst_transitions})
    return (df_with_dst,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.5 Cyclical encoding

    Temporal features like hour, day of week, and month are **periodic** — the distance
    between hour 23 and hour 0 should be small (not 23!), just like the distance between
    Sunday (day 7) and Monday (day 1).

    We encode these periodic features using **sine and cosine transformations**:
    - `sin_hour`, `cos_hour`: Capture hourly patterns
    - `sin_dow`, `cos_dow`: Capture weekly patterns (Monday-Sunday)
    - `sin_month`, `cos_month`: Capture seasonal patterns (January-December)
    - `sin_doy`, `cos_doy`: Capture annual day-of-year patterns (day 1-366)

    This creates a circular representation where the "distance" between adjacent points on the cycle is correctly captured by the model. These features are new numerical features, not to be mistaken with the earlier ordinal features (basic temporal encoding from timestamps).
    """)
    return


@app.cell(hide_code=True)
def _(add_cyclical_features, df_with_dst, mo):
    df_with_cyclical = add_cyclical_features(df_with_dst)
    mo.accordion(
        {
            "Cyclical encoded features (UTC-based)": df_with_cyclical.select(
                "utc_timestamp",
                "local_timestamp",
                "load",
                "local_hour",
                "local_day_of_week",
                "local_month",
                "local_day_of_year",
                "utc_sin_hour",
                "utc_cos_hour",
                "utc_sin_dow",
                "utc_cos_dow",
                "utc_sin_month",
                "utc_cos_month",
                "utc_sin_doy",
                "utc_cos_doy",
            )
        }
    )
    return (df_with_cyclical,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.6 Estimated total solar capacity

    Estimate the evolution of installed PV capacity over time using the ratio of total solar production to irradiation.

    **Problem**: `forecast_global_radiation` is Global Horizontal Irradiance (GHI), but panels are tilted. The ratio $P_{pv} / (GHI \times \Delta t)$ has seasonal bias: inflated in winter (tilted panels capture more than horizontal), deflated in summer.

    **Three irradiance bases compared** (all using isotonic P90 for monotonic capacity):
    1. **Raw GHI** (baseline): $C = P_{pv} / (GHI / 1000 \times 0.25)$
    2. **Empirical monthly** $\eta$: two-pass — first estimate capacity, then compute monthly correction factors from reconstruction residuals, re-estimate
    3. **pvlib POA**: transpose GHI to Plane-of-Array irradiance using solar geometry (Sion 46.23N, panel tilt/azimuth configurable)

    **Additional method**:
    4. **Rolling yield ratio**: compute $Y = P_{solar} / GHI_{forecast}$ during daytime, take rolling median over a configurable window, then $\hat{P}(t) = \tilde{Y}_{window}(t) \times GHI(t)$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    panel_tilt = mo.ui.slider(
        start=0, stop=60, value=30, step=1,
        label="Panel tilt (degrees)",
    )
    panel_azimuth = mo.ui.slider(
        start=90, stop=270, value=180, step=5,
        label="Panel azimuth (degrees, 180=south)",
    )
    yield_window = mo.ui.slider(
        start=7, stop=90, value=30, step=1,
        label="Yield ratio window (days)",
    )
    mo.hstack([panel_tilt, panel_azimuth, yield_window])
    return panel_azimuth, panel_tilt, yield_window


@app.cell(hide_code=True)
def _(compute_poa_irradiance, df_with_cyclical, panel_azimuth, panel_tilt):
    df_with_poa = compute_poa_irradiance(
        df_with_cyclical,
        tilt=panel_tilt.value,
        azimuth=panel_azimuth.value,
    )
    return (df_with_poa,)


@app.cell(hide_code=True)
def _(
    IsotonicRegression,
    alt,
    df_with_poa,
    estimate_solar_capacity,
    mo,
    np,
    pl,
    yield_window,
):
    # --- 1. Module-based feature computation (Causal) -------------------------
    _df_features = estimate_solar_capacity(
        df_with_poa,
        threshold=200.0,
        window_days=30,
        min_periods=96,
    )

    # --- 2. Ad-hoc diagnostic computation (includes non-causal monthly eta) ---
    _df = df_with_poa.sort("utc_timestamp").with_columns(
        pl.sum_horizontal(
            [
                "solar_central_valais",
                "solar_sion",
                "solar_sierre",
                "solar_remote",
            ]
        ).alias("_solar_total")
    )

    _window = 2880  # 30 days at 15-min intervals
    _min_periods = 96
    _threshold = 200

    def _isotonic_p90(df, ratio_col, out_col):
        df = df.with_columns(
            pl.col(ratio_col)
            .rolling_quantile(
                quantile=0.9,
                window_size=_window,
                min_samples=_min_periods,
                center=False,
            )
            .forward_fill()
            .alias("_rp90_tmp")
        )
        _vals = df["_rp90_tmp"].to_numpy().astype(np.float64)
        _mask = ~np.isnan(_vals)
        _x = np.arange(len(_vals))
        _iso = IsotonicRegression(increasing=True)
        _result = np.full(len(_vals), np.nan)
        _result[_mask] = _iso.fit_transform(_x[_mask], _vals[_mask])
        df = df.with_columns(
            pl.Series(out_col, _result).forward_fill().backward_fill()
        ).drop("_rp90_tmp")
        return df

    # Raw GHI
    _df = _df.with_columns(
        pl.when(pl.col("sion_forecast_global_radiation") > _threshold)
        .then(
            pl.col("_solar_total")
            / (pl.col("sion_forecast_global_radiation") / 1000 * 0.25)
        )
        .otherwise(None)
        .alias("_ratio_ghi")
    )
    _df = _isotonic_p90(_df, "_ratio_ghi", "cap_ghi")

    # Monthly eta (diagnostic only)
    _df = _df.with_columns(
        pl.col("utc_timestamp").dt.month().alias("_month_num")
    )
    _eta_df = (
        _df.filter(
            pl.col("sion_forecast_global_radiation") > _threshold,
            pl.col("cap_ghi").is_not_null(),
            pl.col("_solar_total").is_not_null(),
        )
        .with_columns(
            (
                pl.col("_solar_total")
                / (
                    pl.col("cap_ghi")
                    * pl.col("sion_forecast_global_radiation")
                    / 1000
                    * 0.25
                )
            ).alias("_eta_raw")
        )
        .group_by("_month_num")
        .agg(pl.col("_eta_raw").median().alias("_eta_month"))
        .sort("_month_num")
    )
    _df = _df.join(_eta_df, on="_month_num", how="left")
    _df = _df.with_columns(
        pl.when(
            (pl.col("sion_forecast_global_radiation") > _threshold)
            & pl.col("_eta_month").is_not_null()
        )
        .then(
            pl.col("_solar_total")
            / (
                pl.col("_eta_month")
                * pl.col("sion_forecast_global_radiation")
                / 1000
                * 0.25
            )
        )
        .otherwise(None)
        .alias("_ratio_eta")
    )
    _df = _isotonic_p90(_df, "_ratio_eta", "cap_eta")

    # POA
    _df = _df.with_columns(
        pl.when(pl.col("poa_irradiance") > _threshold)
        .then(
            pl.col("_solar_total") / (pl.col("poa_irradiance") / 1000 * 0.25)
        )
        .otherwise(None)
        .alias("_ratio_poa")
    )
    _df = _isotonic_p90(_df, "_ratio_poa", "cap_poa")
    # Yield ratio (re-compute for evaluation consistency)
    _yield_rows = yield_window.value * 96
    _df = _df.with_columns(
        pl.when(pl.col("sion_forecast_global_radiation") > _threshold)
        .then(
            pl.col("_solar_total") / pl.col("sion_forecast_global_radiation")
        )
        .otherwise(None)
        .alias("_yield_raw")
    )
    _df = _df.with_columns(
        pl.col("_yield_raw")
        .rolling_median(window_size=_yield_rows, min_samples=_min_periods)
        .forward_fill()
        .alias("solar_yield_30d")
    )

    # Reconstruction
    _ghi_factor = pl.col("sion_forecast_global_radiation") / 1000 * 0.25
    _poa_factor = pl.col("poa_irradiance") / 1000 * 0.25
    _df = _df.with_columns(
        [
            (pl.col("cap_ghi") * _ghi_factor).alias("_pred_ghi"),
            (pl.col("cap_eta") * pl.col("_eta_month") * _ghi_factor).alias(
                "_pred_eta"
            ),
            (pl.col("cap_poa") * _poa_factor).alias("_pred_poa"),
            (
                pl.col("solar_yield_30d")
                * pl.col("sion_forecast_global_radiation")
            ).alias("_pred_yield"),
        ]
    )

    # --- 3. Evaluation and Charts ---------------------------------------------
    _pred_names = ["_pred_ghi", "_pred_eta", "_pred_poa", "_pred_yield"]
    solar_eval_df = (
        _df.select("utc_timestamp", "_solar_total", *_pred_names)
        .fill_nan(None)
        .drop_nulls()
    )
    solar_eval_df = solar_eval_df.with_columns(
        [
            (pl.col(p) - pl.col("_solar_total"))
            .abs()
            .rolling_mean(window_size=24 * 4)
            .alias(p.replace("_pred_", "mae_"))
            for p in _pred_names
        ]
    ).with_columns(
        [
            (pl.col(p) - pl.col("_solar_total")).alias(
                p.replace("_pred_", "err_")
            )
            for p in _pred_names
        ]
    )

    # === Capacity curves chart ===
    _cap_names = ["cap_ghi", "cap_eta", "cap_poa"]
    _cap_chart_data = (
        _df.select("utc_timestamp", *_cap_names)
        .drop_nulls()
        .gather_every(96)
        .unpivot(
            index="utc_timestamp",
            on=_cap_names,
            variable_name="method",
            value_name="capacity",
        )
    )
    _capacity_chart = (
        alt.Chart(_cap_chart_data)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("utc_timestamp:T", title="Date"),
            y=alt.Y("capacity:Q", title="Estimated capacity (kWp x eta)"),
            color=alt.Color(
                "method:N",
                title="Method",
                scale=alt.Scale(
                    domain=_cap_names,
                    range=["#4c78a8", "#54a24b", "#f58518"],
                ),
            ),
        )
        .properties(
            width="container",
            height=300,
            title="Estimated total solar park capacity over time",
        )
        .interactive()
    )

    # === Monthly eta bar chart ===
    _eta_chart = (
        alt.Chart(_eta_df)
        .mark_bar()
        .encode(
            x=alt.X("_month_num:O", title="Month"),
            y=alt.Y("_eta_month:Q", title="Monthly correction factor (eta)"),
        )
        .properties(
            width="container",
            height=200,
            title="Empirical monthly efficiency factors",
        )
    )
    # --- 4. Final Feature Set -------------------------------------------------
    # Merge the clean module-based features into the main pipeline
    df_features_complete = _df_features

    mo.accordion(
        {
            "Capacity curves (3 irradiance bases)": _capacity_chart,
            "Monthly eta factors": _eta_chart,
        }
    )
    return df_features_complete, solar_eval_df


@app.cell(hide_code=True)
def _(mo, solar_eval_df):
    _min_date = solar_eval_df["utc_timestamp"].min().date()
    _max_date = solar_eval_df["utc_timestamp"].max().date()
    _all_methods = {
        "Raw GHI": "ghi",
        "Monthly eta": "eta",
        "pvlib POA": "poa",
        "Yield ratio": "yield",
    }
    solar_error_date_start = mo.ui.date(value=_min_date, label="Start date")
    solar_error_date_end = mo.ui.date(value=_max_date, label="End date")
    solar_method_select = mo.ui.multiselect(
        options=list(_all_methods.keys()),
        value=list(_all_methods.keys()),
        label="Methods",
    )
    solar_show_actual = mo.ui.switch(value=True, label="Show actual production")
    return (
        solar_error_date_end,
        solar_error_date_start,
        solar_method_select,
        solar_show_actual,
    )


@app.cell(hide_code=True)
def _(
    alt,
    mo,
    pl,
    solar_error_date_end,
    solar_error_date_start,
    solar_eval_df,
    solar_method_select,
    solar_show_actual,
):
    _all_methods = {
        "Raw GHI": "ghi",
        "Monthly eta": "eta",
        "pvlib POA": "poa",
        "Yield ratio": "yield",
    }
    _all_colors = {
        "ghi": "#4c78a8",
        "eta": "#54a24b",
        "poa": "#f58518",
        "yield": "#72b7b2",
    }
    _selected = [_all_methods[m] for m in solar_method_select.value]

    # Filter by date range
    _filtered = solar_eval_df.filter(
        pl.col("utc_timestamp")
        .dt.date()
        .is_between(solar_error_date_start.value, solar_error_date_end.value)
    )

    if not _selected:
        _output = mo.md("> Select at least one method to display.")
    else:
        _colors = [_all_colors[s] for s in _selected]

        # --- Chart 1: Estimated (+ actual) production ---
        _prod_cols = [f"_pred_{s}" for s in _selected]
        _prod_filtered = _filtered.select(
            "utc_timestamp", "_solar_total", *_prod_cols
        ).drop_nulls()
        _prod_sampled = _prod_filtered.gather_every(
            max(1, _prod_filtered.height // 1000)
        ).head(1000)
        _layers = []
        # Estimated production lines
        _prod_data = _prod_sampled.select("utc_timestamp", *_prod_cols).unpivot(
            index="utc_timestamp",
            on=_prod_cols,
            variable_name="method",
            value_name="production",
        )
        _layers.append(
            alt.Chart(_prod_data)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("utc_timestamp:T", title="Date"),
                y=alt.Y("production:Q", title="Solar production (kWh)"),
                color=alt.Color(
                    "method:N",
                    title="Method",
                    scale=alt.Scale(
                        domain=_prod_cols,
                        range=_colors,
                    ),
                ),
            )
        )
        # Actual production (optional)
        if solar_show_actual.value:
            _actual_data = _prod_sampled.select(
                "utc_timestamp", "_solar_total"
            ).rename({"_solar_total": "production"})
            _layers.append(
                alt.Chart(_actual_data)
                .mark_line(strokeWidth=1.5, color="black", strokeDash=[4, 2])
                .encode(
                    x=alt.X("utc_timestamp:T"),
                    y=alt.Y("production:Q"),
                )
            )
        _prod_chart = (
            alt.layer(*_layers)
            .properties(
                width="container",
                height=250,
                title="Estimated vs actual solar production",
            )
            .interactive()
        )

        # --- Chart 2: Rolling MAE (absolute) ---
        _mae_names = [f"mae_{s}" for s in _selected]
        _mae_filtered = _filtered.select("utc_timestamp", *_mae_names).drop_nulls()
        _mae_sampled = _mae_filtered.gather_every(
            max(1, _mae_filtered.height // 1000)
        ).head(1000)
        _mae_data = _mae_sampled.unpivot(
            index="utc_timestamp",
            on=_mae_names,
            variable_name="method",
            value_name="MAE",
        )
        _mae_chart = (
            alt.Chart(_mae_data)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("utc_timestamp:T", title="Date"),
                y=alt.Y("MAE:Q", title="Rolling MAE (kWh, 24h window)"),
                color=alt.Color(
                    "method:N",
                    title="Method",
                    scale=alt.Scale(
                        domain=_mae_names,
                        range=_colors,
                    ),
                ),
            )
            .properties(
                width="container",
                height=250,
                title="Reconstruction error: rolling MAE",
            )
            .interactive()
        )

        # --- Chart 3: Raw error (non-absolute, quarter-hourly) ---
        _err_names = [f"err_{s}" for s in _selected]
        _err_filtered = _filtered.select("utc_timestamp", *_err_names).drop_nulls()
        _err_sampled = _err_filtered.gather_every(
            max(1, _err_filtered.height // 1000)
        ).head(1000)
        _err_data = _err_sampled.unpivot(
            index="utc_timestamp",
            on=_err_names,
            variable_name="method",
            value_name="error",
        )
        _err_chart = (
            alt.Chart(_err_data)
            .mark_line(strokeWidth=1, opacity=0.7)
            .encode(
                x=alt.X("utc_timestamp:T", title="Date"),
                y=alt.Y("error:Q", title="Error (kWh, predicted - actual)"),
                color=alt.Color(
                    "method:N",
                    title="Method",
                    scale=alt.Scale(
                        domain=_err_names,
                        range=_colors,
                    ),
                ),
            )
            .properties(
                width="container",
                height=250,
                title="Reconstruction error: quarter-hourly (predicted - actual)",
            )
            .interactive()
        )

        # --- MAE summary text ---
        _mae_parts = []
        for _s in _selected:
            _val = _filtered[f"mae_{_s}"].mean()
            _label = [k for k, v in _all_methods.items() if v == _s][0]
            _mae_parts.append(f"{_label}: `{_val:.1f}`")
        _mae_md = mo.md(
            "**Mean MAE (kWh, selected range)** — " + " · ".join(_mae_parts)
        )

        _output = mo.vstack([_prod_chart, _mae_chart, _err_chart, _mae_md])

    mo.accordion(
        {
            "Reconstruction error": mo.vstack(
                [
                    mo.hstack(
                        [
                            solar_method_select,
                            solar_show_actual,
                            solar_error_date_start,
                            solar_error_date_end,
                        ]
                    ),
                    _output,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(add_remote_yield_ratio, df_features_complete, mo, pl):
    df_with_remote_yield = add_remote_yield_ratio(
        df_features_complete,
        window_days=30,
    )

    mo.accordion(
        {
            "Solar remote yield features (from D-2 to D-32)": df_with_remote_yield.select(
                "utc_timestamp",
                "solar_remote",
                "sion_forecast_global_radiation",
                "solar_remote_yield_ratio",
            ).filter(pl.col("solar_remote_yield_ratio").is_not_null())
        }
    )
    return (df_with_remote_yield,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.7 Lagging features

    Historical patterns from recent days provide strong predictive signal for energy load forecasting. This section creates lagging features that capture:

    **Time periods** (using **fixed daily windows**):
    - **2 days ago**: Statistics from the full day (00:00-23:45) of D-2
    - **3 days ago**: Statistics from the full day (00:00-23:45) of D-3
    - **Week (2-9 days ago)**: Statistics from the 7 full days of D-9 to D-2

    **Statistics computed** for each period:
    - Basic: min, max, mean, std
    - **Volatility**: Coefficient of Variation (CV = std / |mean|)

    **Variables**:
    - Measured load (`load`)
    - Measured temperature from Sion (`sion_measured_temperature`)
    - Measured radiation from Sion (`sion_measured_global_radiation`)
    - Remote solar production (`solar_remote`)

    **Coefficient of Variation (CV)** measures relative volatility:
    - **Low CV**: Stable period, values cluster around the mean
    - **High CV**: Volatile period, high dispersion relative to average magnitude

    **Important**: Fixed daily windows mean all predictions on the same day use the **same** lag values (unlike rolling windows which slide every 15 minutes).

    These 60 new features provide the model with rich historical context.
    """)
    return


@app.cell(hide_code=True)
def _(add_lag_features, df_with_remote_yield):
    df_with_lags = add_lag_features(df_with_remote_yield)

    # Verify the new features were created
    print(f"Original columns: {df_with_remote_yield.width}")
    print(f"New columns: {df_with_lags.width}")
    print(f"Added lag features: {df_with_lags.width - df_with_remote_yield.width}")
    print("Expected: 60 lag features (min, max, mean, std, CV for 4 vars × 3 periods)")
    return (df_with_lags,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Preview of lag features

    This cell displays a sample of the newly created lag features for the `load` variable, focusing on rows with sufficient history (starting from day 10 to avoid initial null values).

    **Columns shown**:
    - Current load value
    - 2 days ago: mean, min, max, std, CV
    - 3 days ago: mean, CV
    - Week (2-9 days ago): mean, CV

    The CV (Coefficient of Variation) values indicate relative volatility:
    - **Low CV**: Stable period
    - **High CV**: Volatile period with high dispersion
    """)
    return


@app.cell(hide_code=True)
def _(df_with_lags, mo):
    # Show all lag features including the new yield ratio
    sample_start = 15 * 96  # Day 15, after sufficient history

    measured_vars = [
        "load",
        "sion_measured_temperature",
        "sion_measured_global_radiation",
        "solar_remote",
    ]

    # Get all lag columns including yield
    lag_cols = []
    for var in measured_vars:
        var_lags = [
            c
            for c in df_with_lags.columns
            if c.startswith(var)
            and any(suffix in c for suffix in ["_2d", "_3d", "_week"])
        ]
        lag_cols.extend(var_lags)


    # Sort columns: measured first, then lag features, yield at end
    def sort_key(col):
        if col in measured_vars:
            return (0, measured_vars.index(col), 0)
        elif col.startswith("load"):
            var_order = 0
        elif "sion_measured_temperature" in col:
            var_order = 1
        elif "sion_measured_global_radiation" in col:
            var_order = 2
        elif "solar_remote" in col:
            var_order = 3
        else:
            var_order = 999

        if "_2d" in col and "_week" not in col:
            period_order = 0
        elif "_3d" in col:
            period_order = 1
        elif "_week" in col:
            period_order = 2
        else:
            period_order = 999

        if "_min_" in col or (col.endswith("_min") and "_week" in col):
            stat_order = 0
        elif "_max_" in col or (col.endswith("_max") and "_week" in col):
            stat_order = 1
        elif "_mean" in col and "_std" not in col and "_cv" not in col:
            stat_order = 2
        elif "_std" in col:
            stat_order = 3
        elif "_cv" in col or col.endswith("_cv"):
            stat_order = 4
        else:
            stat_order = 999

        return (var_order + 1, stat_order, period_order)


    sorted_cols = ["utc_timestamp"] + sorted(measured_vars + lag_cols, key=sort_key
    )

    mo.accordion(
        {
            "Lag features": mo.ui.table(df_with_lags[sample_start :].select(sorted_cols), max_columns= 100),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Data analysis

    ### 5.1 Data quality overview

    Assessing missing values per column to identify data gaps that may affect feature selection and model training.
    """)
    return


@app.cell(hide_code=True)
def _(alt, df_clean, mo, pl):
    # Compute null counts and percentages
    null_counts = df_clean.null_count()
    total_rows = df_clean.height
    quality_df = pl.DataFrame(
        {
            "column": null_counts.columns,
            "null_count": [null_counts[col][0] for col in null_counts.columns],
        }
    ).with_columns(
        (pl.col("null_count") / total_rows * 100).round(1).alias("null_pct"),
        pl.when(pl.col("null_count") / total_rows * 100 > 10)
        .then(pl.lit("high (>10%)"))
        .when(pl.col("null_count") / total_rows * 100 > 1)
        .then(pl.lit("medium (1-10%)"))
        .otherwise(pl.lit("low (<1%)"))
        .alias("severity"),
    )

    # Altair bar chart of null percentages
    quality_chart = (
        alt.Chart(quality_df)
        .mark_bar()
        .encode(
            x=alt.X("column:N", sort="-y", title="Column"),
            y=alt.Y("null_pct:Q", title="Missing values (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(
                    domain=["low (<1%)", "medium (1-10%)", "high (>10%)"],
                    range=["#54a24b", "#f58518", "#e45756"],
                ),
            ),
        )
        .properties(width=600, height=300, title="Missing values per column")
    )

    mo.accordion(
        {
            "Data quality": mo.vstack(
                [
                    mo.md("""
    > **Note**: Data cleaning (forward-fill) was applied in section 2
                    """),
                    quality_chart,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.2 Load profile analysis

    Exploring daily, weekly, and seasonal load patterns to understand which temporal features drive consumption.
    """)
    return


@app.cell(hide_code=True)
def _(alt, df_with_lags, mo, pl):
    # Compute daily profiles using existing calendar features
    # Each day can belong to multiple categories (non-exclusive)
    # Categories: Holiday, Not working day, Weekday, Weekend, Working day
    _daily_profile = (
        df_with_lags
        # Create boolean flags for the 5 categories
        .with_columns(
            [
                pl.col("local_is_holiday").alias("Holiday"),
                (~pl.col("local_is_working_day")).alias("Not working day"),
                (~pl.col("local_is_weekend")).alias("Weekday"),
                pl.col("local_is_weekend").alias("Weekend"),
                pl.col("local_is_working_day").alias("Working day"),
            ]
        )
        # Select only the columns we need
        .select(
            [
                "local_hour",
                "load",
                "Holiday",
                "Not working day",
                "Weekday",
                "Weekend",
                "Working day",
            ]
        )
        # Unpivot only the 5 category columns
        .unpivot(
            index=["local_hour", "load"],
            on=["Holiday", "Not working day", "Weekday", "Weekend", "Working day"],
            variable_name="day_type",
            value_name="is_in_category",
        )
        # Filter to only include rows where the category applies
        .filter(pl.col("is_in_category") == True)
        .group_by("local_hour", "day_type")
        .agg(
            pl.col("load").mean().alias("mean_load"),
            pl.col("load").std().alias("std_load"),
        )
        .sort("local_hour")
    )
    _daily_chart = (
        alt.Chart(_daily_profile)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23])
            ),
            y=alt.Y("mean_load:Q", title="Mean load (standardised)"),
            color=alt.Color("day_type:N", title="Day type"),
            strokeDash=alt.StrokeDash("day_type:N"),
        )
        .properties(width=500, height=300, title="Average daily load profile")
    )
    # Weekly profile using existing day_of_week feature
    _weekly_profile = (
        df_with_lags.group_by("local_day_of_week")
        .agg(pl.col("load").mean().alias("mean_load"))
        .sort("local_day_of_week")
    )
    _weekly_chart = (
        alt.Chart(_weekly_profile)
        .mark_bar()
        .encode(
            x=alt.X("day_of_week:O", title="Day of week (1=Mon, 7=Sun)"),
            y=alt.Y("mean_load:Q", title="Mean load (standardised)"),
        )
        .properties(width=500, height=300, title="Average weekly load profile")
    )
    mo.accordion(
        {"Daily & weekly load profiles": mo.vstack([_daily_chart, _weekly_chart])}
    )
    return


@app.cell(hide_code=True)
def _(alt, df_with_lags, mo, pl):
    # Seasonal heatmap: mean load by month x hour (using existing calendar features)
    seasonal = df_with_lags.group_by("local_month", "local_hour").agg(
        pl.col("load").mean().alias("mean_load")
    )

    seasonal_chart = (
        alt.Chart(seasonal)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("month:O", title="Month"),
            color=alt.Color(
                "mean_load:Q",
                title="Mean load",
                scale=alt.Scale(scheme="blueorange"),
            ),
        )
        .properties(
            width=500, height=300, title="Seasonal load heatmap (month × hour)"
        )
    )
    mo.accordion({"Seasonal load heatmap": seasonal_chart})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.3 Feature–load linear correlation
    Identifying the correlation values of the load with each of the different features (weather forecasts and solar production). We see notably that
    1. The precipitations are not very correlated to the load
    2. The solar values are all correlated at a similar level with the load
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    weather_corr_station = mo.ui.dropdown(
        options=[
            "sion",
            "evionnaz",
            "evolene",
            "montana",
            "visp",
            "basel",
            "bern",
            "geneve",
            "pully",
            "zurich",
        ],
        value="sion",
        label="Station",
    )
    weather_radio = mo.ui.radio(
        [
            "forecast_temperature",
            "forecast_global_radiation",
            "forecast_precipitation",
            "forecast_humidity",
            "forecast_sunshine_duration",
        ],
        value="forecast_temperature",
        label="Weather feature",
    )
    solar_radio = mo.ui.radio(
        ["solar_central_valais", "solar_sion", "solar_sierre", "solar_remote"],
        value="solar_central_valais",
        label="Solar feature",
    )
    return solar_radio, weather_corr_station, weather_radio


@app.cell(hide_code=True)
def _(alt, df_clean, mo, pl, solar_radio, weather_corr_station, weather_radio):
    def _build_scatter(_df, _feature_col):
        """Build a scatter plot of min-max normalised feature vs load with trend line."""
        _data = _df.select(["load", _feature_col]).drop_nulls()
        _min = _data[_feature_col].min()
        _max = _data[_feature_col].max()
        _range = _max - _min
        if _range == 0:
            _data = _data.with_columns(pl.lit(0.0).alias("feature_norm"))
        else:
            _data = _data.with_columns(
                ((pl.col(_feature_col) - _min) / _range).alias("feature_norm")
            )

        # Pearson correlation
        _mean_f = _data["feature_norm"].mean()
        _mean_l = _data["load"].mean()
        _std_f = _data["feature_norm"].std()
        _std_l = _data["load"].std()
        if _std_f == 0 or _std_l == 0:
            _r = 0.0
        else:
            _cov = (
                (_data["feature_norm"] - _mean_f) * (_data["load"] - _mean_l)
            ).mean()
            _r = _cov / (_std_f * _std_l)

        _plot_data = _data.sample(n=min(5000, _data.height), seed=42)

        _points = (
            alt.Chart(_plot_data)
            .mark_circle(size=8, opacity=0.3)
            .encode(
                x=alt.X("feature_norm:Q", title=f"{_feature_col} (normalised)"),
                y=alt.Y("load:Q", title="Load (standardised)"),
            )
        )

        _trend = _points.transform_regression("feature_norm", "load").mark_line(
            color="red", strokeWidth=2
        )

        _chart = (_points + _trend).properties(
            width=350, height=300, title=_feature_col
        )
        return _chart, round(_r, 3)


    _weather_col = f"{weather_corr_station.value}_{weather_radio.value}"
    _weather_chart, _weather_r = _build_scatter(df_clean, _weather_col)
    _solar_chart, _solar_r = _build_scatter(df_clean, solar_radio.value)

    _left = mo.vstack(
        [
            mo.hstack([weather_corr_station, weather_radio]),
            mo.md(f"**Pearson r = {_weather_r}**"),
            _weather_chart,
        ]
    )
    _right = mo.vstack(
        [
            solar_radio,
            mo.md(f"**Pearson r = {_solar_r}**"),
            _solar_chart,
        ]
    )

    mo.accordion(
        {"Feature\u2013load correlation": mo.hstack([_left, _right], gap=2)}
    )
    return


@app.cell(hide_code=True)
def _(df_clean, mo, pl):
    _stations = [
        "sion",
        "evionnaz",
        "evolene",
        "montana",
        "visp",
        "basel",
        "bern",
        "geneve",
        "pully",
        "zurich",
    ]
    _weather_vars = [
        "temperature",
        "global_radiation",
        "precipitation",
        "humidity",
        "sunshine_duration",
    ]

    _all_features = [
        *[f"{s}_forecast_{v}" for s in _stations for v in _weather_vars],
        "solar_central_valais",
        "solar_sion",
        "solar_sierre",
        "solar_remote",
    ]

    _rows = []
    for _feat in _all_features:
        _data = df_clean.select(["load", _feat]).drop_nulls()
        _std_f = _data[_feat].std()
        _std_l = _data["load"].std()
        if _std_f == 0 or _std_l == 0:
            _r = 0.0
        else:
            _cov = (
                (_data[_feat] - _data[_feat].mean())
                * (_data["load"] - _data["load"].mean())
            ).mean()
            _r = _cov / (_std_f * _std_l)
        if "forecast_" in _feat:
            _category = "Weather forecast"
        else:
            _category = "Solar production"
        _rows.append(
            {"feature": _feat, "category": _category, "pearson_r": round(_r, 3)}
        )

    _corr_table = pl.DataFrame(_rows).sort("pearson_r", descending=True)

    mo.accordion({"Correlation summary (all features vs load)": _corr_table})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.4 Non-linear feature importance

    Pearson correlation (4.3) only captures **linear** relationships. Real-world
    energy data often exhibits **non-linear** patterns:
    - Temperature effects may plateau at extremes
    - Solar radiation has threshold effects
    - Calendar patterns interact in complex ways

    This section uses **Mutual Information** to capture non-linear dependencies.
    MI measures any statistical dependency (linear or non-linear) between each
    feature and the target, without assuming a particular functional form.
    """)
    return


@app.cell(hide_code=True)
def _(df_with_lags, mo):
    _stations = [
        "sion",
        "evionnaz",
        "evolene",
        "montana",
        "visp",
        "basel",
        "bern",
        "geneve",
        "pully",
        "zurich",
    ]
    _weather_vars = [
        "temperature",
        "global_radiation",
        "precipitation",
        "humidity",
        "sunshine_duration",
    ]

    # Define feature columns (forecasts only, no current weather)
    feature_cols = [
        # Calendar features (raw)
        "local_hour",
        "local_day_of_week",
        "local_month",
        "local_day_of_year",
        "local_week_of_year",
        "local_is_weekend",
        "local_is_holiday",
        "local_is_working_day",
        # Cyclical features
        "utc_sin_hour",
        "utc_cos_hour",
        "utc_sin_dow",
        "utc_cos_dow",
        "utc_sin_month",
        "utc_cos_month",
        "utc_sin_doy",
        "utc_cos_doy",
        # Weather forecasts (all stations, day-ahead model uses forecasts)
        *[f"{s}_forecast_{v}" for s in _stations for v in _weather_vars],
        # Solar production
        "solar_central_valais",
        "solar_sion",
        "solar_sierre",
        "solar_remote",
        # Derived features
        "poa_irradiance",
        "estimated_solar_capacity_ghi",
        "estimated_solar_capacity_poa",
        "solar_yield_30d",
        "solar_remote_yield_ratio",
    ]

    # Data was already cleaned in section 2, just select needed columns
    df_features = df_with_lags.select(["load", *feature_cols])

    # Convert any float NaN to polars null, then drop rows with nulls
    df_features = df_features.fill_nan(None).drop_nulls()

    # Separate features and target (convert to numpy for sklearn)
    X = df_features.select(feature_cols).to_numpy()
    y = df_features["load"].to_numpy()

    mo.accordion(
        {
            "Feature matrix ready": mo.md(f"""
        **Shape**: {X.shape} ({X.shape[0]:,} samples, {X.shape[1]} features)

        **Features** ({len(feature_cols)}):
        - Calendar (raw): hour, day_of_week, month, day_of_year, week_of_year, is_weekend, is_holiday, is_working_day
        - Cyclical: sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month, sin_doy, cos_doy
        - Weather forecasts ({len(_stations)} stations x {len(_weather_vars)} variables)
        - Solar: solar_central_valais, solar_sion, solar_sierre, solar_remote
        - Derived: poa, cap_ghi, cap_poa, solar_yield_30d, remote_yield_ratio

        **Target**: load (standardised, net of solar)
        """)
        }
    )
    return X, feature_cols, y


@app.cell(hide_code=True)
def _(X, feature_cols, mo, mutual_info_regression, pl, y):
    # Indices of binary features (for discrete_features parameter)
    binary_indices = [6, 7, 8]  # is_weekend, is_holiday, is_working_day

    # Calculate mutual information
    mi_scores = mutual_info_regression(
        X, y, random_state=42, discrete_features=binary_indices
    )

    # Create results dataframe with categories
    mi_df = (
        pl.DataFrame({"feature": feature_cols, "importance": mi_scores})
        .sort("importance", descending=True)
        .with_columns(
            pl.when(pl.col("feature").str.starts_with("is_"))
            .then(pl.lit("Calendar (binary)"))
            .when(
                pl.col("feature").str.starts_with("sin_")
                | pl.col("feature").str.starts_with("cos_")
            )
            .then(pl.lit("Cyclical"))
            .when(pl.col("feature") == "local_hour")
            .then(pl.lit("Calendar (numeric)"))
            .when(
                pl.col("feature").is_in(
                    [
                        "local_day_of_week",
                        "local_month",
                        "local_day_of_year",
                        "local_week_of_year",
                    ]
                )
            )
            .then(pl.lit("Calendar (numeric)"))
            .when(pl.col("feature").str.starts_with("forecast_"))
            .then(pl.lit("Weather forecast"))
            .when(pl.col("feature").str.starts_with("solar_"))
            .then(pl.lit("Solar production"))
            .otherwise(pl.lit("Other"))
            .alias("category")
        )
    )

    # Normalize MI scores to percentage of max
    mi_top = mi_df.head(15).with_columns(
        (pl.col("importance") / pl.col("importance").max() * 100).alias(
            "importance_norm"
        )
    )

    mo.accordion({"Mutual information (MI) scores": mi_df})
    return (mi_top,)


@app.cell(hide_code=True)
def _(alt, mi_top, mo):
    # Create horizontal bar chart
    mi_chart = (
        alt.Chart(mi_top)
        .mark_bar()
        .encode(
            x=alt.X("importance_norm:Q", title="Normalized MI score (%)"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color("category:N", title="Category"),
        )
        .properties(width=500, height=400, title="Mutual Information: feature importance")
    )

    mo.accordion({"Mutual information (MI) chart": mi_chart})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.5 Daily lag autocorrelation

    For day-ahead forecasting, load data up to D-1 is available at prediction
    time. We test Pearson correlation at daily lags t-2d through t-7d (i.e.,
    same quarter-hour, k days earlier) to determine which past daily load
    values carry predictive signal.

    - **t-2d to t-6d**: test recency — does recent load history help?
    - **t-7d**: same weekday last week — captures weekly periodicity
    """)
    return


@app.cell(hide_code=True)
def _(df_with_lags, mo, pl):
    _load_series = df_with_lags["load"]
    _n = len(_load_series)

    # Compute rows per day from the actual timestamp interval
    _interval_minutes = (
        df_with_lags["utc_timestamp"][1] - df_with_lags["utc_timestamp"][0]
    ).total_seconds() / 60
    _rows_per_day = int(24 * 60 / _interval_minutes)
    print(f"The rows per day are {_rows_per_day}")

    _is_weekend = df_with_lags["local_is_weekend"]

    _lag_results = []
    for _k in range(2, 8):
        _shift = _k * _rows_per_day
        _df_corr = pl.DataFrame(
            {
                "original": _load_series[_shift:],
                "lagged": _load_series[: _n - _shift],
                "local_is_weekend": _is_weekend[_shift:],
            }
        )
        # Overall correlation
        _r_all = _df_corr.select(pl.corr("original", "lagged")).item()
        # Weekday only
        _r_wd = (
            _df_corr.filter(pl.col("local_is_weekend") == 0)
            .select(pl.corr("original", "lagged"))
            .item()
        )
        # Weekend only
        _r_we = (
            _df_corr.filter(pl.col("local_is_weekend") == 1)
            .select(pl.corr("original", "lagged"))
            .item()
        )
        _lag_results.append(
            {
                "lag_days": _k,
                "all": round(_r_all, 4),
                "weekday": round(_r_wd, 4),
                "weekend": round(_r_we, 4),
            }
        )

    lag_corr_df = pl.DataFrame(_lag_results)

    mo.accordion(
        {"Daily lag correlations (overall / weekday / weekend)": lag_corr_df}
    )
    return (lag_corr_df,)


@app.cell(hide_code=True)
def _(alt, lag_corr_df, mo):
    # Melt to long format for grouped bar chart
    _lag_long = lag_corr_df.unpivot(
        index="lag_days",
        on=["all", "weekday", "weekend"],
        variable_name="day_type",
        value_name="pearson_r",
    )

    lag_chart = (
        alt.Chart(_lag_long)
        .mark_bar()
        .encode(
            x=alt.X("pearson_r:Q", title="Pearson correlation with load(t)"),
            y=alt.Y("lag_days:O", title="Lag (days)", sort="ascending"),
            color=alt.Color(
                "day_type:N",
                title="Day type",
                scale=alt.Scale(
                    domain=["all", "weekday", "weekend"],
                    range=["#4c78a8", "#54a24b", "#e45756"],
                ),
            ),
            yOffset="day_type:N",
            tooltip=["lag_days:O", "day_type:N", "pearson_r:Q"],
        )
        .properties(
            width=500,
            height=250,
            title="Load autocorrelation at daily lags (t-2d to t-7d)",
        )
    )

    mo.accordion({
        "Daily lag autocorrelation chart": mo.vstack([
            mo.md(
                """
    > Grouped bars show Pearson correlation between load(t) and load(t - k days),
    > split by day type. This reveals whether weekday and weekend load patterns
    > have different lag structures.
            """
            ),
            lag_chart,
        ])
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.6 Weather gradient analysis

    Exploring whether temperature and radiation differences between Sion (Valais) and major Swiss cities correlate with OIKEN load. The hypothesis is that favorable weather in Valais relative to cities could drive tourism-related load increases.

    **Delta features**: `sion_forecast_{variable} - {city}_forecast_{variable}`
    - Positive temperature delta: Valais is warmer than the city
    - Positive radiation delta: Valais has more sunshine than the city
    """)
    return


@app.cell(hide_code=True)
def _(alt, df_clean, mo, pl):
    _cities = ["basel", "bern", "geneve", "pully", "zurich"]
    _variables = ["temperature", "global_radiation"]

    # Compute delta features: sion - city
    _delta_exprs = []
    for _var in _variables:
        for _city in _cities:
            _delta_name = f"delta_{_var}_{_city}"
            _delta_exprs.append(
                (
                    pl.col(f"sion_forecast_{_var}")
                    - pl.col(f"{_city}_forecast_{_var}")
                ).alias(_delta_name)
            )

    df_with_deltas = df_clean.with_columns(_delta_exprs)

    # Overall Pearson correlations
    _rows = []
    for _var in _variables:
        for _city in _cities:
            _delta_name = f"delta_{_var}_{_city}"
            _data = df_with_deltas.select(["load", _delta_name]).drop_nulls()
            _std_d = _data[_delta_name].std()
            _std_l = _data["load"].std()
            if _std_d == 0 or _std_l == 0:
                _r = 0.0
            else:
                _cov = (
                    (_data[_delta_name] - _data[_delta_name].mean())
                    * (_data["load"] - _data["load"].mean())
                ).mean()
                _r = _cov / (_std_d * _std_l)
            _rows.append(
                {
                    "variable": _var,
                    "city": _city,
                    "delta_feature": _delta_name,
                    "pearson_r": round(_r, 3),
                }
            )

    gradient_corr_df = pl.DataFrame(_rows).sort("pearson_r", descending=True)

    # Seasonal correlations
    _seasonal_rows = []
    for _var in _variables:
        for _city in _cities:
            _delta_name = f"delta_{_var}_{_city}"
            for _season, _months in [
                ("Winter", [12, 1, 2]),
                ("Spring", [3, 4, 5]),
                ("Summer", [6, 7, 8]),
                ("Autumn", [9, 10, 11]),
            ]:
                _data = (
                    df_with_deltas.filter(
                        pl.col("utc_timestamp").dt.month().is_in(_months)
                    )
                    .select(["load", _delta_name])
                    .drop_nulls()
                )
                _std_d = _data[_delta_name].std()
                _std_l = _data["load"].std()
                if _std_d == 0 or _std_l == 0:
                    _r = 0.0
                else:
                    _cov = (
                        (_data[_delta_name] - _data[_delta_name].mean())
                        * (_data["load"] - _data["load"].mean())
                    ).mean()
                    _r = _cov / (_std_d * _std_l)
                _seasonal_rows.append(
                    {
                        "variable": _var,
                        "city": _city,
                        "season": _season,
                        "pearson_r": round(_r, 3),
                    }
                )

    _seasonal_df = pl.DataFrame(_seasonal_rows)

    _heatmap = (
        alt.Chart(_seasonal_df)
        .mark_rect()
        .encode(
            x=alt.X("city:N", title="City"),
            y=alt.Y(
                "season:N",
                title="Season",
                sort=["Winter", "Spring", "Summer", "Autumn"],
            ),
            color=alt.Color(
                "pearson_r:Q",
                scale=alt.Scale(scheme="redblue", domainMid=0),
                title="Pearson r",
            ),
            tooltip=["city:N", "season:N", "variable:N", "pearson_r:Q"],
        )
        .properties(width=250, height=200)
        .facet(column=alt.Column("variable:N", title="Variable"))
    )

    mo.accordion(
        {
            "Weather gradient correlations": mo.vstack(
                [
                    mo.md("**Overall correlations (delta vs load)**"),
                    gradient_corr_df,
                    mo.md("**Seasonal breakdown**"),
                    _heatmap,
                ]
            )
        }
    )
    return (df_with_deltas,)


@app.cell(hide_code=True)
def _(mo):
    gradient_city_select = mo.ui.dropdown(
        options=["basel", "bern", "geneve", "pully", "zurich"],
        value="zurich",
        label="City",
    )
    gradient_var_radio = mo.ui.radio(
        ["temperature", "global_radiation"],
        value="temperature",
        label="Variable",
    )
    return gradient_city_select, gradient_var_radio


@app.cell(hide_code=True)
def _(alt, df_with_deltas, gradient_city_select, gradient_var_radio, mo):
    _city = gradient_city_select.value
    _var = gradient_var_radio.value
    _delta_name = f"delta_{_var}_{_city}"

    _data = df_with_deltas.select(["load", _delta_name]).drop_nulls()
    _std_d = _data[_delta_name].std()
    _std_l = _data["load"].std()
    if _std_d == 0 or _std_l == 0:
        _r = 0.0
    else:
        _cov = (
            (_data[_delta_name] - _data[_delta_name].mean())
            * (_data["load"] - _data["load"].mean())
        ).mean()
        _r = _cov / (_std_d * _std_l)

    _plot_data = _data.sample(n=min(5000, _data.height), seed=42)

    _points = (
        alt.Chart(_plot_data)
        .mark_circle(size=8, opacity=0.3)
        .encode(
            x=alt.X(
                f"{_delta_name}:Q", title=f"\u0394 {_var} (Sion \u2212 {_city})"
            ),
            y=alt.Y("load:Q", title="Load (standardised)"),
        )
    )

    _trend = _points.transform_regression(_delta_name, "load").mark_line(
        color="red", strokeWidth=2
    )

    _chart = (
        (_points + _trend)
        .properties(
            width=500, height=350, title=f"\u0394 {_var} (Sion \u2212 {_city})"
        )
        .interactive()
    )

    mo.accordion(
        {
            "Weather gradient scatter": mo.vstack(
                [
                    mo.hstack([gradient_city_select, gradient_var_radio]),
                    mo.md(f"**Pearson r = {round(_r, 3)}**"),
                    _chart,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Model training

    This section trains and evaluates models to forecast the standardised net electricity load (consumption − production) on a day-ahead horizon.

    ### Evaluation metrics
    - **MAE** (Mean Absolute Error): primary metric — average absolute prediction error
    - **RMSE** (Root Mean Square Error): secondary diagnostic — penalises large errors, useful for detecting struggles at peak periods

    ### Train / test split
    A **temporal split** at 1 October 2024 divides the data into:
    - **Training set**: Oct 2022 – Sep 2024 (~2 years)
    - **Test set**: Oct 2024 – Sep 2025 (~1 year)

    This respects the time ordering of the data and avoids data leakage from future observations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.1 Train / test split

    Excluded from the feature set:
    - **Raw timestamps** (`utc_timestamp`, `local_timestamp`) — kept on the split dataframes for sorting and plotting, but not fed to the model. All temporal signal is already captured by derived features (hour, day-of-week, cyclical sin/cos, calendar flags). Including raw datetimes would also cause an extrapolation issue: test timestamps are all larger than any training value, so tree-based models can't split on them meaningfully.
    - **Target and OIKEN baseline** (`load`, `forecast_load`) — target must not leak, and the OIKEN forecast is reserved as a benchmark to compare against
    - **Raw measured weather** — not available at day-ahead prediction time
    - **Raw solar production** (`solar_central_valais`, `solar_sion`, `solar_sierre`, `solar_remote`) — these are actual production values, not forecasts

    Lag features derived from measured variables (load, solar_remote, Sion weather) are **kept** since they use only past observations that are available at prediction time.
    """)
    return


@app.cell(hide_code=True)
def _(SPLIT_DATE, df_with_lags, metrics, mo, model_preparation, pl):
    # Exclude: target, OIKEN forecast, raw measured weather, raw solar production
    # Keep: lag features derived from measured variables (available at prediction time)
    _stations = [
        "sion",
        "evionnaz",
        "evolene",
        "montana",
        "visp",
        "basel",
        "bern",
        "geneve",
        "pully",
        "zurich",
    ]
    _weather_vars = [
        "temperature",
        "global_radiation",
        "precipitation",
        "humidity",
        "sunshine_duration",
    ]
    _raw_measured = {f"{s}_measured_{v}" for s in _stations for v in _weather_vars}
    _raw_solar_prod = {
        "solar_central_valais",
        "solar_sion",
        "solar_sierre",
        "solar_remote",
    }
    _exclude = {
        "utc_timestamp",
        "local_timestamp",
        "load",
        "forecast_load",
        *_raw_measured,
        *_raw_solar_prod,
    }

    model_features = [c for c in df_with_lags.columns if c not in _exclude]

    # Raw temporal split (before warmup clipping)
    df_train_full, df_test_full = model_preparation.split_temporal(
        df_with_lags, SPLIT_DATE
    )

    # Metric helpers
    mae = metrics.mae
    rmse = metrics.rmse

    mo.vstack(
        [
            mo.md(f"""
    **Raw split (before warmup clipping)**

    | | Rows | Period |
    |---|---|---|
    | Train | {df_train_full.height:,} | {df_train_full["utc_timestamp"].min().date()} → {df_train_full["utc_timestamp"].max().date()} |
    | Test | {df_test_full.height:,} | {df_test_full["utc_timestamp"].min().date()} → {df_test_full["utc_timestamp"].max().date()} |

    **Features**: {len(model_features)} columns
    """),
            mo.accordion(
                {
                    "Feature list": pl.DataFrame(
                        {
                            "feature": model_features,
                            "dtype": [
                                str(df_with_lags[c].dtype) for c in model_features
                            ],
                        }
                    )
                }
            ),
        ]
    )
    return df_test_full, df_train_full, mae, model_features, rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.2 Warmup clipping

    Lag features from section 4.7 use fixed daily windows reaching up to **9 days back** (the weekly statistics use daily stats from D-9 to D-2). Two symmetric clips are applied:

    1. **Training set**: drop the first 9 days of data — lag features are undefined (null or partially warmed-up) at the very start of the series.
    2. **Test set**: drop the first 9 days of data — lag features on these rows would reference load/weather values from the training period, which represents an implicit form of data leakage between the two sets.

    Clipping symmetrically keeps train and test conceptually independent in their lag-window lookback.
    """)
    return


@app.cell(hide_code=True)
def _(
    SPLIT_DATE,
    datetime,
    df_test_full,
    df_train_full,
    mo,
    model_features,
    model_preparation,
):
    WARMUP_DAYS = 9

    df_train, df_test = model_preparation.apply_warmup_clipping(
        df_train_full, df_test_full, SPLIT_DATE, warmup_days=WARMUP_DAYS
    )

    # Save snapshots for standalone scripts
    model_preparation.save_prepared_data(
        df_train,
        df_test,
        model_features,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
    )

    # Backward-fill solar_remote_yield_ratio gaps (uses next valid yield ratio).
    # Forward-fill as fallback for any trailing gaps at the end of the series.
    X_train = df_train.select(model_features).to_pandas()
    X_train["solar_remote_yield_ratio"] = (
        X_train["solar_remote_yield_ratio"].bfill().ffill()
    )

    X_test = df_test.select(model_features).to_pandas()
    X_test["solar_remote_yield_ratio"] = (
        X_test["solar_remote_yield_ratio"].bfill().ffill()
    )
    y_test = df_test["load"].to_pandas()

    mo.md(f"""
    **After {WARMUP_DAYS}-day warmup clipping**

    | | Rows | Dropped | Period |
    |---|---|---|---|
    | Train | {df_train.height:,} | {df_train_full.height - df_train.height:,} | {df_train["utc_timestamp"].min()} → {df_train["utc_timestamp"].max()} |
    | Test | {df_test.height:,} | {df_test_full.height - df_test.height:,} | {df_test["utc_timestamp"].min()} → {df_test["utc_timestamp"].max()} |

    **NaNs in X_train**: {int(X_train.isna().sum().sum())} &nbsp;&nbsp; **NaNs in X_test**: {int(X_test.isna().sum().sum())}
    """)
    return X_test, df_test, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.3 Baselines

    Two simple baselines establish reference performance to judge whether more complex models add value:

    1. **Persistence (t − 7 days)**: predict load at time _t_ using the load value from the same quarter-hour, one week earlier. Exploits the weekly periodicity (r ≈ 0.90 from section 5.5).
    2. **OIKEN forecast**: the operator's own day-ahead forecast (`forecast_load` column). This is the production benchmark that any new model should aim to beat.

    Both are evaluated on the same test window (2024-10-10 → 2025-09-29).
    """)
    return


@app.cell(hide_code=True)
def _(df_test, df_with_lags, mae, mo, pl, rmse, y_test):
    # --- Persistence baseline: load at t - 7 days ---------------------------
    _rows_per_day = 96  # 15-min interval
    _lag_shift = 7 * _rows_per_day

    _df_persist = df_with_lags.with_columns(
        pl.col("load").shift(_lag_shift).alias("_load_lag_7d")
    ).filter(pl.col("utc_timestamp") >= df_test["utc_timestamp"].min())

    y_pred_persistence = _df_persist["_load_lag_7d"].to_numpy()

    # --- OIKEN baseline: forecast_load column -------------------------------
    y_pred_oiken = df_test["forecast_load"].to_numpy()
    y_test_np = y_test.to_numpy()

    # --- Evaluate -----------------------------------------------------------
    baseline_results = pl.DataFrame(
        {
            "model": ["Persistence (t-7d)", "OIKEN forecast"],
            "MAE": [
                mae(y_test_np, y_pred_persistence),
                mae(y_test_np, y_pred_oiken),
            ],
            "RMSE": [
                rmse(y_test_np, y_pred_persistence),
                rmse(y_test_np, y_pred_oiken),
            ],
        }
    ).with_columns(
        [
            pl.col("MAE").round(4),
            pl.col("RMSE").round(4),
        ]
    )

    # Store baseline predictions in a dict so they can be reused/compared later
    baseline_predictions = {
        "Persistence (t-7d)": y_pred_persistence,
        "OIKEN forecast": y_pred_oiken,
    }

    mo.vstack(
        [
            mo.md(
                "**Baseline performance on test set (load is standardised, mean 0 / std 1)**"
            ),
            mo.accordion({"Results table": baseline_results}),
        ]
    )
    return (baseline_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.4 Ridge regression

    A linear baseline with L2 regularisation. Ridge is a natural first-choice ML model:
    - **Fast**: closed-form solution, trains in seconds even on 69k × 131
    - **Interpretable**: each feature gets a signed weight
    - **Handles correlated features**: the L2 penalty shrinks collinear coefficients together (we have several correlated weather features across stations)

    **Preprocessing**: features are standardised (zero mean, unit variance) using statistics computed on the **training set only**, then applied to the test set. This prevents information from the test period from leaking into the preprocessing step.

    **Regularisation strength** (α): tuned via a small grid on a time-series cross-validation split inside the training data. For a first pass we use `RidgeCV` which automates this selection.
    """)
    return


@app.cell(hide_code=True)
def _(
    X_test,
    baseline_predictions,
    joblib,
    mae,
    mo,
    model_features,
    np,
    os,
    pl,
    rmse,
    y_test,
):
    _scaler_path = "models/scaler_latest.joblib"
    _ridge_path = "models/ridge_latest.joblib"

    if not os.path.exists(_scaler_path) or not os.path.exists(_ridge_path):
        mo.md(
            "⚠️ **Ridge model or scaler not found!** Run `python scripts/train_ridge.py` first."
        )
        ridge_model = None
        y_pred_ridge = np.zeros_like(y_test.to_numpy())
    else:
        # Load pre-trained scaler and model
        _scaler = joblib.load(_scaler_path)
        ridge_model = joblib.load(_ridge_path)

        # Standardise test features
        _X_test_scaled = _scaler.transform(X_test)
        y_pred_ridge = ridge_model.predict(_X_test_scaled)

    # Evaluate
    ridge_mae = mae(y_test.to_numpy(), y_pred_ridge)
    ridge_rmse = rmse(y_test.to_numpy(), y_pred_ridge)

    # Top features by absolute coefficient (if model exists)
    if ridge_model is not None:
        _coefs = (
            pl.DataFrame(
                {
                    "feature": model_features,
                    "coefficient": ridge_model.coef_,
                    "abs_coef": np.abs(ridge_model.coef_),
                }
            )
            .sort("abs_coef", descending=True)
            .drop("abs_coef")
        )
    else:
        _coefs = pl.DataFrame({"feature": [], "coefficient": []})

    # Combined results table
    all_results = pl.DataFrame(
        {
            "model": ["Persistence (t-7d)", "OIKEN forecast", "Ridge regression"],
            "MAE": [
                mae(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                mae(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                ridge_mae,
            ],
            "RMSE": [
                rmse(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                rmse(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                ridge_rmse,
            ],
        }
    ).with_columns(
        [
            pl.col("MAE").round(4),
            pl.col("RMSE").round(4),
        ]
    )

    baseline_predictions["Ridge regression"] = y_pred_ridge

    mo.vstack(
        [
            mo.md("**Model performance: Ridge regression (linear baseline)**"),
            mo.accordion({"Results table": all_results}),
            mo.accordion({"Ridge top 20 features": _coefs.head(20)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.5 LightGBM

    Gradient-boosted decision trees — the workhorse for tabular time-series forecasting. Unlike Ridge, LightGBM:
    - Captures **non-linear relationships** (e.g. temperature thresholds above which cooling kicks in)
    - Captures **feature interactions** automatically (e.g. `hour × is_working_day × temperature`)
    - **No scaling required** — trees are invariant to monotonic feature transformations
    - Handles **hundreds of features** efficiently via histogram-based splits

    **Validation strategy**: reserve the last 3 months of training data (Jul-Sep 2024) as a validation set. Use **early stopping** on this set to pick the optimal number of boosting rounds — this mimics a realistic deployment scenario where the latest data validates the model.

    **Hyperparameters** (first pass, reasonable defaults):
    - `n_estimators=2000` with early stopping patience 50
    - `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=20`
    - `reg_lambda=0.1` (mild L2 regularisation)
    - `objective="regression_l1"` (MAE objective, matches our primary metric)
    """)
    return


@app.cell(hide_code=True)
def _(X_test, baseline_predictions, lgb, mae, mo, np, os, pl, rmse, y_test):
    _lgb_path = "models/lgb_default_latest.txt"

    if not os.path.exists(_lgb_path):
        mo.md(
            "⚠️ **LightGBM baseline model not found!** Run `python scripts/train_lgbm_baseline.py` first."
        )
        lgb_model = None
        y_pred_lgb = np.zeros_like(y_test.to_numpy())
    else:
        # Load pre-trained booster
        lgb_model = lgb.Booster(model_file=_lgb_path)
        y_pred_lgb = lgb_model.predict(X_test)

    # --- Evaluate -------------------------------------------------------------
    lgb_mae = mae(y_test.to_numpy(), y_pred_lgb)
    lgb_rmse = rmse(y_test.to_numpy(), y_pred_lgb)

    baseline_predictions["LightGBM"] = y_pred_lgb

    # Combined results
    lgb_results = pl.DataFrame(
        {
            "model": [
                "Persistence (t-7d)",
                "OIKEN forecast",
                "Ridge regression",
                "LightGBM",
            ],
            "MAE": [
                mae(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                mae(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                mae(y_test.to_numpy(), baseline_predictions["Ridge regression"]),
                lgb_mae,
            ],
            "RMSE": [
                rmse(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                rmse(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                rmse(y_test.to_numpy(), baseline_predictions["Ridge regression"]),
                lgb_rmse,
            ],
        }
    ).with_columns(
        [
            pl.col("MAE").round(4),
            pl.col("RMSE").round(4),
        ]
    )

    if lgb_model is not None:
        lgb_importance = pl.DataFrame(
            {
                "feature": lgb_model.feature_name(),
                "gain": lgb_model.feature_importance(importance_type="gain"),
            }
        ).sort("gain", descending=True)
    else:
        lgb_importance = pl.DataFrame({"feature": [], "gain": []})

    mo.vstack(
        [
            mo.md("**Model performance: LightGBM (baseline)**"),
            mo.accordion({"Results table": lgb_results}),
            mo.accordion({"Top 20 features by gain": lgb_importance.head(20)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.6 LightGBM hyperparameter tuning

    To ensure robust performance across different seasonal patterns, hyperparameter tuning is performed outside this notebook using the `scripts/tune_lgbm.py` script.

    **Tuning Strategy:**
    - **Objective**: Minimize the Mean Absolute Error (MAE).
    - **Cross-Validation**: 3-fold expanding-window CV covering the first three quarters of 2024.
    - **Search Space**: Tunes learning rate, tree complexity, regularization, and sampling ratios using Optuna's TPE sampler.
    - **Refit**: After tuning, the model is refit on the full training set using the best parameters via `scripts/train_lgbm_tuned.py`.

    This section loads the resulting hyperparameters and the final tuned model for evaluation on the untouched Q4 2024 test set.
    """)
    return


@app.cell(hide_code=True)
def _(X_test, baseline_predictions, lgb, mae, mo, np, os, pl, rmse, y_test):
    import json

    _params_path = "tuning_results/best_params.json"
    _lgb_tuned_path = "models/lgb_tuned_latest.txt"

    if not os.path.exists(_params_path) or not os.path.exists(_lgb_tuned_path):
        mo.md(
            """
            ⚠️ **Tuned model or parameters not found!** 
            Run the following in your terminal:
            1. `uv run python scripts/tune_lgbm.py --trials 100`
            2. `uv run python scripts/train_lgbm_tuned.py`
            """
        )
        lgb_tuned_model = None
        y_pred_lgb_tuned = np.zeros_like(y_test.to_numpy())
        best_params = {}
    else:
        # Load best parameters for display
        with open(_params_path) as f:
            best_params = json.load(f)

        # Load pre-trained tuned booster
        lgb_tuned_model = lgb.Booster(model_file=_lgb_tuned_path)
        y_pred_lgb_tuned = lgb_tuned_model.predict(X_test)

    # --- Evaluate -------------------------------------------------------------
    lgb_tuned_mae = mae(y_test.to_numpy(), y_pred_lgb_tuned)
    lgb_tuned_rmse = rmse(y_test.to_numpy(), y_pred_lgb_tuned)

    baseline_predictions["LightGBM (tuned)"] = y_pred_lgb_tuned

    # --- Results summary ------------------------------------------------------
    tuned_results = pl.DataFrame(
        {
            "model": [
                "Persistence (t-7d)",
                "OIKEN forecast",
                "Ridge regression",
                "LightGBM (default)",
                "LightGBM (tuned)",
            ],
            "MAE": [
                mae(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                mae(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                mae(y_test.to_numpy(), baseline_predictions["Ridge regression"]),
                mae(y_test.to_numpy(), baseline_predictions["LightGBM"]),
                lgb_tuned_mae,
            ],
            "RMSE": [
                rmse(y_test.to_numpy(), baseline_predictions["Persistence (t-7d)"]),
                rmse(y_test.to_numpy(), baseline_predictions["OIKEN forecast"]),
                rmse(y_test.to_numpy(), baseline_predictions["Ridge regression"]),
                rmse(y_test.to_numpy(), baseline_predictions["LightGBM"]),
                lgb_tuned_rmse,
            ],
        }
    ).with_columns(
        [
            pl.col("MAE").round(4),
            pl.col("RMSE").round(4),
        ]
    )

    _best_params_df = pl.DataFrame(
        {
            "parameter": list(best_params.keys()),
            "value": [
                f"{v:.4g}" if isinstance(v, float) else str(v)
                for v in best_params.values()
            ],
        }
    )

    if lgb_tuned_model is not None:
        lgb_tuned_importance = pl.DataFrame(
            {
                "feature": lgb_tuned_model.feature_name(),
                "gain": lgb_tuned_model.feature_importance(importance_type="gain"),
            }
        ).sort("gain", descending=True)
    else:
        lgb_tuned_importance = pl.DataFrame({"feature": [], "gain": []})

    mo.vstack(
        [
            mo.md("**Model performance: LightGBM (tuned)**"),
            mo.accordion({"Results table": tuned_results}),
            mo.accordion({"Best hyperparameters (from local tuning)": _best_params_df}),
            mo.accordion({"Top 20 features by gain": lgb_tuned_importance.head(20)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Model evaluation

    In this section, we explore the results of our forecasting models in more detail, analyzing error distributions, seasonal performance, and specific failure modes.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
