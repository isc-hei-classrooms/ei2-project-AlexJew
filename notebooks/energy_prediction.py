import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl
    from sklearn.feature_selection import mutual_info_regression

    return alt, mo, mutual_info_regression, pl


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

    There are two data sources available

    | File | Description | Resolution |
    |------|-------------|------------|
    | `oiken_data.csv` | Electricity load (standardised) and solar production by area | 15-min |
    | `sion_forecast.csv` | Weather forecasts from MeteoSwiss (Sion) | 1-h |

    OIKEN data variables
    - **standardised load [-]**: Net electricity consumption (standardised)
    - **standardised forecast load [-]**: Forecasted load
    - **Solar production [kWh]**: Central Valais, Sion, Sierre, Remote areas

    Weather variables
    - **Forecasts (PRED_*)**: 24-hour predictions for multiple weather variables taken from the prediction made at 9AM the day before
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
    weather_df = pl.read_csv("data/sion_forecast_2026-03-24_18-21.csv", try_parse_dates=True)
    mo.accordion({"Weather raw data": weather_df})
    return (weather_df,)


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
    weather_renamed = weather_df.rename(
        {
            "PRED_T_2M_ctrl": "forecast_temperature",
            "PRED_GLOB_ctrl": "forecast_global_radiation",
            "PRED_TOT_PREC_ctrl": "forecast_precipitation",
            "PRED_RELHUM_2M_ctrl": "forecast_humidity",
            "PRED_DURSUN_ctrl": "forecast_sunshine_duration",
        }
    )
    # Drop columns with _0 suffix if present (polars pivot artefacts)
    weather_renamed = weather_renamed.select(
        [c for c in weather_renamed.columns if not c.endswith("_0")]
    )
    # Reorder: timestamp, historical values, then forecast values
    weather_renamed = weather_renamed.select(
        "timestamp",
        "forecast_temperature",
        "forecast_global_radiation",
        "forecast_precipitation",
        "forecast_humidity",
        "forecast_sunshine_duration",
    )
    mo.accordion({"Weather renamed": weather_renamed})
    return (weather_renamed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Merging of the data sets
    The weather forecast timestamps are converted from UTC to Swiss local time (Europe/Zurich) to match the OIKEN data, then the two datasets are merged on timestamp using a full outer join.
    """)
    return


@app.cell(hide_code=True)
def _(mo, oiken_renamed, pl, weather_renamed):
    # Convert weather UTC timestamps to Swiss local time (naive)
    weather_local = weather_renamed.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Europe/Zurich").dt.replace_time_zone(None)
    )

    # Merge OIKEN and weather datasets on timestamp (outer join)
    merged_df = oiken_renamed.join(
        weather_local,
        on="timestamp",
        how="full",
        coalesce=True,
    ).sort("timestamp")
    mo.accordion(
        {"Merged dataset": merged_df}
    )
    return (merged_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Cleaning of the data set
    The null values are cleaned and filled with through a forward fill. We look for null values, empty values and negative counts (where there shouldn't be any).

    Important: The forward fill drops all the rows for which contain a null cell for which there is no earlier data available (the first 8 rows of load up to the )
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
    # Replace negative forecasted values by zero
    df_clean = merged_df.with_columns([
        pl.col("forecast_global_radiation").clip(lower_bound=0),
        pl.col("forecast_precipitation").clip(lower_bound=0),
        pl.col("forecast_humidity").clip(lower_bound=0),
        pl.col("forecast_sunshine_duration").clip(lower_bound=0)
    ])

    # Handle missing values with time-series aware forward-fill
    df_clean = merged_df.fill_null(strategy="forward")

    # Drop any remaining nulls at the beginning of the dataset
    df_clean = df_clean.drop_nulls()

    # Show cleaning summary
    original_nulls = merged_df.null_count().row(0)
    remaining_nulls = df_clean.null_count().row(0)
    total_cleaned = sum(original_nulls) - sum(remaining_nulls)

    mo.accordion({
        "Data cleaning applied": mo.vstack(
            [
                mo.md(f"""
                **Cleaned**: {total_cleaned:,} values filled
                """), 
                df_clean
            ]
        )
    })
    return (df_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Calendar features

    Calendar features capture temporal patterns in electricity consumption:
    - **Daily patterns**: hour of day (morning peak, evening dip)
    - **Weekly patterns**: weekday vs weekend
    - **Seasonal patterns**: month, day of year
    - **Special days**: holidays, working days

    ### Feature extraction strategy
    1. Basic temporal features (hour, day of week, month, etc.)
    2. Swiss holiday calendar (Valais region)
    3. Working day classification
    4. Cyclical encoding (sin/cos) for periodic features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.1 Basic temporal features

    Based on the timestamps, it is possible to extract basic temporal features (hour, day of week, month). This results in ordinal features _hour_, _day_of_week_, _month_, and a binary feature _is_weekend_
    """)
    return


@app.cell(hide_code=True)
def _(df_clean, mo, pl):
    def add_temporal_features(df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
        """Extract basic temporal features from timestamp column."""
        return df.with_columns(
            [
                # Hour of day (0-23)
                pl.col(timestamp_col).dt.hour().alias("hour"),
                # Day of week (1=Monday, 7=Sunday)
                pl.col(timestamp_col).dt.weekday().alias("day_of_week"),
                # Month (1-12)
                pl.col(timestamp_col).dt.month().alias("month"),
                # Day of year (1-366)
                pl.col(timestamp_col).dt.ordinal_day().alias("day_of_year"),
                # Week of year (1-53)
                pl.col(timestamp_col).dt.week().alias("week_of_year"),
                # Binary flags
                (pl.col(timestamp_col).dt.weekday() > 5).alias("is_weekend"),  # Sat=6, Sun=7
            ]
        )

    # Test on cleaned data
    df_with_temporal = add_temporal_features(df_clean)
    mo.accordion({
        "Temporal features preview": df_with_temporal.select(
            "timestamp", "hour", "day_of_week", "month", "is_weekend"
        )
    })
    return (df_with_temporal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.2 Swiss holiday calendar

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
def _(df_with_temporal, mo, pl):
    def get_swiss_holidays(year: int) -> set:
        """Get set of Swiss holidays (national + Valais-specific) for a given year."""
        import holidays

        # Swiss holidays (national)
        ch_holidays = holidays.CH(years=year)

        # Add Valais-specific cantonal holidays
        # These are already included in holidays.CH for VS (Valais)
        ch_holidays.update(holidays.CH(years=year, prov="VS"))

        # Return set of dates for fast lookup
        return set(ch_holidays.keys())

    def add_holiday_features(df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
        """Add holiday flags to dataframe."""
        # Get all years in the dataset
        years = df.select(pl.col(timestamp_col).dt.year().unique()).to_series().to_list()

        # Build holiday set for all years
        holiday_dates = set()
        for year in years:
            holiday_dates.update(get_swiss_holidays(year))

        # Add is_holiday flag (check if date is in holiday set)
        return df.with_columns(
            pl.col(timestamp_col).dt.date().is_in(holiday_dates).alias("is_holiday")
        )

    # Test holiday feature
    df_with_holidays = add_holiday_features(df_with_temporal)

    # Show some examples
    holiday_examples = (
        df_with_holidays.filter(pl.col("is_holiday") == True)
        .select("timestamp", "is_holiday", "is_weekend")
    )
    mo.accordion({"Holiday examples": holiday_examples})
    return (df_with_holidays,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.3 Working day classification

    A "working day" is typically defined as:
    - NOT a weekend (Saturday/Sunday)
    - NOT a public holiday

    Working days have different consumption patterns than non-working days
    (industrial/commercial activity is higher).
    """)
    return


@app.cell(hide_code=True)
def _(df_with_holidays, mo, pl):
    def add_working_day_flag(df: pl.DataFrame) -> pl.DataFrame:
        """Add is_working_day flag (not weekend AND not holiday)."""
        return df.with_columns(
            (~pl.col("is_weekend") & ~pl.col("is_holiday")).alias("is_working_day")
        )

    # Apply working day flag
    df_working_days = add_working_day_flag(df_with_holidays)

    # Summary statistics - count unique days, not timestamps
    calendar_summary = (
        df_working_days.group_by(pl.col("timestamp").dt.date().alias("date"))
        .agg(
            pl.col("is_weekend").max().alias("is_weekend_day"),
            pl.col("is_holiday").max().alias("is_holiday_day"),
            pl.col("is_working_day").max().alias("is_working_day"),
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
    ### 3.4 Cyclical encoding

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
def _(df_working_days, mo, pl):
    def add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add sin/cos encoding for periodic temporal features."""
        return df.with_columns(
            [
                # Hour (0-23) -> 2π/24
                (pl.col("hour") * 2 * 3.14159 / 24).sin().alias("sin_hour"),
                (pl.col("hour") * 2 * 3.14159 / 24).cos().alias("cos_hour"),
                # Day of week (1-7) -> 2π/7 (adjusted for 1-based indexing)
                ((pl.col("day_of_week") - 1) * 2 * 3.14159 / 7).sin().alias("sin_dow"),
                ((pl.col("day_of_week") - 1) * 2 * 3.14159 / 7).cos().alias("cos_dow"),
                # Month (1-12) -> 2π/12 (shifted by 1 to start at 0)
                ((pl.col("month") - 1) * 2 * 3.14159 / 12).sin().alias("sin_month"),
                ((pl.col("month") - 1) * 2 * 3.14159 / 12).cos().alias("cos_month"),
                # Day of year (1-366) -> 2π/366 (shifted by 1)
                ((pl.col("day_of_year") - 1) * 2 * 3.14159 / 366).sin().alias("sin_doy"),
                ((pl.col("day_of_year") - 1) * 2 * 3.14159 / 366).cos().alias("cos_doy"),
            ]
        )

    # Apply cyclical encoding to create the final calendar features dataframe
    df_calendar_complete = add_cyclical_features(df_working_days)
    mo.accordion({"Cyclical encoded features": df_calendar_complete.select("timestamp", "load", "hour", "day_of_week", "month", "day_of_year", "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month", "sin_doy", "cos_doy")})
    return (df_calendar_complete,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Data analysis

    ### 4.1 Data quality overview

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
    ### 4.2 Load profile analysis

    Exploring daily, weekly, and seasonal load patterns to understand which temporal features drive consumption.
    """)
    return


@app.cell(hide_code=True)
def _(alt, df_calendar_complete, mo, pl):
    # Compute daily profiles using existing calendar features
    # Each day can belong to multiple categories (non-exclusive)
    # Categories: Holiday, Not working day, Weekday, Weekend, Working day
    _daily_profile = (
        df_calendar_complete
        # Create boolean flags for the 5 categories
        .with_columns(
            [
                pl.col("is_holiday").alias("Holiday"),
                (~pl.col("is_working_day")).alias("Not working day"),
                (~pl.col("is_weekend")).alias("Weekday"),
                pl.col("is_weekend").alias("Weekend"),
                pl.col("is_working_day").alias("Working day"),
            ]
        )
        # Select only the columns we need
        .select(["hour", "load", "Holiday", "Not working day", "Weekday", "Weekend", "Working day"])
        # Unpivot only the 5 category columns
        .unpivot(
            index=["hour", "load"],
            on=["Holiday", "Not working day", "Weekday", "Weekend", "Working day"],
            variable_name="day_type",
            value_name="is_in_category",
        )
        # Filter to only include rows where the category applies
        .filter(pl.col("is_in_category") == True)
        .group_by("hour", "day_type")
        .agg(
            pl.col("load").mean().alias("mean_load"),
            pl.col("load").std().alias("std_load"),
        )
        .sort("hour")
    )
    _daily_chart = (
        alt.Chart(_daily_profile)
        .mark_line(point=True)
        .encode(
            x=alt.X("hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23])),
            y=alt.Y("mean_load:Q", title="Mean load (standardised)"),
            color=alt.Color("day_type:N", title="Day type"),
            strokeDash=alt.StrokeDash("day_type:N"),
        )
        .properties(width=500, height=300, title="Average daily load profile")
    )
    # Weekly profile using existing day_of_week feature
    _weekly_profile = (
        df_calendar_complete.group_by("day_of_week")
        .agg(pl.col("load").mean().alias("mean_load"))
        .sort("day_of_week")
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
    mo.accordion({"Daily & weekly load profiles": mo.vstack([_daily_chart, _weekly_chart])})
    return


@app.cell(hide_code=True)
def _(alt, df_calendar_complete, mo, pl):
    # Seasonal heatmap: mean load by month x hour (using existing calendar features)
    seasonal = (
        df_calendar_complete.group_by("month", "hour")
        .agg(pl.col("load").mean().alias("mean_load"))
    )

    seasonal_chart = (
        alt.Chart(seasonal)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("month:O", title="Month"),
            color=alt.Color(
                "mean_load:Q", title="Mean load", scale=alt.Scale(scheme="blueorange")
            ),
        )
        .properties(width=500, height=300, title="Seasonal load heatmap (month × hour)")
    )
    mo.accordion({"Seasonal load heatmap": seasonal_chart})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.3 Feature–load linear correlation
    Identifying the correlation values of the load with each of the different features (weather forecasts and solar production). We see notably that
    1. The precipitations are not very correlated to the load
    2. The solar values are all correlated at a similar level with the load
    """)
    return


@app.cell(hide_code=True)
def _(mo):
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
        [
            "solar_central_valais",
            "solar_sion",
            "solar_sierre",
            "solar_remote",
        ],
        value="solar_central_valais",
        label="Solar feature",
    )
    return solar_radio, weather_radio


@app.cell(hide_code=True)
def _(alt, df_clean, mo, pl, solar_radio, weather_radio):
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
            _cov = ((_data["feature_norm"] - _mean_f) * (_data["load"] - _mean_l)).mean()
            _r = _cov / (_std_f * _std_l)

        # Sample for performance (scatter with 100k+ points is slow)
        _plot_data = _data.sample(n=min(5000, _data.height), seed=42)

        _points = (
            alt.Chart(_plot_data)
            .mark_circle(size=8, opacity=0.3)
            .encode(
                x=alt.X("feature_norm:Q", title=f"{_feature_col} (normalised)"),
                y=alt.Y("load:Q", title="Load (standardised)"),
            )
        )

        _trend = _points.transform_regression(
            "feature_norm", "load"
        ).mark_line(color="red", strokeWidth=2)

        _chart = (_points + _trend).properties(width=350, height=300, title=_feature_col)
        return _chart, round(_r, 3)

    _weather_chart, _weather_r = _build_scatter(df_clean, weather_radio.value)
    _solar_chart, _solar_r = _build_scatter(df_clean, solar_radio.value)

    _left = mo.vstack([
        weather_radio,
        mo.md(f"**Pearson r = {_weather_r}**"),
        _weather_chart,
    ])
    _right = mo.vstack([
        solar_radio,
        mo.md(f"**Pearson r = {_solar_r}**"),
        _solar_chart,
    ])

    mo.accordion({"Feature–load correlation": mo.hstack([_left, _right], gap=2)})
    return


@app.cell(hide_code=True)
def _(df_clean, mo, pl):
    _all_features = [
        "forecast_temperature",
        "forecast_global_radiation",
        "forecast_precipitation",
        "forecast_humidity",
        "forecast_sunshine_duration",
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
        _category = "Weather forecast" if _feat.startswith("forecast_") else "Solar production"
        _rows.append(
            {"feature": _feat, "category": _category, "pearson_r": round(_r, 3)}
        )

    _corr_table = (
        pl.DataFrame(_rows)
        .sort("pearson_r", descending=True)
    )

    mo.accordion({"Correlation summary (all features vs load)": _corr_table})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.4 Non-linear feature importance

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
def _(df_calendar_complete, mo):
    # Define feature columns (forecasts only, no current weather)
    feature_cols = [
        # Calendar features (raw)
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "is_holiday",
        "is_working_day",
        # Cyclical features
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "sin_month",
        "cos_month",
        "sin_doy",
        "cos_doy",
        # Weather forecasts only (day-ahead model uses forecasts)
        "forecast_temperature",
        "forecast_global_radiation",
        "forecast_precipitation",
        "forecast_humidity",
        "forecast_sunshine_duration",
        # Solar production
        "solar_central_valais",
        "solar_sion",
        "solar_sierre",
        "solar_remote",
    ]

    # Data was already cleaned in section 2, just select needed columns
    df_features = df_calendar_complete.select(["load", *feature_cols])

    # Final check for any unexpected nulls (should be none after section 2 cleaning)
    df_features = df_features.drop_nulls()

    # Separate features and target (convert to numpy for sklearn)
    X = df_features.select(feature_cols).to_numpy()
    y = df_features["load"].to_numpy()

    mo.accordion({
        f"Feature matrix ready": mo.md(f"""
        **Shape**: {X.shape} ({X.shape[0]:,} samples, {X.shape[1]} features)

        **Features** ({len(feature_cols)}):
        - Calendar (raw): hour, day_of_week, month, day_of_year, week_of_year, is_weekend, is_holiday, is_working_day
        - Cyclical: sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month, sin_doy, cos_doy
        - Weather forecasts: forecast_temperature, forecast_global_radiation, forecast_precipitation, forecast_humidity, forecast_sunshine_duration
        - Solar: solar_central_valais, solar_sion, solar_sierre, solar_remote

        **Target**: load (standardised, net of solar)
        """)
    })
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
    mi_df = pl.DataFrame({"feature": feature_cols, "importance": mi_scores}).sort(
        "importance", descending=True
    ).with_columns(
        pl.when(pl.col("feature").str.starts_with("is_"))
        .then(pl.lit("Calendar (binary)"))
        .when(
            pl.col("feature").str.starts_with("sin_")
            | pl.col("feature").str.starts_with("cos_")
        )
        .then(pl.lit("Cyclical"))
        .when(pl.col("feature") == "hour")
        .then(pl.lit("Calendar (numeric)"))
        .when(
            pl.col("feature").is_in(["day_of_week", "month", "day_of_year", "week_of_year"])
        )
        .then(pl.lit("Calendar (numeric)"))
        .when(pl.col("feature").str.starts_with("forecast_"))
        .then(pl.lit("Weather forecast"))
        .when(pl.col("feature").str.starts_with("solar_"))
        .then(pl.lit("Solar production"))
        .otherwise(pl.lit("Other"))
        .alias("category")
    )

    # Normalize MI scores to percentage of max
    mi_top = mi_df.head(15).with_columns(
        (pl.col("importance") / pl.col("importance").max() * 100).alias("importance_norm")
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
    ### 4.5 Daily lag autocorrelation

    For day-ahead forecasting, load data up to D-1 is available at prediction
    time. We test Pearson correlation at daily lags t-2d through t-7d (i.e.,
    same quarter-hour, k days earlier) to determine which past daily load
    values carry predictive signal.

    - **t-2d to t-6d**: test recency — does recent load history help?
    - **t-7d**: same weekday last week — captures weekly periodicity
    """)
    return


@app.cell(hide_code=True)
def _(df_calendar_complete, mo, pl):
    _load_series = df_calendar_complete["load"]
    _n = len(_load_series)

    # Compute rows per day from the actual timestamp interval
    _interval_minutes = (
        df_calendar_complete["timestamp"][1] - df_calendar_complete["timestamp"][0]
    ).total_seconds() / 60
    _rows_per_day = int(24 * 60 / _interval_minutes)
    print(f"The rows per day are {_rows_per_day}")

    _is_weekend = df_calendar_complete["is_weekend"]

    _lag_results = []
    for _k in range(2, 8):
        _shift = _k * _rows_per_day
        _df_corr = pl.DataFrame({
            "original": _load_series[_shift:],
            "lagged": _load_series[: _n - _shift],
            "is_weekend": _is_weekend[_shift:],
        })
        # Overall correlation
        _r_all = _df_corr.select(pl.corr("original", "lagged")).item()
        # Weekday only
        _r_wd = _df_corr.filter(pl.col("is_weekend") == 0).select(
            pl.corr("original", "lagged")
        ).item()
        # Weekend only
        _r_we = _df_corr.filter(pl.col("is_weekend") == 1).select(
            pl.corr("original", "lagged")
        ).item()
        _lag_results.append({
            "lag_days": _k,
            "all": round(_r_all, 4),
            "weekday": round(_r_wd, 4),
            "weekend": round(_r_we, 4),
        })

    lag_corr_df = pl.DataFrame(_lag_results)

    mo.accordion({"Daily lag correlations (overall / weekday / weekend)": lag_corr_df})
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


if __name__ == "__main__":
    app.run()
