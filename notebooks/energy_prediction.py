import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Energy net load forecasting

    This notebook analyzes data for net energy load forecasting.

    ## Available data

    | File | Description | Resolution |
    |------|-------------|------------|
    | `oiken_data.csv` | Electricity load (standardised) and solar production by area | 15-min |
    | `sion_weather.csv` | Weather measurements and forecasts from MeteoSwiss (Sion) | 10-min |

    ### OIKEN data variables
    - **standardised load [-]**: Net electricity consumption (standardised)
    - **standardised forecast load [-]**: Forecasted load
    - **Solar production [kWh]**: Central Valais, Sion, Sierre, Remote areas

    ### Weather variables
    - **Current**: Temperature, pressure, global radiation, wind, precipitation, humidity
    - **Forecasts (PRED_*)**: 12-hour predictions for multiple weather variables
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Data loading

    Loading OIKEN and weather data with Polars...
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
    # Handle two different date formats: try 4-digit year first, then 2-digit
    oiken_df = oiken_df.with_columns(
        pl.col("timestamp")
        .str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False)
        .fill_null(pl.col("timestamp").str.strptime(pl.Datetime, "%d/%m/%y %H:%M", strict=False))
        .alias("timestamp")
    )
    mo.accordion({"OIKEN raw data": oiken_df})
    return (oiken_df,)


@app.cell(hide_code=True)
def _(mo, pl):
    weather_df = pl.read_csv("data/sion_weather_2026-03-19_15-28.csv", try_parse_dates=True)

    # Strip UTC timezone to match OIKEN data format (both naive datetimes)
    weather_df = weather_df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    mo.accordion({"Weather raw data": weather_df})
    return (weather_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Data processing
    - Renaming of the columns
    - Merging of the two datasets
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
            "Air temperature 2m above ground (current value)": "temperature",
            "Global radiation (ten minutes mean)": "global_radiation",
            "Precipitation (ten minutes total)": "precipitation",
            "Relative air humidity 2m above ground (current value)": "humidity",
            "Sunshine duration (ten minutes total)": "sunshine_duration",
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
        "temperature",
        "global_radiation",
        "precipitation",
        "humidity",
        "sunshine_duration",
        "forecast_temperature",
        "forecast_global_radiation",
        "forecast_precipitation",
        "forecast_humidity",
        "forecast_sunshine_duration",
    )
    mo.accordion({"Weather renamed": weather_renamed})
    return (weather_renamed,)


@app.cell(hide_code=True)
def _(mo, oiken_renamed, weather_renamed):
    # Merge OIKEN and weather datasets on timestamp (both naive datetimes)
    merged_df = oiken_renamed.join(
        weather_renamed,
        on="timestamp",
        how="inner",
    )
    mo.accordion(
        {f"Merged dataset ({merged_df.height:,} rows, {merged_df.width} columns)": merged_df}
    )
    return (merged_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Data analysis

    ### 3.1 Data quality overview

    Assessing missing values per column to identify data gaps that may affect feature selection and model training.
    """)
    return


@app.cell(hide_code=True)
def _(alt, merged_df, mo, pl):
    # Compute null counts and percentages
    null_counts = merged_df.null_count()
    total_rows = merged_df.height
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
    chart = (
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
    > **Note**: The inner join between OIKEN (15-min) and weather (10-min) data
    > only aligns at shared timestamps, dropping rows. Resampling will be
    > addressed separately.
                    """),
                    chart,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.2 Load profile analysis

    Exploring daily, weekly, and seasonal load patterns to understand which temporal features drive consumption.
    """)
    return


@app.cell(hide_code=True)
def _(alt, merged_df, mo, pl):
    # Compute daily profiles
    _daily_profile = (
        merged_df.with_columns(
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.when(pl.col("timestamp").dt.weekday() >= 5)
            .then(pl.lit("Weekend"))
            .otherwise(pl.lit("Weekday"))
            .alias("day_type"),
        )
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
    _weekly_profile = (
        merged_df.with_columns(pl.col("timestamp").dt.weekday().alias("dow"))
        .group_by("dow")
        .agg(pl.col("load").mean().alias("mean_load"))
        .sort("dow")
    )
    _weekly_chart = (
        alt.Chart(_weekly_profile)
        .mark_bar()
        .encode(
            x=alt.X("dow:O", title="Day of week (1=Mon, 7=Sun)"),
            y=alt.Y("mean_load:Q", title="Mean load (standardised)"),
        )
        .properties(width=500, height=300, title="Average weekly load profile")
    )
    mo.accordion({"Daily & weekly load profiles": mo.hstack([_daily_chart, _weekly_chart])})
    return


@app.cell(hide_code=True)
def _(alt, merged_df, mo, pl):
    # Seasonal heatmap: mean load by month x hour
    seasonal = (
        merged_df.with_columns(
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.month().alias("month"),
        )
        .group_by("month", "hour")
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
    ## 4. Calendar features

    Calendar features capture temporal patterns in electricity consumption:
    - **Daily patterns**: hour of day (morning peak, evening dip)
    - **Weekly patterns**: weekday vs weekend
    - **Seasonal patterns**: month, day of year
    - **Special days**: holidays, working days

    ### Feature Extraction Strategy
    1. Basic temporal features (hour, day of week, month, etc.)
    2. Cyclical encoding (sin/cos) for periodic features
    3. Swiss holiday calendar (Valais region)
    4. Working day classification
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.1 Basic Temporal Features

    Extracting temporal features from timestamp...
    """)
    return


@app.cell
def _(oiken_df, pl):
    def add_temporal_features(df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
        """Extract basic temporal features from timestamp column."""
        return df.with_columns(
            [
                # Hour of day (0-23)
                pl.col(timestamp_col).dt.hour().alias("hour"),
                # Day of week (0=Monday, 6=Sunday)
                pl.col(timestamp_col).dt.weekday().alias("day_of_week"),
                # Month (1-12)
                pl.col(timestamp_col).dt.month().alias("month"),
                # Day of year (1-366)
                pl.col(timestamp_col).dt.ordinal_day().alias("day_of_year"),
                # Week of year (1-53)
                pl.col(timestamp_col).dt.week().alias("week_of_year"),
                # Binary flags
                (pl.col(timestamp_col).dt.weekday() >= 5).alias("is_weekend"),  # Sat=5, Sun=6
            ]
        )

    # Test on OIKEN data
    oiken_with_temporal = add_temporal_features(oiken_df)
    oiken_with_temporal.select("timestamp", "hour", "day_of_week", "month", "is_weekend").head(10)
    return (oiken_with_temporal,)


@app.cell
def _(mo):
    mo.md("""
    ### 4.2 Cyclical Encoding

    Cyclical encoding ensures that periodic features wrap around correctly:
    - 23:00 should be close to 00:00 (not 23 units apart!)
    - Sunday (day 6) should be close to Monday (day 0)

    Formula: `sin_feature = sin(2π * feature / max_value)`
    """)
    return


@app.cell
def _(oiken_with_temporal, pl):
    def add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add sin/cos encoding for periodic temporal features."""
        return df.with_columns(
            [
                # Hour (0-23) -> 2π/24
                (pl.col("hour") * 2 * 3.14159 / 24).sin().alias("sin_hour"),
                (pl.col("hour") * 2 * 3.14159 / 24).cos().alias("cos_hour"),
                # Day of week (0-6) -> 2π/7
                (pl.col("day_of_week") * 2 * 3.14159 / 7).sin().alias("sin_dow"),
                (pl.col("day_of_week") * 2 * 3.14159 / 7).cos().alias("cos_dow"),
                # Month (1-12) -> 2π/12 (shifted by 1 to start at 0)
                ((pl.col("month") - 1) * 2 * 3.14159 / 12).sin().alias("sin_month"),
                ((pl.col("month") - 1) * 2 * 3.14159 / 12).cos().alias("cos_month"),
                # Day of year (1-366) -> 2π/366 (shifted by 1)
                ((pl.col("day_of_year") - 1) * 2 * 3.14159 / 366).sin().alias("sin_doy"),
                ((pl.col("day_of_year") - 1) * 2 * 3.14159 / 366).cos().alias("cos_doy"),
            ]
        )

    # Test on data with temporal features
    oiken_with_cyclical = add_cyclical_features(oiken_with_temporal)
    oiken_with_cyclical.select(
        "hour", "sin_hour", "cos_hour", "day_of_week", "sin_dow", "cos_dow"
    ).head(10)
    return (oiken_with_cyclical,)


@app.cell
def _(mo):
    mo.md("""
    ### 4.3 Swiss Holiday Calendar

    Adding Swiss holidays (focusing on Valais region)...

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


@app.cell
def _(oiken_with_cyclical, pl):
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
    oiken_with_holidays = add_holiday_features(oiken_with_cyclical)

    # Show some examples
    holiday_examples = (
        oiken_with_holidays.filter(pl.col("is_holiday") == True)
        .select("timestamp", "is_holiday", "is_weekend")
        .head(10)
    )
    holiday_examples
    return (oiken_with_holidays,)


@app.cell
def _(mo):
    mo.md("""
    ### 4.4 Working Day Classification

    A "working day" is typically defined as:
    - NOT a weekend (Saturday/Sunday)
    - NOT a public holiday

    Working days have different consumption patterns than non-working days
    (industrial/commercial activity is higher).
    """)
    return


@app.cell
def _(oiken_with_holidays, pl):
    def add_working_day_flag(df: pl.DataFrame) -> pl.DataFrame:
        """Add is_working_day flag (not weekend AND not holiday)."""
        return df.with_columns(
            (~pl.col("is_weekend") & ~pl.col("is_holiday")).alias("is_working_day")
        )

    # Apply working day flag
    oiken_calendar_complete = add_working_day_flag(oiken_with_holidays)

    # Summary statistics
    calendar_summary = oiken_calendar_complete.select(
        [
            pl.col("is_weekend").sum().alias("weekend_days"),
            pl.col("is_holiday").sum().alias("holidays"),
            pl.col("is_working_day").sum().alias("working_days"),
        ]
    )
    calendar_summary
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.5 Complete Calendar Feature Pipeline

    Combining all calendar feature functions into a single pipeline...
    """)
    return


@app.cell
def _(oiken_df, pl):
    def build_calendar_features(
        df: pl.DataFrame, timestamp_col: str = "timestamp"
    ) -> pl.DataFrame:
        """
        Complete calendar feature engineering pipeline.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with timestamp column
        timestamp_col : str
            Name of timestamp column (default: "timestamp")

        Returns
        -------
        pl.DataFrame
            Dataframe with all calendar features added
        """
        # Step 1: Basic temporal features
        df = df.with_columns(
            [
                pl.col(timestamp_col).dt.hour().alias("hour"),
                pl.col(timestamp_col).dt.weekday().alias("day_of_week"),
                pl.col(timestamp_col).dt.month().alias("month"),
                pl.col(timestamp_col).dt.ordinal_day().alias("day_of_year"),
                pl.col(timestamp_col).dt.week().alias("week_of_year"),
                (pl.col(timestamp_col).dt.weekday() >= 5).alias("is_weekend"),
            ]
        )

        # Step 2: Cyclical encoding
        df = df.with_columns(
            [
                (pl.col("hour") * 2 * 3.14159 / 24).sin().alias("sin_hour"),
                (pl.col("hour") * 2 * 3.14159 / 24).cos().alias("cos_hour"),
                (pl.col("day_of_week") * 2 * 3.14159 / 7).sin().alias("sin_dow"),
                (pl.col("day_of_week") * 2 * 3.14159 / 7).cos().alias("cos_dow"),
                ((pl.col("month") - 1) * 2 * 3.14159 / 12).sin().alias("sin_month"),
                ((pl.col("month") - 1) * 2 * 3.14159 / 12).cos().alias("cos_month"),
                ((pl.col("day_of_year") - 1) * 2 * 3.14159 / 366).sin().alias("sin_doy"),
                ((pl.col("day_of_year") - 1) * 2 * 3.14159 / 366).cos().alias("cos_doy"),
            ]
        )

        # Step 3: Holiday features
        import holidays

        years = df.select(pl.col(timestamp_col).dt.year().unique()).to_series().to_list()
        holiday_dates = set()
        for year in years:
            holiday_dates.update(holidays.CH(years=year))
            holiday_dates.update(holidays.CH(years=year, prov="VS"))

        df = df.with_columns(
            pl.col(timestamp_col).dt.date().is_in(holiday_dates).alias("is_holiday")
        )

        # Step 4: Working day flag
        df = df.with_columns(
            (~pl.col("is_weekend") & ~pl.col("is_holiday")).alias("is_working_day")
        )

        return df

    # Apply complete pipeline to OIKEN data
    oiken_calendar = build_calendar_features(oiken_df)

    # Show final feature set
    feature_cols = [
        c
        for c in oiken_calendar.columns
        if c.startswith(("hour", "day", "month", "week", "is_", "sin_", "cos_"))
    ]
    print(f"Calendar features added: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")

    oiken_calendar.select("timestamp", *feature_cols[:5]).head(10)
    return


if __name__ == "__main__":
    app.run()
