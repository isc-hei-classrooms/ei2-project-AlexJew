"""Feature engineering for the net load forecasting pipeline.

Functions can be imported individually in notebooks or scripts:
    from utils.feature_engineering import (
        add_temporal_features,
        add_holiday_features,
        add_working_day_flag,
        add_dst_feature,
        add_cyclical_features,
        add_lag_features,
    )
"""

import math
import holidays
import polars as pl


def add_temporal_features(
    df: pl.DataFrame, timestamp_col: str = "local_timestamp"
) -> pl.DataFrame:
    """Extract basic temporal features from local timestamp column."""
    return df.with_columns(
        [
            pl.col(timestamp_col).dt.hour().alias("local_hour"),
            pl.col(timestamp_col).dt.weekday().alias("local_day_of_week"),
            pl.col(timestamp_col).dt.month().alias("local_month"),
            pl.col(timestamp_col).dt.ordinal_day().alias("local_day_of_year"),
            pl.col(timestamp_col).dt.week().alias("local_week_of_year"),
            (pl.col(timestamp_col).dt.weekday() > 5).alias("local_is_weekend"),
        ]
    )


def get_swiss_holidays(year: int) -> set:
    """Get set of Swiss holidays (national + Valais-specific) for a given year."""
    ch_holidays = holidays.CH(years=year)
    ch_holidays.update(holidays.CH(years=year, prov="VS"))
    return set(ch_holidays.keys())


def add_holiday_features(
    df: pl.DataFrame, timestamp_col: str = "local_timestamp"
) -> pl.DataFrame:
    """Add holiday flags to dataframe."""
    years = (
        df.select(pl.col(timestamp_col).dt.year().unique())
        .to_series()
        .to_list()
    )

    holiday_dates = set()
    for year in years:
        holiday_dates.update(get_swiss_holidays(year))

    return df.with_columns(
        pl.col(timestamp_col)
        .dt.date()
        .is_in(holiday_dates)
        .alias("local_is_holiday")
    )


def add_working_day_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Add local_is_working_day flag (not weekend AND not holiday)."""
    return df.with_columns(
        (~pl.col("local_is_weekend") & ~pl.col("local_is_holiday")).alias(
            "local_is_working_day"
        )
    )


def add_dst_feature(df: pl.DataFrame) -> pl.DataFrame:
    """Add local_is_summer_time flag based on UTC-to-local offset."""
    offset_hours = (
        pl.col("local_timestamp") - pl.col("utc_timestamp")
    ).dt.total_hours()
    return df.with_columns((offset_hours == 2).alias("local_is_summer_time"))


def add_cyclical_features(
    df: pl.DataFrame, timestamp_col: str = "utc_timestamp"
) -> pl.DataFrame:
    """Add sin/cos encoding for periodic temporal features based on UTC timestamp."""
    two_pi = 2 * math.pi
    return df.with_columns(
        [
            (pl.col(timestamp_col).dt.hour() * two_pi / 24)
            .sin()
            .alias("utc_sin_hour"),
            (pl.col(timestamp_col).dt.hour() * two_pi / 24)
            .cos()
            .alias("utc_cos_hour"),
            ((pl.col(timestamp_col).dt.weekday() - 1) * two_pi / 7)
            .sin()
            .alias("utc_sin_dow"),
            ((pl.col(timestamp_col).dt.weekday() - 1) * two_pi / 7)
            .cos()
            .alias("utc_cos_dow"),
            ((pl.col(timestamp_col).dt.month() - 1) * two_pi / 12)
            .sin()
            .alias("utc_sin_month"),
            ((pl.col(timestamp_col).dt.month() - 1) * two_pi / 12)
            .cos()
            .alias("utc_cos_month"),
            ((pl.col(timestamp_col).dt.ordinal_day() - 1) * two_pi / 366)
            .sin()
            .alias("utc_sin_doy"),
            ((pl.col(timestamp_col).dt.ordinal_day() - 1) * two_pi / 366)
            .cos()
            .alias("utc_cos_doy"),
        ]
    )


def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add lagging features: min, max, mean, std, and coefficient of variation (CV).

    Computes statistics for 3 time periods using FIXED daily windows:
    - 2 days ago: statistics from the full day (00:00-23:45) of D-2
    - 3 days ago: statistics from the full day (00:00-23:45) of D-3
    - Week (2-9 days ago): statistics from the 7 full days of D-9 to D-2

    All predictions on the same day use the SAME lag values (fixed, not rolling).

    Args:
        df: DataFrame with utc_timestamp and target columns

    Returns:
        DataFrame with 60 new lagging feature columns
    """
    variables = [
        "load",
        "sion_measured_temperature",
        "sion_measured_global_radiation",
        "solar_remote",
    ]

    df_with_date = df.with_columns(
        pl.col("utc_timestamp").dt.date().alias("_date")
    )

    agg_exprs = []
    for var in variables:
        agg_exprs.extend(
            [
                pl.col(var).min().alias(f"{var}_daily_min"),
                pl.col(var).max().alias(f"{var}_daily_max"),
                pl.col(var).mean().alias(f"{var}_daily_mean"),
                pl.col(var).std().alias(f"{var}_daily_std"),
            ]
        )
    daily_stats = df_with_date.group_by("_date").agg(agg_exprs)

    df_with_lags = df_with_date.with_columns(
        [
            (pl.col("_date") - pl.duration(days=2)).alias("_date_2d"),
            (pl.col("_date") - pl.duration(days=3)).alias("_date_3d"),
        ]
    )

    for period in ["2d", "3d"]:
        stats = daily_stats.rename({"_date": f"_date_{period}_join"})
        for var in variables:
            stats = stats.rename(
                {
                    f"{var}_daily_min": f"{var}_min_{period}",
                    f"{var}_daily_max": f"{var}_max_{period}",
                    f"{var}_daily_mean": f"{var}_mean_{period}",
                    f"{var}_daily_std": f"{var}_std_{period}",
                }
            )
        select_cols = [f"_date_{period}_join"] + [
            f"{v}_{s}_{period}"
            for v in variables
            for s in ["min", "max", "mean", "std"]
        ]
        df_with_lags = df_with_lags.join(
            stats.select(select_cols),
            left_on=f"_date_{period}",
            right_on=f"_date_{period}_join",
            how="left",
            suffix=f"_tmp_{period}",
        )

    week_exprs = []
    for var in variables:
        week_exprs.extend(
            [
                pl.col(f"{var}_daily_min")
                .rolling_mean(window_size=7, min_samples=1)
                .alias(f"{var}_week_min"),
                pl.col(f"{var}_daily_max")
                .rolling_mean(window_size=7, min_samples=1)
                .alias(f"{var}_week_max"),
                pl.col(f"{var}_daily_mean")
                .rolling_mean(window_size=7, min_samples=1)
                .alias(f"{var}_week_mean"),
                pl.col(f"{var}_daily_mean")
                .rolling_std(window_size=7, min_samples=1)
                .alias(f"{var}_week_std"),
            ]
        )
    week_stats = daily_stats.sort("_date").with_columns(week_exprs)
    df_with_lags = df_with_lags.join(
        week_stats.select(
            ["_date"]
            + [
                f"{v}_week_{s}"
                for v in variables
                for s in ["min", "max", "mean", "std"]
            ]
        ),
        left_on="_date_2d",
        right_on="_date",
        how="left",
        suffix="_week_tmp",
    )

    temp_cols = [
        c
        for c in df_with_lags.columns
        if c.startswith("_date") or "_tmp_" in c or "_week_tmp" in c
    ]
    df_with_lags = df_with_lags.drop(temp_cols)

    cv_exprs = []
    for var in variables:
        for period in ["2d", "3d"]:
            cv_exprs.append(
                pl.when(pl.col(f"{var}_mean_{period}").abs() > 1e-10)
                .then(
                    pl.col(f"{var}_std_{period}")
                    / pl.col(f"{var}_mean_{period}").abs()
                )
                .otherwise(None)
                .alias(f"{var}_cv_{period}")
            )
        cv_exprs.append(
            pl.when(pl.col(f"{var}_week_mean").abs() > 1e-10)
            .then(pl.col(f"{var}_week_std") / pl.col(f"{var}_week_mean").abs())
            .otherwise(None)
            .alias(f"{var}_cv_week")
        )

    return df_with_lags.with_columns(cv_exprs)
