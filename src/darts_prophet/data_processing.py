"""
Data loading and processing utilities for time series forecasting.

Handles CSV data loading with date/hour column merging and
TimeSeries conversion for Darts models.
"""

import pandas as pd
from darts.timeseries import TimeSeries
from darts.utils.missing_values import fill_missing_values


def load_data(
    path: str,
    target_col: str = "price",
    date_col: str = "date",
    hour_col: str = "hour",
) -> pd.DataFrame:
    """
    Load CSV data and merge date + hour into a datetime column.
    
    Args:
        path: Path to the CSV file.
        target_col: Name of the target value column.
        date_col: Name of the date column (YYYY-MM-DD format).
        hour_col: Name of the hour column (1–24).
    
    Returns:
        DataFrame with 'datetime' and target columns. Aggregates by mean
        if there are duplicate timestamps.
    """
    df = pd.read_csv(path)

    # Combine date + hour into a single datetime column
    # hour is 1–24, so we treat hour 24 as 00:00 of next day
    df[hour_col] = df[hour_col].astype(int)
    df["datetime"] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[hour_col] - 1, unit="h")

    # If there are duplicate timestamps (multiple rows per hour), aggregate by mean
    df = df.groupby("datetime")[target_col].mean().reset_index()
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"✅ Loaded {len(df)} rows  |  Range: {df['datetime'].min()} → {df['datetime'].max()}")
    return df


def build_series(df: pd.DataFrame, target_col: str = "price") -> TimeSeries:
    """
    Build a Darts TimeSeries from a DataFrame.
    
    Args:
        df: DataFrame with 'datetime' column and target column.
        target_col: Name of the target value column.
    
    Returns:
        Darts TimeSeries object with hourly frequency and filled missing values.
    """
    series = TimeSeries.from_dataframe(
        df,
        time_col="datetime",
        value_cols=target_col,
        freq="h",          # hourly frequency
        fill_missing_dates=True,
    )
    # Fill any gaps introduced by fill_missing_dates
    series = fill_missing_values(series)
    return series
