"""
Data loading and processing utilities for time series forecasting.

Handles CSV data loading with day_of_week + hour indexing (no date information).
Preserves ALL data points by assigning them to repeating weekly cycles.
"""

import pandas as pd
from datetime import datetime, timedelta
from darts.timeseries import TimeSeries
from darts.utils.missing_values import fill_missing_values


def load_data(
    path: str,
    target_col: str = "price",
    day_of_week_col: str = "day of week",
    hour_col: str = "hour",
) -> pd.DataFrame:
    """
    Load CSV and create repeating weekly index, keeping ALL observations.
    
    Preserves all data by sorting by (day_of_week, hour) and sequentially
    assigning each row to repeating weekly cycles.
    
    Args:
        path: Path to the CSV file.
        target_col: Name of the target value column.
        day_of_week_col: Name of the day-of-week column (0=Monday, 6=Sunday).
        hour_col: Name of the hour column (0–23).
    
    Returns:
        DataFrame with 'datetime' (repeating weekly) and target columns.
        All data rows preserved, no aggregation.
    """
    df = pd.read_csv(path, low_memory=False)

    df[day_of_week_col] = df[day_of_week_col].astype(int)
    df[hour_col] = df[hour_col].astype(int)
    
    # Sort by day_of_week and hour to group similar times together
    # This keeps all observations for each (day, hour) combo adjacent
    df = df.sort_values([day_of_week_col, hour_col]).reset_index(drop=True)
    
    # Create repeating weekly index
    reference_monday = datetime(2024, 1, 1)  # Monday
    
    # Assign each row to a repeating weekly cycle (168 hours per week)
    df['cycle'] = df.index // 168
    df['position_in_cycle'] = df.index % 168
    
    # Create datetime: cycle through weeks, position is hour within week
    df['datetime'] = reference_monday + pd.to_timedelta(
        df['cycle'] * 168 + df['position_in_cycle'],
        unit='h'
    )

    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"✅ Loaded {len(df)} total rows")
    print(f"   Weekly cycles: {df['cycle'].max() + 1}")
    print(f"   Row distribution: {len(df)} observations across repeating weeks")
    return df


def build_series(df: pd.DataFrame, target_col: str = "price") -> TimeSeries:
    """
    Build a Darts TimeSeries from DataFrame with repeating weekly index.
    
    Args:
        df: DataFrame with 'datetime' column (repeating weekly) and target column.
        target_col: Name of the target value column.
    
    Returns:
        Darts TimeSeries object with hourly frequency.
    """
    series = TimeSeries.from_dataframe(
        df,
        time_col="datetime",
        value_cols=target_col,
        freq="h",
        fill_missing_dates=False,  # Don't fill - we want exact indices
    )
    return series


def load_timeseries(
    path: str,
    target_cols: list[str] = ['price'],
    day_of_week_col: str = "day of week",
    hour_col: str = "hour",
    repeat_cycles: int = None,
) -> TimeSeries:
    """
    Load CSV and create a repeating weekly TimeSeries with NO data loss.
    
    Preserves ALL observations by:
    1. Sorting by (day_of_week, hour)
    2. Sequentially assigning to repeating weekly cycles
    3. Creating synthetic datetime that cycles through weeks
    
    Multiple observations for same (day, hour) combo are kept in different cycles.
    
    Args:
        path: Path to the CSV file.
        target_cols: List of target column names to extract.
        day_of_week_col: Name of day-of-week column (0=Monday, 6=Sunday).
        hour_col: Name of hour column (0–23).
        repeat_cycles: Number of times to repeat pattern (ignored - determined by data).
    
    Returns:
        Darts TimeSeries with all data preserved, repeating weekly index.
    """
    df = pd.read_csv(path, low_memory=False)

    df[day_of_week_col] = df[day_of_week_col].astype(int)
    df[hour_col] = df[hour_col].astype(int)
    
    # Sort by (day_of_week, hour) to group similar times together
    # This keeps all observations - each time slot can have multiple rows
    df = df.sort_values([day_of_week_col, hour_col]).reset_index(drop=True)
    
    reference_monday = datetime(2024, 1, 1)  # Monday
    
    # Assign each row sequentially to repeating weekly cycles
    # If we have 1512 rows: rows 0-167 → week 1, rows 168-335 → week 2, etc.
    df['cycle'] = df.index // 168
    df['position_in_cycle'] = df.index % 168
    
    # Create datetime with repeating weekly pattern
    df['datetime'] = reference_monday + pd.to_timedelta(
        df['cycle'] * 168 + df['position_in_cycle'],
        unit='h'
    )
    
    # Sort by datetime and keep all target columns
    df = (
        df[['datetime'] + target_cols]
        .sort_values('datetime')
        .reset_index(drop=True)
    )
    
    print(f"✅ Loaded {len(df)} rows (NO aggregation)")
    print(f"   Distributed across {df.groupby('cycle').ngroups} weekly cycles")

    series = TimeSeries.from_dataframe(
        df,
        time_col="datetime",
        value_cols=target_cols,
        freq="h",
        fill_missing_dates=False,  # Keep exact indices, no filling
    )

    return series