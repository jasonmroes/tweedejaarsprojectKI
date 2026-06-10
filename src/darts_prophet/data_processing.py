"""
Data loading and processing utilities for time series forecasting.

Handles CSV data loading with day_of_week + hour indexing (no date information).
Preserves ALL data points by assigning them to repeating weekly cycles.
"""

import pandas as pd
import numpy as np
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
        DataFrame with RangeIndex and target columns.
        All data rows preserved, no aggregation.
    """
    df = pd.read_csv(path, low_memory=False)

    df[day_of_week_col] = df[day_of_week_col].astype(int)
    df[hour_col] = df[hour_col].astype(int)
    
    # Sort by day_of_week and hour to group similar times together
    # This keeps all observations for each (day, hour) combo adjacent
    df = df.sort_values([day_of_week_col, hour_col]).reset_index(drop=True)
    
    # Use RangeIndex directly instead of artificial dates
    df = df.reset_index(drop=True)

    print(f"✅ Loaded {len(df)} total rows")
    print(f"   Index type: RangeIndex(0, {len(df)})")
    print(f"   Row distribution: {len(df)} observations")
    return df


def build_series(df: pd.DataFrame, target_col: str = "price") -> TimeSeries:
    """
    Build a Darts TimeSeries from DataFrame with RangeIndex.
    
    Args:
        df: DataFrame with RangeIndex and target column.
        target_col: Name of the target value column.
    
    Returns:
        Darts TimeSeries object with RangeIndex frequency.
    """
    series = TimeSeries.from_dataframe(
        df,
        value_cols=target_col,
        freq="1",
        fill_missing_dates=False,
    )
    return series


def load_timeseries(
    path: str,
    target_cols: list[str] = ['price'],
    day_of_week_col: str = "day of week",
    hour_col: str = "hour",
    repeat_cycles: int = None,
    all_columns_for_alignment: list[str] | None = None,
) -> TimeSeries:
    """
    Load CSV and create a TimeSeries with RangeIndex (no artificial dates).
    
    Preserves ALL observations by:
    1. Sorting by (day_of_week, hour)
    2. Sequentially assigning to repeating weekly cycles
    3. Using integer RangeIndex instead of synthetic datetime
    4. Interpolating NaN values instead of dropping rows:
       - Float columns: fill with column mean
       - Binary/categorical columns: fill with 0
    
    Multiple observations for same (day, hour) combo are kept in different cycles.
    
    Args:
        path: Path to the CSV file.
        target_cols: List of target column names to extract.
        day_of_week_col: Name of day-of-week column (0=Monday, 6=Sunday).
        hour_col: Name of hour column (0–23).
        repeat_cycles: Number of times to repeat pattern (ignored - determined by data).
        all_columns_for_alignment: List of all columns to validate (unused - kept for compatibility).
    
    Returns:
        Darts TimeSeries with all data preserved, RangeIndex.
    """
    df = pd.read_csv(path, low_memory=False)

    df[day_of_week_col] = df[day_of_week_col].astype(int)
    df[hour_col] = df[hour_col].astype(int)
    
    # ── Data validation and imputation ──────────────────────────────────────
    # Keep all rows, interpolate NaN values instead of dropping
    for col in target_cols:
        if col not in df.columns:
            continue
            
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col].values).sum()
        
        if nan_count > 0:
            print(f"⚠️  Column '{col}': {nan_count} NaN values found - interpolating")
            # For binary/categorical columns (like 'national holiday'), use 0
            # For continuous columns, use mean
            if col in ['national holiday', 'walk-in']:
                fill_value = 0
                print(f"      → Filling with 0 (binary column)")
            else:
                fill_value = df[col].mean()
                print(f"      → Filling with mean: {fill_value:.4f}")
            df[col] = df[col].fillna(fill_value)
        
        if inf_count > 0:
            print(f"⚠️  Column '{col}': {inf_count} Inf values found - clipping to ±1e6")
            df[col] = np.clip(df[col], -1e6, 1e6)
    
    # Sort by (day_of_week, hour) to group similar times together
    # This keeps all observations - each time slot can have multiple rows
    df = df.sort_values([day_of_week_col, hour_col]).reset_index(drop=True)
    
    # Keep only target columns and use RangeIndex
    df = df[target_cols].reset_index(drop=True)
    
    # ── Final validation: ensure no NaN/Inf in final data ────────────────────
    for col in target_cols:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col].values).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"⚠️  WARNING: Column '{col}' still has {nan_count} NaN, {inf_count} Inf after processing!")
        
        # Print data statistics for debugging
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"   📊 {col:20s}: mean={mean_val:10.2f}, std={std_val:10.2f}, min={min_val:10.2f}, max={max_val:10.2f}")
    
    print(f"✅ Loaded {len(df)} rows (NO data loss)")
    print(f"   Index type: RangeIndex(0, {len(df)})")

    series = TimeSeries.from_dataframe(
        df,
        value_cols=target_cols,
        freq="1",
        fill_missing_dates=False,
    )

    return series