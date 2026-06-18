"""
Data loading and processing utilities for time series forecasting.

Handles CSV data loading with day_of_week + hour indexing (no date information).
Preserves ALL data points by assigning them to repeating weekly cycles.
"""

import pandas as pd
import numpy as np
from darts.timeseries import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler
import yaml

def ensure_chronological_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe is ordered oldest → newest.

    Detection priority:
        1. date
        2. createdAt

    If neither column exists, the dataframe is returned unchanged.

    Returns
    -------
    DataFrame in chronological order.
    """

    date_col = None

    for candidate in ("date", "createdAt"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        print(
            "⚠️  No 'date' or 'createdAt' column found. "
            "Assuming data is already chronological."
        )
        return df.reset_index(drop=True)

    dates = pd.to_datetime(df[date_col], errors="coerce")

    valid_dates = dates.dropna()

    if len(valid_dates) < 2:
        print(
            f"⚠️  Could not determine chronology from '{date_col}'. "
            "Assuming data is already chronological."
        )
        return df.reset_index(drop=True)

    first_date = valid_dates.iloc[0]
    last_date = valid_dates.iloc[-1]

    if first_date > last_date:
        print(
            f"↩️  Reverse chronological order detected "
            f"({date_col}: {first_date} → {last_date}). "
            "Reversing dataset."
        )
        return df.iloc[::-1].reset_index(drop=True)

    print(
        f"➡️  Chronological order detected "
        f"({date_col}: {first_date} → {last_date})."
    )

    return df.reset_index(drop=True)

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
        freq=1,
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
    sort_by_day_hour: bool = True
) -> TimeSeries:
    """
    Load CSV and create a TimeSeries with RangeIndex (no artificial dates).
    
    Preserves ALL observations by:
    1. Sorting chronologically
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

    df = ensure_chronological_order(df)

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
            # For continuous columns, use linear interpolation (mean of surrounding values)
            if col in ['national holiday', 'walk-in']:
                fill_value = 0
                print(f"      → Filling with 0 (binary column)")
                df[col] = df[col].fillna(fill_value)
            else:
                print(f"      → Using forward linear interpolation (based on preceding values only)")
                df[col] = df[col].interpolate(method='linear', limit_direction='forward')
                # For any remaining NaN at the very start (no preceding value exists), forward-fill only
                df[col] = df[col].ffill()
        
        if inf_count > 0:
            print(f"⚠️  Column '{col}': {inf_count} Inf values found - clipping to ±1e6")
            df[col] = np.clip(df[col], -1e6, 1e6)
    
    if "month" in target_cols:
        # Replace raw month integer with sin/cos encoding
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12).astype(np.float32)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12).astype(np.float32)
        # Replace 'month' in target_cols with the two encoded columns
        target_cols = list(dict.fromkeys(
        [c for c in target_cols if c != "month"]
        + ["month_sin", "month_cos"]))

        keep_cols = list(dict.fromkeys(
            target_cols + [day_of_week_col, hour_col]
        ))

        df = df[keep_cols]

        # Always set sort by day hour false here
        sort_by_day_hour = False

    if len(target_cols) != len(set(target_cols)):
        duplicates = [
            c for c in target_cols
            if target_cols.count(c) > 1
        ]

        raise ValueError(
            f"Duplicate target columns detected: {sorted(set(duplicates))}"
        )

    # Preserve chronological ordering. Never sort by weekday/hour.
    df = df.reset_index(drop=True)   # preserve original order
    
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


def normalize_series(series: TimeSeries, scaler: Scaler | None = None) -> tuple[TimeSeries, Scaler]:
    """
    Normalize a TimeSeries using StandardScaler (zero mean, unit variance).
    
    Args:
        series: Input TimeSeries to normalize.
        scaler: Optional pre-fitted Scaler. If None, a new one is created and fitted.
    
    Returns:
        (normalized_series, scaler) — the normalized series and the fitted scaler
        (can be used to transform other series like covariates).
    """
    if scaler is None:
        scaler = Scaler()
        series = scaler.fit_transform(series)
    else:
        series = scaler.transform(series)
    
    return series, scaler