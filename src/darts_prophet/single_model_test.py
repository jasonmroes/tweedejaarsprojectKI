"""
single_model_test.py
====================
Trains and evaluates any Darts forecasting model on a price time series.

Key features
------------
* Works with any Darts model class.
* Automatically filters constructor kwargs to only those the model accepts.
* Passes past_covariates / future_covariates only when the model supports them.
* Optional month (sin/cos) future covariate — enable in YAML.
* Produces a 4-panel diagnostic figure and exports forecasts to CSV.
* Backtest model  : trained on (train / val), evaluated on test.
* Future model    : trained on (train / val) over the full dataset —
                    no test holdout, val used for early stopping only.

Split layout
------------
    Backtest model:  |── train ──|── val ──|── test ──|
    Full model:      |──── train ────|──── val ────|

Required YAML keys (see configs/no_past.yaml for a full example)
-----------------------------------------------------------------
    data:
      data_path:            path/to/data.csv
      future_data_path:     path/to/future_data.csv   # optional
      target_col:           price
      past_covariates:      []
      future_covariates:    []
      validation_fraction:  0.1
      repeat_cycles:        4
      use_month_covariate:  false   # set true to add month sin/cos
      month_covariate:             # only needed when use_month_covariate: true
        start_month:   1
        start_day:     1
        start_hour:    0
        hours_per_row: 1

    model:
      model_cls:                TiDEModel
      input_chunk_length:       168
      output_chunk_length:      336
      test_checkpoint_filepath: experiments/TiDEModel_backtest.pt
      full_checkpoint_filepath: experiments/TiDEModel_future.pt

    training:
      save_dir:      experiments/
      epochs:        50
      batch_size:    32
      learning_rate: 1e-4
      weight_decay:  0.01

Usage
-----
    python single_model_test.py
"""

from __future__ import annotations

import os
import yaml
from typing import Any, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts.metrics import smape, rmse, mae
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries

from data_processing import normalize_series
from model_utils import (
    filter_kwargs,
    is_torch_model,
    build_torch_kwargs,
    build_fit_kwargs,
    build_predict_kwargs,
    load_all_series,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "configs/no_past.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

TARGET_COL = cfg["data"].get("target_col", "price")
SAVE_DIR   = cfg["training"]["save_dir"]
# ─────────────────────────────────────────────────────────────────────────────


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _ts_to_array(ts: TimeSeries) -> np.ndarray:
    """Extract the first component of a TimeSeries as a flat float32 array."""
    return ts.values(copy=False)[:, 0].astype(np.float32)


def _make_tick_labels(
    time_index, n_positions: int, max_labels: int = 10
) -> tuple[list[int], list[str]]:
    """
    Return (positions, labels) for x-axis ticks.

    Positions are plain integers (0..n_positions-1) — always safe to pass
    to ax.set_xticks() regardless of the Darts index type.

    Supports integer weekly-cycle indices, pd.Timestamp, and numpy datetime64.
    """
    day_names        = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    reference_monday = pd.Timestamp(2024, 1, 1)

    step      = max(1, n_positions // max_labels)
    positions = list(range(0, n_positions, step))
    labels: list[str] = []

    for pos in positions:
        ts = time_index[pos]

        if isinstance(ts, (int, np.integer)):
            day_of_week = (int(ts) // 24) % 7
            hour        = int(ts) % 24
        elif isinstance(ts, pd.Timestamp):
            dh          = int((ts - reference_monday).total_seconds() / 3600)
            day_of_week = (dh // 24) % 7
            hour        = dh % 24
        else:
            ts_pd       = pd.Timestamp(ts)
            dh          = int((ts_pd - reference_monday).total_seconds() / 3600)
            day_of_week = (dh // 24) % 7
            hour        = dh % 24

        labels.append(f"{day_names[day_of_week]} {hour:02d}:00")

    return positions, labels


def _apply_time_ticks(ax, time_index, n_positions: int, max_labels: int = 10) -> None:
    positions, labels = _make_tick_labels(time_index, n_positions, max_labels)
    if positions:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Day of Week & Hour")


# ── Core ──────────────────────────────────────────────────────────────────────

def run_forecast(
    model_cls:         Type[ForecastingModel],
    series:            TimeSeries,
    model_kwargs:      dict[str, Any]   | None = None,
    past_covariates:   TimeSeries       | None = None,
    future_covariates: TimeSeries       | None = None,
    ckpt_backtest:     str | None = None,
    ckpt_future:       str | None = None,
    test_size:         int  = cfg["model"]["output_chunk_length"],
    forecast_horizon:  int  = cfg["model"]["output_chunk_length"],
    result_path:       str  = cfg["training"]["save_dir"],
    target_col:        str  = TARGET_COL,
    scaler            = None,
) -> tuple[TimeSeries, TimeSeries | None]:
    """
    Train (or load from checkpoint), evaluate, and forecast.

    Parameters
    ----------
    model_cls         : Any Darts model class (not an instance).
    series            : Normalised target TimeSeries (full dataset).
    model_kwargs      : Constructor kwargs; unsupported ones are silently dropped.
    past_covariates   : Optional normalised past-covariate TimeSeries.
    future_covariates : Optional normalised future-covariate TimeSeries.
    ckpt_backtest     : Path to saved backtest checkpoint; skips training when set.
    ckpt_future       : Path to saved future-model checkpoint; skips training when set.
    test_size         : Steps held out for backtest evaluation.
    forecast_horizon  : Steps to forecast beyond the full dataset.
    result_path       : Directory for checkpoints, plots, and CSVs.
    target_col        : Used in plot titles.
    scaler            : Fitted Darts Scaler; used to inverse-transform predictions.

    Returns
    -------
    (backtest_pred, future_pred)
    """
    os.makedirs(result_path, exist_ok=True)
    model_name   = model_cls.__name__
    model_kwargs = model_kwargs or {}

    validation_fraction = cfg["data"]["validation_fraction"]

    # Inverse-transform series once so all plotting / metrics are in price units.
    series_orig = scaler.inverse_transform(series) if scaler is not None else series

    # ── Backtest split ────────────────────────────────────────────────────────
    train_full, test = series[:-test_size], series[-test_size:]
    val_size         = int(len(train_full) * validation_fraction)
    train            = train_full[:-val_size]
    val              = train_full[-val_size:]

    print(
        f"\n📊 Backtest split:"
        f"  Train {len(train)}  |  Val {len(val)}  |  Test {len(test)}"
    )

    # Slice covariates
    past_cov_train = past_cov_val = None
    fut_cov_train  = fut_cov_val  = None

    if past_covariates is not None:
        past_cov_train = past_covariates.slice_intersect(train)
        past_cov_val   = past_covariates.slice_intersect(val)

    if future_covariates is not None:
        fut_cov_train = future_covariates.slice_intersect(train)
        fut_cov_val   = future_covariates.slice_intersect(val)

    safe_kwargs = filter_kwargs(model_cls, model_kwargs)

    # ── Backtest model ────────────────────────────────────────────────────────
    if ckpt_backtest:
        model = model_cls.load(ckpt_backtest)
        print(f"✅ Backtest checkpoint loaded from {ckpt_backtest}")
    else:
        print(f"🔧 Creating {model_name} with kwargs: {safe_kwargs}")
        model = model_cls(**safe_kwargs)
        fit_kw = build_fit_kwargs(
            model, train, val,
            past_cov_train, fut_cov_train,
            past_cov_val,   fut_cov_val,
        )
        print(f"⏳ Fitting {model_name} on backtest split…")
        model.fit(**fit_kw)
        ckpt_backtest = os.path.join(result_path, f"{model_name}_checkpoint_backtest.pt")
        model.save(ckpt_backtest)
        print(f"💾 Backtest checkpoint saved → {ckpt_backtest}")

    # ── Backtest prediction ───────────────────────────────────────────────────
    print("🔮 Generating backtest forecast…")
    pred_kw = build_predict_kwargs(
        model, len(test), train_full,
        past_covariates, future_covariates,
    )
    pred = model.predict(**pred_kw)

    if scaler is not None:
        pred     = scaler.inverse_transform(pred)
        test_inv = scaler.inverse_transform(test)
    else:
        test_inv = test

    # ── Backtest metrics ──────────────────────────────────────────────────────
    smape_val = smape(test_inv, pred)
    rmse_val  = rmse(test_inv, pred)
    mae_val   = mae(test_inv, pred)
    print(f"\n📈 SMAPE : {smape_val:.2f}%")
    print(f"📉 RMSE  : {rmse_val:.4f}")
    print(f"📐 MAE   : {mae_val:.4f}")

    # ── Full-series split (no test holdout) ───────────────────────────────────
    future_val_size = int(len(series) * validation_fraction)
    future_train    = series[:-future_val_size]
    future_val      = series[-future_val_size:]

    print(
        f"\n🔭 Full-series split:"
        f"  Train {len(future_train)}  |  Val {len(future_val)}"
    )

    fut_past_cov_train = fut_past_cov_val = None
    fut_fut_cov_train  = fut_fut_cov_val  = None

    if past_covariates is not None:
        fut_past_cov_train = past_covariates.slice_intersect(future_train)
        fut_past_cov_val   = past_covariates.slice_intersect(future_val)

    if future_covariates is not None:
        fut_fut_cov_train = future_covariates.slice_intersect(future_train)
        fut_fut_cov_val   = future_covariates.slice_intersect(future_val)

    # ── Future model ──────────────────────────────────────────────────────────
    if ckpt_future:
        future_model = model_cls.load(ckpt_future)
        print(f"✅ Future checkpoint loaded from {ckpt_future}")
    else:
        print(f"\n🔭 Fitting {model_name} on full-series split…")
        future_model = model_cls(**safe_kwargs)
        future_fit_kw = build_fit_kwargs(
            future_model, future_train, future_val,
            fut_past_cov_train, fut_fut_cov_train,
            fut_past_cov_val,   fut_fut_cov_val,
        )
        future_model.fit(**future_fit_kw)
        ckpt_future = os.path.join(result_path, f"{model_name}_checkpoint_future.pt")
        future_model.save(ckpt_future)
        print(f"💾 Future checkpoint saved → {ckpt_future}")

    # ── Future prediction ─────────────────────────────────────────────────────
    future_pred_kw = build_predict_kwargs(
        future_model, forecast_horizon, future_train,
        past_covariates, future_covariates,
    )
    future_pred = future_model.predict(**future_pred_kw)

    if scaler is not None:
        future_pred = scaler.inverse_transform(future_pred)

    print(f"\nfuture_pred length : {len(future_pred)}")
    print(f"future_pred start  : {future_pred.start_time()}")
    print(f"future_pred end    : {future_pred.end_time()}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_results(
        model_name       = model_name,
        series           = series_orig,
        test             = test_inv,
        pred             = pred,
        future_pred      = future_pred,
        smape_val        = smape_val,
        rmse_val         = rmse_val,
        mae_val          = mae_val,
        forecast_horizon = forecast_horizon,
        target_col       = target_col,
        result_path      = result_path,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    csv_path = os.path.join(result_path, f"{model_name}_future_forecast.csv")
    future_pred.to_dataframe().to_csv(csv_path)
    print(f"💾 Future forecast saved → {csv_path}")

    return pred, future_pred


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_results(
    model_name:       str,
    series:           TimeSeries,
    test:             TimeSeries,
    pred:             TimeSeries,
    future_pred:      TimeSeries | None,
    smape_val:        float,
    rmse_val:         float,
    mae_val:          float,
    forecast_horizon: int,
    target_col:       str,
    result_path:      str,
) -> None:
    """
    4-panel diagnostic figure (all values in price units).
      1. Backtest     — predicted vs actual (test window only).
      2. Residuals    — signed error over time.
      3. Future       — last 7 days of history + future forecast.
      4. Distribution — histogram of residuals.
    """
    test_vals = _ts_to_array(test)
    pred_vals = _ts_to_array(pred)

    n = min(len(test_vals), len(pred_vals))
    test_vals, pred_vals = test_vals[:n], pred_vals[:n]
    residuals = pred_vals - test_vals
    x         = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"{model_name}  –  '{target_col}' Forecast\n"
        f"SMAPE {smape_val:.2f}%   RMSE {rmse_val:.4f}   MAE {mae_val:.4f}",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Backtest
    ax1 = axes[0, 0]
    ax1.plot(x, test_vals, label="Actual",    color="steelblue",  linewidth=1.2)
    ax1.plot(x, pred_vals, label="Predicted", color="darkorange", linewidth=1.2)
    ax1.set_title("Backtest – Test Set")
    ax1.legend(fontsize=8)
    _apply_time_ticks(ax1, test.time_index, n)

    # Panel 2: Residuals
    ax2 = axes[0, 1]
    ax2.plot(x, residuals, color="tomato", linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.fill_between(x, residuals, 0, alpha=0.25, color="tomato")
    ax2.set_title("Residuals Over Time (predicted − actual)")
    ax2.set_ylabel("Residual")
    _apply_time_ticks(ax2, test.time_index, n)

    # Panel 3: Future forecast
    ax3 = axes[1, 0]
    if future_pred is not None:
        show_n    = min(7 * 24, len(series))
        hist_ts   = series[-show_n:]
        hist_vals = _ts_to_array(hist_ts)
        fut_vals  = _ts_to_array(future_pred)

        x_hist = np.arange(len(hist_vals))
        x_fut  = np.arange(len(hist_vals), len(hist_vals) + len(fut_vals))

        ax3.plot(x_hist, hist_vals, label="Historical (last 7 d)",
                 color="steelblue", linewidth=1.2)
        ax3.plot(x_fut, fut_vals, label=f"Future ({forecast_horizon} steps)",
                 color="darkorange", linewidth=1.2)
        ax3.axvline(len(hist_vals) - 1, color="grey",
                    linewidth=0.8, linestyle="--", alpha=0.6)

        combined_idx = np.concatenate([
            hist_ts.time_index.to_numpy(),
            future_pred.time_index.to_numpy(),
        ])
        _apply_time_ticks(ax3, combined_idx, len(hist_vals) + len(fut_vals))
        ax3.set_title(f"Future Forecast – Next {forecast_horizon} Steps")
        ax3.legend(fontsize=8)
    else:
        ax3.set_title("Future Forecast – Skipped")
        ax3.text(0.5, 0.5, "No future forecast", ha="center", va="center",
                 transform=ax3.transAxes, color="grey", fontsize=10)

    # Panel 4: Residual distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax4.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax4.axvline(float(residuals.mean()), color="tomato", linewidth=1.2,
                linestyle="--", label=f"Mean {float(residuals.mean()):.3f}")
    ax4.set_title("Residual Distribution")
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(result_path, f"{model_name}_forecast.png")
    plt.savefig(plot_path, dpi=150)
    print(f"💾 Plot saved → {plot_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import importlib

    darts_models   = importlib.import_module("darts.models")
    model_cls_name = cfg["model"]["model_cls"]
    MODEL_CLS      = getattr(darts_models, model_cls_name)

    # Build model kwargs
    MODEL_KWARGS: dict[str, Any] = {}
    if is_torch_model(MODEL_CLS):
        print("🧠 Neural network detected – building kwargs from YAML…")
        MODEL_KWARGS = build_torch_kwargs(cfg)
    else:
        for key in ("seasonality_mode", "yearly_seasonality",
                    "weekly_seasonality", "daily_seasonality"):
            if key in cfg.get("model", {}):
                MODEL_KWARGS[key] = cfg["model"][key]

    # Load data — month covariate appended to future_cov when enabled in YAML
    print("📂 Loading data…")
    series, scaler, past_cov, future_cov = load_all_series(cfg)
    series = series.astype(np.float32)

    print(
        f"✅ Series: {len(series)} steps"
        + (f"  |  past_cov: {past_cov.n_components} component(s)"
           if past_cov  is not None else "")
        + (f"  |  future_cov: {future_cov.n_components} component(s)"
           if future_cov is not None else "")
    )

    run_forecast(
        model_cls         = MODEL_CLS,
        series            = series,
        model_kwargs      = MODEL_KWARGS,
        past_covariates   = past_cov,
        future_covariates = future_cov,
        ckpt_backtest     = cfg["model"].get("test_checkpoint_filepath") or None,
        ckpt_future       = cfg["model"].get("full_checkpoint_filepath") or None,
        test_size         = cfg["model"]["output_chunk_length"],
        forecast_horizon  = cfg["model"]["output_chunk_length"],
        result_path       = cfg["training"]["save_dir"],
        target_col        = TARGET_COL,
        scaler            = scaler,
    )
