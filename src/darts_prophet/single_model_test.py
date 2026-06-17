"""
single_model_test.py
====================
Trains and evaluates a single Darts model on the train / val / test split
defined in the config YAML.

Split layout
------------
    |── train ──|── val ──|── test ──|

The test set is held out entirely and never seen during training.
Val is used for early stopping only.

Metrics reported (on test set)
-------------------------------
    Normalised  : RMSE, MAE   (in scaler units — comparable across configs)
    Unnormalised: RMSE, MAE   (in original price units — interpretable)
    SMAPE                     (scale-independent, same either way)

Output
------
    <save_dir>/<ModelName>_backtest.png    — 4-panel diagnostic plot
    <save_dir>/<ModelName>_backtest.csv    — test predictions (price units)
    <save_dir>/<ModelName>_backtest_metrics.json

Usage
-----
    python single_model_test.py --config configs/TiDE_standard.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import pickle
from darts.timeseries import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel

from data_processing import normalize_series
from model_utils import (
    filter_kwargs,
    is_torch_model,
    build_torch_kwargs,
    build_fit_kwargs,
    build_predict_kwargs,
    load_all_series,
)


# ── Plotting helpers (unchanged from original) ────────────────────────────────

def _ts_to_array(ts: TimeSeries) -> np.ndarray:
    return ts.values(copy=False)[:, 0].astype(np.float32)


def _make_tick_labels(
    time_index, n_positions: int, max_labels: int = 10
) -> tuple[list[int], list[str]]:
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

def _daily_mae(residuals: np.ndarray, chunk: int = 24) -> list[float]:
    n_chunks = len(residuals) // chunk
    return [
        float(np.abs(residuals[i * chunk:(i + 1) * chunk]).mean())
        for i in range(n_chunks)
    ]
# ── Core ──────────────────────────────────────────────────────────────────────

def run_backtest(
    cfg:               dict,
    model_cls:         Type[ForecastingModel],
    series:            TimeSeries,
    scaler,
    model_kwargs:      dict[str, Any]  | None = None,
    past_covariates:   TimeSeries      | None = None,
    future_covariates: TimeSeries      | None = None,
) -> TimeSeries:
    """
    Train on (train+val), evaluate on test, produce plots and metrics.

    Returns the test-set prediction in original price units.
    """
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    data_cfg     = cfg["data"]

    model_name       = model_cls.__name__
    model_kwargs     = model_kwargs or {}
    result_path      = training_cfg["save_dir"]
    target_col       = data_cfg.get("target_col", "price")
    val_fraction     = data_cfg.get("validation_fraction", 0.1)
    test_fraction    = data_cfg.get("test_fraction", 0.1)
    output_chunk_len = model_cfg["output_chunk_length"]
    save_dir = training_cfg["save_dir"]


    os.makedirs(result_path, exist_ok=True)

    # ── Split ─────────────────────────────────────────────────────────────────
    test_size    = int(len(series) * test_fraction)
    train_full   = series[:-test_size]
    test         = series[-test_size:]

    val_size     = int(len(train_full) * val_fraction)
    train        = train_full[:-val_size]
    val          = train_full[-val_size:]

    print(
        f"\n📊 Split  —  Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}"
        f"  (val={val_fraction}, test={test_fraction})"
    )

    # ── Slice covariates ──────────────────────────────────────────────────────
    past_cov_train = past_cov_val = None
    fut_cov_train  = fut_cov_val  = None

    if past_covariates is not None:
        past_cov_train = past_covariates.slice_intersect(train)
        past_cov_val   = past_covariates.slice_intersect(val)

    if future_covariates is not None:
        fut_cov_train = future_covariates.slice_intersect(train)
        fut_cov_val   = future_covariates.slice_intersect(val)

    # ── Build and train model ─────────────────────────────────────────────────
    safe_kwargs = filter_kwargs(model_cls, model_kwargs)
    print(f"🔧 Constructor kwargs: {safe_kwargs}\n")

    ckpt_path = model_cfg.get("test_checkpoint_filepath") or None
    if ckpt_path:
        model = model_cls.load(ckpt_path)
        print(f"✅ Checkpoint loaded from {ckpt_path}")
    elif model_cfg.get("resume"):
        model = model_cls.load_from_checkpoint(
            model_name=model_cls_name,
            work_dir=save_dir,
            best=False,
        )
        print(f"✅ Resumed from checkpoint in {save_dir}")
    else:
        model = model_cls(**safe_kwargs)
        fit_kw = build_fit_kwargs(
            model, train, val,
            past_cov_train, fut_cov_train,
            past_cov_val,   fut_cov_val,
        )
        print(f"⏳ Training {model_name}…")
        model.fit(**fit_kw)
        ckpt_path = os.path.join(result_path, f"{model_name}_backtest.pt")
        model.save(ckpt_path)
        print(f"💾 Checkpoint saved → {ckpt_path}")

        # also save the scaler
        with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    # ── Predict on test set ───────────────────────────────────────────────────
    # Predict output_chunk_length steps at a time across the test window.
    # If test_size > output_chunk_length, use multiple windows.

    # ── Predict on test set ───────────────────────────────────────────────────
    past_cov_context = past_covariates.slice_intersect(train_full) if past_covariates is not None else None
    fut_cov_context  = future_covariates  # Must cover train_full + n steps ahead; keep full series

    pred_kw = build_predict_kwargs(
        model, output_chunk_len, train_full,
        past_cov_context, fut_cov_context,
    )
    pred_norm = model.predict(**pred_kw)

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Align lengths (test may be longer than output_chunk_length)
    n = min(len(pred_norm), len(test))
    pred_norm_vals = pred_norm.values()[:n, 0]
    test_norm_vals = test.values()[:n, 0]
    residuals_norm = pred_norm_vals - test_norm_vals

    # Normalised metrics
    rmse_norm  = float(np.sqrt((residuals_norm ** 2).mean()))
    mae_norm   = float(np.abs(residuals_norm).mean())

    # Unnormalised metrics (price units)
    pred_inv = scaler.inverse_transform(pred_norm)
    test_inv = scaler.inverse_transform(test[:n])

    pred_inv_vals = pred_inv.values()[:, 0]
    test_inv_vals = test_inv.values()[:, 0]
    residuals_inv = pred_inv_vals - test_inv_vals

    rmse_inv  = float(np.sqrt((residuals_inv ** 2).mean()))
    mae_inv   = float(np.abs(residuals_inv).mean())

    # SMAPE (scale-independent, compute on unnormalised for interpretability)
    denom     = (np.abs(test_inv_vals) + np.abs(pred_inv_vals)) / 2
    smape_val = float(np.mean(np.where(denom == 0, 0, np.abs(residuals_inv) / denom)) * 100)

    print(f"\n📈 Test-set metrics (first {n} steps, out of {len(test)} test steps):")
    print(f"   SMAPE          : {smape_val:.2f}%")
    print(f"   RMSE (norm)    : {rmse_norm:.6f}")
    print(f"   MAE  (norm)    : {mae_norm:.6f}")
    print(f"   RMSE (price)   : {rmse_inv:.4f}")
    print(f"   MAE  (price)   : {mae_inv:.4f}")

# ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "n_steps_evaluated":  n,
        "smape":              round(smape_val, 4),
        "rmse_norm":          round(rmse_norm, 6),
        "mae_norm":           round(mae_norm,  6),
        "rmse_price":         round(rmse_inv,  4),
        "mae_price":          round(mae_inv,   4),
    }
    metrics_path = os.path.join(result_path, f"{model_name}_backtest_metrics.json")

# ── Extended diagnostics ──────────────────────────────────────────────────
    print("\n📐 Extended diagnostics…")

    chunk = 24
    n_chunks = n // chunk

    if n_chunks > 0:

        # ------------------------------------------------------------------
        # Daily MAE (all hours)
        # ------------------------------------------------------------------
        chunk_maes_price = _daily_mae(residuals_inv, chunk)
        chunk_maes_norm  = _daily_mae(residuals_norm, chunk)

        mae_std_price = float(np.std(chunk_maes_price))
        mae_std_norm  = float(np.std(chunk_maes_norm))

        print(
            f"\n   MAE/day (price)          : "
            f"{[round(x, 2) for x in chunk_maes_price]}"
        )
        print(f"   MAE std/day (price)      : {mae_std_price:.4f}")
        print(f"   MAE day 1 (price)        : {chunk_maes_price[0]:.4f}")
        print(f"   MAE day {n_chunks} (price): {chunk_maes_price[-1]:.4f}")

        print(
            f"\n   MAE/day (norm)           : "
            f"{[round(x, 4) for x in chunk_maes_norm]}"
        )
        print(f"   MAE std/day (norm)       : {mae_std_norm:.6f}")
        print(f"   MAE day 1 (norm)         : {chunk_maes_norm[0]:.6f}")
        print(f"   MAE day {n_chunks} (norm): {chunk_maes_norm[-1]:.6f}")

        metrics.update({
            # summary
            "mae_std_per_day_price": round(mae_std_price, 4),
            "mae_day1_price": round(chunk_maes_price[0], 4),
            "mae_last_day_price": round(chunk_maes_price[-1], 4),

            "mae_std_per_day_norm": round(mae_std_norm, 6),
            "mae_day1_norm": round(chunk_maes_norm[0], 6),
            "mae_last_day_norm": round(chunk_maes_norm[-1], 6),

            # full daily series
            "daily_mae_price": [
                round(x, 4) for x in chunk_maes_price
            ],

            "daily_mae_norm": [
                round(x, 6) for x in chunk_maes_norm
            ],
        })

        # ------------------------------------------------------------------
        # Daily MAE (non-zero actual-price hours only)
        # ------------------------------------------------------------------
        chunk_maes_price_nonzero = []
        chunk_maes_norm_nonzero = []

        for i in range(n_chunks):

            start = i * chunk
            end   = (i + 1) * chunk

            day_true = test_inv_vals[start:end]
            day_mask = day_true != 0

            if day_mask.any():

                day_res_price = residuals_inv[start:end][day_mask]
                day_res_norm  = residuals_norm[start:end][day_mask]

                chunk_maes_price_nonzero.append(
                    float(np.abs(day_res_price).mean())
                )

                chunk_maes_norm_nonzero.append(
                    float(np.abs(day_res_norm).mean())
                )

            else:

                chunk_maes_price_nonzero.append(np.nan)
                chunk_maes_norm_nonzero.append(np.nan)

        valid_price = [
            x for x in chunk_maes_price_nonzero
            if not np.isnan(x)
        ]

        valid_norm = [
            x for x in chunk_maes_norm_nonzero
            if not np.isnan(x)
        ]

        mae_std_price_nonzero = (
            float(np.std(valid_price))
            if valid_price else None
        )

        mae_std_norm_nonzero = (
            float(np.std(valid_norm))
            if valid_norm else None
        )

        print("\n   Daily MAE (non-zero hours only)")

        print(
            f"   Price: "
            f"{[round(x,2) if not np.isnan(x) else None for x in chunk_maes_price_nonzero]}"
        )

        print(
            f"   Norm : "
            f"{[round(x,4) if not np.isnan(x) else None for x in chunk_maes_norm_nonzero]}"
        )

        if mae_std_price_nonzero is not None:
            print(
                f"   MAE std/day (price, non-zero): "
                f"{mae_std_price_nonzero:.4f}"
            )

        if mae_std_norm_nonzero is not None:
            print(
                f"   MAE std/day (norm, non-zero) : "
                f"{mae_std_norm_nonzero:.6f}"
            )

        metrics.update({

            "mae_std_per_day_price_nonzero":
                None if mae_std_price_nonzero is None
                else round(mae_std_price_nonzero, 4),

            "mae_std_per_day_norm_nonzero":
                None if mae_std_norm_nonzero is None
                else round(mae_std_norm_nonzero, 6),

            "mae_day1_price_nonzero":
                None if np.isnan(chunk_maes_price_nonzero[0])
                else round(chunk_maes_price_nonzero[0], 4),

            "mae_last_day_price_nonzero":
                None if np.isnan(chunk_maes_price_nonzero[-1])
                else round(chunk_maes_price_nonzero[-1], 4),

            "mae_day1_norm_nonzero":
                None if np.isnan(chunk_maes_norm_nonzero[0])
                else round(chunk_maes_norm_nonzero[0], 6),

            "mae_last_day_norm_nonzero":
                None if np.isnan(chunk_maes_norm_nonzero[-1])
                else round(chunk_maes_norm_nonzero[-1], 6),

            "daily_mae_price_nonzero": [
                None if np.isnan(x)
                else round(x, 4)
                for x in chunk_maes_price_nonzero
            ],

            "daily_mae_norm_nonzero": [
                None if np.isnan(x)
                else round(x, 6)
                for x in chunk_maes_norm_nonzero
            ],
        })

    else:
        print(
            f"\n   Not enough data for daily diagnostics "
            f"({n} hours < {chunk} hours)."
        )

    nonzero_mask = test_inv_vals != 0
    n_nonzero    = int(nonzero_mask.sum())
    if n_nonzero > 0:
        pred_nz  = pred_inv_vals[nonzero_mask]
        true_nz  = test_inv_vals[nonzero_mask]
        denom_nz = (np.abs(true_nz) + np.abs(pred_nz)) / 2
        res_nz_price = residuals_inv[nonzero_mask]
        res_nz_norm  = residuals_norm[nonzero_mask]

        # Price units
        rmse_nz_price = float(np.sqrt((res_nz_price ** 2).mean()))
        mae_nz_price  = float(np.abs(res_nz_price).mean())

        # Normalised units
        rmse_nz_norm = float(np.sqrt((res_nz_norm ** 2).mean()))
        mae_nz_norm  = float(np.abs(res_nz_norm).mean())
    
        smape_nz = float(
            np.mean(
                np.where(
                    denom_nz == 0,
                    0,
                    np.abs(res_nz_price) / denom_nz
                )
            ) * 100
        )

        print(f"\n   Metrics on non-zero price hours only ({n_nonzero}/{n} steps):")
        print(f"   SMAPE           : {smape_nz:.2f}%")
        print(f"   RMSE (norm)     : {rmse_nz_norm:.6f}")
        print(f"   MAE  (norm)     : {mae_nz_norm:.6f}")
        print(f"   RMSE (price)    : {rmse_nz_price:.4f}")
        print(f"   MAE  (price)    : {mae_nz_price:.4f}")


        metrics.update({
        "n_nonzero_steps": n_nonzero,

        "smape_nonzero": round(smape_nz, 4),

        "rmse_norm_nonzero": round(rmse_nz_norm, 6),
        "mae_norm_nonzero": round(mae_nz_norm, 6),

        "rmse_price_nonzero": round(rmse_nz_price, 4),
        "mae_price_nonzero": round(mae_nz_price, 4),
    })
    else:
        print("   ℹ️  All test hours have price == 0, skipping non-zero filter.")

    # Single save — all fields guaranteed to be present
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📄 Metrics saved → {metrics_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_results(
        model_name  = model_name,
        series      = scaler.inverse_transform(series),
        test_inv    = test_inv,
        pred_inv    = pred_inv,
        train       = scaler.inverse_transform(train),
        val         = scaler.inverse_transform(val),
        smape_val   = smape_val,
        rmse_norm   = rmse_norm,
        mae_norm    = mae_norm,
        rmse_inv    = rmse_inv,
        mae_inv     = mae_inv,
        target_col  = target_col,
        result_path = result_path,
    )

    return pred_inv


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_results(
    model_name:  str,
    series:      TimeSeries,   # full series, price units
    test_inv:    TimeSeries,   # test set, price units
    pred_inv:    TimeSeries,   # prediction, price units
    train:       TimeSeries,   # train split, price units
    val:         TimeSeries,   # val split, price units
    smape_val:   float,
    rmse_norm:   float,
    mae_norm:    float,
    rmse_inv:    float,
    mae_inv:     float,
    target_col:  str,
    result_path: str,
) -> None:
    """
    4-panel diagnostic figure:
      1. Full series with train / val / test split highlighted.
      2. Test set — predicted vs actual (price units).
      3. Residuals over time with zero line.
      4. Residual distribution histogram.
    """
    test_vals = _ts_to_array(test_inv)
    pred_vals = _ts_to_array(pred_inv)
    n = min(len(test_vals), len(pred_vals))
    test_vals, pred_vals = test_vals[:n], pred_vals[:n]
    residuals = pred_vals - test_vals
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"{model_name}  –  '{target_col}' Backtest\n"
        f"SMAPE {smape_val:.2f}%   "
        f"RMSE {rmse_norm:.4f} (norm) / {rmse_inv:.4f} (price)   "
        f"MAE {mae_norm:.4f} (norm) / {mae_inv:.4f} (price)",
        fontsize=11, fontweight="bold",
    )

    # Panel 1: Full series with split regions highlighted
    ax1 = axes[0, 0]
    train_vals = _ts_to_array(train)
    val_vals   = _ts_to_array(val)
    full_vals  = np.concatenate([train_vals, val_vals, test_vals])
    n_train    = len(train_vals)
    n_val      = len(val_vals)
    n_full     = len(full_vals)
    x_full     = np.arange(n_full)

    ax1.plot(x_full, full_vals, color="steelblue", linewidth=0.8, label=target_col)
    ax1.axvspan(0,                   n_train,          alpha=0.10, color="steelblue", label="Train")
    ax1.axvspan(n_train,             n_train + n_val,  alpha=0.15, color="orange",    label="Val")
    ax1.axvspan(n_train + n_val,     n_full,           alpha=0.15, color="tomato",    label="Test")
    ax1.set_title("Full Series — Train / Val / Test Split")
    ax1.legend(fontsize=8)
    _apply_time_ticks(ax1, series.time_index, n_full, max_labels=8)

    # Panel 2: Predicted vs actual on test set
    ax2 = axes[0, 1]
    ax2.plot(x, test_vals, label="Actual",    color="steelblue",  linewidth=1.2)
    ax2.plot(x, pred_vals, label="Predicted", color="darkorange", linewidth=1.2, linestyle="--")
    ax2.set_title(f"Test Set — Predicted vs Actual ({n} steps)")
    ax2.legend(fontsize=8)
    _apply_time_ticks(ax2, test_inv.time_index, n)

    # Panel 3: Residuals over time
    ax3 = axes[1, 0]
    ax3.plot(x, residuals, color="tomato", linewidth=1)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.fill_between(x, residuals, 0, alpha=0.2, color="tomato")
    ax3.set_title("Residuals Over Time (predicted − actual)")
    ax3.set_ylabel(f"Residual ({target_col})")
    _apply_time_ticks(ax3, test_inv.time_index, n)

    # Panel 4: Residual distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax4.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax4.axvline(
        float(residuals.mean()), color="tomato", linewidth=1.2, linestyle="--",
        label=f"Mean {float(residuals.mean()):.3f}",
    )
    ax4.axvline(
        float(np.median(residuals)), color="orange", linewidth=1.2, linestyle=":",
        label=f"Median {float(np.median(residuals)):.3f}",
    )
    ax4.set_title("Residual Distribution")
    ax4.set_xlabel(f"Residual ({target_col})")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(result_path, f"{model_name}_backtest.png")
    plt.savefig(plot_path, dpi=150)
    print(f"💾 Plot saved → {plot_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import importlib

    parser = argparse.ArgumentParser(
        description="Train and evaluate a Darts model on the train/val/test split."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML config (shared with train.py and inference.py).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_cls_name = cfg["model"]["model_cls"]
    darts_models   = importlib.import_module("darts.models")
    MODEL_CLS      = getattr(darts_models, model_cls_name)

    print(f"\n🧠 Model : {model_cls_name}")

    MODEL_KWARGS: dict[str, Any] = {}
    if is_torch_model(MODEL_CLS):
        print("⚙️  Neural network detected — building kwargs from YAML…")
        MODEL_KWARGS = build_torch_kwargs(cfg)
    else:
        for key in ("seasonality_mode", "yearly_seasonality",
                    "weekly_seasonality", "daily_seasonality"):
            if key in cfg.get("model", {}):
                MODEL_KWARGS[key] = cfg["model"][key]

    print("📂 Loading data…")
    series, scaler, past_cov, future_cov = load_all_series(cfg)

    series = series.astype(np.float32)

    print(
        f"✅ Series: {len(series)} steps"
        + (f"  |  past_cov: {past_cov.n_components} component(s)"   if past_cov   is not None else "")
        + (f"  |  future_cov: {future_cov.n_components} component(s)" if future_cov is not None else "")
    )

    run_backtest(
        cfg               = cfg,
        model_cls         = MODEL_CLS,
        series            = series,
        scaler            = scaler,
        model_kwargs      = MODEL_KWARGS,
        past_covariates   = past_cov,
        future_covariates = future_cov,
    )