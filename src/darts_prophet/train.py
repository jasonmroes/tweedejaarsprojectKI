"""
train.py
--------
Trains a Darts forecasting model for future inference.

Reads all configuration from a single YAML file — the same config used by
single_model_test.py and inference.py.

Split layout (no test holdout)
------------------------------
    |──────────── train ────────────|──── val ────|
    ^                                              ^
    start of data                         end of data

The val window provides a genuine early-stopping signal over the tail of
the data — the region most relevant to future forecasting.

Output
------
    <model.full_checkpoint_filepath>   — loadable by inference.py

Usage
-----
    python train.py --config configs/no_past.yaml
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any

import numpy as np
import yaml
import torch
# Use nvidia gpu
torch.set_float32_matmul_precision('high') 

from data_processing import normalize_series
from model_utils import (
    filter_kwargs,
    is_torch_model,
    build_torch_kwargs,
    build_fit_kwargs,
    build_predict_kwargs,
    supports_covariates,
    load_all_series,
)


def train(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_cfg     = cfg["data"]
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]

    target_col      = data_cfg.get("target_col", "price")
    val_fraction    = data_cfg.get("validation_fraction", 0.1)
    save_dir        = training_cfg["save_dir"]
    model_cls_name  = model_cfg["model_cls"]
    ckpt_path = model_cfg.get("full_checkpoint_filepath") or None

    if not ckpt_path:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"{model_cls_name}_checkpoint_future.pt")
        print(f"ℹ️  full_checkpoint_filepath not set — saving to {ckpt_path}")

    os.makedirs(save_dir, exist_ok=True)

    # ── Resolve model class ───────────────────────────────────────────────────
    import importlib
    model_cls = getattr(importlib.import_module("darts.models"), model_cls_name)
    print(f"\n🧠 Model : {model_cls_name}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("📂 Loading data…")
    import pickle
    series, scaler, past_cov, future_cov = load_all_series(cfg)
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    series = series.astype(np.float32)

    print(
        f"✅ Series: {len(series)} steps"
        + (f"  |  past_cov: {past_cov.n_components} component(s)"
           if past_cov  is not None else "")
        + (f"  |  future_cov: {future_cov.n_components} component(s)"
           if future_cov is not None else "")
    )

    # ── Train / val split (no test) ───────────────────────────────────────────
    val_size    = int(len(series) * val_fraction)
    train_ser   = series[:-val_size]
    val_ser     = series[-val_size:]

    print(
        f"\n📊 Split  —  Train: {len(train_ser)}  |  Val: {len(val_ser)}"
        f"  (val_fraction={val_fraction})"
    )

    # Slice covariates
    past_cov_train = past_cov_val = None
    fut_cov_train  = fut_cov_val  = None

    if past_cov is not None:
        past_cov_train = past_cov.slice_intersect(train_ser)
        past_cov_val   = past_cov.slice_intersect(val_ser)

    if future_cov is not None:
        fut_cov_train = future_cov.slice_intersect(train_ser)
        fut_cov_val   = future_cov.slice_intersect(val_ser)

    # ── Build model kwargs ────────────────────────────────────────────────────
    model_kwargs: dict[str, Any] = {}
    if is_torch_model(model_cls):
        print("⚙️  Neural network detected — building kwargs from YAML…")
        model_kwargs = build_torch_kwargs(cfg)
    else:
        for key in ("seasonality_mode", "yearly_seasonality",
                    "weekly_seasonality", "daily_seasonality"):
            if key in model_cfg:
                model_kwargs[key] = model_cfg[key]

    safe_kwargs = filter_kwargs(model_cls, model_kwargs)
    print(f"🔧 Constructor kwargs: {safe_kwargs}\n")

    # ── Fit ───────────────────────────────────────────────────────────────────
    # ── Resume from checkpoint if one is provided ─────────────────────────
    resume_ckpt = model_cfg.get("resume")

    if resume_ckpt:
        print(f"⏩ Resuming from checkpoint: {save_dir}")
        # load_from_checkpoint() reconstructs the model AND restores the
        # trainer state (epoch count, optimiser state, LR scheduler state).
        model = model_cls.load_from_checkpoint(
            model_name=model_cls_name,
            work_dir=save_dir,
            best=False,
        )
    else:
        model = model_cls(**safe_kwargs)

    fit_kw = build_fit_kwargs(
        model,
        train_ser, val_ser,
        past_cov_train, fut_cov_train,
        past_cov_val,   fut_cov_val,
    )

    print(f"⏳ Training {model_cls_name}…")
    model.fit(**fit_kw)
    print("✅ Training complete.")

    # ── Val-set metrics (price units) ─────────────────────────────────────────
    print("\n📐 Computing val-set metrics…")
    try:
        n_eval = model_cfg.get("output_chunk_length", 336)

        val_pred_kw = build_predict_kwargs(
            model, n_eval, train_ser,
            past_cov, future_cov,   # full covariates, not train-sliced
        )
        val_pred     = model.predict(**val_pred_kw)
        val_pred_inv = scaler.inverse_transform(val_pred)
        # Compare against the first n_eval steps of the val set
        val_ser_inv  = scaler.inverse_transform(val_ser[:n_eval])

        pred_vals = val_pred_inv.values()[:, 0]
        true_vals = val_ser_inv.values()[:len(pred_vals), 0]
        residuals = pred_vals - true_vals

        mae_val   = float(np.abs(residuals).mean())
        rmse_val  = float(np.sqrt((residuals ** 2).mean()))
        denom     = (np.abs(true_vals) + np.abs(pred_vals)) / 2
        smape_val = float(
            np.mean(np.where(denom == 0, 0, np.abs(residuals) / denom)) * 100
        )

        print(f"\n   SMAPE : {smape_val:.2f}%")
        print(f"   RMSE  : {rmse_val:.4f}")
        print(f"   MAE   : {mae_val:.4f}")

        metrics = {
            "smape": round(smape_val, 4),
            "rmse":  round(rmse_val,  4),
            "mae":   round(mae_val,   4),
        }

        import json
        metrics_path = os.path.join(save_dir, f"{model_cls_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📄 Metrics saved → {metrics_path}")
    except Exception as exc:
        print(f"⚠️  Could not compute val metrics: {exc}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(ckpt_path)
    print(f"\n💾 Checkpoint saved → {ckpt_path}")
    print(f"\n▶  Run inference:")
    print(f"   python inference.py --config {config_path} --horizon 336")
    print(f"   python inference.py --config {config_path} --horizon 336 --offset 24")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Darts model for future inference.")
    parser.add_argument("--config", required=True, nargs='+',
                        help="Path(s) to YAML config file(s). Pass multiple to train in sequence.")
    args = parser.parse_args()

    for config_path in args.config:
        print(f"\n{'='*60}")
        print(f"▶  Training with config: {config_path}")
        print(f"{'='*60}\n")
        train(config_path)
