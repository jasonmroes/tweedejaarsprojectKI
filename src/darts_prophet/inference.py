"""
inference.py
------------
Load a checkpoint produced by train.py and forecast a window in the future.

The forecast window is defined by two parameters:

    --horizon N    Number of steps to predict.
                   Defaults to model.output_chunk_length from the config.

    --offset N     Gap in steps between the END of the training data and the
                   START of the desired forecast window.
                   Default 0 — forecast begins immediately after the last
                   training point.

                   Example:
                     Training data ends at step 8760 (end of year).
                     --offset 24  → forecast starts 24 h into the future,
                                    i.e. at step 8784.
                     --offset 0   → forecast starts at step 8761.

How it works
------------
Darts' predict(n) always produces the next n consecutive steps after the
context series.  To introduce a gap we predict (offset + horizon) steps
total, then discard the first `offset` steps, keeping only the target window.

    |─────────── training data ──────────────|  offset  |──── horizon ────|
                                             ^          ^                  ^
                                         data end   window start       window end

The context series passed to predict() is always the full normalised series
(same data used during training).  This gives the model maximum history to
condition on.

Covariate note
--------------
If the model uses future covariates AND offset > 0, the covariate series
must extend at least (offset + horizon) steps beyond the end of the training
data.  The month sin/cos covariate is extended automatically.  CSV-backed
future covariates must be supplied by the user at sufficient length.

Output
------
    <save_dir>/<ModelName>_inference_offset<N>.csv

Usage
-----
    python inference.py --config configs/no_past.yaml --horizon 336
    python inference.py --config configs/no_past.yaml --horizon 336 --offset 24
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import yaml
import torch
torch.set_float32_matmul_precision("high")

from model_utils import (
    build_predict_kwargs,
    load_all_series,
    extend_future_covariates,
)


def run_inference(config_path: str, horizon: int | None, offset: int) -> None:
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    data_cfg     = cfg["data"]

    model_cls_name = model_cfg["model_cls"]


    ckpt_path = model_cfg.get("full_checkpoint_filepath") or None
    save_dir       = training_cfg["save_dir"]
    if not ckpt_path:
        ckpt_path = os.path.join(save_dir, f"{model_cls_name}_checkpoint_future.pt")
        print(f"ℹ️  full_checkpoint_filepath not set — using default: {ckpt_path}")


    if horizon is None:
        horizon = model_cfg["output_chunk_length"]

    total_steps = offset + horizon

    # ── Load model ────────────────────────────────────────────────────────────
    import importlib
    model_cls = getattr(importlib.import_module("darts.models"), model_cls_name)
    model = model_cls.load_from_checkpoint(
        model_name=model_cls_name,
        work_dir=save_dir,
        best=True,
    )

    print(f"✅ Loaded {model_cls_name} from {ckpt_path}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("📂 Loading data…")
    series, _, past_cov, future_cov = load_all_series(cfg)
    # inference.py — replace the scaler from load_all_series
    import pickle
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"No scaler found at {scaler_path}. Run train.py first."
        )
    series = series.astype(np.float32)

    # ── Validate covariate length ─────────────────────────────────────────────
    # Future covariates must cover the prediction window beyond the series end.
    # The month covariate is extended automatically; CSV covariates are checked.
    if future_cov is not None and offset > 0:
        csv_future_cols = data_cfg.get("future_covariates", []) or []
        if csv_future_cols:
            # CSV-backed covariates have the same length as the training series.
            # They cannot cover an offset gap automatically.
            available = len(future_cov) - len(series)
            if available < total_steps:
                raise ValueError(
                    f"future_covariates only extends {available} step(s) beyond "
                    f"the training data, but offset + horizon = {total_steps}. "
                    f"Provide a future_data_path CSV covering at least "
                    f"{total_steps} extra steps beyond the training data."
                )

        # Extend month covariate columns automatically if needed
        future_cov = extend_future_covariates(future_cov, cfg, extra_steps=total_steps)

    # ── Context window ────────────────────────────────────────────────────────
    # Always use the full training series as context so the model has the
    # maximum available history when conditioning its prediction.
    context = series

    min_context = model_cfg.get("input_chunk_length", 1)
    if len(context) < min_context:
        raise ValueError(
            f"Context length {len(context)} < input_chunk_length {min_context}."
        )

    print(
        f"\n📍 Forecast window:"
        f"\n   Data ends at step : {len(series)}"
        f"\n   Offset (gap)       : {offset} step(s)"
        f"\n   Horizon            : {horizon} step(s)"
        f"\n   Predicting steps   : {len(series) + offset + 1}"
        f" → {len(series) + offset + horizon}"
    )

    # ── Predict ───────────────────────────────────────────────────────────────
    # Predict total_steps = offset + horizon, then slice off the gap.
    pred_kw = build_predict_kwargs(
        model, total_steps, context,
        past_cov, future_cov,
    )
    full_pred = model.predict(**pred_kw)
    full_pred = scaler.inverse_transform(full_pred)

    # Discard the gap; keep only the target window
    pred = full_pred[offset:] if offset > 0 else full_pred

    print(f"\n✅ Forecast produced: {len(pred)} step(s)")

    # ── Export ────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    csv_name = f"{model_cls_name}_inference_offset{offset}_horizon{horizon}.csv"
    csv_path = os.path.join(save_dir, csv_name)
    pred.to_dataframe().to_csv(csv_path)
    print(f"💾 Forecast saved → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a checkpoint produced by train.py."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML config (shared with train.py).",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help=(
            "Number of steps to forecast. "
            "Defaults to model.output_chunk_length from the config."
        ),
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help=(
            "Gap in steps between the end of the training data and the start "
            "of the forecast window.  0 = forecast begins immediately after "
            "the last training point.  E.g. --offset 24 starts the forecast "
            "24 hours after the training data ends."
        ),
    )
    args = parser.parse_args()
    run_inference(args.config, args.horizon, args.offset)
