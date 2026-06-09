from __future__ import annotations
import single_model_test


import yaml
from typing import Any

from data_processing import load_timeseries
# ── CONFIG ─────────────────────────────────────────────────────────────────────
CONFIG_PATH      = "configs/no_past_config.yaml"
RESULT_PATH      = "experiments/"
TARGET_COL       = "price"

with open(CONFIG_PATH) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Covariate column groups
PAST_COV_COLS: list[str]    = cfg['data']['past_covariates']       # e.g. ["consumption", "temp_avg"]
FUTURE_COV_COLS: list[str]  = cfg['data']['future_covariates']         # e.g. ["national holiday", "walk-in"]
FUTURE_CSV_PATH             = cfg['data']['future_data_path']
SAVE_PATH                   = cfg['training']['save_dir']

# ───────────────────────────────────────────────────────────────────────────────

# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Choose your model ──────────────────────────────────────────────────
    # Swap this import + class for any other Darts model, e.g.:
    #   from darts.models import LightGBMModel, NBEATSModel, TFTModel, ...
    from darts.models import CatBoostModel

    MODEL_CLS = CatBoostModel

    # ── Model-specific kwargs ──────────────────────────────────────────────
    # Unknown kwargs are silently dropped, so it is safe to keep a superset
    # here and switch MODEL_CLS without editing anything else.
    MODEL_KWARGS: dict[str, Any] = {
        # Prophet
        "seasonality_mode":   "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality":  True,
        # LightGBM / regression models
        "lags":                24,
        # NBEATS / deep learning
        "input_chunk_length":  48,
        # TFT
        "hidden_size":         64,
    }

    # ── Load target series ─────────────────────────────────────────────────
    series = load_timeseries(FUTURE_CSV_PATH, target_cols=[TARGET_COL])

    # ── Optionally load covariates ─────────────────────────────────────────
    past_cov   = None
    future_cov = None

    if PAST_COV_COLS:
        past_cov = load_timeseries(FUTURE_CSV_PATH, target_cols=PAST_COV_COLS)
        print(f"📎 Past covariates loaded: {PAST_COV_COLS}")

    if FUTURE_COV_COLS:
        future_cov = load_timeseries(FUTURE_CSV_PATH, target_cols=FUTURE_COV_COLS)
        print(f"📎 Future covariates loaded: {FUTURE_COV_COLS}")

    # ── Run ────────────────────────────────────────────────────────────────
    _, future = single_model_test.run_forecast(
        model_cls         = MODEL_CLS,
        series            = series,
        model_kwargs      = MODEL_KWARGS,
        past_covariates   = past_cov,
        future_covariates = future_cov,
        ckpt_backtest     = "",
        ckpt_future       = f"{SAVE_PATH}/checkpoint_future.pt",
    )