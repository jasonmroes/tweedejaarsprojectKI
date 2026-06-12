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
    #   from darts.models import LightGBMModel, NBEATSModel, TFTModel, NHiTSModel, 
    # All accepted models are as follows:
    # NaiveSeasonal,NaiveDrift,NaiveMovingAverage,NaiveEnsembleModel,ExponentialSmoothing,
    # StatsForecastModel,TBATS,Theta,FourTheta,FFT, AutoARIMA,Prophet,KalmanForecaster,RegressionModel,
    # RandomForest,LightGBMModel,XGBModel,CatBoostModel,Croston,LinearRegressionModel,
    # And NN Models:
    # RNNModel, BlockRNNModel, NBEATSModel, NHiTSModel, TCNModel, TransformerModel,TFTModel,TiDEModel,
    from darts.models import TiDEModel

    MODEL_CLS = TiDEModel

    # ── Build kwargs from YAML ─────────────────────────────────────────────
    # Start with any model-specific static kwargs you want to hard-code
    # (these take priority over YAML values when keys clash).
    MODEL_KWARGS: dict[str, Any] = {
        # Prophet-specific – ignored automatically for other models
        "seasonality_mode":   cfg.get("model", {}).get("seasonality_mode",   "multiplicative"),
        "yearly_seasonality": cfg.get("model", {}).get("yearly_seasonality", True),
        "weekly_seasonality": cfg.get("model", {}).get("weekly_seasonality", True),
        "daily_seasonality":  cfg.get("model", {}).get("daily_seasonality",  True),
    }

    # If the chosen model is a neural network, merge the full torch kwargs
    # derived from the model + training blocks in the YAML.
    if single_model_test._is_torch_model(MODEL_CLS):
        print("🧠 Neural network model detected – loading training config from YAML…")
        torch_kwargs = single_model_test._build_torch_kwargs(cfg)
        # YAML-derived values are the base; any explicit MODEL_KWARGS above
        # override them so hand-tuned values always win.
        MODEL_KWARGS = {**torch_kwargs, **MODEL_KWARGS}

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