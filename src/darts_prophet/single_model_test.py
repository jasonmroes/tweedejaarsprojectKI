"""
Darts – Generic Model Forecast
================================
Trains and evaluates any Darts forecasting model on a price time series.

Key features
------------
* Works with any Darts model class.
* Automatically filters constructor kwargs to only those the model accepts.
* Passes past_covariates / future_covariates only when the model supports them.
* Produces a 4-panel diagnostic figure and exports forecasts to CSV.

Requirements
------------
    pip install darts

Usage
-----
    python darts_forecast.py

    Or import and call run_forecast() directly:

        from darts_forecast import run_forecast
        from darts.models import LightGBMModel
        run_forecast(
            model_cls=LightGBMModel,
            series=my_series,
            model_kwargs={"lags": 24, "output_chunk_length": 24},
            past_covariates=my_past_cov,
        )
"""

from __future__ import annotations

import inspect
import os
import yaml
from typing import Any, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts.metrics import smape, rmse, mae
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries

from data_processing import load_timeseries

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CONFIG_PATH      = "configs/standard_config.yaml"
RESULT_PATH      = "experiments/"
TARGET_COL       = "price"
FORECAST_HORIZON = 24     # hours ahead for the future forecast
TEST_SIZE        = 48     # last N hours held out for evaluation

with open(CONFIG_PATH) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Covariate column groups
PAST_COV_COLS: list[str]    = cfg['data']['past_covariates']       # e.g. ["consumption", "temp_avg"]
FUTURE_COV_COLS: list[str]  = cfg['data']['future_covariates']         # e.g. ["national holiday", "walk-in"]
CSV_PATH                    = cfg['data']['data_path']

# ───────────────────────────────────────────────────────────────────────────────


# ── Helpers ────────────────────────────────────────────────────────────────────

def _accepted_params(model_cls: Type) -> set[str] | None:
    """
    Return the set of parameter names accepted by model_cls.__init__.
 
    Walks the MRO so that Darts wrapper classes (which often declare
    ``**kwargs`` and forward them to an underlying library) are handled
    correctly: we only return None ("allow everything") when the *concrete*
    class at the bottom of the chain genuinely accepts arbitrary kwargs,
    not just because an intermediate wrapper re-exports them.
 
    Returns None when inspection is impossible or the class truly accepts
    arbitrary keyword arguments.
    """
    collected: set[str] = set()
    has_var_keyword = False
 
    for cls in model_cls.__mro__:
        if cls is object:
            continue
        try:
            # Only look at __init__ defined *directly* on this class
            if "__init__" not in cls.__dict__:
                continue
            sig = inspect.signature(cls.__init__)
            for name, p in sig.parameters.items():
                if name == "self":
                    continue
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    has_var_keyword = True   # this level uses **kwargs
                elif p.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,  # *args – skip
                ):
                    collected.add(name)
        except (ValueError, TypeError):
            continue
 
    # If no **kwargs anywhere in the chain we have a complete picture
    if not has_var_keyword:
        return collected if collected else None
 
    # There IS a **kwargs somewhere. Check whether any class in the MRO
    # also defines a tighter __init__ that enumerates explicit params
    # (e.g. the underlying prophet.Prophet class beneath darts.Prophet).
    # If we collected named params from any level, prefer those.
    if collected:
        return collected
 
    # Truly open-ended – cannot filter safely via inspection alone.
    return None
 



def _filter_kwargs(model_cls: Type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Return only the kwargs that model_cls.__init__ actually accepts.
 
    Strategy
    --------
    1. Try static inspection via _accepted_params.
    2. If inspection says "allow all" (None), fall back to trial-and-
       elimination: repeatedly attempt to instantiate the model, catch
       TypeError for unexpected keyword arguments, strip those keys, and
       retry until the constructor succeeds or we run out of kwargs.
    """
    accepted = _accepted_params(model_cls)
 
    if accepted is not None:
        # Static inspection gave us a definitive list – use it.
        filtered  = {k: v for k, v in kwargs.items() if k in accepted}
        discarded = {k for k in kwargs if k not in accepted}
        if discarded:
            print(f"⚠️  Ignored unsupported kwargs for {model_cls.__name__}: {discarded}")
        return filtered
 
    # ── Fallback: trial-and-elimination ──────────────────────────────────────
    # The model uses **kwargs internally (e.g. darts.Prophet forwards to
    # prophet.Prophet), so static inspection cannot tell us what is valid.
    # We probe by attempting instantiation and stripping offending keys.
    candidate = dict(kwargs)
    discarded: set[str] = set()
 
    while True:
        try:
            model_cls(**candidate)   # probe only – result discarded
            break                    # success: candidate is safe
        except TypeError as exc:
            msg = str(exc)
            # CPython error format: "got an unexpected keyword argument 'foo'"
            import re
            match = re.search(r"unexpected keyword argument '(\w+)'", msg)
            if match:
                bad_key = match.group(1)
                discarded.add(bad_key)
                candidate.pop(bad_key, None)
            else:
                # Can't parse the error – stop stripping to avoid infinite loop
                break
 
    if discarded:
        print(f"⚠️  Ignored unsupported kwargs for {model_cls.__name__}: {discarded}")
    return candidate

def _is_torch_model(model_cls: Type) -> bool:
    """Return True if model_cls is a Darts TorchForecastingModel subclass."""
    try:
        from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
        return issubclass(model_cls, TorchForecastingModel)
    except ImportError:
        return False
 
 
def _build_torch_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Translate the YAML ``model`` and ``training`` blocks into Darts
    TorchForecastingModel constructor kwargs.
 
    YAML structure expected (all keys optional):
 
        model:
          input_chunk_length:  96
          output_chunk_length: 24
          hidden_dim:          256      # or hidden_size – both tried
          num_layers:          2
          num_attention_heads: 4        # Transformer / TFT
          dropout:             0.1      # maps from dropout_rate too
          activation:          "relu"
          use_reversible_instance_norm: false
 
        training:
          epochs:        50
          batch_size:    32             # also read from data.batch_size
          learning_rate: 1e-4
          weight_decay:  0.01
          scheduler:
            step_size: 5
            gamma:     0.1
 
    The scheduler block, when present, is wired up as a
    ``torch.optim.lr_scheduler.StepLR`` passed via ``lr_scheduler_cls``
    and ``lr_scheduler_kwargs``.  Swap the class name in the YAML to use
    a different scheduler (see ``_SCHEDULER_MAP`` below).
    """
    import torch
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_sched
 
    # ── Friendly name → class mappings ───────────────────────────────────────
    _OPTIMIZER_MAP: dict[str, type] = {
        "adam":     optim.Adam,
        "adamw":    optim.AdamW,
        "sgd":      optim.SGD,
        "rmsprop":  optim.RMSprop,
    }
    _SCHEDULER_MAP: dict[str, type] = {
        "steplr":        lr_sched.StepLR,
        "multisteplr":   lr_sched.MultiStepLR,
        "cosineannealinglr": lr_sched.CosineAnnealingLR,
        "reducelronplateau": lr_sched.ReduceLROnPlateau,
        "exponentiallr": lr_sched.ExponentialLR,
    }
 
    model_cfg    = cfg.get("model",    {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    data_cfg     = cfg.get("data",     {}) or {}
    sched_cfg    = training_cfg.get("scheduler", {}) or {}
 
    kwargs: dict[str, Any] = {}
 
    # ── Architecture ─────────────────────────────────────────────────────────
    # Chunk lengths – mandatory for all TorchForecastingModels
    for key in ("input_chunk_length", "output_chunk_length"):
        if key in model_cfg:
            kwargs[key] = int(model_cfg[key])
 
    # Hidden size: YAML may say hidden_dim (your convention) or hidden_size
    # (Darts convention).  We write both so _filter_kwargs keeps whichever
    # the target model actually accepts.
    hidden = model_cfg.get("hidden_dim") or model_cfg.get("hidden_size")
    if hidden is not None:
        kwargs["hidden_dim"]  = int(hidden)
        kwargs["hidden_size"] = int(hidden)   # alias for e.g. TFT
 
    if "num_layers" in model_cfg:
        kwargs["num_layers"] = int(model_cfg["num_layers"])
 
    if "num_attention_heads" in model_cfg:
        kwargs["num_attention_heads"] = int(model_cfg["num_attention_heads"])
 
    # Dropout: prefer explicit "dropout" key
    dropout = model_cfg.get("dropout")
    if dropout is not None:
        kwargs["dropout"] = float(dropout)
 
    if "activation" in model_cfg:
        kwargs["activation"] = str(model_cfg["activation"])
 
    if "use_reversible_instance_norm" in model_cfg:
        kwargs["use_reversible_instance_norm"] = bool(
            model_cfg["use_reversible_instance_norm"]
        )
 
    # ── Training loop ─────────────────────────────────────────────────────────
    if "epochs" in training_cfg:
        kwargs["n_epochs"] = int(training_cfg["epochs"])
 
    # batch_size: training block takes priority, then data block
    batch_size = training_cfg.get("batch_size")
    if batch_size is not None:
        kwargs["batch_size"] = int(batch_size)
 
    # Optimizer
    lr           = training_cfg.get("learning_rate")
    weight_decay = training_cfg.get("weight_decay")
    opt_name     = str(training_cfg.get("optimizer", "adamw")).lower()
    opt_cls      = _OPTIMIZER_MAP.get(opt_name, optim.AdamW)
 
    opt_kwargs: dict[str, Any] = {}
    if lr           is not None: opt_kwargs["lr"]           = float(lr)
    if weight_decay is not None: opt_kwargs["weight_decay"] = float(weight_decay)
 
    kwargs["optimizer_cls"]    = opt_cls
    kwargs["optimizer_kwargs"] = opt_kwargs
 
    # ── LR Scheduler ─────────────────────────────────────────────────────────
    if sched_cfg:
        sched_name = str(sched_cfg.get("type", "steplr")).lower()
        sched_cls  = _SCHEDULER_MAP.get(sched_name, lr_sched.StepLR)
 
        # Pass all remaining keys from the scheduler block as kwargs,
        # excluding the "type" meta-key itself.
        sched_kwargs = {
            k: v for k, v in sched_cfg.items() if k != "type"
        }
 
        kwargs["lr_scheduler_cls"]    = sched_cls
        kwargs["lr_scheduler_kwargs"] = sched_kwargs
 
    # ── Reproducibility ───────────────────────────────────────────────────────
    if "seed" in cfg:
        kwargs["random_state"] = int(cfg["seed"])
 
    return kwargs
 


def _supports_covariates(model_instance, cov_type: str) -> bool:
    """
    Check whether a fitted/unfitted model instance supports a covariate type.
    cov_type: 'past' | 'future'
    """
    attr = f"supports_{cov_type}_covariates"
    # Some models expose this as a property, others as a class attribute
    val = getattr(model_instance, attr, False)
    return bool(val)


def _build_fit_kwargs(
    model,
    train: TimeSeries,
    past_covariates:   TimeSeries | None,
    future_covariates: TimeSeries | None,
) -> dict[str, Any]:
    """Build the kwargs dict for model.fit(), respecting covariate support."""
    fit_sig  = set(inspect.signature(model.fit).parameters.keys())
    kwargs: dict[str, Any] = {"series": train}

    if past_covariates is not None and _supports_covariates(model, "past"):
        if "past_covariates" in fit_sig:
            kwargs["past_covariates"] = past_covariates
    elif past_covariates is not None:
        print(f"⚠️  {type(model).__name__} does not support past_covariates – skipping.")

    if future_covariates is not None and _supports_covariates(model, "future"):
        if "future_covariates" in fit_sig:
            kwargs["future_covariates"] = future_covariates
    elif future_covariates is not None:
        print(f"⚠️  {type(model).__name__} does not support future_covariates – skipping.")

    return kwargs


def _build_predict_kwargs(
    model,
    n: int,
    past_covariates:   TimeSeries | None,
    future_covariates: TimeSeries | None,
) -> dict[str, Any]:
    """Build the kwargs dict for model.predict(), respecting covariate support."""
    pred_sig = set(inspect.signature(model.predict).parameters.keys())
    kwargs: dict[str, Any] = {"n": n}

    if past_covariates is not None and _supports_covariates(model, "past"):
        if "past_covariates" in pred_sig:
            kwargs["past_covariates"] = past_covariates

    if future_covariates is not None and _supports_covariates(model, "future"):
        if "future_covariates" in pred_sig:
            kwargs["future_covariates"] = future_covariates

    return kwargs


# ── Core ───────────────────────────────────────────────────────────────────────

def run_forecast(
    model_cls:         Type[ForecastingModel],
    series:            TimeSeries,
    model_kwargs:      dict[str, Any]   | None = None,
    past_covariates:   TimeSeries       | None = None,
    future_covariates: TimeSeries       | None = None,
    test_size:         int  = TEST_SIZE,
    forecast_horizon:  int  = FORECAST_HORIZON,
    result_path:       str  = RESULT_PATH,
    target_col:        str  = TARGET_COL,
) -> tuple[TimeSeries, TimeSeries]:
    """
    Train, evaluate, and forecast using any Darts ForecastingModel.

    Parameters
    ----------
    model_cls         : Any Darts model class (not an instance).
    series            : Target TimeSeries (full, including test period).
    model_kwargs      : Constructor kwargs for the model (unsupported ones are
                        silently dropped).
    past_covariates   : Optional past covariates TimeSeries.
    future_covariates : Optional future covariates TimeSeries.
    test_size         : Number of time steps held out for evaluation.
    forecast_horizon  : Number of steps to forecast beyond the dataset.
    result_path       : Directory where plots/CSVs are saved.
    target_col        : Label used in plot titles.

    Returns
    -------
    (pred, future_pred) – backtest and future forecast TimeSeries objects.
    """
    os.makedirs(result_path, exist_ok=True)
    model_name = model_cls.__name__
    model_kwargs = model_kwargs or {}

    # ── Train / test split ───────────────────────────────────────────────────
    train, test = series[:-test_size], series[-test_size:]
    print(f"\n📊 Train: {len(train)} steps  |  Test: {len(test)} steps")

    # ── Instantiate model (filtered kwargs) ──────────────────────────────────
    safe_kwargs = _filter_kwargs(model_cls, model_kwargs)
    print(f"🔧 Creating {model_name} with kwargs: {safe_kwargs}")
    model = model_cls(**safe_kwargs)

    # ── Fit ──────────────────────────────────────────────────────────────────
    fit_kwargs = _build_fit_kwargs(model, train, past_covariates, future_covariates)
    print(f"⏳ Fitting {model_name}…")
    model.fit(**fit_kwargs)

    # ── Backtest predict ─────────────────────────────────────────────────────
    print("🔮 Generating backtest forecast…")
    pred_kwargs = _build_predict_kwargs(
        model, len(test), past_covariates, future_covariates
    )
    pred = model.predict(**pred_kwargs)

    # ── Metrics ──────────────────────────────────────────────────────────────
    smape_val = smape(test, pred)
    rmse_val  = rmse(test, pred)
    mae_val   = mae(test, pred)
    print(f"\n📈 SMAPE : {smape_val:.2f}%")
    print(f"📉 RMSE  : {rmse_val:.4f}")
    print(f"📐 MAE   : {mae_val:.4f}")

    # ── Future forecast (refit on full series) ───────────────────────────────
    print(f"\n🔭 Fitting {model_name} on full series for future forecast…")
    future_model = model_cls(**safe_kwargs)
    future_fit_kw = _build_fit_kwargs(
        future_model, series, past_covariates, future_covariates
    )
    future_model.fit(**future_fit_kw)

    future_pred_kw = _build_predict_kwargs(
        future_model, forecast_horizon, past_covariates, future_covariates
    )
    future_pred = future_model.predict(**future_pred_kw)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_results(
        model_name=model_name,
        series=series,
        train=train,
        test=test,
        pred=pred,
        future_pred=future_pred,
        smape_val=smape_val,
        rmse_val=rmse_val,
        mae_val=mae_val,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
        result_path=result_path,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    csv_path = os.path.join(result_path, f"{model_name}_future_forecast.csv")
    future_pred.to_dataframe().to_csv(csv_path)
    print(f"💾 Future forecast saved → {csv_path}")

    return pred, future_pred


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_results(
    model_name:      str,
    series:          TimeSeries,
    train:           TimeSeries,
    test:            TimeSeries,
    pred:            TimeSeries,
    future_pred:     TimeSeries,
    smape_val:       float,
    rmse_val:        float,
    mae_val:         float,
    forecast_horizon: int,
    target_col:      str,
    result_path:     str,
) -> None:
    """
    4-panel diagnostic figure:
      1. Backtest – predicted vs actual over the test window
      2. Residuals – signed error over time (predicted − actual)
      3. Future forecast – history + forecast beyond the dataset
      4. Error distribution – histogram of residuals
    """
    # Align series to a common time index for residual calculation
    pred_df   = pred.to_dataframe()
    test_df   = test.to_dataframe()
    common_ix = pred_df.index.intersection(test_df.index)
    residuals = (
        pred_df.loc[common_ix].iloc[:, 0] - test_df.loc[common_ix].iloc[:, 0]
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"{model_name}  –  '{target_col}' Forecast\n"
        f"SMAPE {smape_val:.2f}%   RMSE {rmse_val:.4f}   MAE {mae_val:.4f}",
        fontsize=13,
        fontweight="bold",
    )

    # ── Panel 1: Backtest ────────────────────────────────────────────────────
    ax1 = axes[0, 0]
    show_n = min(7 * 24, len(train))
    train[-show_n:].plot(ax=ax1, label="Train (last 7 d)")
    test.plot(ax=ax1, label="Actual (test)")
    try:
        pred.plot(ax=ax1, label="Predicted", low_quantile=0.1, high_quantile=0.9)
    except Exception:
        pred.plot(ax=ax1, label="Predicted")
    ax1.set_title("Backtest – Test Set")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("")

    # ── Panel 2: Residuals over time ─────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.plot(residuals.index, residuals.values, color="tomato", linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.fill_between(residuals.index, residuals.values, 0, alpha=0.25, color="tomato")
    ax2.set_title("Residuals Over Time  (predicted − actual)")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("")

    # ── Panel 3: Future forecast ─────────────────────────────────────────────
    ax3 = axes[1, 0]
    show_n_future = min(7 * 24, len(series))
    series[-show_n_future:].plot(ax=ax3, label="Historical (last 7 d)")
    try:
        future_pred.plot(
            ax=ax3,
            label=f"Future ({forecast_horizon} h)",
            low_quantile=0.1,
            high_quantile=0.9,
        )
    except Exception:
        future_pred.plot(ax=ax3, label=f"Future ({forecast_horizon} h)")
    ax3.set_title(f"Future Forecast – Next {forecast_horizon} Hours")
    ax3.legend(fontsize=8)
    ax3.set_xlabel("")

    # ── Panel 4: Error distribution ──────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.hist(residuals.values, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax4.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax4.axvline(
        residuals.mean(), color="tomato", linewidth=1.2,
        linestyle="--", label=f"Mean {residuals.mean():.3f}"
    )
    ax4.set_title("Residual Distribution")
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(result_path, f"{model_name}_forecast.png")
    plt.savefig(plot_path, dpi=150)
    print(f"💾 Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    # ── Choose your model ──────────────────────────────────────────────────
    # Swap this import + class for any other Darts model, e.g.:
    #   from darts.models import LightGBMModel, NBEATSModel, TFTModel, NHiTSModel, ...
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
    if _is_torch_model(MODEL_CLS):
        print("🧠 Neural network model detected – loading training config from YAML…")
        torch_kwargs = _build_torch_kwargs(cfg)
        # YAML-derived values are the base; any explicit MODEL_KWARGS above
        # override them so hand-tuned values always win.
        MODEL_KWARGS = {**torch_kwargs, **MODEL_KWARGS}
 
    # ── Load target series ─────────────────────────────────────────────────
    series = load_timeseries(CSV_PATH, target_cols=[TARGET_COL])
 
    # ── Optionally load covariates ─────────────────────────────────────────
    past_cov   = None
    future_cov = None
 
    if PAST_COV_COLS:
        past_cov = load_timeseries(CSV_PATH, target_cols=PAST_COV_COLS)
        print(f"📎 Past covariates loaded: {PAST_COV_COLS}")
 
    if FUTURE_COV_COLS:
        future_cov = load_timeseries(CSV_PATH, target_cols=FUTURE_COV_COLS)
        print(f"📎 Future covariates loaded: {FUTURE_COV_COLS}")
 
    # ── Run ────────────────────────────────────────────────────────────────
    pred, future = run_forecast(
        model_cls         = MODEL_CLS,
        series            = series,
        model_kwargs      = MODEL_KWARGS,
        past_covariates   = past_cov,
        future_covariates = future_cov,
    )
