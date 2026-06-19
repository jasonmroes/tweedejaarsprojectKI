"""
model_utils.py
--------------
Shared helpers for single_model_test.py, train.py, and inference.py.

Covers:
  - Model-kwarg inspection and filtering
  - TorchForecastingModel kwarg construction from YAML
  - fit / predict kwarg builders (covariate-aware)
  - Month covariate generation
  - Data loading convenience wrapper
"""

from __future__ import annotations

import inspect
import os
import pickle
import re
from typing import Any, Type

import numpy as np
import pandas as pd
from darts.timeseries import TimeSeries

from data_processing import load_timeseries, normalize_series, ensure_chronological_order
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback, EarlyStopping

class EMALossCallback(Callback):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self._alpha = alpha
        self._ema: float | None = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        raw = trainer.callback_metrics.get("val_loss")
        if raw is None:
            return
        v = float(raw)
        self._ema = (
            v if self._ema is None
            else self._alpha * v + (1.0 - self._alpha) * self._ema
        )
        pl_module.log("val_loss_ema", self._ema, prog_bar=False)
# ── kwarg inspection ──────────────────────────────────────────────────────────
def _assert_aligned(series: TimeSeries, other: TimeSeries, name: str) -> None:
    if len(series) != len(other):
        raise ValueError(
            f"Length mismatch: target series has {len(series)} steps "
            f"but '{name}' has {len(other)}. They must match exactly."
        )


def compute_split_sizes(series_len: int, cfg: dict[str, Any]) -> tuple[int, int]:
    """
    Return (val_size, test_size) in steps, both as fractions of the full
    series length, read directly from cfg['data']. Single source of truth
    for the train/val/test boundary so every script agrees on the same
    split: [... train ...][... val ...][... test ...].
    """
    val_fraction  = cfg["data"]["validation_fraction"]
    test_fraction = cfg["data"]["test_fraction"]
    return int(series_len * val_fraction), int(series_len * test_fraction)

def accepted_params(model_cls: Type) -> set[str] | None:
    """
    Return the set of parameter names accepted by model_cls.__init__.

    Walks the MRO so Darts wrapper classes that forward **kwargs to an
    underlying library are handled correctly.  Returns None when the
    class genuinely accepts arbitrary keyword arguments.
    """
    collected: set[str] = set()
    has_var_keyword = False

    for cls in model_cls.__mro__:
        if cls is object:
            continue
        try:
            if "__init__" not in cls.__dict__:
                continue
            sig = inspect.signature(cls.__init__)
            for name, p in sig.parameters.items():
                if name == "self":
                    continue
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    has_var_keyword = True
                elif p.kind != inspect.Parameter.VAR_POSITIONAL:
                    collected.add(name)
        except (ValueError, TypeError):
            continue

    if not has_var_keyword:
        return collected if collected else None
    if collected:
        return collected
    return None


def filter_kwargs(model_cls: Type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Return only the kwargs that model_cls.__init__ actually accepts.
    If the class uses **kwargs (e.g. wraps an external library), all kwargs
    are passed through — filtering is not possible without instantiation.
    """
    accepted = accepted_params(model_cls)
    if accepted is None:
        # **kwargs present: cannot filter statically; pass everything through.
        return dict(kwargs)

    filtered  = {k: v for k, v in kwargs.items() if k in accepted}
    discarded = {k for k in kwargs if k not in accepted}
    if discarded:
        print(f"⚠️  Ignored unsupported kwargs for {model_cls.__name__}: {discarded}")
    return filtered


def is_torch_model(model_cls: Type) -> bool:
    """Return True if model_cls is a Darts TorchForecastingModel subclass."""
    try:
        from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
        return issubclass(model_cls, TorchForecastingModel)
    except ImportError:
        return False


def supports_covariates(model_instance, cov_type: str) -> bool:
    """Return True if the model supports 'past' or 'future' covariates."""
    return bool(getattr(model_instance, f"supports_{cov_type}_covariates", False))


# ── fit / predict builders ────────────────────────────────────────────────────

def build_fit_kwargs(
    model,
    train:                 TimeSeries,
    val:                   TimeSeries,
    past_covariates:       TimeSeries | None = None,
    future_covariates:     TimeSeries | None = None,
    val_past_covariates:   TimeSeries | None = None,
    val_future_covariates: TimeSeries | None = None,
) -> dict[str, Any]:
    """Build model.fit() kwargs, respecting per-model covariate support."""
    fit_sig = set(inspect.signature(model.fit).parameters.keys())
    kwargs: dict[str, Any] = {"series": train, "val_series": val}

    if past_covariates is not None and supports_covariates(model, "past"):
        if "past_covariates" in fit_sig:
            kwargs["past_covariates"] = past_covariates
        if "val_past_covariates" in fit_sig and val_past_covariates is not None:
            kwargs["val_past_covariates"] = val_past_covariates
    elif past_covariates is not None:
        print(f"⚠️  {type(model).__name__} does not support past_covariates – skipping.")

    if future_covariates is not None and supports_covariates(model, "future"):
        if "future_covariates" in fit_sig:
            kwargs["future_covariates"] = future_covariates
        if "val_future_covariates" in fit_sig and val_future_covariates is not None:
            kwargs["val_future_covariates"] = val_future_covariates
    elif future_covariates is not None:
        print(f"⚠️  {type(model).__name__} does not support future_covariates – skipping.")

    return kwargs


def build_predict_kwargs(
    model,
    n:                 int,
    series:            TimeSeries,
    past_covariates:   TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
) -> dict[str, Any]:
    """
    Build model.predict() kwargs, respecting per-model covariate support.
    `series` is always included — required by global models (e.g. TiDEModel)
    to anchor the prediction context window.
    """
    pred_sig = set(inspect.signature(model.predict).parameters.keys())
    kwargs: dict[str, Any] = {"n": n, "series": series}

    if past_covariates is not None and supports_covariates(model, "past"):
        if "past_covariates" in pred_sig:
            kwargs["past_covariates"] = past_covariates

    if future_covariates is not None and supports_covariates(model, "future"):
        if "future_covariates" in pred_sig:
            kwargs["future_covariates"] = future_covariates

    return kwargs


# ── Torch kwargs from YAML ────────────────────────────────────────────────────

def build_torch_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Translate the YAML model + training blocks into TorchForecastingModel
    constructor kwargs.

    YAML keys consumed
    ------------------
    model:
      input_chunk_length, output_chunk_length,
      hidden_dim (or hidden_size), num_layers, num_attention_heads,
      dropout, activation, use_reversible_instance_norm

    training:
      epochs, batch_size, learning_rate, weight_decay, optimizer,
      scheduler:
        type, (scheduler-specific keys), ema_alpha

    tensorboard:
      enabled, log_dir, log_every_n_steps

    seed
    """
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_sched
    from pytorch_lightning.callbacks import LearningRateMonitor

    _OPTIMIZER_MAP: dict[str, type] = {
        "adam":    optim.Adam,
        "adamw":   optim.AdamW,
        "sgd":     optim.SGD,
        "rmsprop": optim.RMSprop,
    }
    _SCHEDULER_MAP: dict[str, type] = {
        "steplr":            lr_sched.StepLR,
        "multisteplr":       lr_sched.MultiStepLR,
        "cosineannealinglr": lr_sched.CosineAnnealingLR,
        "reducelronplateau": lr_sched.ReduceLROnPlateau,
        "exponentiallr":     lr_sched.ExponentialLR,
    }

    model_cfg    = cfg.get("model",    {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    sched_cfg    = training_cfg.get("scheduler", {}) or {}
    kwargs: dict[str, Any] = {}

    # Architecture
    for key in ("input_chunk_length", "output_chunk_length"):
        if key in model_cfg:
            kwargs[key] = int(model_cfg[key])

    hidden = model_cfg.get("hidden_dim") or model_cfg.get("hidden_size")
    if hidden is not None:
        kwargs["hidden_dim"]  = int(hidden)
        kwargs["hidden_size"] = int(hidden)

    for key in ("num_layers", "num_attention_heads"):
        if key in model_cfg:
            kwargs[key] = int(model_cfg[key])

    if "dropout" in model_cfg:
        kwargs["dropout"] = float(model_cfg["dropout"])
    if "activation" in model_cfg:
        kwargs["activation"] = str(model_cfg["activation"])
    if "use_reversible_instance_norm" in model_cfg:
        kwargs["use_reversible_instance_norm"] = bool(
            model_cfg["use_reversible_instance_norm"]
        )

    # Training loop
    if "epochs" in training_cfg:
        kwargs["n_epochs"] = int(training_cfg["epochs"])
    if training_cfg.get("batch_size") is not None:
        kwargs["batch_size"] = int(training_cfg["batch_size"])
    num_workers = int(training_cfg.get("num_workers", 0))
    if training_cfg.get("num_workers") is not None:
        kwargs["dataloader_kwargs"] = {
            "num_workers": num_workers,
        }

    # Optimizer
    lr           = training_cfg.get("learning_rate")
    weight_decay = training_cfg.get("weight_decay")
    opt_cls      = _OPTIMIZER_MAP.get(
        str(training_cfg.get("optimizer", "adamw")).lower(), optim.AdamW
    )
    opt_kwargs: dict[str, Any] = {}
    if lr           is not None: opt_kwargs["lr"]           = float(lr)
    if weight_decay is not None: opt_kwargs["weight_decay"] = float(weight_decay)
    kwargs["optimizer_cls"]    = opt_cls
    kwargs["optimizer_kwargs"] = opt_kwargs

    # Early stopping (stops training when val_loss stops improving)
    early_stop_cfg       = training_cfg.get("early_stop", {}) or {}
    early_stop_patience  = int(early_stop_cfg.get("patience", 10))
    early_stop_min_delta = float(early_stop_cfg.get("min_delta", 0.0))
    early_stopper = EarlyStopping(
        monitor   = "val_loss",
        patience  = early_stop_patience,
        min_delta = early_stop_min_delta,
        mode      = "min",
    )

    # LR Scheduler
    pl_callbacks: list = [LearningRateMonitor(logging_interval="step"), early_stopper]

    if sched_cfg:
        sched_name   = str(sched_cfg.get("type", "steplr")).lower()
        sched_cls    = _SCHEDULER_MAP.get(sched_name, lr_sched.StepLR)
        sched_kwargs = {k: v for k, v in sched_cfg.items() if k != "type"}
        kwargs["lr_scheduler_cls"]    = sched_cls
        kwargs["lr_scheduler_kwargs"] = sched_kwargs

        if sched_name == "reducelronplateau":
            alpha = float(sched_cfg.get("ema_alpha", 0.1))
            pl_callbacks.append(EMALossCallback(alpha=alpha))   # module-level class
            kwargs["lr_scheduler_kwargs"]["monitor"] = "val_loss_ema"
            print(f"📉 ReduceLROnPlateau monitoring val_loss_ema (α={alpha})")

    save_dir = cfg['training']['save_dir']
    model_name = cfg['model']['model_cls']

# Superfluous, as darts already handles checkpointing
#     checkpoint_callback = ModelCheckpoint(
#     dirpath   = save_dir,           # needs to be passed in — see below
#     filename  = "{epoch:04d}",
#     every_n_epochs = 1,
#     save_top_k     = -1,            # -1 = keep all checkpoints
# )

#     pl_callbacks.append(checkpoint_callback)

    # Tell Darts to use its own internal checkpointing mechanism as well,
    # which is what model.load_from_checkpoint() reads.
    kwargs["save_checkpoints"] = True
    kwargs["work_dir"]         = save_dir
    kwargs["model_name"]       = model_name

    # TensorBoard
    tb_cfg = cfg.get("tensorboard", {}) or {}
    if tb_cfg.get("enabled", False):
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            log_dir     = cfg["tensorboard"]["log_dir"]
            log_name    = cfg.get("experiment_name", "experiment")
            log_every_n = int(tb_cfg.get("log_every_n_steps", 50))
            logger      = TensorBoardLogger(
                save_dir=log_dir, name=log_name, default_hp_metric=False
            )
            kwargs["pl_trainer_kwargs"] = {
                "logger"           : logger,
                "callbacks"        : pl_callbacks,
                "log_every_n_steps": log_every_n,
            }
            print(f"📊 TensorBoard → {log_dir}/{log_name}")
            print(f"   tensorboard --logdir {log_dir}")
        except ImportError:
            print("⚠️  pytorch_lightning not found – TensorBoard disabled.")
    else:
        # Still attach LR monitor and EMA callbacks even without TensorBoard
        kwargs["pl_trainer_kwargs"] = {"callbacks": pl_callbacks}

    if "seed" in cfg:
        kwargs["random_state"] = int(cfg["seed"])

    return kwargs

def soft_clip(series: TimeSeries, margin: float = 0.2) -> TimeSeries:
    """Softly bound scaled-space model predictions before inverse-transforming.

    The model is unconstrained and can predict outside the scaler's fitted
    [0, 1] range; left uncorrected, scaler.inverse_transform() extrapolates
    linearly into unrealistic prices. A hard clip to [0, 1] avoids that but
    flattens every prediction above the historical max to the same value,
    losing legitimate "higher than usual" signal.

    Values within [0, 1 + margin] pass through unchanged (so a
    reasonable overshoot up to `margin` beyond the trained range is fully
    preserved); beyond that, a tanh saturation smoothly bends the curve
    towards an asymptote instead of truncating it, so larger overshoots are
    compressed rather than clipped flat.
    """
    lo, hi = 0, 1.0 + margin # Never predict negative

    def _squash(x: np.ndarray) -> np.ndarray:
        out = x
        out = np.where(out > hi, hi + margin * np.tanh((out - hi) / margin), out)
        out = np.where(out < lo, lo - margin * np.tanh((lo - out) / margin), out)
        return out

    return series.map(_squash)

# ── Scaler persistence ────────────────────────────────────────────────────────
# All scalers (target + covariates) are bundled into a single scalers.pkl so
# that inference and resumed training reuse the exact transforms the model was
# trained with, instead of re-fitting (which silently desyncs if the config or
# data drifts). A legacy bare-target scaler.pkl is still read as a fallback so
# older checkpoints keep working (their covariate scalers are then re-fit).

SCALERS_FILENAME = "scalers.pkl"
LEGACY_TARGET_SCALER_FILENAME = "scaler.pkl"


def save_scalers(
    save_dir: str,
    target_scaler: Any,
    past_scaler: Any = None,
    future_scaler: Any = None,
) -> None:
    """Persist all fitted scalers as a bundle under save_dir/scalers.pkl."""
    os.makedirs(save_dir, exist_ok=True)
    bundle = {
        "target":  target_scaler,
        "past":    past_scaler,
        "future":  future_scaler,
    }
    with open(os.path.join(save_dir, SCALERS_FILENAME), "wb") as f:
        pickle.dump(bundle, f)


def load_scalers(save_dir: str) -> dict[str, Any] | None:
    """
    Load the scaler bundle from save_dir/scalers.pkl.

    Falls back to the legacy bare-target scaler.pkl (covariate scalers None)
    so checkpoints saved before bundling still load. Returns None if neither
    file exists.
    """
    bundle_path = os.path.join(save_dir, SCALERS_FILENAME)
    if os.path.exists(bundle_path):
        with open(bundle_path, "rb") as f:
            return pickle.load(f)

    legacy_path = os.path.join(save_dir, LEGACY_TARGET_SCALER_FILENAME)
    if os.path.exists(legacy_path):
        with open(legacy_path, "rb") as f:
            target_scaler = pickle.load(f)
        print(
            f"⚠️  Found legacy {LEGACY_TARGET_SCALER_FILENAME} (target only) — "
            f"covariate scalers will be re-fit. Retrain to persist them."
        )
        return {"target": target_scaler, "past": None, "future": None}

    return None

# ── Data loading convenience wrapper ─────────────────────────────────────────

def load_all_series(
    cfg: dict[str, Any],
    target_scaler: Any | None = None,
    past_scaler: Any | None = None,
    future_scaler: Any | None = None,
) -> tuple[
    TimeSeries,           # series (normalised)
    Any,                  # target scaler
    TimeSeries | None,    # past_cov (normalised, or None)
    TimeSeries | None,    # future_cov (normalised, or None)
    Any,                  # past scaler (or None)
    Any,                  # future scaler (or None)
]:
    """
    Load and normalise the target series and all covariates as specified in
    the config dict.  Month covariate is appended to future_cov when
    month column is available in csv.

    target_scaler / past_scaler / future_scaler: optional pre-fitted scalers
    (e.g. the ones saved during training). When given, each is applied as-is
    instead of being re-fit here — required at inference time, and when
    resuming training, so every series is scaled identically to what the
    model was trained on even if the config (split fractions, data path, ...)
    has since changed. Any scaler left as None is fit fresh on the train split.

    Returns (series, target_scaler, past_cov, future_cov, past_scaler,
    future_scaler). The returned scalers are the ones actually used (the
    passed-in objects when supplied, otherwise the freshly-fit ones) so the
    caller can persist them.
    """
    data_cfg        = cfg["data"]
    target_col      = data_cfg.get("target_col", "price")
    past_cov_cols   = data_cfg.get("past_covariates",  []) or []
    future_cov_cols = data_cfg.get("future_covariates", []) or []
    csv_path        = data_cfg["data_path"]
    future_csv      = data_cfg.get("future_data_path")
    repeat_cycles   = data_cfg.get("repeat_cycles", 4)
    day_of_week_col = data_cfg.get("day_of_week_col", "day of week")
    hour_col        = data_cfg.get("hour_col", "hour")
    
    if future_cov_cols and not future_csv:
        raise ValueError(
            "future_covariates specified but future_data_path is empty."
        )
     # ── Target series ─────────────────────────────────────────────────────────
    raw_series = load_timeseries(
        csv_path,
        target_cols=[target_col],
        day_of_week_col=day_of_week_col,
        hour_col=hour_col,
        repeat_cycles=repeat_cycles,
    ).astype(np.float32)

    val_size, test_size = compute_split_sizes(len(raw_series), cfg)
    train_end = len(raw_series) - val_size - test_size

    if target_scaler is not None:
        scaler = target_scaler
        series = scaler.transform(raw_series)              # apply pre-fit scaler to full
    else:
        series, scaler = normalize_series(raw_series[:train_end])  # fit on train only
        series         = scaler.transform(raw_series)              # apply to full

    # ── Past covariates ───────────────────────────────────────────────────────
    past_cov: TimeSeries | None = None
    if past_cov_cols:
        raw_past = load_timeseries(
            csv_path,
            target_cols=past_cov_cols,
            day_of_week_col=day_of_week_col,
            hour_col=hour_col,
            repeat_cycles=repeat_cycles,
        ).astype(np.float32)

        if past_scaler is not None:
            past_cov = past_scaler.transform(raw_past)              # apply pre-fit scaler to full
        else:
            past_cov, past_scaler = normalize_series(raw_past[:train_end])  # fit on train only
            past_cov              = past_scaler.transform(raw_past)          # apply to full

        print(f"📎 Past covariates  : {past_cov_cols}")
        _assert_aligned(series, past_cov, "past_covariates")

    # ── Future covariates ─────────────────────────────────────────────────────
    future_cov: TimeSeries | None = None
    if future_cov_cols:
        # Always load training future covariates from the main CSV first
        raw_future = load_timeseries(
            csv_path,
            target_cols=future_cov_cols,
            day_of_week_col=day_of_week_col,
            hour_col=hour_col,
            repeat_cycles=repeat_cycles,
        ).astype(np.float32)

        if future_scaler is not None:
            training_future_cov = future_scaler.transform(raw_future)              # apply pre-fit scaler to full
        else:
            training_future_cov, future_scaler = normalize_series(raw_future[:train_end])  # fit on train only
            training_future_cov                = future_scaler.transform(raw_future)        # apply to full

        if future_csv and future_csv != csv_path:
            # Separate future CSV — align index to follow training data
            future_cov = load_future_covariate_csv(
                future_csv,
                cfg=cfg,
                past_cov=past_cov if past_cov is not None else series,
                training_future_cov=training_future_cov,
            )
        else:
            future_cov = training_future_cov
            _assert_aligned(series, future_cov, "future_covariates")

        print(f"📎 Future covariates: {future_cov_cols}")

    return series, scaler, past_cov, future_cov, past_scaler, future_scaler

def load_future_covariate_csv(
    future_csv_path: str,
    cfg: dict,
    past_cov: TimeSeries,
    training_future_cov: TimeSeries,
    day_of_week_col: str = "day of week",
    hour_col: str = "hour",
) -> TimeSeries:
    col_names = cfg["data"].get("future_covariates", [])
    input_chunk_length = cfg["model"].get("input_chunk_length", 672)

    future_ts = load_timeseries(
        path=future_csv_path,
        target_cols=col_names,
        day_of_week_col=day_of_week_col,
        hour_col=hour_col,
        is_future_covariate=True,
    )

    offset = int(past_cov.time_index[-1]) + 1
    df = future_ts.to_dataframe().reset_index(drop=True)
    df.index = range(offset, offset + len(df))

    lookback = training_future_cov[-input_chunk_length:].to_dataframe()
    combined = pd.concat([lookback, df])

    print(f"📅 Future covariates: {len(df)} rows of new data, "
          f"prepended {input_chunk_length} steps of training history, "
          f"index {int(combined.index[0])} → {int(combined.index[-1])}")

    return TimeSeries.from_dataframe(
        combined,
        freq="1",
        fill_missing_dates=False,
    ).astype(np.float32)

def extend_future_covariates(
    future_cov: TimeSeries,
    cfg: dict,
    extra_steps: int,
) -> TimeSeries:
    """
    Extend future covariates by `extra_steps` steps beyond their current end.

    - month_sin / month_cos: extended by continuing the cyclical pattern.
    - Any other column: cannot be extrapolated — raises ValueError.

    Parameters
    ----------
    future_cov  : TimeSeries to extend (RangeIndex, float32).
    cfg         : Full config dict (unused currently, kept for future use).
    extra_steps : Number of steps to append.
    """
    if extra_steps <= 0:
        return future_cov

    df = future_cov.to_dataframe()
    cols = set(df.columns)
    non_cyclical = cols - {"month_sin", "month_cos"}

    if non_cyclical:
        raise ValueError(
            f"Cannot auto-extend non-cyclical future covariate column(s): "
            f"{sorted(non_cyclical)}. Supply a future_data_path CSV that "
            f"covers at least {extra_steps} extra steps beyond the training data."
        )

    # Extend the integer index
    last_idx = df.index[-1]
    new_idx  = range(last_idx + 1, last_idx + 1 + extra_steps)
    extension = {}

    if "month_sin" in cols and "month_cos" in cols:
        # Infer the period from the existing pattern by finding zero-crossings
        # of month_sin, then continue the cycle arithmetically.
        # Simpler and more robust: just continue the angle from where we left off.
        # Each step advances the angle by 2π / 12 months * (steps_per_month).
        # We derive steps_per_month from the total series length ÷ number of months.
        n = len(df)
        # Recover the angle of the last point directly from the data
        last_sin = float(df["month_sin"].iloc[-1])
        last_cos = float(df["month_cos"].iloc[-1])
        last_angle = np.arctan2(last_sin, last_cos)

        # Infer angular step size from consecutive differences
        if n >= 2:
            prev_sin = float(df["month_sin"].iloc[-2])
            prev_cos = float(df["month_cos"].iloc[-2])
            prev_angle = np.arctan2(prev_sin, prev_cos)
            delta = last_angle - prev_angle
            # Wrap to (-π, π]
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
        else:
            delta = 2 * np.pi / (12 * 30 * 24)  # fallback: hourly data

        angles = last_angle + delta * np.arange(1, extra_steps + 1)
        extension["month_sin"] = np.sin(angles).astype(np.float32)
        extension["month_cos"] = np.cos(angles).astype(np.float32)

    ext_df = pd.DataFrame(extension, index=new_idx)
    combined = pd.concat([df, ext_df])
    return TimeSeries.from_dataframe(combined, freq="1", fill_missing_dates=False)
