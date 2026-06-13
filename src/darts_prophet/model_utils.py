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
import re
from typing import Any, Type

import numpy as np
import pandas as pd
from darts.timeseries import TimeSeries

from data_processing import load_timeseries, normalize_series
from pytorch_lightning.callbacks import ModelCheckpoint

# ── kwarg inspection ──────────────────────────────────────────────────────────

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

    Uses static inspection first; falls back to trial-and-elimination for
    models that forward **kwargs internally (e.g. darts.Prophet).
    """
    accepted = accepted_params(model_cls)

    if accepted is not None:
        filtered  = {k: v for k, v in kwargs.items() if k in accepted}
        discarded = {k for k in kwargs if k not in accepted}
        if discarded:
            print(f"⚠️  Ignored unsupported kwargs for {model_cls.__name__}: {discarded}")
        return filtered

    candidate = dict(kwargs)
    discarded: set[str] = set()

    while True:
        try:
            model_cls(**candidate)
            break
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
            if match:
                bad_key = match.group(1)
                discarded.add(bad_key)
                candidate.pop(bad_key, None)
            else:
                break

    if discarded:
        print(f"⚠️  Ignored unsupported kwargs for {model_cls.__name__}: {discarded}")
    return candidate


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

    # LR Scheduler
    pl_callbacks: list = [LearningRateMonitor(logging_interval="step")]

    if sched_cfg:
        sched_name   = str(sched_cfg.get("type", "steplr")).lower()
        sched_cls    = _SCHEDULER_MAP.get(sched_name, lr_sched.StepLR)
        sched_kwargs = {k: v for k, v in sched_cfg.items() if k != "type"}
        kwargs["lr_scheduler_cls"]    = sched_cls
        kwargs["lr_scheduler_kwargs"] = sched_kwargs

        if sched_name == "reducelronplateau":
            from pytorch_lightning.callbacks import Callback

            class _EMALossCallback(Callback):
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

            alpha = float(sched_cfg.get("ema_alpha", 0.1))
            pl_callbacks.append(_EMALossCallback(alpha=alpha))
            kwargs["lr_scheduler_kwargs"]["monitor"] = "val_loss_ema"
            print(f"📉 ReduceLROnPlateau monitoring val_loss_ema (α={alpha})")

    save_dir = cfg['training']['save_dir']
    model_name = cfg['model']['model_cls']
    checkpoint_callback = ModelCheckpoint(
    dirpath   = save_dir,           # needs to be passed in — see below
    filename  = "{epoch:04d}",
    every_n_epochs = 1,
    save_top_k     = 3,            # -1 = keep all checkpoints. Otherwise, save n newest checkpoints
)


    kwargs["pl_trainer_kwargs"].setdefault("callbacks", []).append(checkpoint_callback)

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


# ── Month covariate ───────────────────────────────────────────────────────────

def make_month_covariate(
    n_steps:       int,
    hours_per_row: int,
    start_month:   int = 1,
    start_day:     int = 1,
    start_hour:    int = 0,
) -> TimeSeries:
    """
    Build a normalised month covariate TimeSeries with a RangeIndex matching
    the target series.

    Month is encoded as two sine/cosine components to preserve cyclical
    continuity (December → January has no discontinuity).

    Parameters
    ----------
    n_steps       : Total number of time steps (must match the target series).
    hours_per_row : How many real-world hours each index step represents.
                    Typically 1 for hourly data.
    start_month   : Calendar month (1–12) of the first row in the dataset.
    start_day     : Calendar day-of-month of the first row.
    start_hour    : Hour-of-day of the first row.

    Returns
    -------
    Darts TimeSeries with two components: month_sin and month_cos,
    both in [-1, 1], with a RangeIndex identical to the target series.
    """
    # Reconstruct calendar timestamps for each step
    base = pd.Timestamp(year=2000, month=start_month, day=start_day, hour=start_hour)
    timestamps = pd.date_range(base, periods=n_steps, freq=f"{hours_per_row}h")
    months = timestamps.month.to_numpy().astype(np.float32)   # 1–12

    # Sine / cosine encoding: maps 1..12 onto a full circle
    month_sin = np.sin(2 * np.pi * (months - 1) / 12).astype(np.float32)
    month_cos = np.cos(2 * np.pi * (months - 1) / 12).astype(np.float32)

    df = pd.DataFrame(
        {"month_sin": month_sin, "month_cos": month_cos},
        index=pd.RangeIndex(n_steps),
    )
    return TimeSeries.from_dataframe(df, freq="1", fill_missing_dates=False)


# ── Data loading convenience wrapper ─────────────────────────────────────────

def load_all_series(cfg: dict[str, Any]) -> tuple[
    TimeSeries,           # series (normalised)
    Any,                  # scaler
    TimeSeries | None,    # past_cov (normalised, or None)
    TimeSeries | None,    # future_cov (normalised, or None)
]:
    """
    Load and normalise the target series and all covariates as specified in
    the config dict.  Month covariate is appended to future_cov when
    ``data.use_month_covariate`` is true.

    Returns (series, scaler, past_cov, future_cov).
    """
    data_cfg        = cfg["data"]
    target_col      = data_cfg.get("target_col", "price")
    past_cov_cols   = data_cfg.get("past_covariates",  []) or []
    future_cov_cols = data_cfg.get("future_covariates", []) or []
    csv_path        = data_cfg["data_path"]
    future_csv      = data_cfg.get("future_data_path")
    repeat_cycles   = data_cfg.get("repeat_cycles", 4)
    use_month       = data_cfg.get("use_month_covariate", False)

    # Target
    series, scaler = normalize_series(
        load_timeseries(
            csv_path,
            target_cols=[target_col],
            day_of_week_col="day of week",
            hour_col="hour",
            repeat_cycles=repeat_cycles,
        ).astype(np.float32)
    )

    # Past covariates
    past_cov: TimeSeries | None = None
    if past_cov_cols:
        past_cov, _ = normalize_series(
            load_timeseries(
                csv_path,
                target_cols=past_cov_cols,
                day_of_week_col="day of week",
                hour_col="hour",
                repeat_cycles=repeat_cycles,
            ).astype(np.float32)
        )
        print(f"📎 Past covariates  : {past_cov_cols}")

    # Future covariates from CSV
    future_cov: TimeSeries | None = None
    if future_cov_cols and future_csv:
        future_cov, _ = normalize_series(
            load_timeseries(
                future_csv,
                target_cols=future_cov_cols,
                day_of_week_col="day of week",
                hour_col="hour",
                repeat_cycles=repeat_cycles,
            ).astype(np.float32)
        )
        print(f"📎 Future covariates: {future_cov_cols}")

    # Month covariate (future, always known in advance)
    if use_month:
        month_cfg     = data_cfg.get("month_covariate", {}) or {}
        start_month   = int(month_cfg.get("start_month",   1))
        start_day     = int(month_cfg.get("start_day",     1))
        start_hour    = int(month_cfg.get("start_hour",    0))
        hours_per_row = int(month_cfg.get("hours_per_row", 1))

        month_ts = make_month_covariate(
            n_steps       = len(series),
            hours_per_row = hours_per_row,
            start_month   = start_month,
            start_day     = start_day,
            start_hour    = start_hour,
        )

        if future_cov is not None:
            # Stack month columns onto the existing future covariate series
            future_cov = future_cov.stack(month_ts)
        else:
            future_cov = month_ts

        print(f"📅 Month covariate added (sin/cos, start {start_month:02d}-{start_day:02d} {start_hour:02d}:00)")

    return series, scaler, past_cov, future_cov


def extend_future_covariates(
    future_cov:   TimeSeries | None,
    cfg:          dict[str, Any],
    extra_steps:  int,
) -> TimeSeries | None:
    """
    Extend future covariates by `extra_steps` beyond their current end so
    that inference with an offset does not run out of covariate data.

    Currently handles the month covariate automatically.  CSV-backed
    covariates are left unchanged — those must be supplied by the user.

    If future_cov is None and use_month_covariate is True, a pure month
    covariate series is created at the required length.
    """
    data_cfg  = cfg["data"]
    use_month = data_cfg.get("use_month_covariate", False)

    if not use_month or extra_steps <= 0:
        return future_cov

    month_cfg     = data_cfg.get("month_covariate", {}) or {}
    start_month   = int(month_cfg.get("start_month",   1))
    start_day     = int(month_cfg.get("start_day",     1))
    start_hour    = int(month_cfg.get("start_hour",    0))
    hours_per_row = int(month_cfg.get("hours_per_row", 1))

    if future_cov is None:
        return None   # no future covariates at all; caller handles

    total_needed = len(future_cov) + extra_steps
    extended_month = make_month_covariate(
        n_steps       = total_needed,
        hours_per_row = hours_per_row,
        start_month   = start_month,
        start_day     = start_day,
        start_hour    = start_hour,
    )

    # Determine which components of future_cov are month columns
    month_cols  = {"month_sin", "month_cos"}
    other_cols  = [c for c in future_cov.components if c not in month_cols]
    has_month   = any(c in month_cols for c in future_cov.components)

    if not has_month:
        # No month in existing future_cov — nothing to extend here
        return future_cov

    if other_cols:
        # Keep non-month components at original length; stack extended month
        other_ts = future_cov[other_cols]
        return other_ts.stack(extended_month[:len(other_ts)])
    else:
        return extended_month
