# Forecasting Pipeline — Low-Level Documentation

This document describes how data flows through `data_processing.py`,
`model_utils.py`, `single_model_test.py`, `train.py`, and `inference.py` to
produce a trained price-forecasting model and its predictions.

All five scripts are driven by a single YAML config (e.g.
`configs/TiDE_hid256_dr0.5.yaml`) with three top-level blocks: `data`,
`model`, `training` (plus optional `tensorboard` and `seed`).

## 1. High-level overview

The pipeline forecasts a target column (default `price`) from an hourly CSV
that has no real dates — only a `day of week` (0–6) and `hour` (0–23)
column. Rows are kept in their original chronological order and treated as
one continuous `RangeIndex` series (step 0, 1, 2, …), repeating the weekly
pattern as many times as the data actually contains. A step size is determined by the time gap between rows preprocessed data. Currently, a step equals one hour. 

There are three entry-point scripts, meant to be run in this order:

1. **`single_model_test.py`** — splits the series into
   `[train][val][test]`, trains on `train`, early-stops on `val`, and
   reports honest generalization metrics on the held-out `test` set plus a
   4-panel diagnostic plot. This is the only script whose metrics should be
   trusted to answer "is this config good?".
2. **`train.py`** — once a config is trusted, retrains the same model on
   `[train][val]` (more data, `test` is carved out of the series but never
   used) to produce the checkpoint that gets deployed. Val metrics here are only used to inform early stopping, and should not be taken as absolute truth.
3. **`inference.py`** — loads the checkpoint `train.py` produced and
   forecasts a future window, optionally with a gap (`--offset`) between
   the end of training data and the start of the forecast window.

`data_processing.py` and `model_utils.py` are shared libraries with no CLI
of their own; every entry-point script calls into them for loading,
scaling, model-kwarg construction, and metrics.

### Data flow at a glance

```
CSV (day_of_week, hour, price, covariate cols...)
        │  load_timeseries()                      [data_processing.py]
        ▼
raw TimeSeries (RangeIndex, NaN/Inf cleaned)
        │  normalize_series() / scaler.transform() [data_processing.py]
        ▼
normalised target series + past_cov + future_cov   [model_utils.load_all_series]
        │  split by validation_fraction / test_fraction
        ▼
train / val (/ test)  ──fit()──▶ trained Darts model ──predict()──▶ scaled forecast
        │                                                                  │
        └─────────────────────── scaler.inverse_transform() ◀─────────────┘
                                          │
                                          ▼
                         price-unit predictions → metrics / CSV / plots
```

### Key shared conventions

- **Split boundary**: `compute_split_sizes()` produces the correct splits as defined in the config.
  `val_size = int(len * validation_fraction)`, `test_size = int(len *
  test_fraction)`, both as fractions of the **full** series — never of an
  intermediate slice. Layout is always `[train][val][test]`.
- **Scalers persist together**: `save_scalers()` / `load_scalers()` bundle
  the target, past-covariate, and future-covariate `Scaler` objects into one
  `scalers.pkl` per `save_dir`, so resumed training and inference reuse
  exactly the transform the model was trained with instead of re-fitting.
- **Soft clipping**: `soft_clip()` bounds scaled-space predictions to
  `[0, 1 + margin]` with a tanh saturation beyond that range before
  inverse-transforming, so the model can't extrapolate into absurd prices
  while still preserving "higher than usual" signal.
- **Covariate-aware kwargs**: `build_fit_kwargs()` / `build_predict_kwargs()`
  inspect each model's `fit`/`predict` signature and `supports_past/future_covariates`
  flags so the same calling code works across Darts model classes that do or
  don't accept covariates. This allows for easily swapping out and trying different Darts models

---

## 2. Detailed reference

### 2.1 `data_processing.py`

| Function | Inputs | Output |
|---|---|---|
| `ensure_chronological_order(df)` | `df: DataFrame` (looks for `date` or `createdAt` column) | Same `df`, reversed if descending, reset index. Unchanged + warning if no date column. |
| `build_series(df, target_col="price")` | `df` with RangeIndex; `target_col: str` | `TimeSeries` built from `df[target_col]`, `freq=1`, no date filling. |
| `load_timeseries(path, target_cols, day_of_week_col, hour_col, repeat_cycles, all_columns_for_alignment, sort_by_day_hour, is_future_covariate)` | CSV `path`; `target_cols: list[str]`; column names for day/hour; `is_future_covariate: bool` | `TimeSeries` (RangeIndex, `float32`, one column per `target_col`). Reads CSV → chronological order → cast day/hour to int → per-column NaN/Inf handling → (if `"month"` in `target_cols`) replace with `month_sin`/`month_cos` for proper handling of cyclical nature of time → drop to `target_cols` only → validate no duplicate columns → build `TimeSeries`. |
| `normalize_series(series, scaler=None)` | `series: TimeSeries`; optional pre-fit `Scaler` | `(normalized_series, scaler)` — fits a new `Scaler` (zero mean / unit variance) if none given, else applies the given one. |

**NaN/Inf handling inside `load_timeseries`** (per target column):
- Binary columns (`national holiday`, `walk-in`): NaN → `0`.
- `is_future_covariate=True`: linear interpolation **only between** known
  values (`limit_area='inside'`); any NaN that can't be interpolated this
  way (i.e. trailing/extrapolated values) is filled with the column's
  historical mean — never forward-filled, since freezing a stale forecast
  value would feed an out-of-distribution constant to every later step.
- Otherwise (historical/past data): forward-direction linear interpolation,
  then `ffill()` for any leading NaN.
- Inf values are clipped to `±1e6`.

**Month encoding**: if `"month"` is requested as a target/covariate column,
it's replaced by `month_sin` / `month_cos` (`sin`/`cos` of
`2π(month-1)/12`) so the model sees a continuous cyclical feature instead of
a discrete integer.

### 2.2 `model_utils.py`

#### Kwarg inspection / filtering

| Function | Inputs | Output |
|---|---|---|
| `accepted_params(model_cls)` | Darts model class | `set[str]` of accepted `__init__` kwarg names (walks the MRO), or `None` if the class accepts arbitrary `**kwargs`. |
| `filter_kwargs(model_cls, kwargs)` | model class; `dict` of candidate kwargs | Subset of `kwargs` the class actually accepts (passes everything through if the class uses `**kwargs`); prints a warning listing discarded keys. |
| `is_torch_model(model_cls)` | model class | `bool` — is it a Darts `TorchForecastingModel` subclass. |
| `supports_covariates(model_instance, cov_type)` | model instance; `"past"` or `"future"` | `bool` — reads `model_instance.supports_{cov_type}_covariates`. |

#### Fit / predict kwarg builders

| Function | Inputs | Output |
|---|---|---|
| `build_fit_kwargs(model, train, val, past_covariates, future_covariates, val_past_covariates, val_future_covariates)` | model instance; train/val `TimeSeries`; optional past/future covariates (train + val) | `dict` for `model.fit(**kwargs)` — includes `series`, `val_series` (if `fit()` accepts it), and covariates only if the model supports that covariate type **and** its `fit()` signature accepts the kwarg. Prints a warning if covariates were supplied but unsupported. |
| `build_predict_kwargs(model, n, series, past_covariates, future_covariates)` | model instance; horizon `n`; context `series`; optional covariates | `dict` for `model.predict(**kwargs)` — `n`, `series` (if accepted — required by global models like TiDE, rejected by local models like `ExponentialSmoothing`), and covariates filtered the same way as above. |

#### Torch model kwargs from YAML

| Function | Inputs | Output |
|---|---|---|
| `build_torch_kwargs(cfg)` | full config dict | `dict` of `TorchForecastingModel` constructor kwargs: architecture (`input_chunk_length`, `output_chunk_length`, `hidden_dim`/`hidden_size`, `num_layers`, `num_attention_heads`, `dropout`, `activation`, `use_reversible_instance_norm`), training loop (`n_epochs`, `batch_size`, `dataloader_kwargs`), `optimizer_cls`/`optimizer_kwargs` (mapped from `training.optimizer` name), `lr_scheduler_cls`/`lr_scheduler_kwargs` (mapped from `training.scheduler.type`), an `EarlyStopping` callback (`training.early_stop.patience`/`min_delta`), `save_checkpoints=True`, `work_dir`/`model_name` (so `load_from_checkpoint()` can find it later), and optional `TensorBoardLogger` + `pl_trainer_kwargs`. Also attaches an `EMALossCallback` when the scheduler is `reducelronplateau`, so the plateau scheduler watches a smoothed `val_loss_ema` instead of the raw noisy `val_loss`. |
| `EMALossCallback(alpha)` | PL `Callback` | Each validation epoch end, updates and logs an exponential moving average of `val_loss` as `val_loss_ema`. |

#### Prediction post-processing

| Function | Inputs | Output |
|---|---|---|
| `soft_clip(series, margin=0.2)` | scaled-space `TimeSeries` prediction; `margin: float` | `TimeSeries` with values `> 1+margin` or `< 0` smoothly tanh-saturated instead of hard-clipped, before `scaler.inverse_transform()` is applied downstream. |

#### Metrics

| Function | Inputs | Output |
|---|---|---|
| `compute_extended_metrics(pred_norm_vals, test_norm_vals, pred_inv_vals, test_inv_vals, chunk=24, verbose=True)` | aligned `np.ndarray`s: scaled-space pred/true and price-unit pred/true | `dict` with overall SMAPE/RMSE/MAE (normalised and price units), per-`chunk` (default daily) MAE diagnostics (`daily_mae_price`, `daily_mae_norm`, std/day1/last-day variants), and the same set restricted to steps where the true price is non-zero (`*_nonzero` keys) — used because closing hours are literal-zero price and otherwise dilute the error signal during actual open hours. |

#### Scaler persistence

| Function | Inputs | Output |
|---|---|---|
| `save_scalers(save_dir, target_scaler, past_scaler=None, future_scaler=None)` | fitted scaler objects | Writes `{save_dir}/scalers.pkl` containing `{"target":…, "past":…, "future":…}`. |
| `load_scalers(save_dir)` | `save_dir: str` | `dict` as above, or `None` if nothing found. Falls back to a legacy `scaler.pkl` (target-only) for old checkpoints, re-fitting covariate scalers in that case. |

#### Data loading wrapper

| Function | Inputs | Output |
|---|---|---|
| `load_all_series(cfg, target_scaler=None, past_scaler=None, future_scaler=None)` | full config; optional pre-fit scalers | `(series, scaler, past_cov, future_cov, past_scaler, future_scaler)` — all `TimeSeries` normalised, `float32`. Loads the target via `load_timeseries`, computes `train_end` via `compute_split_sizes`, fits (or reuses) the target scaler **on the train slice only**, then applies it to the **full** series. Repeats the same pattern for past covariates (from the same CSV) and future covariates (from the same CSV, optionally extended with a separate `future_data_path` CSV via `load_future_covariate_csv`). |
| `load_future_covariate_csv(future_csv_path, cfg, past_cov, training_future_cov, ...)` | path to a future-only CSV (e.g. weather forecast); training future covariates; `past_cov` (for offset alignment) | `TimeSeries` of new future-covariate rows, indexed to start right after the training data ends, prepended with `input_chunk_length` steps of training history (`lookback`) so the model has continuous context across the train/forecast boundary. |
| `extend_future_covariates(future_cov, cfg, extra_steps)` | future covariate `TimeSeries`; `extra_steps: int` | Same `TimeSeries` extended by `extra_steps`. Only `month_sin`/`month_cos` can be auto-extended (by continuing their cyclical angle); any other column raises `ValueError` — those must come from a real forecast CSV. |

### 2.3 `single_model_test.py`

Purpose: genuine train/val/test backtest to evaluate whether a config is
trustworthy, with diagnostic plots.

| Function | Inputs | Output |
|---|---|---|
| `_ts_to_array(ts)` | `TimeSeries` | 1-D `float32` `np.ndarray` of its first column. |
| `_index_to_hour_offset(time_index)` | RangeIndex or DatetimeIndex | `np.ndarray` of integer hour-offsets from a fixed reference Monday (2024-01-01). |
| `_make_calendar_tick_labels(time_index, n_positions, hour_step=12)` | time index; number of plotted points; tick spacing in hours | `(positions, labels)` — labels like `"Tue 12:00"` at every `hour_step`-hour boundary. |
| `_apply_calendar_ticks(ax, time_index, n_positions, hour_step=12)` | matplotlib axis + above | Sets x-ticks/labels on `ax` in place. |
| `_count_unique_dates(csv_path, date_col="date")` | raw CSV path | `int` or `None` — true number of distinct calendar dates spanned (row count overstates this when multiple rows share one date). |
| `_apply_relative_time_ticks(ax, time_index, n_positions, max_labels=8, total_days=None)` | matplotlib axis; time index; `total_days` from `_count_unique_dates` | Sets x-ticks as `"Year N.N"` relative-time labels, scaled by `total_days` when given (more accurate than assuming 1 step = 1 hour for long multi-row-per-day datasets). |
| `run_backtest(cfg, model_cls, series, scaler, model_kwargs, past_covariates, future_covariates, past_scaler, future_scaler)` | normalised full series + covariates + scalers; resolved model class; model kwargs | Returns `pred_inv: TimeSeries` (test-set prediction, price units). Side effects: splits `[train][val][test]` via `compute_split_sizes`; builds/resumes model; `model.fit(**build_fit_kwargs(...))`; saves checkpoint `{save_dir}/{ModelName}_backtest.pt`; predicts `output_chunk_length` steps anchored on `train_full`; computes metrics via `compute_extended_metrics`; saves `{save_dir}/{ModelName}_backtest_metrics.json`; calls `_plot_results`; saves `{save_dir}/{ModelName}_backtest.png`. |
| `_plot_results(model_name, series, test_inv, pred_inv, train, val, smape_val, rmse_norm, mae_norm, rmse_inv, mae_inv, target_col, result_path, total_days)` | price-unit series/splits/predictions + metric scalars | No return — saves a 4-panel PNG: (1) full series with train/val/test regions shaded, (2) test actual vs. predicted, (3) residuals over time, (4) residual histogram. |

CLI (`__main__`): reads `--config`, resolves the Darts model class, builds
kwargs via `build_torch_kwargs` (for NN models) or seasonality kwargs (for
classical models), loads data via `load_all_series` (reusing scalers only if
`model.resume` is set), then calls `run_backtest`.

### 2.4 `train.py`

Purpose: retrain the trusted config on `train+val` to produce the
deployable checkpoint.

| Function | Inputs | Output |
|---|---|---|
| `train(config_path)` | path to YAML config | No return. Side effects: loads config; resolves model class; loads/reuses scalers (`load_scalers` if `model.resume`); `load_all_series()`; splits into `[train][val]` only (`test` region size is still computed and excluded from `train_ser`/`val_ser`, but never read) via `compute_split_sizes`; builds model kwargs (`build_torch_kwargs` or seasonality kwargs); constructs or resumes (`model_cls.load_from_checkpoint`) the model; `model.fit(**build_fit_kwargs(...))`; computes val-set SMAPE/RMSE/MAE in price units as an early-stopping sanity check (wrapped in `try/except`, not fatal if it fails) and writes `{save_dir}/{ModelName}_metrics.json`; saves the deployable checkpoint to `{save_dir}/{ModelName}_checkpoint_future.pt` via `model.save()`. |

CLI (`__main__`): `--config` accepts one or more paths (`nargs='+'`) and
trains each sequentially by calling `train(config_path)` in a loop.

### 2.5 `inference.py`

Purpose: load a `train.py` checkpoint and forecast a future window, with an
optional gap (`offset`) between the end of training data and the start of
the desired window.

| Function | Inputs | Output |
|---|---|---|
| `run_inference(config_path, horizon, offset, no_future_cov=False)` | config path; `horizon: int \| None` (defaults to `model.output_chunk_length`); `offset: int` (gap in steps); `no_future_cov: bool` | No return. Side effects: loads model via `model_cls.load_from_checkpoint`; loads persisted scalers via `load_scalers` (raises `FileNotFoundError` if absent — inference cannot proceed without the exact training-time transform); `load_all_series()` using those scalers; if `no_future_cov`, tiles the future covariate CSV cyclically to cover the needed horizon instead of disabling it outright; validates/extends future covariates for the `offset` gap via `extend_future_covariates` (raises `ValueError` if a non-cyclical CSV covariate doesn't extend far enough); predicts `offset + horizon` steps anchored on the full series via `build_predict_kwargs` + `model.predict`; applies `soft_clip` then `scaler.inverse_transform`; discards the first `offset` predicted steps, keeping only the target window; writes `{save_dir}/{ModelName}_inference_offset{N}_horizon{M}.csv`. |

CLI (`__main__`): `--config` (required), `--horizon` (default: model's
`output_chunk_length`), `--offset` (default `0`), `--no-future-cov` flag.

**Why predict `offset + horizon` and discard the gap**: Darts'
`model.predict(n)` always returns the next `n` consecutive steps
immediately following the context series — there's no native way to ask
for a forecast that starts partway into the future. So `inference.py`
always predicts from the end of the training series, requests
`total_steps = offset + horizon`, and slices off the first `offset` steps
to keep only the desired window.

---

## 3. End-to-end example

```bash
# 1. Decide if the config is good enough (honest test-set metrics + plots)
python single_model_test.py --config configs/TiDE_hid256_dr0.5.yaml

# 2. Retrain on train+val for deployment
python train.py --config configs/TiDE_hid256_dr0.5.yaml

# 3. Forecast the next 336 steps right after training data ends
python inference.py --config configs/TiDE_hid256_dr0.5.yaml --horizon 336

# 3b. Forecast 336 steps starting 24 steps after training data ends
python inference.py --config configs/TiDE_hid256_dr0.5.yaml --horizon 336 --offset 24
```

All three scripts read `data`, `model`, `training` (and optionally
`tensorboard`, `seed`) from the same YAML, so the split fractions, scaling,
and model architecture stay consistent across the backtest, the deployed
checkpoint, and inference.
