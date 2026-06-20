"""
benchmark_stat_models.py

Benchmark statistical and tree-based Darts models using the same
data-loading and validation pipeline as the main training script.

Results are written to:

    benchmark_results.csv  (full metrics, one row per model)
    metrics_summary.csv    (same column format as extract_metrics.py's
                            output for the global models, for cross-comparison)

and printed to console.

Usage:

    python benchmark_stat_models.py configs/my_config.yaml
"""

from pathlib import Path
import argparse
import csv
import traceback

import pandas as pd
import yaml

from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    ExponentialSmoothing,
    # AutoARIMA, #takes very long
    LinearRegressionModel,
    #RandomForest, Takes very long
    LightGBMModel,
    XGBModel,
)

from model_utils import (
    load_all_series,
    build_fit_kwargs,
    build_predict_kwargs,
    compute_split_sizes,
    compute_extended_metrics,
    soft_clip,
)


# Same columns/order as extract_metrics.py's metrics_summary.csv for the
# global (deep-learning) models, so both can be compared side by side.
METRICS_SUMMARY_COLUMNS = [
    "smape",
    "rmse_norm",
    "mae_norm",
    "mae_std_per_day_norm",
    "smape_nonzero",
    "rmse_norm_nonzero",
    "mae_norm_nonzero",
    "mae_std_per_day_norm_nonzero",
]


# ------------------------------------------------------------------
# Candidate Models
# ------------------------------------------------------------------

def build_models(cfg):

    input_chunk = cfg["model"]["input_chunk_length"]
    output_chunk = cfg["model"]["output_chunk_length"]

    return {

        # Baselines
        "NaiveDaily": lambda: NaiveSeasonal(K=24),
        "NaiveWeekly": lambda: NaiveSeasonal(K=168),
        "NaiveDrift": lambda: NaiveDrift(),

        # Classical
        "ExponentialSmoothing": lambda: ExponentialSmoothing(),
        "AutoARIMA": lambda: AutoARIMA(),

        # Tree / ML
        "LinearRegression": lambda: LinearRegressionModel(
            lags=input_chunk,
            output_chunk_length=output_chunk,
        ),

        "RandomForest": lambda: RandomForest(
            lags=input_chunk,
            output_chunk_length=output_chunk,
            n_estimators=200,
        ),

        "LightGBM": lambda: LightGBMModel(
            lags=input_chunk,
            output_chunk_length=output_chunk,
        ),

        "XGBoost": lambda: XGBModel(
            lags=input_chunk,
            output_chunk_length=output_chunk,
        ),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def benchmark(config_path: str):

    print(f"\nLoading config: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("\nLoading data...")

    series, scaler, past_cov, future_cov, _, _ = load_all_series(cfg)

    # Same boundary convention as load_all_series()/train.py: [train][val][test].
    # The test region is carved out and left unused here (this script never
    # evaluates a held-out test set — its rmse/mae/smape are val metrics).
    val_size, test_size = compute_split_sizes(len(series), cfg)

    if val_size <= 0:
        raise ValueError("Validation set size is zero.")

    train_full = series[:-test_size] if test_size > 0 else series
    train_ser = train_full[:-val_size]
    val_ser = train_full[-val_size:]

    # --------------------------------------------------------------
    # Covariate splits
    # --------------------------------------------------------------

    past_cov_train = None
    past_cov_val = None

    if past_cov is not None:
        past_cov_train = past_cov.slice_intersect(train_ser)
        past_cov_val = past_cov.slice_intersect(val_ser)

    fut_cov_train = None
    fut_cov_val = None

    if future_cov is not None:
        fut_cov_train = future_cov.slice_intersect(train_ser)
        fut_cov_val = future_cov.slice_intersect(val_ser)

    print("\nDataset summary")
    print(f"Train length: {len(train_ser)}")
    print(f"Val length  : {len(val_ser)}")

    soft_clip_margin = cfg["training"].get("soft_clip_margin", 0.2)

    results = []

    models = build_models(cfg)

    # --------------------------------------------------------------
    # Benchmark loop
    # --------------------------------------------------------------

    for model_name, model_factory in models.items():

        print("\n" + "=" * 70)
        print(f"Training: {model_name}")
        print("=" * 70)

        try:

            model = model_factory()

            fit_kw = build_fit_kwargs(
                model,
                train_ser,
                val_ser,
                past_cov_train,
                fut_cov_train,
                past_cov_val,
                fut_cov_val,
            )

            model.fit(**fit_kw)

            pred_kw = build_predict_kwargs(
                model,
                len(val_ser),
                train_ser,
                past_cov_train,
                fut_cov_train,
            )

            pred = model.predict(**pred_kw)

            # Align lengths (model predictions may be shorter than val_ser)
            n = min(len(pred), len(val_ser))
            pred_norm_vals = pred.values()[:n, 0]
            val_norm_vals = val_ser.values()[:n, 0]

            pred_inv = scaler.inverse_transform(soft_clip(pred, margin=soft_clip_margin))
            val_inv = scaler.inverse_transform(val_ser[:n])

            pred_inv_vals = pred_inv.values()[:, 0]
            val_inv_vals = val_inv.values()[:, 0]

            metrics = compute_extended_metrics(
                pred_norm_vals, val_norm_vals,
                pred_inv_vals,  val_inv_vals,
                verbose=False,
            )

            print(
                f"SMAPE={metrics['smape']:.4f} | "
                f"RMSE(price)={metrics['rmse_price']:.4f} | "
                f"MAE(price)={metrics['mae_price']:.4f} | "
                f"RMSE(norm)={metrics['rmse_norm']:.6f} | "
                f"MAE(norm)={metrics['mae_norm']:.6f}"
            )

            if "smape_nonzero" in metrics:
                print(
                    f"  Non-zero price only ({metrics['n_nonzero_steps']}/{metrics['n_steps_evaluated']} steps): "
                    f"SMAPE={metrics['smape_nonzero']:.4f} | "
                    f"RMSE(price)={metrics['rmse_price_nonzero']:.4f} | "
                    f"MAE(price)={metrics['mae_price_nonzero']:.4f}"
                )

            # Daily breakdown lists aren't useful in a flat comparison table.
            scalar_metrics = {
                k: v for k, v in metrics.items() if not isinstance(v, list)
            }

            results.append({
                "model": model_name,
                "status": "OK",
                **scalar_metrics,
            })

        except Exception as e:

            print(f"FAILED: {model_name}")
            print(e)

            traceback.print_exc()

            results.append({
                "model": model_name,
                "status": str(e),
            })

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    results_df = pd.DataFrame(results)

    successful = results_df.dropna(subset=["rmse_price"])

    if len(successful):

        successful = successful.sort_values(
            "rmse_price",
            ascending=True,
        )

        print("\n")
        print("=" * 70)
        print("FINAL RANKING (sorted by RMSE, price units)")
        print("=" * 70)

        display_cols = [
            col for col in [
                "model",
                "rmse_price", "mae_price", "smape",
                "rmse_price_nonzero", "mae_price_nonzero", "smape_nonzero",
                "rmse_norm", "mae_norm",
            ]
            if col in successful.columns
        ]

        print(
            successful[display_cols].to_string(index=False)
        )


    output_file = "stats_benchmark_results.csv"

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(f"\nResults saved to:\n{output_file}")

    # --------------------------------------------------------------
    # metrics_summary.csv — same column format as extract_metrics.py
    # produces for the global models, for direct cross-comparison.
    # --------------------------------------------------------------

    summary_rows = [
        {
            "Model": r["model"],
            **{col: r.get(col, "N/A") for col in METRICS_SUMMARY_COLUMNS},
        }
        for r in results
        if r.get("status") == "OK"
    ]

    if summary_rows:

        summary_file = "stats_metrics_summary.csv"

        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Model"] + METRICS_SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"Metrics summary saved to:\n{summary_file}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config",
    )

    args = parser.parse_args()

    benchmark(args.config)