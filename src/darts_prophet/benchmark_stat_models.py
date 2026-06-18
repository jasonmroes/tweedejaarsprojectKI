"""
benchmark_stat_models.py

Benchmark statistical and tree-based Darts models using the same
data-loading and validation pipeline as the main training script.

Results are written to:

    benchmark_results.csv

and printed to console.

Usage:

    python benchmark_stat_models.py configs/my_config.yaml
"""

from pathlib import Path
import argparse
import traceback

import pandas as pd
import yaml

from darts.metrics import rmse, mae, smape

from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    ExponentialSmoothing,
    AutoARIMA, #takes very long
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
)


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

            model_rmse = float(rmse(val_ser, pred))
            model_mae = float(mae(val_ser, pred))
            model_smape = float(smape(val_ser, pred))

            results.append({
                "model": model_name,
                "rmse": model_rmse,
                "mae": model_mae,
                "smape": model_smape,
                "status": "OK",
            })

            print(
                f"RMSE={model_rmse:.4f} | "
                f"MAE={model_mae:.4f} | "
                f"SMAPE={model_smape:.4f}"
            )

        except Exception as e:

            print(f"FAILED: {model_name}")
            print(e)

            traceback.print_exc()

            results.append({
                "model": model_name,
                "rmse": None,
                "mae": None,
                "smape": None,
                "status": str(e),
            })

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    results_df = pd.DataFrame(results)

    successful = results_df.dropna(subset=["rmse"])

    if len(successful):

        successful = successful.sort_values(
            "rmse",
            ascending=True,
        )

        print("\n")
        print("=" * 70)
        print("FINAL RANKING")
        print("=" * 70)

        print(
            successful[
                ["model", "rmse", "mae", "smape"]
            ].to_string(index=False)
        )

    save_dir = Path(cfg["training"]["save_dir"])

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = save_dir / "benchmark_results.csv"

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(f"\nResults saved to:\n{output_file}")


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