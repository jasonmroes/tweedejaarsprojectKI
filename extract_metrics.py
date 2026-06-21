#!/usr/bin/env python3
import json
import csv
from pathlib import Path

# Define the metrics to extract
METRICS = [
    "smape",
    "rmse_norm",
    "mae_norm",
    "mae_std_per_day_norm",
    "smape_nonzero",
    "rmse_norm_nonzero",
    "mae_norm_nonzero",
    "mae_std_per_day_norm_nonzero",
]

# Path to results directory
results_dir = Path("experiments/results")

# Collect all metrics
data = []

for subdir in sorted(results_dir.iterdir()):
    if not subdir.is_dir():
        continue

    json_file = subdir / "TiDEModel_backtest_metrics.json"

    if not json_file.exists():
        print(f"Warning: {json_file} not found, skipping {subdir.name}")
        continue

    with open(json_file, "r") as f:
        metrics = json.load(f)

    # Extract the required metrics
    row = {"Model": subdir.name}
    for metric in METRICS:
        row[metric] = metrics.get(metric, "N/A")

    data.append(row)

# Write to CSV
output_file = Path("metrics_summary.csv")
if data:
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model"] + METRICS)
        writer.writeheader()
        writer.writerows(data)
    print(f"✓ Metrics saved to {output_file}")
else:
    print("No data found to write")
