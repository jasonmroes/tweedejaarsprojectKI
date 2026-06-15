import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_inference(
    inference_csv: str,
    data_csv: str,
    target_col: str = "price",
    title: str | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot the inference forecast alongside the preceding known data window.

    The context window shown is forecast_horizon * 2 data points taken from
    the *end* of the training CSV (before the forecast begins).

    Parameters
    ----------
    inference_csv : str
        Path to the CSV produced by inference.py (index = integer step).
    data_csv : str
        Path to the original training CSV so we can show the recent history.
    target_col : str
        Name of the target column in data_csv.
    title : str, optional
        Plot title. Defaults to the inference CSV filename.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    matplotlib Figure (also calls plt.show()).
    """
    # ── Load forecast ──────────────────────────────────────────────────────
    forecast_df = pd.read_csv(inference_csv, index_col=0)
    forecast_col = forecast_df.columns[0]           # e.g. "price"
    forecast_vals = forecast_df[forecast_col].values
    forecast_horizon = len(forecast_vals)

    # ── Load known history ─────────────────────────────────────────────────
    history_df = pd.read_csv(data_csv, low_memory=False)
    if target_col not in history_df.columns:
        raise ValueError(f"Column '{target_col}' not found in {data_csv}.")

    context_len = forecast_horizon * 2
    history_vals = history_df[target_col].dropna().values[-context_len:]

    # Build a continuous integer x-axis:
    #   history occupies steps  [0, context_len)
    #   forecast occupies steps [context_len, context_len + forecast_horizon)
    history_x  = np.arange(len(history_vals))
    forecast_x = np.arange(len(history_vals), len(history_vals) + forecast_horizon)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(history_x,  history_vals,  color="steelblue", linewidth=1.4,
            label=f"Known history (last {context_len} steps)")
    ax.plot(forecast_x, forecast_vals, color="tomato",    linewidth=1.6,
            linestyle="--", label=f"Forecast ({forecast_horizon} steps)")

    # Vertical line at the boundary
    boundary = len(history_vals) - 0.5
    ax.axvline(boundary, color="gray", linewidth=1, linestyle=":")
    ax.text(boundary + 0.3, ax.get_ylim()[1],
            "forecast\nstart", fontsize=8, color="gray", va="top")

    ax.set_xlabel("Step")
    ax.set_ylabel(target_col.capitalize())
    ax.set_title(title or Path(inference_csv).stem)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot an inference forecast against recent known history."
    )
    parser.add_argument(
        "--inference-csv", required=True,
        help="Path to the forecast CSV produced by inference.py.",
    )
    parser.add_argument(
        "--data-csv", required=True,
        help="Path to the original training CSV (for historical context).",
    )
    parser.add_argument(
        "--target-col", default="price",
        help="Name of the target column in the training CSV (default: price).",
    )
    parser.add_argument(
        "--title", default=None,
        help="Optional plot title. Defaults to the inference CSV filename.",
    )
    args = parser.parse_args()

    plot_inference(
        inference_csv=args.inference_csv,
        data_csv=args.data_csv,
        target_col=args.target_col,
        title=args.title,
    )