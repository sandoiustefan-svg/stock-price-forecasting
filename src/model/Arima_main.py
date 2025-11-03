# Arima_main.py
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Arima_model import ArimaBaseline

METHOD_MAP = {
    1: "STL",
    2: "stats_decompose_additive",
    3: "stats_decompose_multiplicative",
}

def _ensure_results_dirs(method_label: str) -> Path:
    """
    Create and return: src/results/arima/<method_label> with subfolders val/ and test/
    """
    base = Path("src/results") / "arima" / method_label
    (base / "val").mkdir(parents=True, exist_ok=True)
    (base / "test").mkdir(parents=True, exist_ok=True)
    return base

def _load_split(base_dir: Path, split: str) -> pd.DataFrame:
    """Load residuals for a split: expects data/preprocessed/<method>/<split>/residuals.csv"""
    f = base_dir / split / "residuals.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["Series", "date", "residuals"])

def _plot_residual_series(
    series_id: str,
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    out_path: Path,
    split_name: str,
) -> None:
    """Plot z-scored residuals: prediction vs truth for one Series and save PNG."""
    merged = (
        truth_df[truth_df["Series"].astype(str) == str(series_id)]
        .merge(
            pred_df[pred_df["Series"].astype(str) == str(series_id)],
            on=["Series", "date"],
            how="inner",
        )
        .sort_values("date")
    )
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(merged["date"], merged["residuals"], lw=1.2, label="truth (z-resid)")
    ax.plot(merged["date"], merged["resid_pred"], lw=1.2, ls="--", label="ARIMA pred")
    ax.axhline(0.0, ls="--", lw=0.8, alpha=0.6)
    ax.set_title(f"{split_name.upper()} · Series {series_id}")
    ax.set_xlabel("date")
    ax.set_ylabel("z-scored residuals")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _evaluate_and_save(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    out_csv: Path,
    split_name: str,
) -> None:
    metrics = ArimaBaseline.evaluate(truth_df, pred_df)
    metrics.to_csv(out_csv, index=False)
    g = metrics.loc[metrics["Series"] == "GLOBAL"].iloc[0]
    print(f"[{split_name}] n={int(g['n'])} | MAE={g['MAE']:.4f} | RMSE={g['RMSE']:.4f} | sMAPE={g['sMAPE']:.4f}")


def main():
    choice = int(input("Decomposition (1=STL, 2=Stats Add., 3=Stats Mult.): ").strip())
    method_label = METHOD_MAP.get(choice)
    if method_label is None:
        raise ValueError("Invalid choice. Use 1, 2, or 3.")

    data_dir = Path("data/preprocessed") / method_label
    train_resid = _load_split(data_dir, "train")
    val_resid   = _load_split(data_dir, "val")
    test_resid  = _load_split(data_dir, "test")

    results_dir = _ensure_results_dirs(method_label)

    arima = ArimaBaseline(
        seasonal=True, s=12,
        order_grid=[(0,0,0),(1,0,0),(0,0,1),(1,0,1),(2,0,1)],
        seasonal_grid=[(0,0,0,12),(1,0,0,12),(0,0,1,12),(1,0,1,12)],
        maxiter=200,
    )
    arima.fit(train_resid)

    val_template  = val_resid[["Series", "date"]].copy()
    test_template = test_resid[["Series", "date"]].copy()

    val_fc  = arima.forecast_for_split(val_template,  split_name="VAL")
    test_fc = arima.forecast_for_split(test_template, split_name="TEST")

    val_fc.to_csv(results_dir / "val" / "pred_residuals.csv", index=False)
    test_fc.to_csv(results_dir / "test" / "pred_residuals.csv", index=False)

    _evaluate_and_save(val_resid,  val_fc,  results_dir / "metrics_val.csv",  "VAL")
    _evaluate_and_save(test_resid, test_fc, results_dir / "metrics_test.csv", "TEST")

    ids = sorted(
        set(train_resid["Series"].astype(str))
        & set(val_resid["Series"].astype(str))
        & set(test_resid["Series"].astype(str))
    )
    ids_to_plot: List[str] = ids[:5] if len(ids) > 24 else ids

    for sid in ids_to_plot:
        _plot_residual_series(
            sid, val_resid, val_fc, results_dir / "val" / f"val_{sid}.png", "val"
        )
        _plot_residual_series(
            sid, test_resid, test_fc, results_dir / "test" / f"test_{sid}.png", "test"
        )

if __name__ == "__main__":
    main()
