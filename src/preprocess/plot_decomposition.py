# src/plots/decomp_plotter.py
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DecompPlotter:
    """
    Plot decomposition panels (value, trend, seasonal, residuals) for TRAIN/VAL/TEST.
    - Expects per-split DataFrames with columns:
        TRAIN: ['Series','date','value','trend','seasonal','residuals']
        VAL/TEST: either ['Series','date','value','trend','seasonal','residuals']
                  or      ['Series','date','value','trend_fc','seasonal','residuals']
    - Saves one PNG per Series into:
        src/results/STL           (method_label == 'STL')
        src/results/additive      (method_label == 'stats_decompose_additive')
        src/results/multiplicative(method_label == 'stats_decompose_multiplicative')
      with subfolders train/, val/, test/
    """

    def __init__(
        self,
        out_root: str = "src/results",
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 8)
    ):
        self.out_root = Path(out_root)
        self.dpi = dpi
        self.figsize = figsize

    # ---------- public API ----------

    def plot_all(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        test_df: Optional[pd.DataFrame],
        method_label: str,
        series_ids: Optional[Iterable[str]] = None,
        max_series: Optional[int] = None,
        log_domain: bool = False,
    ) -> None:
        base = self._method_dir(method_label)
        (base / "train").mkdir(parents=True, exist_ok=True)
        if val_df is not None:
            (base / "val").mkdir(parents=True, exist_ok=True)
        if test_df is not None:
            (base / "test").mkdir(parents=True, exist_ok=True)

        # Decide which IDs to plot (intersection across provided splits)
        ids = self._choose_series_ids(train_df, val_df, test_df, series_ids, max_series)

        if len(ids) == 0:
            print("[DecompPlotter] No series IDs to plot.")
            return

        # Plot each split
        self._plot_split(train_df, ids, base / "train", split_name="train", log_domain=log_domain)
        if val_df is not None:
            self._plot_split(val_df, ids, base / "val", split_name="val", log_domain=log_domain)
        if test_df is not None:
            self._plot_split(test_df, ids, base / "test", split_name="test", log_domain=log_domain)

        print(f"[DecompPlotter] Saved plots under: {base}")

    # ---------- internals ----------

    def _method_dir(self, method_label: str) -> Path:
        # Map to folder names
        if method_label == "STL":
            sub = "STL"
        elif method_label == "stats_decompose_additive":
            sub = "additive"
        elif method_label == "stats_decompose_multiplicative":
            sub = "multiplicative"
        else:
            # fallback: use label itself
            sub = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(method_label)).strip("_")
        return self.out_root / sub

    def _choose_series_ids(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        test_df: Optional[pd.DataFrame],
        series_ids: Optional[Iterable[str]],
        max_series: Optional[int],
    ) -> list:
        def ids_in(df: Optional[pd.DataFrame]) -> set:
            if df is None or "Series" not in df.columns:
                return set()
            return set(map(str, pd.unique(df["Series"].dropna())))

        if series_ids is not None:
            chosen = list(dict.fromkeys(map(str, series_ids)))
        else:
            # start from TRAIN, then keep only those present in other splits (if supplied)
            chosen_set = ids_in(train_df)
            if val_df is not None:
                chosen_set &= ids_in(val_df)
            if test_df is not None:
                chosen_set &= ids_in(test_df)
            chosen = sorted(chosen_set)

        if max_series is not None and len(chosen) > max_series:
            chosen = chosen[:max_series]
        return chosen

    def _safe_name(self, s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s)).strip("_")

    def _pick_trend_column(self, df: pd.DataFrame) -> str:
        if "trend" in df.columns:
            return "trend"
        if "trend_fc" in df.columns:
            return "trend_fc"
        raise KeyError("No 'trend' or 'trend_fc' column found for plotting.")

    def _plot_split(
        self,
        df: pd.DataFrame,
        series_ids: Iterable[str],
        out_dir: Path,
        split_name: str,
        log_domain: bool,
    ) -> None:
        # basic checks
        needed = {"Series", "date", "seasonal", "residuals"}
        missing = needed - set(df.columns)
        if missing:
            raise KeyError(f"[DecompPlotter] Missing columns in {split_name} df: {missing}")

        trend_col = self._pick_trend_column(df)

        # ensure datetime
        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["Series", "date"])

        for sid in series_ids:
            sub = work.loc[work["Series"].astype(str) == str(sid)].sort_values("date").copy()
            if sub.empty:
                continue

            # Drop rows with all-NaN on the plotted components to avoid blank panels
            # (value may be missing in some edge cases; handle gracefully)
            have_value = "value" in sub.columns
            cols = ["seasonal", trend_col, "residuals"] + (["value"] if have_value else [])
            valid_mask = ~sub[cols].isna().all(axis=1)
            sub = sub.loc[valid_mask]
            if sub.empty:
                continue

            # Build figure
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=self.figsize, sharex=True)
            ax_idx = 0

            # Panel 1: Original value (if present)
            if have_value:
                axes[ax_idx].plot(sub["date"], sub["value"], lw=1.2)
                axes[ax_idx].set_ylabel("value" + (" (log)" if log_domain else ""))
                axes[ax_idx].grid(True, alpha=0.3)
                ax_idx += 1
            else:
                # if no original value, leave an empty panel with message
                axes[ax_idx].text(0.5, 0.5, "value not provided", ha="center", va="center", transform=axes[ax_idx].transAxes)
                axes[ax_idx].set_axis_off()
                ax_idx += 1

            # Panel 2: Trend
            axes[ax_idx].plot(sub["date"], sub[trend_col], lw=1.2)
            axes[ax_idx].set_ylabel("trend" if trend_col == "trend" else "trend_fc")
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

            # Panel 3: Seasonal
            axes[ax_idx].plot(sub["date"], sub["seasonal"], lw=1.2)
            axes[ax_idx].set_ylabel("seasonal")
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

            # Panel 4: Residuals
            axes[ax_idx].plot(sub["date"], sub["residuals"], lw=1.2)
            axes[ax_idx].axhline(0.0, ls="--", lw=0.8)
            axes[ax_idx].set_ylabel("residuals")
            axes[ax_idx].grid(True, alpha=0.3)

            fig.suptitle(f"{split_name.upper()} · Series {sid}", y=0.98)
            axes[-1].set_xlabel("date")
            fig.tight_layout()

            # save
            out_path = out_dir / f"{split_name}_{self._safe_name(sid)}.png"
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
