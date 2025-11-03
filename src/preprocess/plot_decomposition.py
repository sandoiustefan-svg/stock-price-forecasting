from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DecompPlotter:
    """
    Plot decomposition panels (value, trend, seasonal, residuals) for TRAIN/VAL/TEST,
    and overlay the continuation/reconstruction:
      - STL/additive:   y_hat_log = trend(or trend_fc) + seasonal,  y_hat_lin = exp(y_hat_log)
      - Multiplicative: y_hat     = trend(or trend_fc) * seasonal

    For VAL/TEST this is the forecasted continuation; for TRAIN it's a reconstruction.

    Expected columns:
      TRAIN:    ['Series','date','value','trend','seasonal','residuals']
      VAL/TEST: ['Series','date','value','trend_fc','seasonal','residuals'] (or 'trend' already)
    """

    def __init__(
        self,
        out_root: str = "src/results",
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 8),
    ):
        self.out_root = Path(out_root)
        self.dpi = dpi
        self.figsize = figsize


    def plot_all(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        test_df: Optional[pd.DataFrame],
        method_label: str,
        series_ids: Optional[Iterable[str]] = None,
        max_series: Optional[int] = None,
        log_domain: bool = False,   # True only for STL/additive
    ) -> None:
        base = self._method_dir(method_label)
        (base / "train").mkdir(parents=True, exist_ok=True)
        if val_df is not None:
            (base / "val").mkdir(parents=True, exist_ok=True)
        if test_df is not None:
            (base / "test").mkdir(parents=True, exist_ok=True)

        ids = self._choose_series_ids(train_df, val_df, test_df, series_ids, max_series)
        if len(ids) == 0:
            print("[DecompPlotter] No series IDs to plot.")
            return

        is_multiplicative = (method_label == "stats_decompose_multiplicative")

        self._plot_split(train_df, ids, base / "train", split_name="train",
                         log_domain=log_domain, is_multiplicative=is_multiplicative)
        if val_df is not None:
            self._plot_split(val_df, ids, base / "val", split_name="val",
                             log_domain=log_domain, is_multiplicative=is_multiplicative)
        if test_df is not None:
            self._plot_split(test_df, ids, base / "test", split_name="test",
                             log_domain=log_domain, is_multiplicative=is_multiplicative)

        print(f"[DecompPlotter] Saved plots under: {base}")


    def _method_dir(self, method_label: str) -> Path:
        if method_label == "STL":
            sub = "STL"
        elif method_label == "stats_decompose_additive":
            sub = "additive"
        elif method_label == "stats_decompose_multiplicative":
            sub = "multiplicative"
        else:
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

    def _rmse(self, a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if not m.any():
            return np.inf
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

    def _plot_split(
        self,
        df: pd.DataFrame,
        series_ids: Iterable[str],
        out_dir: Path,
        split_name: str,
        log_domain: bool,
        is_multiplicative: bool,   
    ) -> None:
        needed = {"Series", "date", "seasonal", "residuals"}
        missing = needed - set(df.columns)
        if missing:
            raise KeyError(f"[DecompPlotter] Missing columns in {split_name} df: {missing}")

        trend_col = self._pick_trend_column(df)

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["Series", "date"])

        for sid in series_ids:
            sub = work.loc[work["Series"].astype(str) == str(sid)].sort_values("date").copy()
            if sub.empty:
                continue

            have_value = "value" in sub.columns

            if log_domain:
                yhat_log = pd.to_numeric(sub[trend_col], errors="coerce") + pd.to_numeric(sub["seasonal"], errors="coerce")
                with np.errstate(over="ignore", invalid="ignore"):
                    yhat_lin = np.exp(yhat_log)
            else:
                tr = pd.to_numeric(sub[trend_col], errors="coerce")
                se = pd.to_numeric(sub["seasonal"], errors="coerce")
                yhat_lin = tr * se
                yhat_log = None

            if have_value:
                v = pd.to_numeric(sub["value"], errors="coerce")
                if log_domain:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        v_log = np.log(v)
                    rmse_log = self._rmse(v_log, yhat_log)
                    rmse_lin = self._rmse(v, yhat_lin)
                    if rmse_log <= rmse_lin:
                        y_true, y_hat, val_label = v_log, yhat_log, "value (log)"
                    else:
                        y_true, y_hat, val_label = v, yhat_lin, "value"
                else:
                    y_true, y_hat, val_label = v, yhat_lin, "value"
            else:
                y_true, y_hat, val_label = None, yhat_lin, "value"

            cols_to_check = ["seasonal", trend_col, "residuals"] + (["value"] if have_value else [])
            valid_mask = ~sub[cols_to_check].isna().all(axis=1)
            sub = sub.loc[valid_mask]
            y_hat = pd.Series(y_hat, index=sub.index).loc[valid_mask]
            if y_true is not None:
                y_true = pd.Series(y_true, index=sub.index).loc[valid_mask]
            if sub.empty:
                continue

            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=self.figsize, sharex=True)
            ax_idx = 0

            if y_true is not None:
                axes[ax_idx].plot(sub["date"], y_true, lw=1.2, label="true")
            axes[ax_idx].plot(sub["date"], y_hat, lw=1.2, ls="--",
                              label=("forecast (trend+season)" if log_domain else "forecast (trend×season)"))
            axes[ax_idx].legend(loc="upper left", fontsize=9, frameon=False)
            axes[ax_idx].set_ylabel(val_label)
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

            axes[ax_idx].plot(sub["date"], sub[trend_col], lw=1.2)
            axes[ax_idx].set_ylabel("trend" if trend_col == "trend" else "trend_fc")
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

            axes[ax_idx].plot(sub["date"], sub["seasonal"], lw=1.2)
            axes[ax_idx].set_ylabel("seasonal")
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

            axes[ax_idx].plot(sub["date"], sub["residuals"], lw=1.2)
            baseline = 1.0 if is_multiplicative else 0.0
            axes[ax_idx].axhline(baseline, ls="--", lw=0.8)
            axes[ax_idx].set_ylabel("residual factor" if is_multiplicative else "residuals")
            axes[ax_idx].grid(True, alpha=0.3)

            fig.suptitle(f"{split_name.upper()} · Series {sid}", y=0.98)
            axes[-1].set_xlabel("date")
            fig.tight_layout()

            out_path = out_dir / f"{split_name}_{self._safe_name(sid)}.png"
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
