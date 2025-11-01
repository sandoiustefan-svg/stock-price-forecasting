import pandas as pd
import numpy as np
from typing import Literal
from statsmodels.tsa.holtwinters import ExponentialSmoothing 

Method = Literal["STL", "stats_decompose_additive", "stats_decompose_multiplicative"]

class HoldoutDecomposer:
    def __init__(self, train_decomp: pd.DataFrame, method: Method,
                 mean_var_per_id: pd.DataFrame | None, period: int = 12):
        self.method = method
        self.mean_var_per_id = None if mean_var_per_id is None else mean_var_per_id.copy()
        self.period = period

        self.train_decomp = train_decomp.copy()
        self.train_decomp["date"] = pd.to_datetime(self.train_decomp["date"], errors="coerce")
        self.train_decomp = self.train_decomp.dropna(subset=["Series", "date"])

        # build per-ID seasonal template once
        self._seas_tpl = self._build_seasonal_template(self.train_decomp, method=self.method, period=self.period)

    @staticmethod
    def _phase(dates: pd.Series, period: int = 12) -> pd.Series:
        d = pd.to_datetime(dates, errors="coerce")
        return d.dt.month if period == 12 else ((d - d.min()).dt.days % period) + 1

    def _add_sinusoids_to_df(self, df: pd.DataFrame, period: int = 12,
                             harmonics: int = 3, date_col: str = "date",
                             keep_phase: bool = False) -> pd.DataFrame:
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col])
        phase = out[date_col].dt.month if period == 12 else ((out[date_col] - out[date_col].min()).dt.days % period) + 1
        x = (phase - 1) / float(period)
        for k in range(1, harmonics + 1):
            theta = 2.0 * np.pi * k * x
            out[f"sin_{k}"] = np.sin(theta)
            out[f"cos_{k}"] = np.cos(theta)
        if keep_phase:
            out["phase"] = phase
        return out

    def _build_seasonal_template(self, train_decomp: pd.DataFrame, method: Method, period: int = 12) -> pd.DataFrame:
        temp = train_decomp[["Series", "date", "seasonal"]].dropna().copy()
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp["phase"] = self._phase(temp["date"], period)
        pivot = temp.pivot_table(index="Series", columns="phase", values="seasonal", aggfunc="mean")
        if method in ("STL", "stats_decompose_additive"):
            pivot = pivot.subtract(pivot.mean(axis=1), axis=0)
        return pivot.stack().rename("season_tpl").reset_index()

    def _forecast_trend_for(self, target_df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for sid, tr in self.train_decomp.groupby("Series", sort=False):
            horizon = int((target_df["Series"] == sid).sum())
            if horizon == 0:
                continue
            y_trend = pd.to_numeric(tr["trend"], errors="coerce").dropna().to_numpy()
            if y_trend.size == 0:
                fc = np.zeros(horizon)
            else:
                try:
                    model = ExponentialSmoothing(y_trend, trend="add", seasonal=None, damped_trend=True)
                    fit = model.fit(optimized=True)
                    fc = fit.forecast(horizon)
                except Exception:
                    fc = np.full(horizon, float(y_trend[-1]))
            if self.method == "stats_decompose_multiplicative":
                fc = np.maximum(fc, 1e-12)
            t_dates = target_df.loc[target_df["Series"] == sid, "date"].values
            out.append(pd.DataFrame({"Series": sid, "date": t_dates, "trend_fc": fc}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["Series", "date", "trend_fc"])

    def _apply_per_id_z(self, residuals_df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_var_per_id is None:
            return residuals_df
        out = residuals_df.merge(self.mean_var_per_id[["Series", "mu", "sigma"]], on="Series", how="left")
        valid = out["mu"].notna() & out["sigma"].notna() & (out["sigma"] != 0)
        out.loc[valid, "residuals"] = (out.loc[valid, "residuals"] - out.loc[valid, "mu"]) / out.loc[valid, "sigma"]
        return out.drop(columns=["mu", "sigma"])

    def transform(self, target_df: pd.DataFrame, harmonics: int = 3):
        df = target_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["Series", "date", "value"]).sort_values(["Series", "date"])
        df["phase"] = self._phase(df["date"], self.period)
        df = df.merge(self._seas_tpl, on=["Series", "phase"], how="left")          # per-ID seasonal
        df = df.merge(self._forecast_trend_for(df), on=["Series", "date"], how="left")  # per-ID trend

        if self.method in ("STL", "stats_decompose_additive"):
            if (df["value"] <= 0).any():
                raise ValueError("Log-domain path requires strictly positive values in target.")
            y_log = np.log(pd.to_numeric(df["value"], errors="coerce"))
            seas  = pd.to_numeric(df["season_tpl"], errors="coerce").fillna(0.0)
            trnd  = pd.to_numeric(df["trend_fc"],   errors="coerce").ffill().bfill()
            df["seasonal"]         = seas
            df["value_deseasoned"] = y_log - seas
            df["residuals"]        = df["value_deseasoned"] - trnd
        else:  # multiplicative
            y    = pd.to_numeric(df["value"], errors="coerce")
            seas = pd.to_numeric(df["season_tpl"], errors="coerce").fillna(1.0).clip(lower=1e-12)
            trnd = pd.to_numeric(df["trend_fc"],   errors="coerce").ffill().bfill().clip(lower=1e-12)
            df["seasonal"]         = seas
            df["value_deseasoned"] = y / seas
            df["residuals"]        = df["value_deseasoned"] / trnd

        residuals_df = df[["Series", "date", "residuals"]].copy()
        residuals_df = self._apply_per_id_z(residuals_df)
        residuals_df = self._add_sinusoids_to_df(residuals_df, period=self.period, harmonics=harmonics)

        trend_df    = df[["Series", "date", "trend_fc"]].rename(columns={"trend_fc": "trend"})
        seasonal_df = df[["Series", "date", "seasonal"]]
        decomp_full = df[["Series", "date", "value", "phase", "seasonal", "trend_fc", "value_deseasoned"]] \
                        .merge(residuals_df, on=["Series", "date"], how="left")

        return decomp_full, trend_df, seasonal_df, residuals_df
