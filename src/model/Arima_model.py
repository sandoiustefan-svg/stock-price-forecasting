import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from dataclasses import dataclass

@dataclass(frozen=True)
class Spec:
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int] | None

class ArimaBaseline:
    
    def __init__(
            self,
            seasonal: bool = True,
            s: int = 12,
            order_grid: Optional[Iterable[Tuple[int,int,int]]] = None,
            seasonal_grid: Optional[Iterable[Tuple[int,int,int,int]]] = None,
            maxiter: int = 200,
            ):
        
        self.seasonal = seasonal
        self.s = s

        self.order_grid = list(order_grid) if order_grid else [
            (0,0,0), (1,0,0), (0,0,1), (1,0,1), (2,0,1)
        ]

        if self.seasonal:
            self.seasonal_grid = list(seasonal_grid) if seasonal_grid else [
                (0,0,0,s), (1,0,0,s), (0,0,1,s), (1,0,1,s)
            ]
        else:
            self.seasonal_grid = [None]

        self.maxiter = maxiter
        self.models_: Dict[str, object] = {}
        self.specs_: Dict[str, Spec] = {}

    def _best_spec(self, y: pd.Series) ->  Spec | None:
        best = None
        best_aic = np.inf
        for order in self.order_grid:
            for seas in self.seasonal_grid:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        warnings.simplefilter("ignore", category=UserWarning)
                        m = SARIMAX(
                            y, order=order,
                            seasonal_order=(seas or (0,0,0,0)),
                            enforce_stationarity=True,
                            enforce_invertibility=True,
                        ).fit(disp=False, maxiter=self.maxiter)
                    aic = float(m.aic)
                except Exception:
                    continue
                if np.isfinite(aic) and aic < best_aic:
                    best_aic, best = aic, (order, seas)
        
        return Spec(order=best[0], seasonal_order=best[1]) if best else None
    
    def fit(self, train_residuals: pd.DataFrame):
        df = train_residuals.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["Series", "date", "residuals"])

        for id, sequence in df.groupby("Series", sort=False):
            seq = sequence.sort_values("date").copy()
            idx = pd.DatetimeIndex(seq["date"])
            yv  = pd.to_numeric(seq["residuals"], errors="coerce").values
            y   = pd.Series(yv, index=idx)

            freq = pd.infer_freq(y.index)
            if freq is None:
                if (y.index.day == 1).all():
                    freq = "MS"
                elif (y.index.day == y.index.days_in_month).all():
                    freq = "M"
                else:
                    freq = "MS"
            y = y.asfreq(freq).dropna()

            if y.size == 0:
                continue

            spec = self._best_spec(y)
            if spec is None:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                m = SARIMAX(
                    y,
                    order=spec.order,
                    seasonal_order=(spec.seasonal_order or (0, 0, 0, 0)),
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False, maxiter=self.maxiter)

            self.models_[id] = m
            self.specs_[id]  = spec


    def forecast_for_split(self, template_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        template_df: DataFrame with ['Series','date'] rows for the split (VAL/TEST)
        Returns: ['Series','date','resid_pred'] (still z-scored)
        """

        td = template_df.copy()
        td["date"] = pd.to_datetime(td["date"], errors="coerce")
        td = td.dropna(subset=["Series","date"]).sort_values(["Series","date"])

        out_frames = []
        for id, sequence in td.groupby("Series", sort=False):
            h = len(sequence)
            m = self.models_.get(id)

            try:
                fc = m.forecast(steps=h)
                pred = np.asarray(fc, dtype=float)
            except Exception:
                pred = np.zeros(h)
            out_frames.append(pd.DataFrame({"Series": id, "date": sequence["date"].values,
                                    "resid_pred": pred}))
            
        res = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(columns=["Series","date","resid_pred"])
        res["split"] = split_name
        return res

    @staticmethod
    def evaluate(
        truth_df: pd.DataFrame, pred_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        truth_df: ['Series','date','residuals']  (z-scored)
        pred_df:  ['Series','date','resid_pred']
        Returns per-ID metrics and a 'GLOBAL' row.
        """
        merged = truth_df.merge(pred_df, on=["Series","date"], how="inner")
        merged["ae"]  = (merged["residuals"] - merged["resid_pred"]).abs()
        merged["se"]  = (merged["residuals"] - merged["resid_pred"])**2
        merged["sape"] = (merged["residuals"] - merged["resid_pred"]).abs() / (merged["residuals"].abs() + 1e-8)

        per_id = merged.groupby("Series").agg(
            n=("residuals","size"),
            MAE=("ae","mean"),
            RMSE=("se", lambda s: float(np.sqrt(np.mean(s)))),
            sMAPE=("sape","mean"),
        ).reset_index()

        global_row = pd.DataFrame([{
            "Series":"GLOBAL",
            "n": int(merged.shape[0]),
            "MAE": float(merged["ae"].mean()),
            "RMSE": float(np.sqrt(merged["se"].mean())),
            "sMAPE": float(merged["sape"].mean()),
        }])
        return pd.concat([per_id, global_row], ignore_index=True)



        
        