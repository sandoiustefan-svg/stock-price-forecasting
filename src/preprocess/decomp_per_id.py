import pandas as pd
from statsmodels.tsa.seasonal import STL
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Literal

class Decompose:
    def __init__(self, data: pd.DataFrame, decomp_type: str = "STL"):
        self.data = data
        self.decomp_type = decomp_type  

    def _drop_na(self) -> pd.DataFrame:
        df = self.data.copy()
        df = df.dropna(subset=["Series", "date", "value"])

        return df
    
    def _sort_on_time(self, sequence: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
        sequence = sequence.sort_values("date")
        value = sequence["value"].astype(float).to_numpy()
        return sequence, value

    def check_value_bigger_0_per_id(self):
        df = self._drop_na().copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        value = df.groupby("Series", sort=False)["value"]

        ok_per_id = value.apply(lambda s: (s > 0).all())
        has_negative = value.apply(lambda s: (s < 0).any())
        has_zero = value.apply(lambda s: (s == 0).any())

        all_ok = bool(ok_per_id.all())
        ids_with_neg  = has_negative[has_negative].index.tolist()
        ids_with_zero = has_zero[has_zero].index.tolist()

        if ids_with_neg or ids_with_zero:
            print(f"IDs with negative values (< 0): {len(ids_with_neg)} values")
            print(f"IDs with zeros (== 0): {len(ids_with_zero)} values")
        else:
            print(f"IDs with negative values (< 0): 0 values")
            print(f"IDs with zeros (== 0): 0 values")

        return all_ok, {"negative_ids": ids_with_neg, "zero_ids": ids_with_zero}
    
    def log_transform(self):
        df = self._drop_na().copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["value"] = np.log(df["value"])
        self.data = df
               
    
    def stl_decompose_per_id(
        self,
        robust: bool = True,
        seasonal: int | None = 11,
        trend: int | None = 21,
        period: int = 12,
    ) -> pd.DataFrame:
        """
        Run STL per Series ID and return a single concatenated DataFrame:
        columns = ['Series','date','value','trend','seasonal','residuals'].
        """
        df = self._drop_na()

        out_frames: list[pd.DataFrame] = []

        for series_id, sequence in df.groupby("Series", sort=False):
            sequence, value = self._sort_on_time(sequence=sequence) 

            stl = STL(value, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()

            comp = pd.DataFrame({
                "Series": series_id,
                "date": sequence["date"].values,
                "value": value,
                "trend": stl.trend,
                "seasonal": stl.seasonal,
                "residuals": stl.resid,
            })
            out_frames.append(comp)

        if not out_frames:
            return pd.DataFrame(columns=["Series","date","value","trend","seasonal","residuals"])

        return pd.concat(out_frames, ignore_index=True)
    
    def normal_decompose_per_id(
            self,
            period: int = 12,
            require_positive: bool = True,
            model: Literal["additive", "multiplicative"] = "multiplicative",
            drop_edge_na: bool = True,
            ) -> pd.DataFrame:
        
        if require_positive is None:
            require_positive = (model == "multiplicative")

        df = self._drop_na()

        out_frames: list[pd.DataFrame] = []

        for series_id, sequence in df.groupby('Series', sort=False):
            sequence, value = self._sort_on_time(sequence=sequence)

            if require_positive and np.any(value <= 0):
                raise ValueError(
                    f"Series {series_id}: multiplicative decomposition requires strictly positive values."
                )
            
            try:
                res = seasonal_decompose(
                    value, model=model, period=period, filt=None, two_sided=True, extrapolate_trend=0
                )
            except Exception as e:
                raise RuntimeError(f"Series {series_id}: seasonal_decompose failed: {e}") from e
            
            comp = pd.DataFrame(
            {
                "Series": series_id,
                "date": sequence["date"].values,
                "value": value,
                "trend": res.trend,
                "seasonal": res.seasonal,
                "residuals": res.resid,
            })

            if drop_edge_na:
                comp = comp.dropna(subset=["trend", "residuals"])
            else:
                # s = pd.Series([1.0, None, None, 4.0, None])
                # s.ffill()  # -> [1.0, 1.0, 1.0, 4.0, 4.0]
                # s.bfill()  # -> [1.0, 4.0, 4.0, 4.0, NaN]
                comp[["trend","residuals"]] = comp[["trend","residuals"]].ffill().bfill()
        
            out_frames.append(comp)

        return pd.concat(out_frames, ignore_index=True)
            


