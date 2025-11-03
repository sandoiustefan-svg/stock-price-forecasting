from __future__ import annotations
from typing import Sequence, Tuple, Optional, List
import numpy as np
import pandas as pd


class ResidualWindowGenerator:
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        test_df: Optional[pd.DataFrame],
        *,
        id_col: str = "Series",
        time_col: str = "date",
        feature_cols: Sequence[str] = ("residuals", "sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3"),
        label_col: str = "residuals",
        T: int = 36,
        H: int = 18,
    ):
        self.id_col = id_col
        self.time_col = time_col
        self.feature_cols = tuple(feature_cols)
        self.label_col = label_col
        self.T = int(T)
        self.H = int(H)

        self.train = self._prep_df(train_df)
        self.val   = self._prep_df(val_df)   if val_df is not None else None
        self.test  = self._prep_df(test_df)  if test_df is not None else None

        self._require_columns(self.train, need={self.id_col, self.time_col, self.label_col, *self.feature_cols})
        if self.val is not None:
            self._require_columns(self.val, need={self.id_col, self.time_col, self.label_col, *self.feature_cols})
        if self.test is not None:
            self._require_columns(self.test, need={self.id_col, self.time_col, self.label_col, *self.feature_cols})

    def build_train_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for sid, g in self.train.groupby(self.id_col, sort=False):
            g = g.sort_values(self.time_col)
            Xf = g.loc[:, self.feature_cols].to_numpy(dtype=float, copy=False)
            yv = g.loc[:, self.label_col].to_numpy(dtype=float, copy=False)

            n = len(g)
            for t in range(self.T, n - self.H + 1):
                X_list.append(Xf[t - self.T : t, :])
                y_list.append(yv[t : t + self.H])

        X = np.stack(X_list) if X_list else np.empty((0, self.T, len(self.feature_cols)), dtype=float)
        y = np.stack(y_list) if y_list else np.empty((0, self.H), dtype=float)
        return X, y


    def forecast_val(self, model) -> pd.DataFrame:
        if self.val is None:
            return pd.DataFrame(columns=[self.id_col, self.time_col, "resid_pred"])

        out_frames: List[pd.DataFrame] = []
        for sid, g_val in self.val.groupby(self.id_col, sort=False):
            X_in = self._history_input(self.train, sid)
            if X_in is None:
                continue
            yhat = model.predict(X_in, verbose=0)  # (1, H) expected
            yhat = np.asarray(yhat).reshape(-1)
            dates = g_val.sort_values(self.time_col)[self.time_col].values
            h = min(len(dates), self.H)
            out_frames.append(pd.DataFrame({
                self.id_col: sid,
                self.time_col: dates[:h],
                "resid_pred": yhat[:h]
            }))
        return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(columns=[self.id_col, self.time_col, "resid_pred"])

    def forecast_test(self, model) -> pd.DataFrame:
        if self.test is None:
            return pd.DataFrame(columns=[self.id_col, self.time_col, "resid_pred"])

        out_frames: List[pd.DataFrame] = []
        for sid, g_test in self.test.groupby(self.id_col, sort=False):
            hist = self._concat_hist(self.train, self.val, sid)
            X_in = self._history_input(hist, sid, already_filtered=True)
            if X_in is None:
                continue
            yhat = model.predict(X_in, verbose=0)  # (1, H) expected
            yhat = np.asarray(yhat).reshape(-1)
            dates = g_test.sort_values(self.time_col)[self.time_col].values
            h = min(len(dates), self.H)
            out_frames.append(pd.DataFrame({
                self.id_col: sid,
                self.time_col: dates[:h],
                "resid_pred": yhat[:h]
            }))
        return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(columns=[self.id_col, self.time_col, "resid_pred"])


    def _prep_df(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        out = df.copy()
        out[self.time_col] = pd.to_datetime(out[self.time_col], errors="coerce")
        out = out.dropna(subset=[self.id_col, self.time_col])
        return out.sort_values([self.id_col, self.time_col])

    @staticmethod
    def _require_columns(df: pd.DataFrame, need: set) -> None:
        missing = need - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

    def _history_input(self, df: pd.DataFrame, sid, already_filtered: bool = False) -> Optional[np.ndarray]:
        if not already_filtered:
            g = df.loc[df[self.id_col] == sid].sort_values(self.time_col)
        else:
            g = df.sort_values(self.time_col)  
        if g.empty or len(g) < self.T:
            return None
        Xf = g.loc[:, self.feature_cols].to_numpy(dtype=float, copy=False)
        return Xf[-self.T:, :][None, :, :]

    def _concat_hist(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], sid) -> pd.DataFrame:
        if val_df is None:
            return train_df.loc[train_df[self.id_col] == sid]
        return (
            pd.concat(
                [
                    train_df.loc[train_df[self.id_col] == sid],
                    val_df.loc[val_df[self.id_col] == sid],
                ],
                ignore_index=True,
            )
        )
