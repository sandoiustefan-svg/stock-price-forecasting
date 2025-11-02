import pandas as pd
import numpy as np

class NormalizeMean:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def compute_mean_variance_per_id(self, ddof: int = 1) -> pd.DataFrame:
        df = self.data[["Series", "residuals"]].copy()
        df["residuals"] = pd.to_numeric(df["residuals"], errors="coerce")
        df = df.dropna(subset=["Series", "residuals"])

        sequence = df.groupby("Series", sort=False)["residuals"]
        stats = sequence.agg(mu="mean", var=lambda s: float(np.var(s.to_numpy(), ddof=ddof)), n="size").reset_index()
        stats["sigma"] = np.sqrt(stats["var"])
        stats["sigma"] = stats["sigma"].replace(0.0, 1e-8)
        return stats[["Series", "mu", "sigma", "n"]]


    def normalize_each_value_per_id(self, per_id_stats: pd.DataFrame):
        df = self.data.copy()
        df["residuals"] = pd.to_numeric(df["residuals"], errors="coerce")

        merged = df.merge(per_id_stats[["Series", "mu", "sigma"]], on="Series", how="left")

        z = (merged["residuals"] - merged["mu"]) / merged["sigma"]
        df["value"] = z

        return df
       

    