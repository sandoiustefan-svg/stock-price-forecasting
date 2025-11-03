from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METHOD = "stats_decompose_multiplicative"   # "STL" | "stats_decompose_additive" | "stats_decompose_multiplicative"
MODEL  = "LSTM"  # "LSTM" | "arima"

def load_scalers(base_dir: Path) -> pd.DataFrame:
    s = pd.read_csv(base_dir / "mean_var_per_id.csv")
    s = s.rename(columns={"mu": "resid_mean", "sigma": "resid_std"})
    s["resid_std"] = s["resid_std"].replace(0.0, 1.0).fillna(1.0)
    return s[["Series", "resid_mean", "resid_std"]]


def _load_one(base_dir: Path, split: str, name: str, col: str) -> pd.DataFrame:
    f = base_dir / split / f"{name}.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")
    df = pd.read_csv(f)
    if "date" not in df.columns or "Series" not in df.columns or col not in df.columns:
        raise KeyError(f"{f} must contain columns: ['Series','date','{col}']")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["Series", "date"])
    return df[["Series", "date", col]]


def load_components(base_dir: Path, split: str) -> pd.DataFrame:
    tr = _load_one(base_dir, split, "trend", "trend")
    se = _load_one(base_dir, split, "seasonal", "seasonal")
    re = _load_one(base_dir, split, "residuals", "residuals").rename(columns={"residuals": "residuals_z"})

    df = tr.merge(se, on=["Series", "date"], how="inner").merge(re, on=["Series", "date"], how="inner")
    return df.sort_values(["Series", "date"]).reset_index(drop=True)


def load_pred(method_label: str, split: str) -> pd.DataFrame:
    if MODEL.lower() == "lstm":
        f = Path("src/results") / "LSTM" / method_label / split / "pred_residuals.csv"
    else:
        f = Path("src/results") / "arima" / method_label / split / "pred_residuals.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing predictions file: {f}")
    df = pd.read_csv(f)
    if not {"Series", "date", "resid_pred"}.issubset(df.columns):
        raise KeyError(f"{f} must contain ['Series','date','resid_pred']")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[["Series", "date", "resid_pred"]].dropna()


def _denorm_residuals(df: pd.DataFrame, scalers: pd.DataFrame, col_in: str, col_out: str) -> pd.DataFrame:
    m = df.merge(scalers, on="Series", how="left")
    if m[["resid_mean", "resid_std"]].isna().any().any():
        missing = m[m["resid_mean"].isna() | m["resid_std"].isna()]["Series"].unique()
        raise RuntimeError(f"Missing μ/σ for IDs: {missing[:10]}")
    m[col_out] = m[col_in] * m["resid_std"] + m["resid_mean"]
    return m.drop(columns=["resid_mean", "resid_std"])


def _compose_y(method_label: str, trend: np.ndarray, seasonal: np.ndarray, resid_raw: np.ndarray) -> np.ndarray:
    if method_label == "stats_decompose_multiplicative":
        return trend * seasonal * resid_raw
    else:
        return np.exp(trend + seasonal + resid_raw)


def compose_and_score(method_label: str,
                      comps: pd.DataFrame,
                      preds_raw: pd.DataFrame,
                      scalers: pd.DataFrame) -> pd.DataFrame:
    true_raw = _denorm_residuals(
        comps.rename(columns={"residuals_z": "resid_z"})[["Series", "date", "resid_z"]],
        scalers,
        col_in="resid_z",
        col_out="resid_true",
    )

    c = comps.merge(true_raw, on=["Series", "date"], how="inner")
    y_true = _compose_y(
        method_label,
        c["trend"].to_numpy(dtype=float),
        c["seasonal"].to_numpy(dtype=float),
        c["resid_true"].to_numpy(dtype=float),
    )
    c["y"] = y_true

    pred_raw = _denorm_residuals(preds_raw, scalers, col_in="resid_pred", col_out="resid_pred_raw")

    m = c.merge(pred_raw, on=["Series", "date"], how="inner")
    m["y_hat"] = _compose_y(
        method_label,
        m["trend"].to_numpy(dtype=float),
        m["seasonal"].to_numpy(dtype=float),
        m["resid_pred_raw"].to_numpy(dtype=float),
    )
    return m.sort_values(["Series", "date"]).reset_index(drop=True)


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-8
    err = df["y"] - df["y_hat"]
    df_ = df.copy()
    df_["ae"] = err.abs()
    df_["se"] = err.pow(2)
    df_["smape"] = 2 * df_["ae"] / (df_["y"].abs() + df_["y_hat"].abs() + eps)

    per_id = df_.groupby("Series").agg(
        n=("y", "size"),
        MAE=("ae", "mean"),
        MSE=("se", "mean"),
        RMSE=("se", lambda s: float(np.sqrt(np.mean(s)))),
        sMAPE=("smape", "mean"),
    ).reset_index()

    g = pd.DataFrame([{
        "Series": "GLOBAL",
        "n": int(df_.shape[0]),
        "MAE": float(df_["ae"].mean()),
        "MSE": float(df_["se"].mean()),
        "RMSE": float(np.sqrt(df_["se"].mean())),
        "sMAPE": float(df_["smape"].mean()),
    }])
    return pd.concat([per_id, g], ignore_index=True)


def plot_series(df: pd.DataFrame, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["date"], df["y"], lw=1.2, label="actual")
    ax.plot(df["date"], df["y_hat"], lw=1.2, ls="--", label="predicted")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(split: str):
    method_label = METHOD
    base_dir = Path("data/preprocessed") / method_label

    scalers = load_scalers(base_dir)
    comps   = load_components(base_dir, split)     
    preds   = load_pred(method_label, split)      

    comp = compose_and_score(method_label, comps, preds, scalers)

    out_data = base_dir / MODEL.lower() / "composed"
    out_data.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_data / f"composed_{split}.csv", index=False)

    m = metrics(comp)
    m.to_csv(out_data / f"metrics_total_{split}.csv", index=False)
    g = m.loc[m["Series"] == "GLOBAL"].iloc[0]
    print(f"[{MODEL} {method_label} {split}] "
          f"n={int(g['n'])}  MAE={g['MAE']:.4f}  MSE={g['MSE']:.4f}  RMSE={g['RMSE']:.4f}  sMAPE={g['sMAPE']:.4f}")

    out_plots = Path("src/results") / MODEL.upper() / method_label / "composed" / split / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)
    for sid in comp["Series"].astype(str).unique()[:12]:
        d = comp[comp["Series"].astype(str) == sid].sort_values("date")
        plot_series(d, out_plots / f"{split}_{sid}.png", f"{method_label} {split.upper()} {sid}")


if __name__ == "__main__":
    for sp in ("val", "test"):
        run(sp)
