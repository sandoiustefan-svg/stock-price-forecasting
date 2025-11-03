from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from LSTM_model import ResidualLSTM
from src.Window_Generator import ResidualWindowGenerator


METHOD_MAP = {
    1: "STL",
    2: "stats_decompose_additive",
    3: "stats_decompose_multiplicative",
}

N_PLOTS = 10


def _ensure_dirs(base: Path) -> None:
    (base / "lstm").mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        (Path("src/results") / "LSTM" / base.name / split).mkdir(parents=True, exist_ok=True)


def _split_dir(method_label: str) -> Path:
    return Path("data/preprocessed") / method_label


def _load_split(base_dir: Path, split: str) -> pd.DataFrame:
    f = base_dir / split / "residuals.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["Series", "date", "residuals"])
    return df.sort_values(["Series", "date"]).reset_index(drop=True)


def _ensure_harmonics(df: pd.DataFrame, period: int = 12, harmonics: int = 3) -> pd.DataFrame:
    out = df.copy()
    if all(c in out.columns for c in ("sin_1", "cos_1")):
        return out

    d = pd.to_datetime(out["date"], errors="coerce")
    if period == 12:
        phase = d.dt.month.astype(float)
    else:
        phase = ((d - d.min()).dt.days % period + 1).astype(float)

    x = (phase - 1.0) / float(period)
    for k in range(1, harmonics + 1):
        theta = 2.0 * np.pi * k * x
        out[f"sin_{k}"] = np.sin(theta)
        out[f"cos_{k}"] = np.cos(theta)
    return out


def _detect_feature_cols(df: pd.DataFrame) -> Tuple[List[str], int]:
    feats = ["residuals"]
    for k in (1, 2, 3):
        s, c = f"sin_{k}", f"cos_{k}"
        if s in df.columns and c in df.columns:
            feats += [s, c]
    return feats, len(feats)


def _evaluate(truth_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    merged = truth_df.merge(pred_df, on=["Series", "date"], how="inner")
    err = merged["residuals"] - merged["resid_pred"]
    eps = 1e-8
    merged["ae"] = err.abs()
    merged["se"] = err.pow(2)
    merged["sape"] = merged["ae"] / (merged["residuals"].abs() + eps)

    per_id = merged.groupby("Series").agg(
        n=("residuals", "size"),
        MAE=("ae", "mean"),
        RMSE=("se", lambda s: float(np.sqrt(np.mean(s)))),
        sMAPE=("sape", "mean"),
    ).reset_index()

    global_row = pd.DataFrame([{
        "Series": "GLOBAL",
        "n": int(merged.shape[0]),
        "MAE": float(merged["ae"].mean()),
        "RMSE": float(np.sqrt(merged["se"].mean())),
        "sMAPE": float(merged["sape"].mean()),
    }])
    return pd.concat([per_id, global_row], ignore_index=True)


def _plot_residual_series(
    series_id: str,
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    out_path: Path,
    split_name: str,
) -> None:
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
    ax.plot(merged["date"], merged["resid_pred"], lw=1.2, ls="--", label="LSTM pred")
    ax.axhline(0.0, ls="--", lw=0.8, alpha=0.6)
    ax.set_title(f"{split_name.upper()} · Series {series_id}")
    ax.set_xlabel("date")
    ax.set_ylabel("z-scored residuals")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    choice = int(input("Decomposition (1=STL, 2=Stats Add., 3=Stats Mult.): ").strip())
    method_label = METHOD_MAP.get(choice)
    if method_label is None:
        raise ValueError("Invalid choice. Use 1, 2, or 3.")

    base_dir = _split_dir(method_label)
    _ensure_dirs(base_dir)

    train_df = _load_split(base_dir, "train")
    val_df   = _load_split(base_dir, "val")
    test_df  = _load_split(base_dir, "test")

    train_df = _ensure_harmonics(train_df, period=12, harmonics=3)
    val_df   = _ensure_harmonics(val_df,   period=12, harmonics=3)
    test_df  = _ensure_harmonics(test_df,  period=12, harmonics=3)

    sample_for_feats = pd.concat([train_df.head(1), val_df.head(1), test_df.head(1)], ignore_index=True)
    feature_cols, n_features = _detect_feature_cols(sample_for_feats)

    T, H = 36, 18
    wg = ResidualWindowGenerator(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        id_col="Series",
        time_col="date",
        feature_cols=tuple(feature_cols),
        label_col="residuals",
        T=T,
        H=H,
    )

    X_train, y_train = wg.build_train_windows()

    X_val_inf, y_val_inf = [], []
    for sid, gtr in train_df.groupby("Series", sort=False):
        gtr = gtr.sort_values("date")
        if len(gtr) < T:
            continue
        x_hist = gtr.loc[:, feature_cols].to_numpy(dtype=float)[-T:, :]
        gval = val_df[val_df["Series"] == sid].sort_values("date")
        if len(gval) < H:
            continue
        y_tgt = gval["residuals"].to_numpy(dtype=float)[:H]

        X_val_inf.append(x_hist[None, :, :])  
        y_val_inf.append(y_tgt[None, :])

    X_val_inf = np.concatenate(X_val_inf, axis=0) if X_val_inf else np.empty((0, T, n_features))
    y_val_inf = np.concatenate(y_val_inf, axis=0) if y_val_inf else np.empty((0, H))

    if X_val_inf.shape[0] > 0:
        X_tr, y_tr = X_train, y_train
        X_val, y_val = X_val_inf, y_val_inf
        print(f"[VAL-INF] Using inference-style validation: X_val={X_val.shape}, y_val={y_val.shape}")
    else:
        split = int(0.9 * len(X_train))
        X_tr, y_tr = X_train[:split], y_train[:split]
        X_val, y_val = X_train[split:], y_train[split:]
        print(f"[VAL-SPLIT] Using 90/10 train split: X_val={X_val.shape}, y_val={y_val.shape}")

    lstm = ResidualLSTM(input_width=T, label_width=H, n_features=n_features)
    lstm.fit(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        epochs=100,
        batch_size=128,
        patience=10,
        verbose=1,
        shuffle=False,
    )

    val_fc  = wg.forecast_val(lstm.model)   
    test_fc = wg.forecast_test(lstm.model)

    lstm_dir = base_dir / "lstm"
    val_metrics  = _evaluate(val_df,  val_fc)
    test_metrics = _evaluate(test_df, test_fc)
    val_metrics.to_csv(lstm_dir / "metrics_val.csv", index=False)
    test_metrics.to_csv(lstm_dir / "metrics_test.csv", index=False)

    out_results_base = Path("src/results") / "LSTM" / base_dir.name
    (out_results_base / "val").mkdir(parents=True, exist_ok=True)
    (out_results_base / "test").mkdir(parents=True, exist_ok=True)
    val_fc.to_csv(out_results_base / "val" / "pred_residuals.csv", index=False)
    test_fc.to_csv(out_results_base / "test" / "pred_residuals.csv", index=False)

    vg = val_metrics.loc[val_metrics["Series"] == "GLOBAL"].iloc[0]
    tg = test_metrics.loc[test_metrics["Series"] == "GLOBAL"].iloc[0]
    print(f"[VAL ] n={int(vg['n'])} | MAE={vg['MAE']:.4f} | RMSE={vg['RMSE']:.4f} | sMAPE={vg['sMAPE']:.4f}")
    print(f"[TEST] n={int(tg['n'])} | MAE={tg['MAE']:.4f} | RMSE={tg['RMSE']:.4f} | sMAPE={tg['sMAPE']:.4f}")

    out_val_dir  = out_results_base / "val"
    out_test_dir = out_results_base / "test"
    ids_all = sorted(
        set(train_df["Series"].astype(str))
        & set(val_df["Series"].astype(str))
        & set(test_df["Series"].astype(str))
    )
    ids_to_plot = ids_all[:N_PLOTS]
    print(f"[PLOTS] Will plot {len(ids_to_plot)} series: {ids_to_plot}")

    for sid in ids_to_plot:
        _plot_residual_series(sid, val_df,  val_fc,  out_val_dir  / f"val_{sid}.png",  "val")
        _plot_residual_series(sid, test_df, test_fc, out_test_dir / f"test_{sid}.png", "test")

    (lstm_dir / "model").mkdir(parents=True, exist_ok=True)
    lstm.save(lstm_dir / "model")


if __name__ == "__main__":
    main()
