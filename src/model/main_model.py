import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Window_Generator import WindowGenerator
from util_cross_val import make_cross_split_val_ds

import tensorflow as tf
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Data ----------------
train_resid = pd.read_csv("/Users/bogdansandoiu/Documents/Neural Networks/Stocks Forecasting/Stocks-forecasting/data/preprocessed/stl/train/residuals.csv")
val_resid   = pd.read_csv("/Users/bogdansandoiu/Documents/Neural Networks/Stocks Forecasting/Stocks-forecasting/data/preprocessed/stl/val/residuals.csv")
test_resid  = pd.read_csv("/Users/bogdansandoiu/Documents/Neural Networks/Stocks Forecasting/Stocks-forecasting/data/preprocessed/stl/test/residuals.csv")

DROP_COLS = ["d_residuals", "d-residuals"]
for df in (train_resid, val_resid, test_resid):
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

w = WindowGenerator(
    input_width=36, label_width=18, shift=18,      # 36 -> 18 horizon
    train_df=train_resid, val_df=val_resid, test_df=test_resid,  # we won't use w.val directly
    label_columns=['residuals'],
    id_col='Series',
    time_col='date',
    batch_size=32,
    shuffle=False
)

train_ds = w.train
n_ids_total = train_resid["Series"].nunique()
print(f"TRAIN IDs total: {n_ids_total}")

# Cross-split VAL: last 36 from TRAIN + first 18 from VAL (1 per eligible ID)
val_ds, used_ids_val, skipped_val = make_cross_split_val_ds(w, train_resid, val_resid)
print(f"VAL windows total (cross-split): {len(used_ids_val)}  (1 per eligible ID)")
print(f"VAL skipped IDs: {skipped_val if skipped_val else 'none'}\n")

# Optional pipeline boosts
train_ds_perf = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds_perf   = val_ds.cache().prefetch(tf.data.AUTOTUNE)

tf.random.set_seed(42)
np.random.seed(42)

n_features = len(w.feature_columns)
n_labels   = len(w.label_columns)
L          = w.label_width

inputs = tf.keras.Input(shape=(w.input_width, n_features))  # (36, F)
x = tf.keras.layers.LSTM(48, return_sequences=False)(inputs)
x = tf.keras.layers.RepeatVector(L)(x)                      # 18 steps
x = tf.keras.layers.LSTM(48, return_sequences=True)(x)
x = tf.keras.layers.LayerNormalization()(x)                 # light norm
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_labels))(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
    loss="mse",
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae_all_dims")]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
]

history = model.fit(
    train_ds_perf,
    validation_data=val_ds_perf,
    epochs=100,
    verbose=2,
    callbacks=callbacks,
)

val_loss, val_mae = model.evaluate(val_ds_perf, verbose=0)
print(f"Final VAL — loss (MSE/Huber): {val_loss:.4f} | MAE: {val_mae:.4f}")

# ---------------- Predictions & Plot (VAL) ----------------
y_true_all, y_pred_all = [], []
for xb, yb in val_ds:
    yhat = model.predict(xb, verbose=0)
    y_true_all.append(yb.numpy())
    y_pred_all.append(yhat)

y_true_all = np.concatenate(y_true_all, axis=0)   # (M, 18, n_labels)
y_pred_all = np.concatenate(y_pred_all, axis=0)   # (M, 18, n_labels)
print("VAL arrays shapes -> y_true:", y_true_all.shape, " y_pred:", y_pred_all.shape)

# Plot first few IDs: GT vs Pred over the 18-step horizon
k = min(6, y_true_all.shape[0])  # show up to 6 series
h = np.arange(1, L + 1)

fig, axes = plt.subplots(k, 1, figsize=(8, 2.6 * k), sharex=True)
if k == 1:
    axes = [axes]

for i in range(k):
    ax = axes[i]
    gt = y_true_all[i, :, 0]   # assumes n_labels == 1 ('residuals')
    pr = y_pred_all[i, :, 0]
    ax.plot(h, gt, marker='o', label=f"GT — ID {used_ids_val[i]}")
    ax.plot(h, pr, marker='x', linestyle='--', label="Prediction")
    ax.set_ylabel("residuals")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Horizon (steps ahead)")
axes[0].set_title("Validation — Ground Truth vs Prediction (first few IDs, 36→18)")
plt.tight_layout()

# Save plot to src/results_predictions/
outdir = (Path(__file__).resolve().parents[1] / "results_predictions")
outdir.mkdir(parents=True, exist_ok=True)
plot_path = outdir / f"val_gt_vs_pred_first6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot -> {plot_path}")

plt.show()
