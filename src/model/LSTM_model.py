# src/model/residual_lstm.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

class ResidualLSTM:
    """
    Direct multi-step LSTM forecaster for residuals (global model).
    - Input shape per sample:  (input_width, n_features)
    - Output shape per sample: (label_width,)   # direct multi-horizon
    """

    def __init__(self, input_width: int, label_width: int, n_features: int = 1):
        self.input_width = int(input_width)
        self.label_width = int(label_width)
        self.n_features = int(n_features)
        self.model: tf.keras.Model | None = None 

    def build(self) -> tf.keras.Model:
        inp = tf.keras.layers.Input(shape=(self.input_width, self.n_features))
        x = inp

        # LSTM block 1
        x = tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        # LSTM block 2
        x = tf.keras.layers.LSTM(
            64,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
            recurrent_dropout=0.1,
        )(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        # small dense head before the horizon
        x = tf.keras.layers.Dense(128, activation="relu",
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        out = tf.keras.layers.Dense(self.label_width, name="residuals_hat")(x)

        self.model = tf.keras.models.Model(inputs=inp, outputs=out, name="ResidualLSTM")

        # robust loss + smaller LR + gradient clipping
        huber = tf.keras.losses.Huber(delta=1.0)
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)

        self.model.compile(optimizer=opt, loss=huber, metrics=["mae"])
        return self.model





    def fit(
        self,
        X_train: np.ndarray | tf.Tensor,
        y_train: np.ndarray | tf.Tensor,
        X_val: np.ndarray | tf.Tensor | None = None,
        y_val: np.ndarray | tf.Tensor | None = None,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 20,
        verbose: int = 1,
        shuffle: bool = False,   
    ) -> tf.keras.callbacks.History:
        """
        Train on pre-windowed arrays/tensors.
        X_* shapes: (batch, input_width, n_features)
        y_* shapes: (batch, label_width)
        """
        if self.model is None:
            self.build()

        cbs: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, patience // 3), min_lr=1e-6),
        ]

        val = (X_val, y_val) if (X_val is not None and y_val is not None) else None

        return self.model.fit(
            X_train, y_train,
            validation_data=val,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=verbose,
            callbacks=cbs,
        )


    def predict(self, X: np.ndarray | tf.Tensor, batch_size: int = 256) -> np.ndarray:
        """Return predictions with shape (batch, label_width)."""
        if self.model is None:
            raise RuntimeError("Model is not built. Call build() or fit() first.")
        return np.asarray(self.model.predict(X, batch_size=batch_size, verbose=0))


    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute MAE, RMSE, sMAPE (denominator uses |y_true| + eps, suited for residuals).
        Inputs: y_true, y_pred shape (batch, label_width)
        """
        eps = 1e-8
        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        smape = float(np.mean(np.abs(err) / (np.abs(y_true) + eps)))
        return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}

    def evaluate_arrays(self, X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> Dict[str, float]:
        """Predict on X and compute metrics vs y."""
        y_hat = self.predict(X, batch_size=batch_size)
        return self._metrics(y, y_hat)


    def save(self, folder: str | Path) -> None:
        """Save the model and its minimal metadata."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Nothing to save: model is not built.")
        self.model.save(folder / "model.keras")
        meta = {
            "input_width": self.input_width,
            "label_width": self.label_width,
            "n_features": self.n_features,
        }
        import json
        with open(folder / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, folder: str | Path) -> "ResidualLSTM":
        """Load the model and metadata saved via `save()`."""
        folder = Path(folder)
        import json
        with open(folder / "meta.json", "r") as f:
            meta = json.load(f)
        inst = cls(**meta)
        inst.model = tf.keras.models.load_model(folder / "model.keras")
        return inst
