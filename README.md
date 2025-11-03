# Stocks-forecasting

# 1) Preprocess (decompose per ID, split, normalize)
python -m preprocess.main_preprocess

# 2a) Train the global LSTM on residuals
python -m model.LSTM_main

# 2b) Train per-ID ARIMA baselines on residuals
python -m model.Arima_main

# 3) Recompose predictions to the original scale + evaluate
python main.py

.
├─ model/
│  ├─ Arima_main.py          # Orchestrates per-ID SARIMA training & forecasting on residuals
│  ├─ Arima_model.py         # Statsmodels SARIMA wrapper + small AIC grid
│  ├─ LSTM_main.py           # Orchestrates global LSTM training & validation on residuals
│  └─ LSTM_model.py          # Keras model definition (direct H-step output head)
│
├─ preprocess/
│  ├─ main_preprocess.py     # END-TO-END PREPROCESS: wide→long, splits, decomposition, scaling
│  ├─ data_formating.py      # Wide CSV → tidy long (Series, date, value)
│  ├─ decomp_per_id.py       # Per-ID decomposition (STL / additive / multiplicative)
│  ├─ holdout_decomposer_per_id.py # Trend continuation + seasonal templating for VAL/TEST
│  ├─ normalize_per_id.py    # Train-only z-score per ID; apply to val/test
│  └─ plot_decomposition.py  # Optional diagnostics plots per ID
│
├─ results/
│  ├─ Window_Generator.py    # Sliding-window builder for LSTM (T × F → H)
│  └─ ...                    # Artifacts: processed tables, predictions, metrics, figures
│
├─ main.py                   # Recompose (trend + season + residual) and compute metrics
├─ pyproject.toml            # Project metadata & dependencies
└─ README.md
