# Stock Price Forecasting

An end-to-end time-series forecasting pipeline combining **statistical** and **deep learning** models to predict stock prices.  
The approach decomposes each time series into **trend, seasonality, and residuals**, models residual dynamics with both **ARIMA** (per-asset) and a **global LSTM**, and then recomposes predictions back to the original scale for evaluation.

---

## 📄 Research Paper
This repository accompanies the research paper:

**📎 Paper:** [Stock Price Forecasting with Hybrid ARIMA–LSTM Models](Neural_Networks_Project (3).pdf)

The paper details:
- the decomposition strategy,
- the modeling choices (ARIMA vs. global LSTM),
- experimental setup and evaluation,
- and a comparison of forecasting performance across assets.

---

## 🚀 Pipeline Overview

1. **Preprocessing**
   - Decompose each time series (per ID)
   - Train/validation/test split
   - Normalize residuals using train-only statistics

2. **Modeling**
   - **Global LSTM** trained on residuals across all assets
   - **Per-ID ARIMA/SARIMA** baselines on residuals

3. **Recomposition & Evaluation**
   - Recompose predictions (trend + seasonality + residual)
   - Compute forecasting metrics on the original scale

---

## ▶️ How to Run

```bash
# 1) Preprocess (decompose per ID, split, normalize)
python -m preprocess.main_preprocess

# 2a) Train the global LSTM on residuals
python -m model.LSTM_main

# 2b) Train per-ID ARIMA baselines on residuals
python -m model.Arima_main

# 3) Recompose predictions and evaluate
python main.py
