# S&P 500 Directional Trading with xLSTM (Experiment ONLY)

PRECAUTION: This repo is only an experiment fueled by my curiosity please note the model's performance is basically negligible...

A walk-forward ensemble system that predicts next-day S&P 500 direction (up/down) using an **xLSTM** architecture with **causal wavelet denoising**, multimodal feature engineering (technical, macro, sentiment, VIX), and quarterly-retrained walk-forward backtesting from 2021–2024.

---

## Pipeline Overview

```
Raw OHLCV Data (yfinance)
        │
        ▼
┌───────────────────────────┐
│  00 — NYT Headline Update │  ← NYT News Headlines Archive API (Used to  update news data only)
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  01 — Wavelet Denoising   │  ← Causal db8, level-4, soft threshold
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  02 — Feature Engineering │  ← 50+ Features
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  03 — xLSTM Model         │  ← mLSTM + sLSTM blocks
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  04 — Walk-Forward Train  │  ← Quarterly retrain, no lookahead
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  05 — Evaluation          │  ← Accuracy, Sharpe, drawdown
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  06 — Next-Day Inference  │  ← Live prediction
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  07 — DCA Comparison      │  ← vs Buy & Hold, Long/Short, Long/Flat
└───────────────────────────┘
```

---

## Theory

### Causal Wavelet Denoising

Raw financial time series are contaminated with high-frequency noise (microstructure, random fluctuations) that obscures the underlying price signal. Wavelet transforms decompose a signal into multiple frequency bands (approximation + detail coefficients). By thresholding the detail coefficients, noise is suppressed while preserving the trend.

**Why causal?** Standard wavelet denoising is symmetric—it looks both forward and backward in time. That introduces lookahead bias. This pipeline uses a *causal* implementation: each timestep is denoised using only past data within a sliding window (lookback = 256 days). This guarantees no future information leaks into the denoised signal.

**Configuration:** Daubechies-8 (`db8`) wavelet, 4th-level decomposition, soft thresholding with the Universal (VisuShrink) threshold, where the threshold σ is estimated via the Median Absolute Deviation (MAD) of the finest detail coefficients.

### xLSTM Architecture

The **extended LSTM (xLSTM)** was introduced by Beck et al. (2024) as an architectural evolution over the original LSTM. It combines two block types:

- **mLSTM (Matrix LSTM):** Replaces the scalar cell state with a matrix-valued memory
  $$C_t = f_t \odot C_{t-1} + i_t \, (v_t \, k_t^\top),$$
  enabling the model to store and retrieve key-value associations. This gives exponential memory capacity. The block computes
  $$h_t = q_t^\top C_t,$$
  which is a differentiable attention-like retrieval over the stored matrix.

- **sLSTM (Scalar LSTM):** Retains the scalar cell structure but introduces **exponential gating**
  $$i_t = \exp(\tilde{i}_t), \quad f_t = \exp(\tilde{f}_t),$$
  which allows the forget gate to completely flush or retain memory. A stabilizer normalizer $n_t$ prevents numerical explosion. Combined with a gated MLP residual, this block provides stable, sharp state transitions.

Both blocks use:
- **Causal convolution** (kernel size 4) for local temporal context before gating.
- **Group normalization** for stable training across varied input distributions.
- **Gated MLP residuals** (SwiGLU style) for capacity.

The model alternates mLSTM and sLSTM blocks (2 blocks total, configurable). A sinusoidal positional encoding is added to embed sequence order. The final timestep's representation is passed through a LayerNorm + MLP head to produce a single logit for binary direction classification.

### Stationary Feature Engineering

All features are strictly **stationary** (ratios, returns, or bounded oscillators). No absolute price or volume levels are used as model inputs—this prevents the model from memorizing price regimes rather than learning dynamics. Features include:

| Category | Features |
|----------|----------|
| **Returns** | Log return, daily return, multi-horizon returns (1d, 3d, 5d, 10d, 21d) |
| **Trend** | Price/EMA ratios (12, 26), Price/SMA ratios (7, 14, 30, 60) |
| **Momentum** | RSI-14, Stochastic %K/%D |
| **Volatility** | ATR%, historical volatility, Bollinger width & position |
| **Volume** | Volume ratios, OBV rate-of-change |
| **Macroeconomic** | GDP YoY z-score, gold returns, unemployment delta, interest rate delta, inflation delta (all lagged for publication delay) |
| **Sentiment** | NYT headline sentiment z-score (rolling 63-day) |
| **VIX** | VIX return, VIX high-low spread |
| **Calendar** | Cyclical day-of-week, month, day-of-year (sin/cos) |

Macro indicators are shifted forward to account for publication lag (GDP: 44 trading days, unemployment/inflation: 22 days). Sentiment scores are converted to z-scores over a 63-day rolling window.

### Walk-Forward Backtesting

The pipeline uses a **daily walk-forward** methodology that mirrors real trading conditions:

1. The model is retrained at the start of each calendar quarter during the test period (2021–2024).
2. At each retrain point, training data starts from 2008 and ends before the validation window.
3. The validation window is the 2 years immediately preceding the retrain quarter.
4. The **label-known date** constraint ensures no future labels leak: a sample dated T with a 1-day-forward target only becomes eligible for training after the next trading day's close.
5. The `StandardScaler` is fit exclusively on training data and applied to validation and test data—never updated with future information.
6. Sequences are built so that the context window for any prediction never extends into the future.

---

## Data Sources

| File | Source | Description |
|------|--------|-------------|
| `yfinance_sp500.csv` | Yahoo Finance (yfinance) | Daily OHLCV for S&P 500 |
| `vix.csv` | CBOE | VIX index |
| `GDP.csv` | FRED | Quarterly US GDP |
| `inflation rate.csv` | BLS | CPI-based inflation |
| `interest rate.csv` | FRED | Federal funds rate |
| `Unemployment.csv` | BLS | Unemployment rate |
| `gold_price.csv` | Market data | Gold closing prices |
| `sp500 headlines 2008 to 2024.csv` | NYT Archive API | Market-relevant headlines with sentiment |
| `merged_sp500_dataset.csv` | Merged | All macro + sentiment aligned by date |

---

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set NYT API key (for headline updates)
echo "NYT_KEY=your_api_key_here" > .env

# Run the full pipeline in order:
python src/00_update_nyt_headlines.py   # Fetch NYT headlines
python src/01_wavelet_denoising.py       # Denoise OHLCV
python src/02_feature_engineering.py      # Build stationary features
python src/03_xlstm_model.py             # (model definition — imported by train)
python src/04_train.py                   # Walk-forward training
python src/05_evaluate.py                # Evaluation + plots
python src/06_inference.py               # Next-day prediction
python src/07_dca_comparison.py           # Strategy comparison
```

All configuration (wavelet parameters, model hyperparameters, date ranges, etc.) is in `config.py`.

---

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `D_MODEL` | 48 | Embedding dimension |
| `N_BLOCKS` | 2 | Alternating mLSTM / sLSTM blocks |
| `NUM_HEADS` | 4 | Attention heads (mLSTM) and groups (sLSTM) |
| `EXPAND_FACTOR` | 2 | MLP inner dimension multiplier |
| `DROPOUT` | 0.15 | Dropout rate |
| `SEQUENCE_LENGTH` | 60 | Lookback window (trading days) |
| `LEARNING_RATE` | 5e-4 | AdamW learning rate |
| `WEIGHT_DECAY` | 1e-3 | L2 regularization |
| `BATCH_SIZE` | 32 | Training batch size |
| `MAX_EPOCHS` | 100 | Maximum training epochs per fold |
| `PATIENCE` | 15 | Early stopping patience |
| `LABEL_SMOOTHING` | 0.08 | Label smoothing factor |
| `GRAD_CLIP` | 1.0 | Gradient norm clipping |
| `SEED` | 42 | Random seed |

---

## Output Structure

```
outputs/
├── denoised_sp500.csv              # Wavelet-denoised OHLCV
├── features_directional.csv         # Final feature matrix
├── models/                          # Quarterly-retrained model checkpoints
│   └── xlstm_retrain_YYYY_QN.pt
├── results/
│   ├── fold_results.json            # Per-fold predictions + metadata
│   ├── metrics.json                 # Aggregate evaluation metrics
│   ├── dca_comparison.json          # Strategy comparison metrics
│   ├── evaluation_summary.png       # Confusion matrix, equity, drawdown
│   ├── wavelet_comparison.png       # Original vs denoised prices
│   ├── dca_comparison.png           # Cumulative returns comparison
│   ├── portfolio_growth.png         # Dollar portfolio growth
│   ├── sp500_trade_signals.png      # Price chart with trade markers
│   ├── sp500_prediction_confidence.png  # Price + model confidence overlay
│   └── training_history_*.png       # Per-fold loss/accuracy curves
└── logs/
    └── predictions.csv              # Inference prediction log
```

---

## Strategy Definitions

- **Long/Short:** Go long if P(up) > 0.5, short otherwise. Captures both up and down moves.
- **Long/Flat:** Go long if P(up) > 0.5, move to cash otherwise. Avoids short-selling risk.
- **DCA ($10k/month):** Dollar-cost average on the first trading day of each month. Reported as return on deployed capital (excluding new inflows).
- **Buy & Hold:** Hold S&P 500 for the full test period.

All return series are computed on the same basis (daily return on capital at risk) for fair comparison.

---

## Key Design Decisions

1. **No lookahead** — Causal wavelet denoising, label-known-date constraints, and scaler fit on training data only.
2. **Stationary features only** — All inputs are ratios, returns, or bounded values. No raw prices.
3. **Walk-forward retraining** — Quarterly retrain prevents stale models and simulates production use.
4. **Binary direction target** — The target uses raw (undenoised) close prices, since we trade actual market direction.
5. **Publication lag adjustment** — Macro indicators are shifted to reflect when data becomes available, not when the period ends.
   
