import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from config import (
    RAW_PRICE_CSV, DENOISED_CSV, MERGED_CSV, FEATURES_CSV,
    DATE_COL, PRICE_COLS, VOLUME_COL,
    MA_WINDOWS, RSI_WINDOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_WINDOW, BB_STD,
    ATR_WINDOW, VOLATILITY_WINDOW, STOCH_WINDOW,
    RETURN_LAG_WINDOWS, SEQUENCE_LENGTH, SEED,
)
np.random.seed(SEED)


def compute_all_features(df):
    """All features computed on denoised OHLCV — all in STATIONARY form (ratios/percentages)."""
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    epsilon = 1e-10

    # -- Returns --
    df["log_return"] = np.log(c / (c.shift(1) + epsilon))
    df["daily_return"] = c.pct_change()

    # -- Moving average RATIOS (price/MA instead of raw MA) --
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["price_EMA12_ratio"] = c / (ema12 + epsilon)
    df["price_EMA26_ratio"] = c / (ema26 + epsilon)
    for w in MA_WINDOWS:
        ma_w = c.rolling(w).mean()
        df[f"price_MA{w}_ratio"] = c / (ma_w + epsilon)
        df[f"volume_MA{w}_ratio"] = v / (v.rolling(w).mean() + epsilon)

    # -- RSI (already bounded 0-100, stationary) --
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()
    rs = avg_gain / (avg_loss + epsilon)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # -- MACD as percentage of close (stationary) --
    macd = ema12 - ema26
    df["MACD_pct"] = macd / (c + epsilon)
    macd_signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_signal_pct"] = macd_signal / (c + epsilon)
    df["MACD_histogram_pct"] = (macd - macd_signal) / (c + epsilon)

    # -- Bollinger Bands (only keep stationary: width & position) --
    bb_middle = c.rolling(BB_WINDOW).mean()
    bb_std = c.rolling(BB_WINDOW).std()
    bb_upper = bb_middle + BB_STD * bb_std
    bb_lower = bb_middle - BB_STD * bb_std
    df["BB_width"] = (bb_upper - bb_lower) / (bb_middle + epsilon)
    df["BB_position"] = (c - bb_lower) / (bb_upper - bb_lower + epsilon)

    # -- ATR as percentage (stationary) --
    high_low = h - l
    high_close = np.abs(h - c.shift(1))
    low_close = np.abs(l - c.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_pct"] = tr.rolling(ATR_WINDOW).mean() / (c + epsilon)

    # -- Historical volatility (already stationary) --
    df["volatility"] = df["log_return"].rolling(VOLATILITY_WINDOW).std()

    # -- Stochastic oscillator (already bounded 0-100) --
    low_min = l.rolling(STOCH_WINDOW).min()
    high_max = h.rolling(STOCH_WINDOW).max()
    df["stoch_k"] = 100.0 * (c - low_min) / (high_max - low_min + epsilon)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # -- OBV rate of change (stationary) instead of cumulative OBV --
    obv = (v * np.sign(c.diff())).cumsum()
    df["OBV_roc_10"] = obv.pct_change(10)
    df["OBV_roc_21"] = obv.pct_change(21)

    # -- Volume ratio (already stationary) --
    df["volume_ratio"] = v / (v.rolling(MA_WINDOWS[-1]).mean() + epsilon)

    # -- Day of week, month (cyclical encoding) --
    date_series = pd.to_datetime(df[DATE_COL])
    df["dayofweek_sin"] = np.sin(2 * np.pi * date_series.dt.dayofweek / 5)
    df["dayofweek_cos"] = np.cos(2 * np.pi * date_series.dt.dayofweek / 5)
    df["month_sin"] = np.sin(2 * np.pi * date_series.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * date_series.dt.month / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * date_series.dt.dayofyear / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * date_series.dt.dayofyear / 365)

    # -- Lag features: RETURNS only (not absolute prices/volumes) --
    for lag in RETURN_LAG_WINDOWS:
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)
        df[f"volume_lag_{lag}_ratio"] = v.shift(lag) / (v.rolling(MA_WINDOWS[-1]).mean() + epsilon)

    # -- Returns over various horizons (already stationary) --
    for horizon in [1, 3, 5, 10, 21]:
        df[f"ret_{horizon}d"] = c.pct_change(horizon)

    # -- High-Low spread (already stationary) --
    df["hl_spread"] = (h - l) / (c + epsilon)
    df["close_open_ratio"] = c / (o + epsilon)

    return df


def merge_raw_prices(df, raw_path=RAW_PRICE_CSV):
    """Merge raw (actual market) OHLC from yfinance for target and evaluation."""
    raw = pd.read_csv(raw_path, parse_dates=[DATE_COL])
    raw = raw[raw[DATE_COL].isin(df[DATE_COL])].set_index(DATE_COL)
    df = df.set_index(DATE_COL)
    for col in PRICE_COLS:
        if col in raw.columns:
            df[f"raw_{col.lower()}"] = raw[col]
    df["raw_daily_return"] = raw["Close"].pct_change()
    df["raw_log_return"] = np.log(raw["Close"] / (raw["Close"].shift(1) + 1e-10))
    df = df.reset_index()
    return df


def merge_macro_sentiment(df, path=MERGED_CSV):
    """Merge macro + sentiment — all converted to stationary (YoY % change or delta)."""
    macro = pd.read_csv(path, parse_dates=[DATE_COL])
    macro_cols = [DATE_COL, "Sentiment_Score", "GDP", "Gold_Price",
                  "Unemployment_Rate", "Interest_Rate", "Inflation_Rate"]
    macro = macro[macro_cols].drop_duplicates(subset=DATE_COL, keep="first")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    macro[DATE_COL] = pd.to_datetime(macro[DATE_COL])

    # Shift macro data forward to account for publication lag
    macro = macro.sort_values(DATE_COL).reset_index(drop=True)
    macro["GDP"] = macro["GDP"].shift(22)
    macro["Unemployment_Rate"] = macro["Unemployment_Rate"].shift(22)
    macro["Inflation_Rate"] = macro["Inflation_Rate"].shift(22)

    # Convert absolute macro to stationary forms
    macro["GDP_yoy"] = macro["GDP"].pct_change(252)
    macro["Gold_return"] = macro["Gold_Price"].pct_change()
    macro["Unemployment_delta"] = macro["Unemployment_Rate"].diff(22)
    macro["Interest_rate_delta"] = macro["Interest_Rate"].diff(22)
    macro["Inflation_delta"] = macro["Inflation_Rate"].diff(22)

    # Sentiment: use z-score relative to rolling window (stationary)
    macro["Sentiment_zscore"] = (macro["Sentiment_Score"] - macro["Sentiment_Score"].rolling(63).mean()) / (macro["Sentiment_Score"].rolling(63).std() + 1e-10)

    # GDP_yoy: z-score relative to rolling window to handle regime drift
    macro["GDP_yoy_zscore"] = (macro["GDP_yoy"] - macro["GDP_yoy"].rolling(252).mean()) / (macro["GDP_yoy"].rolling(252).std() + 1e-10)

    # Drop raw absolute columns
    macro.drop(columns=["GDP", "Gold_Price", "Unemployment_Rate",
                        "Interest_Rate", "Inflation_Rate",
                        "GDP_yoy", "Sentiment_Score"], inplace=True)

    df = df.merge(macro, on=DATE_COL, how="left")

    macro_fill = ["GDP_yoy_zscore", "Gold_return", "Unemployment_delta",
                  "Interest_rate_delta", "Inflation_delta", "Sentiment_zscore"]
    df[macro_fill] = df[macro_fill].ffill()
    return df


def create_target(df, raw_close):
    """
    Binary direction: 1 if next-day raw Close > current raw Close, else 0.
    Uses RAW (actual market) close for target — we trade actual market direction,
    not the denoised smoothed version.
    """
    raw_close_series = pd.Series(raw_close)
    df["target_direction"] = (raw_close_series.shift(-1).values > raw_close).astype(int)
    return df


def main():
    print("=" * 60)
    print(" Step 2 — Feature Engineering on Denoised Data")
    print("=" * 60)

    df = pd.read_csv(DENOISED_CSV, parse_dates=[DATE_COL])
    print(f"  Loaded {DENOISED_CSV.name}  ({len(df)} rows)")

    df = compute_all_features(df)
    print(f"  Computed technical indicators ({len(df.columns)} columns)")

    df = merge_macro_sentiment(df)
    print(f"  Merged macro + sentiment data (with publication lag)")

    df = merge_raw_prices(df)
    print(f"  Merged raw market prices for target + evaluation")

    raw_close = df["raw_close"].values
    df = create_target(df, raw_close)
    print(f"  Created binary direction target (from RAW close)")

    # Drop absolute-level columns that are non-stationary (keep only ratios/returns/percentages)
    abs_cols_to_drop = ["Open", "High", "Low", "Close", "Volume"]
    df.drop(columns=[c for c in abs_cols_to_drop if c in df.columns], inplace=True)
    print(f"  Dropped absolute price/volume columns: {abs_cols_to_drop}")

    # Drop rows with NaN from rolling calculations
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Dropped NaN rows:  {before} -> {len(df)}")

    # Keep DATE_COL but don't use as feature
    df.to_csv(FEATURES_CSV, index=False)
    print(f"  Saved to {FEATURES_CSV}")

    print(f"  Final shape: {df.shape}")
    print(f"  Target distribution:\n{df['target_direction'].value_counts()}")
    print("Feature engineering complete.\n")


if __name__ == "__main__":
    main()
