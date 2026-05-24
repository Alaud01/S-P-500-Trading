import sys
import importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from config import (
    RAW_PRICE_CSV, VIX_CSV, MODEL_DIR, LOG_DIR, DATE_COL, PRICE_COLS, VOLUME_COL,
    SEQUENCE_LENGTH, PREDICTION_HORIZON, SEED,
)

wvd = importlib.import_module('src.01_wavelet_denoising')
fe = importlib.import_module('src.02_feature_engineering')
xlstm = importlib.import_module('src.03_xlstm_model')

causal_wavelet_denoise_series = wvd.causal_wavelet_denoise_series
compute_all_features = fe.compute_all_features
merge_macro_sentiment = fe.merge_macro_sentiment
merge_vix = fe.merge_vix
XLSTMTSModel = xlstm.XLSTMTSModel

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_latest_features():
    from sklearn.preprocessing import StandardScaler

    df_raw = pd.read_csv(RAW_PRICE_CSV, parse_dates=[DATE_COL])
    df_raw = df_raw.sort_values(DATE_COL).reset_index(drop=True)

    denoised = df_raw[[DATE_COL, VOLUME_COL]].copy()
    for col in PRICE_COLS:
        if col in df_raw.columns:
            original = df_raw[col].values.astype(np.float64)
            denoised_col = causal_wavelet_denoise_series(original)
            denoised[col] = denoised_col
            print(f"  Causal denoised {col} (inference)")
    df = denoised.copy()

    df = compute_all_features(df)
    df = merge_macro_sentiment(df)
    df = merge_vix(df)

    raw_close = df_raw.set_index(DATE_COL)['Close']
    df = df.set_index(DATE_COL)
    df['raw_close'] = raw_close
    df = df.reset_index()

    abs_cols_to_drop = ["Open", "High", "Low", "Close", "Volume"]
    df.drop(columns=[c for c in abs_cols_to_drop if c in df.columns], inplace=True)

    feature_cols = [c for c in df.columns if c not in [DATE_COL, 'raw_close', 'raw_open',
                     'raw_high', 'raw_low', 'raw_daily_return', 'raw_log_return',
                     'forward_1d_return', 'target_direction']]
    df_features_only = df[feature_cols]
    nan_rows = df_features_only.isna().any(axis=1)
    df = df[~nan_rows].reset_index(drop=True)

    model_files = sorted(MODEL_DIR.glob("xlstm_retrain_*.pt"))
    if not model_files:
        model_files = sorted(MODEL_DIR.glob("xlstm_fold_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No trained model found in {MODEL_DIR}")
    model_path = model_files[-1]
    print(f"  Model: {model_path.name}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    feature_cols = checkpoint['feature_cols']
    scaler = checkpoint['scaler']

    # Exclude raw_* and target columns (not used as features during training)
    exclude_cols = {'target_direction', 'raw_close', 'raw_open', 'raw_high',
                    'raw_low', 'raw_daily_return', 'raw_log_return', 'forward_1d_return'}
    available = [c for c in feature_cols if c in df.columns and c not in exclude_cols]
    X_all = df[available].values.astype(np.float32)

    X_scaled = scaler.transform(X_all)

    last_seq = X_scaled[-SEQUENCE_LENGTH:]
    if len(last_seq) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough data for {SEQUENCE_LENGTH}-day sequence")

    last_seq_t = torch.from_numpy(last_seq).unsqueeze(0)

    n_features = last_seq.shape[1]
    model = XLSTMTSModel(n_features=n_features)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    with torch.no_grad():
        logit = model(last_seq_t)
        prob = torch.sigmoid(logit).item()

    prediction = 1 if prob > 0.5 else 0
    direction = "UP" if prediction == 1 else "DOWN"

    latest_date = df[DATE_COL].iloc[-1]
    # If latest data date is today (market closed), predict for tomorrow
    today = pd.Timestamp.now().normalize()
    data_date = pd.Timestamp(latest_date).normalize()
    if data_date >= today:
        pred_date = today + timedelta(days=1)
        while pred_date.weekday() >= 5:
            pred_date += timedelta(days=1)
    else:
        pred_date = data_date + timedelta(days=1)
        while pred_date.weekday() >= 5:
            pred_date += timedelta(days=1)

    result = {
        'date': str(latest_date.date()),
        'prediction_date': str(pred_date.date()),
        'direction': direction,
        'confidence': float(prob),
        'probability_up': float(prob),
    }

    log_file = LOG_DIR / "predictions.csv"
    log_entry = pd.DataFrame([{
        'timestamp': datetime.now().isoformat(),
        'data_date': str(latest_date.date()),
        'prediction_date': str(pred_date.date()),
        'prediction': prediction,
        'confidence': prob,
    }])

    if log_file.exists():
        existing = pd.read_csv(log_file)
        log_entry = pd.concat([existing, log_entry], ignore_index=True)
    log_entry.to_csv(log_file, index=False)

    return result


def main():
    print("=" * 60)
    print(" Step 6 — Next-Day Inference (Causal Denoising)")
    print("=" * 60)

    try:
        result = get_latest_features()
        print(f"\n  Data date:       {result['date']}")
        print(f"  Prediction date: {result['prediction_date']}")
        print(f"  Direction:       {result['direction']}")
        print(f"  Confidence:      {result['confidence']:.4f}")
        print(f"\n  Logged to {LOG_DIR / 'predictions.csv'}")
        print("Inference complete.\n")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Please run training (04_train.py) first.\n")


if __name__ == "__main__":
    main()
