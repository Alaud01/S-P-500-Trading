# Run command: source .venv/bin/activate && python train_meta.py --verbose --threshold-metric accuracy --logreg-grid 0.01,0.1,1.0,10.0
import os
import json
import time
import math
import argparse
import warnings
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    return df.fillna(method='ffill').fillna(method='bfill')


def select_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    feature_df = df.select_dtypes(include=[np.number]).copy()
    if 'Target' not in feature_df.columns:
        raise ValueError("Target column 'Target' not found in dataset. Run built_dataset.py first.")
    y = feature_df['Target'].astype(int).values
    X = feature_df.drop(columns=['Target'])
    return X, y, list(X.columns)


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {
        'auc': float(auc) if not (isinstance(auc, float) and math.isnan(auc)) else 0.0,
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def find_best_threshold(y_prob: np.ndarray, y_true: np.ndarray, metric: str = 'f1') -> Tuple[float, float]:
    best_thr = 0.5
    best_score = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        preds = (y_prob >= thr).astype(int)
        if metric == 'youden':
            tp = np.logical_and(preds == 1, y_true == 1).sum()
            tn = np.logical_and(preds == 0, y_true == 0).sum()
            fp = np.logical_and(preds == 1, y_true == 0).sum()
            fn = np.logical_and(preds == 0, y_true == 1).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            score = tpr - fpr
        elif metric == 'accuracy':
            score = accuracy_score(y_true, preds)
        else:
            _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
            score = float(f1)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


# ---- LSTM base model (minimal, for generating probabilities) ----
class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        out_features = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            last_hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            last_hidden = hn[-1]
        logits = self.fc(self.dropout(last_hidden))
        return logits.squeeze(-1)


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    num_rows = X.shape[0]
    if num_rows < seq_len + 1:
        raise ValueError(f"Not enough rows ({num_rows}) to build sequences of length {seq_len}.")
    num_seq = num_rows - seq_len + 1
    X_seq = np.zeros((num_seq, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros((num_seq,), dtype=np.int64)
    for i in range(num_seq):
        X_seq[i] = X[i:i + seq_len]
        y_seq[i] = y[i + seq_len - 1]
    return X_seq, y_seq


def compute_class_pos_weight(labels: np.ndarray) -> float:
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    if pos == 0:
        return 1.0
    return max(neg / max(pos, 1.0), 1.0)


@torch.no_grad()
def lstm_predict_probs(model: nn.Module, X_seq: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    probs: List[float] = []
    for i in range(0, len(X_seq), batch_size):
        xb = torch.from_numpy(X_seq[i:i+batch_size]).float().to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(p.tolist())
    return np.asarray(probs, dtype=np.float32)


def predict_lstm_from_checkpoint_for_mask(
    checkpoint_path: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    dates_series: Optional[pd.Series],
    prices_series: Optional[pd.Series],
    mask: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, int]:
    """Load saved LSTM checkpoint and generate probabilities for rows where mask is True.
    Returns: dates_eff, prices_eff (or None), probs, y_eff, seq_len
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    params = ckpt.get('params', {})
    seq_len = int(ckpt.get('seq_len', params.get('seq_len', 60)))
    input_size = X_full.shape[1]

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=int(params.get('hidden_size', 64)),
        num_layers=int(params.get('num_layers', 2)),
        dropout=float(params.get('dropout', 0.2)),
        bidirectional=bool(params.get('bidirectional', False)),
    ).to(device)
    state_key = 'model_state_dict' if 'model_state_dict' in ckpt else 'model'
    model.load_state_dict(ckpt[state_key])
    model.eval()

    scaler_mean = np.asarray(ckpt.get('scaler_mean', np.zeros(input_size)), dtype=np.float32)
    scaler_scale = np.asarray(ckpt.get('scaler_scale', np.ones(input_size)), dtype=np.float32)

    # Select by mask
    X_sel = X_full[mask]
    y_sel = y_full[mask]
    dates_sel = dates_series[mask].values if dates_series is not None else np.arange(mask.sum())
    prices_sel = prices_series[mask].values if (prices_series is not None) else None

    # Scale using saved scaler from checkpoint
    X_scaled = ((X_sel - scaler_mean) / np.where(scaler_scale == 0.0, 1.0, scaler_scale)).astype(np.float32)

    # Build sequences and predict
    X_seq, y_seq = build_sequences(X_scaled, y_sel, seq_len)
    probs = lstm_predict_probs(model, X_seq, batch_size, device)

    # Align dates/prices for effective index
    dates_eff = dates_sel[seq_len - 1:]
    prices_eff = prices_sel[seq_len - 1:] if prices_sel is not None else None
    return dates_eff, prices_eff, probs, y_seq.astype(int), seq_len


# ---- XGBoost inputs: use existing holdout predictions CSV ----

def _find_latest(path_dir: str, suffix: str) -> Optional[str]:
    if not os.path.isdir(path_dir):
        return None
    candidates = [os.path.join(path_dir, f) for f in os.listdir(path_dir) if f.endswith(suffix)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def predict_xgb_for_all(
    model_path: str,
    report_path: Optional[str],
    df_full: pd.DataFrame,
    X_df: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    import xgboost as xgb

    feature_names: List[str]
    use_scaler = False
    if report_path and os.path.exists(report_path):
        rpt = _load_json(report_path)
        feature_names = rpt.get('feature_names', list(X_df.columns))
        params = rpt.get('params', {})
        use_scaler = bool(params.get('use_scaler', False))
    else:
        feature_names = list(X_df.columns)

    # Ensure column order
    X_ordered = X_df[feature_names].values.astype(np.float32)

    # Build a scaler on pre-2022 portion if required, then predict for ALL rows
    if 'Date' in df_full.columns:
        pre_mask = df_full['Date'] < pd.to_datetime('2022-01-01')
        dates = df_full['Date'].values
        prices = df_full['Close'].values if 'Close' in df_full.columns else None
    else:
        pre_mask = np.zeros(len(X_ordered), dtype=bool)
        pre_mask[: int(len(X_ordered) * 0.85)] = True
        dates = np.arange(len(X_ordered))
        prices = None

    X_all = X_ordered
    if use_scaler:
        scaler = StandardScaler().fit(X_all[pre_mask])
        X_all = scaler.transform(X_all).astype(np.float32)

    booster = xgb.Booster()
    booster.load_model(model_path)
    dall = xgb.DMatrix(X_all)
    prob_pos = booster.predict(dall)
    prob_pos = np.asarray(prob_pos).reshape(-1).astype(np.float32)
    return dates, prices, prob_pos, y.astype(int), feature_names


# ---- Meta-model (Logistic Regression) ----


def plot_training_history_meta(history: Dict[str, List[float]], plots_dir: str, verbose: bool = True) -> None:
    # For logistic regression, we do not have epoch-wise history; skip plotting.
    return


def plot_meta_predictions_vs_price(dates: np.ndarray, prices: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, plots_dir: str, test_start_date: str, verbose: bool = True) -> None:
    if not verbose:
        return
    print("  📊 Generating meta-model predictions vs price visualization...")
    os.makedirs(plots_dir, exist_ok=True)

    threshold, _ = find_best_threshold(y_prob, y_true, metric='accuracy')
    pred_binary = (y_prob >= threshold).astype(int)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Meta-Model Predictions vs S&P 500 Price (Test Set: {test_start_date} onwards)', fontsize=16, fontweight='bold')

    if prices is not None:
        ax1.plot(dates, prices, color='blue', linewidth=1.5, alpha=0.8)
        ax1.set_title('S&P 500 Closing Price (Test Set)', fontweight='bold')
        ax1.set_ylabel('Price ($)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if len(prices) > 30:
            ma_30 = np.convolve(prices, np.ones(30)/30, mode='valid')
            ma_dates = dates[29:]
            ax1.plot(ma_dates, ma_30, color='red', linewidth=2, alpha=0.7, label='30-day MA')
            ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Price data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('S&P 500 Closing Price (Test Set)', fontweight='bold')

    ax2.plot(dates, y_prob, color='green', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Decision Threshold ({threshold:.2f})')
    ax2.set_title('Meta Prediction Probability (14-day forward)', fontweight='bold')
    ax2.set_ylabel('Probability', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    correct_up = (pred_binary == 1) & (y_true == 1)
    correct_down = (pred_binary == 0) & (y_true == 0)
    wrong_up = (pred_binary == 1) & (y_true == 0)
    wrong_down = (pred_binary == 0) & (y_true == 1)

    if prices is not None:
        ax3.scatter(dates[correct_up], prices[correct_up], color='green', alpha=0.6, s=20, label='Correct Up Prediction')
        ax3.scatter(dates[correct_down], prices[correct_down], color='blue', alpha=0.6, s=20, label='Correct Down Prediction')
        ax3.scatter(dates[wrong_up], prices[wrong_up], color='red', alpha=0.6, s=20, label='Wrong Up Prediction')
        ax3.scatter(dates[wrong_down], prices[wrong_down], color='orange', alpha=0.6, s=20, label='Wrong Down Prediction')
    else:
        ax3.scatter(dates[correct_up], np.zeros_like(dates[correct_up]), color='green', alpha=0.6, s=20, label='Correct Up Prediction')
        ax3.scatter(dates[correct_down], np.zeros_like(dates[correct_down]), color='blue', alpha=0.6, s=20, label='Correct Down Prediction')
        ax3.scatter(dates[wrong_up], np.zeros_like(dates[wrong_up]), color='red', alpha=0.6, s=20, label='Wrong Up Prediction')
        ax3.scatter(dates[wrong_down], np.zeros_like(dates[wrong_down]), color='orange', alpha=0.6, s=20, label='Wrong Down Prediction')

    ax3.set_title('Prediction Accuracy vs Price (Test Set)', fontweight='bold')
    ax3.set_ylabel('Price ($)' if prices is not None else 'Index', fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    accuracy = np.mean(pred_binary == y_true)
    precision = np.sum((pred_binary == 1) & (y_true == 1)) / max(np.sum(pred_binary == 1), 1)
    recall = np.sum((pred_binary == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
    metrics_text = f'Test Set Metrics:\nAccuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
    ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if len(dates) > 0 and isinstance(dates[0], (np.datetime64, pd.Timestamp)):
        ax3.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
        ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'meta_test_predictions_vs_price.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  📈 Test set predictions vs price plot saved to: {plot_path}")
    plt.show()

    # Detailed analysis similar to train_lstm.py
    fig2, ((bx1, bx2), (bx3, bx4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f'Meta-Model Test Set Analysis ({test_start_date} onwards)', fontsize=16, fontweight='bold')

    bx1.hist(y_prob, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    bx1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    bx1.set_title('Distribution of Prediction Probabilities (Test Set)')
    bx1.set_xlabel('Prediction Probability')
    bx1.set_ylabel('Frequency')
    bx1.legend()
    bx1.grid(True, alpha=0.3)

    confidence = np.abs(y_prob - threshold) * 2
    bx2.plot(dates, confidence, color='purple', alpha=0.7)
    bx2.set_title('Model Confidence Over Time (Test Set)')
    bx2.set_xlabel('Date')
    bx2.set_ylabel('Confidence (0-1)')
    bx2.grid(True, alpha=0.3)

    window_size = min(50, len(pred_binary) // 4)
    if window_size > 10:
        rolling_accuracy = []
        rolling_dates = []
        for i in range(window_size, len(pred_binary)):
            window_acc = np.mean(pred_binary[i-window_size:i] == y_true[i-window_size:i])
            rolling_accuracy.append(window_acc)
            rolling_dates.append(dates[i])
        bx3.plot(rolling_dates, rolling_accuracy, color='green', alpha=0.7)
        bx3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')
        bx3.set_title(f'Rolling Accuracy ({window_size}-day window)')
        bx3.set_xlabel('Date')
        bx3.set_ylabel('Accuracy')
        bx3.legend()
        bx3.grid(True, alpha=0.3)
    else:
        bx3.text(0.5, 0.5, 'Insufficient data for rolling accuracy', ha='center', va='center', transform=bx3.transAxes)
        bx3.set_title('Rolling Accuracy')

    bx4.scatter(y_true, y_prob, alpha=0.5, color='blue')
    bx4.set_title('Predictions vs Actual Values (Test Set)')
    bx4.set_xlabel('Actual (0=Down, 1=Up)')
    bx4.set_ylabel('Predicted Probability')
    bx4.grid(True, alpha=0.3)

    plt.tight_layout()
    detailed_plot_path = os.path.join(plots_dir, 'meta_test_detailed_analysis.png')
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Test set detailed analysis plot saved to: {detailed_plot_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Train Logistic Regression Meta-Model using LSTM and XGBoost predictions')
    parser.add_argument('--data-csv', type=str, default='data/final_dataset_for_modeling.csv')
    parser.add_argument('--models-dir', type=str, default='models/meta')
    parser.add_argument('--plots-dir', type=str, default='plots/meta')
    parser.add_argument('--test-start-date', type=str, default='2022-01-01')
    # Existing base artifacts
    parser.add_argument('--lstm-model-path', type=str, default='')
    parser.add_argument('--xgb-model-path', type=str, default='')
    parser.add_argument('--xgb-report-path', type=str, default='')
    parser.add_argument('--meta-holdout-train-ratio', type=float, default=0.7)
    # Logistic Regression params
    parser.add_argument('--logreg-C', type=float, default=1.0)
    parser.add_argument('--logreg-solver', type=str, default='lbfgs', choices=['lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'])
    parser.add_argument('--logreg-grid', type=str, default='0.01,0.1,1.0,10.0')
    parser.add_argument('--threshold-metric', type=str, default='accuracy', choices=['accuracy', 'f1', 'youden'])
    parser.add_argument('--fixed-threshold', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    if args.verbose:
        print(f"{'='*60}")
        print("META-MODEL TRAINING (Logistic Regression Stacking)")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Data: {args.data_csv}")
        print(f"Models dir: {args.models_dir}")
        print(f"Plots dir: {args.plots_dir}")
        print(f"Test start date: {args.test_start_date}")
        print()

    df = load_dataset(args.data_csv)
    X_df, y, feature_names = select_features_and_target(df)
    X = X_df.values.astype(np.float32)

    # Train/test split by date
    test_start = pd.to_datetime(args.test_start_date)
    if 'Date' in df.columns:
        test_mask = df['Date'] >= test_start
        train_mask = ~test_mask
        dates_test_all = df['Date'][test_mask].values
        prices_test_all = df['Close'][test_mask].values if 'Close' in df.columns else None
    else:
        split_idx = int(len(X) * 0.85)
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[:split_idx] = True
        test_mask = ~train_mask
        dates_test_all = np.arange(test_mask.sum())
        prices_test_all = None

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if args.verbose:
        print(f"Train rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]}")

    # Find latest LSTM checkpoint if not provided
    lstm_model_path = args.lstm_model_path
    if not lstm_model_path:
        lstm_dir = os.path.join('models', 'lstm')
        if os.path.isdir(lstm_dir):
            pt_files = [os.path.join(lstm_dir, f) for f in os.listdir(lstm_dir) if f.endswith('.pt')]
            lstm_model_path = max(pt_files, key=os.path.getmtime) if pt_files else ''
    if not lstm_model_path or not os.path.exists(lstm_model_path):
        raise FileNotFoundError("LSTM checkpoint not found. Provide --lstm-model-path or train LSTM first.")

    # Build masks for CV (<=2021-12-31) and Test (>=2022-01-01)
    if 'Date' in df.columns:
        dates_series = df['Date']
    else:
        # fabricate monotonically increasing integer dates if none
        dates_series = pd.Series(pd.date_range(start='2008-01-01', periods=len(df), freq='D'))

    cv_end = pd.to_datetime('2021-12-31')
    test_start = pd.to_datetime('2022-01-01')
    mask_cv = dates_series <= cv_end
    mask_test = dates_series >= test_start

    # LSTM predictions for CV region and Test region
    dates_lstm_cv, prices_lstm_cv, probs_lstm_cv, y_lstm_cv, seq_len = predict_lstm_from_checkpoint_for_mask(
        checkpoint_path=lstm_model_path,
        X_full=X,
        y_full=y,
        dates_series=dates_series,
        prices_series=df['Close'] if 'Close' in df.columns else None,
        mask=mask_cv.values,
        device=device,
        batch_size=128,
    )
    dates_lstm_te, prices_lstm_te, probs_lstm_te, y_lstm_te, _ = predict_lstm_from_checkpoint_for_mask(
        checkpoint_path=lstm_model_path,
        X_full=X,
        y_full=y,
        dates_series=dates_series,
        prices_series=df['Close'] if 'Close' in df.columns else None,
        mask=mask_test.values,
        device=device,
        batch_size=128,
    )

    # Determine latest XGB model/report if not provided
    xgb_model_path = args.xgb_model_path
    if not xgb_model_path:
        found = _find_latest(os.path.join('models', 'xgboost'), '.json')
        xgb_model_path = found if found else ''
    xgb_report_path = args.xgb_report_path
    if not xgb_report_path:
        found = _find_latest(os.path.join('models', 'xgboost'), '_report.json')
        xgb_report_path = found if found else ''
    if not xgb_model_path or not os.path.exists(xgb_model_path):
        raise FileNotFoundError("XGBoost model (.json) not found in models/xgboost. Provide --xgb-model-path.")

    # Predict XGB probabilities for test (2022+) using the saved model
    dates_xgb_all, _, probs_xgb_all, y_xgb_all, _ = predict_xgb_for_all(
        model_path=xgb_model_path,
        report_path=xgb_report_path,
        df_full=df,
        X_df=X_df,
        y=y,
    )
    # Split XGB predictions to CV and Test by date
    xgb_all_df = pd.DataFrame({'Date': dates_xgb_all, 'prob_xgb': probs_xgb_all})
    xgb_all_df = xgb_all_df.sort_values('Date')
    xgb_cv_df = xgb_all_df[xgb_all_df['Date'] <= cv_end].copy()
    xgb_te_df = xgb_all_df[xgb_all_df['Date'] >= test_start].copy()

    # Build LSTM dataframe
    lstm_cv_df = pd.DataFrame({'Date': dates_lstm_cv, 'prob_lstm': probs_lstm_cv, 'target': y_lstm_cv})
    lstm_te_df = pd.DataFrame({'Date': dates_lstm_te, 'prob_lstm': probs_lstm_te, 'target': y_lstm_te})

    # Merge on Date to align both base predictors (CV and Test separately)
    merged_cv = pd.merge(lstm_cv_df, xgb_cv_df[['Date', 'prob_xgb']], on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    merged_te = pd.merge(lstm_te_df, xgb_te_df[['Date', 'prob_xgb']], on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    if merged_cv.empty or merged_te.empty:
        raise ValueError("No overlapping dates between LSTM predictions and XGBoost holdout predictions. Ensure both are generated for the same period.")

    # Prepare meta features/labels from overlapping holdout only
    # Build Close feature (only) for meta-model
    # Map Date -> Close
    close_map = dict(zip(df['Date'].values, df['Close'].values)) if ('Date' in df.columns and 'Close' in df.columns) else {}

    def add_close(df_in: pd.DataFrame) -> pd.DataFrame:
        if close_map:
            df_in = df_in.copy()
            df_in['Close'] = df_in['Date'].map(close_map)
        else:
            df_in['Close'] = np.nan
        return df_in

    merged_cv = add_close(merged_cv)
    merged_te = add_close(merged_te)

    # Prepare meta features/labels using only base model probabilities (no Close)
    X_meta_cv = np.column_stack([merged_cv['prob_lstm'].values, merged_cv['prob_xgb'].values]).astype(np.float32)
    y_meta_cv = merged_cv['target'].values.astype(int)
    dates_meta_cv = merged_cv['Date'].values

    X_meta_te = np.column_stack([merged_te['prob_lstm'].values, merged_te['prob_xgb'].values]).astype(np.float32)
    y_meta_te = merged_te['target'].values.astype(int)
    dates_meta_te = merged_te['Date'].values
    prices_te = merged_te['Close'].values if 'Close' in merged_te.columns else None

    # Time-series CV on 2008-2021 (merged_cv), then final evaluation on 2022+ (merged_te)
    # Create a simple chronological split for val within CV data (last 10%)
    n_cv = len(X_meta_cv)
    val_tail = max(64, n_cv // 10)
    split_val_idx = max(n_cv - val_tail, 1)
    Xtr_m, ytr_m = X_meta_cv[:split_val_idx], y_meta_cv[:split_val_idx]
    Xval_m, yval_m = X_meta_cv[split_val_idx:], y_meta_cv[split_val_idx:]
    Xte_m, yte_m = X_meta_te, y_meta_te
    dates_te = dates_meta_te

    # Scale meta inputs
    meta_scaler = StandardScaler().fit(Xtr_m)
    Xtr_m_s = meta_scaler.transform(Xtr_m).astype(np.float32)
    Xval_m_s = meta_scaler.transform(Xval_m).astype(np.float32)
    Xte_m_s = meta_scaler.transform(Xte_m).astype(np.float32)

    # Train logistic regression with simple C grid search on CV-train split
    feature_names_meta = ['prob_lstm', 'prob_xgb']
    try:
        c_grid = [float(x) for x in (args.logreg_grid.split(',') if args.logreg_grid else [args.logreg_C])]
    except Exception:
        c_grid = [args.logreg_C]

    best_val_auc_lr = -1.0
    best_C = None
    best_logreg = None
    best_train_probs = None
    best_val_probs = None
    for c_val in c_grid:
        lr = LogisticRegression(C=c_val, solver=args.logreg_solver, class_weight='balanced', max_iter=1000, random_state=args.seed)
        lr.fit(Xtr_m_s, ytr_m)
        tr_probs = lr.predict_proba(Xtr_m_s)[:, 1]
        vl_probs = lr.predict_proba(Xval_m_s)[:, 1]
        vl_auc = roc_auc_score(yval_m, vl_probs) if len(np.unique(yval_m)) > 1 else 0.5
        if vl_auc > best_val_auc_lr:
            best_val_auc_lr = vl_auc
            best_C = c_val
            best_logreg = lr
            best_train_probs = tr_probs
            best_val_probs = vl_probs

    logreg = best_logreg if best_logreg is not None else LogisticRegression(C=args.logreg_C, solver=args.logreg_solver, class_weight='balanced', max_iter=1000, random_state=args.seed).fit(Xtr_m_s, ytr_m)

    # Threshold optimization based on selected metric (default accuracy) on TRAIN set
    train_probs = best_train_probs if best_train_probs is not None else logreg.predict_proba(Xtr_m_s)[:, 1]
    if args.fixed_threshold is not None and str(args.fixed_threshold).strip() != '':
        used_threshold = float(args.fixed_threshold)
    else:
        used_threshold, _ = find_best_threshold(train_probs, ytr_m, metric=args.threshold_metric)

    # Evaluate on training and validation (for diagnosis) and test
    val_probs = best_val_probs if best_val_probs is not None else logreg.predict_proba(Xval_m_s)[:, 1]
    train_metrics = evaluate_binary(ytr_m, train_probs, threshold=used_threshold)
    val_metrics = evaluate_binary(yval_m, val_probs, threshold=used_threshold)

    y_prob_test = logreg.predict_proba(Xte_m_s)[:, 1]
    test_metrics = evaluate_binary(yte_m, y_prob_test, threshold=used_threshold)

    # --- Diagnostic Plots: learning curve, ROC/PR, threshold sweep, calibration ---
    def plot_meta_learning_curves(Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray, chosen_C: float, solver: str, plots_dir: str, verbose: bool = True) -> None:
        if not verbose:
            return
        os.makedirs(plots_dir, exist_ok=True)
        sizes = np.linspace(0.2, 1.0, 5)
        train_auc_list, val_auc_list = [], []
        train_acc_list, val_acc_list = [], []
        for frac in sizes:
            n = max(10, int(len(Xtr) * float(frac)))
            Xtr_sub, ytr_sub = Xtr[:n], ytr[:n]
            lr = LogisticRegression(C=chosen_C, solver=solver, class_weight='balanced', max_iter=1000, random_state=42)
            lr.fit(Xtr_sub, ytr_sub)
            tr_probs = lr.predict_proba(Xtr_sub)[:, 1]
            vl_probs = lr.predict_proba(Xval)[:, 1]
            tr_auc = roc_auc_score(ytr_sub, tr_probs) if len(np.unique(ytr_sub)) > 1 else 0.5
            vl_auc = roc_auc_score(yval, vl_probs) if len(np.unique(yval)) > 1 else 0.5
            tr_thr, _ = find_best_threshold(tr_probs, ytr_sub, metric='accuracy')
            tr_acc = accuracy_score(ytr_sub, (tr_probs >= tr_thr).astype(int))
            vl_acc = accuracy_score(yval, (vl_probs >= tr_thr).astype(int))
            train_auc_list.append(tr_auc)
            val_auc_list.append(vl_auc)
            train_acc_list.append(tr_acc)
            val_acc_list.append(vl_acc)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Meta LR Learning Curves', fontsize=14, fontweight='bold')
        axes[0].plot(sizes, train_auc_list, label='Train AUC', marker='o')
        axes[0].plot(sizes, val_auc_list, label='Val AUC', marker='o')
        axes[0].set_xlabel('Training Fraction')
        axes[0].set_ylabel('AUC')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(sizes, train_acc_list, label='Train Acc', marker='o')
        axes[1].plot(sizes, val_acc_list, label='Val Acc', marker='o')
        axes[1].set_xlabel('Training Fraction')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        path = os.path.join(plots_dir, 'meta_learning_curves.png')
        plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"  📈 Learning curves saved to: {path}")
        plt.show()

    def plot_meta_roc_pr(yv: np.ndarray, pv: np.ndarray, yt: np.ndarray, pt: np.ndarray, plots_dir: str, verbose: bool = True) -> None:
        if not verbose:
            return
        os.makedirs(plots_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Meta LR ROC and PR Curves', fontsize=14, fontweight='bold')
        # ROC
        fpr_v, tpr_v, _ = roc_curve(yv, pv)
        fpr_t, tpr_t, _ = roc_curve(yt, pt)
        auc_v = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.5
        auc_t = roc_auc_score(yt, pt) if len(np.unique(yt)) > 1 else 0.5
        axes[0].plot(fpr_v, tpr_v, label=f'Val AUC={auc_v:.3f}')
        axes[0].plot(fpr_t, tpr_t, label=f'Test AUC={auc_t:.3f}', linestyle='--')
        axes[0].plot([0,1],[0,1], 'k--', alpha=0.3)
        axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        # PR
        prec_v, rec_v, _ = precision_recall_curve(yv, pv)
        prec_t, rec_t, _ = precision_recall_curve(yt, pt)
        ap_v = average_precision_score(yv, pv)
        ap_t = average_precision_score(yt, pt)
        axes[1].plot(rec_v, prec_v, label=f'Val AP={ap_v:.3f}')
        axes[1].plot(rec_t, prec_t, label=f'Test AP={ap_t:.3f}', linestyle='--')
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        path = os.path.join(plots_dir, 'meta_roc_pr.png')
        plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"  📈 ROC/PR curves saved to: {path}")
        plt.show()

    def plot_meta_threshold_sweep(yv: np.ndarray, pv: np.ndarray, yt: np.ndarray, pt: np.ndarray, used_thr: float, plots_dir: str, verbose: bool = True) -> None:
        if not verbose:
            return
        os.makedirs(plots_dir, exist_ok=True)
        thrs = np.linspace(0.05, 0.95, 19)
        def sweep(y, p):
            accs, f1s = [], []
            for t in thrs:
                preds = (p >= t).astype(int)
                accs.append(accuracy_score(y, preds))
                f1s.append(precision_recall_fscore_support(y, preds, average='binary', zero_division=0)[2])
            return np.array(accs), np.array(f1s)
        acc_v, f1_v = sweep(yv, pv)
        acc_t, f1_t = sweep(yt, pt)
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        fig.suptitle('Meta LR Threshold Sweep', fontsize=14, fontweight='bold')
        axes[0].plot(thrs, acc_v, label='Val Acc')
        axes[0].plot(thrs, acc_t, label='Test Acc', linestyle='--')
        axes[0].axvline(x=used_thr, color='red', linestyle=':')
        axes[0].set_xlabel('Threshold'); axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(thrs, f1_v, label='Val F1')
        axes[1].plot(thrs, f1_t, label='Test F1', linestyle='--')
        axes[1].axvline(x=used_thr, color='red', linestyle=':')
        axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('F1'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        path = os.path.join(plots_dir, 'meta_threshold_sweep.png')
        plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"  📈 Threshold sweep saved to: {path}")
        plt.show()

    def plot_meta_calibration(y: np.ndarray, p: np.ndarray, plots_dir: str, verbose: bool = True) -> None:
        if not verbose:
            return
        os.makedirs(plots_dir, exist_ok=True)
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='uniform')
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.plot(mean_pred, frac_pos, marker='o')
        ax.plot([0,1],[0,1], 'k--', alpha=0.3)
        ax.set_title('Meta LR Calibration'); ax.set_xlabel('Mean predicted value'); ax.set_ylabel('Fraction of positives'); ax.grid(True, alpha=0.3)
        path = os.path.join(plots_dir, 'meta_calibration.png')
        plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"  📈 Calibration plot saved to: {path}")
        plt.show()

    # Save artifacts
    os.makedirs(args.models_dir, exist_ok=True)
    # Save logistic regression coefficients and scaler
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(args.models_dir, f"meta_logreg_{timestamp}.json")
    coef = logreg.coef_.reshape(-1).tolist()
    intercept = float(logreg.intercept_.reshape(-1)[0])
    meta_artifact = {
        'type': 'logistic_regression',
        'feature_names': feature_names_meta,
        'coef': coef,
        'intercept': intercept,
        'threshold_metric': args.threshold_metric,
        'used_threshold': used_threshold,
        'scaler_mean': meta_scaler.mean_.tolist(),
        'scaler_scale': meta_scaler.scale_.tolist(),
        'lstm_seq_len': seq_len,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(model_path, 'w') as f:
        json.dump(meta_artifact, f, indent=2)

    report = {
        'device': str(device),
        'dataset_info': {
            'shape': X.shape,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'target_distribution': np.bincount(y).tolist(),
            'positive_class_ratio': float(y.mean())
        },
        'base_artifacts': {
            'lstm_checkpoint': os.path.abspath(lstm_model_path),
            'xgb_model': os.path.abspath(xgb_model_path),
            'xgb_report': os.path.abspath(xgb_report_path) if xgb_report_path else None,
            'seq_len': seq_len,
            'test_start_date': '2022-01-01',
        },
        'meta_model': {
            'type': 'logistic_regression',
            'C': best_C if best_C is not None else args.logreg_C,
            'threshold_metric': args.threshold_metric,
            'used_threshold': used_threshold,
            'coef': coef,
            'intercept': intercept,
        },
        'train_metrics_until_2021-12-31': train_metrics,
        'val_metrics_until_2021-12-31': val_metrics,
        'test_metrics_since_' + args.test_start_date: test_metrics,
        'model_path': model_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(args.models_dir, 'meta_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Save predictions CSV for test
    # Build aligned base probabilities for the test dates from merged_te
    merged_te_indexed = merged_te.set_index('Date')
    prob_lstm_te_aligned = merged_te_indexed.loc[dates_te, 'prob_lstm'].values if len(dates_te) > 0 else np.array([])
    prob_xgb_te_aligned = merged_te_indexed.loc[dates_te, 'prob_xgb'].values if len(dates_te) > 0 else np.array([])
    preds_df = pd.DataFrame({
        'Date': dates_te,
        'prob_meta': y_prob_test,
        'pred_meta': (y_prob_test >= used_threshold).astype(int),
        'target': yte_m,
        'prob_lstm': prob_lstm_te_aligned[:len(y_prob_test)],
        'prob_xgb': prob_xgb_te_aligned[:len(y_prob_test)],
    })
    preds_csv_path = os.path.join(args.models_dir, 'meta_holdout_predictions.csv')
    preds_df.to_csv(preds_csv_path, index=False)

    if args.verbose:
        print(f"\n{'='*60}")
        print("META-MODEL TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"💾 Model saved to: {model_path}")
        print(f"📊 Report saved to: {os.path.join(args.models_dir, 'meta_report.json')}")
        print(f"📄 Predictions saved to: {preds_csv_path}")

    # Plots
    plot_meta_predictions_vs_price(dates_te, prices_te, yte_m[:len(y_prob_test)], y_prob_test, args.plots_dir, args.test_start_date, args.verbose)
    plot_meta_learning_curves(Xtr_m_s, ytr_m, Xval_m_s, yval_m, chosen_C=best_C if best_C is not None else args.logreg_C, solver=args.logreg_solver, plots_dir=args.plots_dir, verbose=args.verbose)
    plot_meta_roc_pr(yval_m, val_probs, yte_m[:len(y_prob_test)], y_prob_test, args.plots_dir, verbose=args.verbose)
    plot_meta_threshold_sweep(yval_m, val_probs, yte_m[:len(y_prob_test)], y_prob_test, used_threshold, args.plots_dir, verbose=args.verbose)
    plot_meta_calibration(yte_m[:len(y_prob_test)], y_prob_test, args.plots_dir, verbose=args.verbose)


if __name__ == '__main__':
    main()


