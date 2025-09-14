# Linear regression model training for S&P 500 prediction
# Usage: python train_linear.py --verbose
# Configurable: regularization, feature selection, cross-validation

import os
import json
import time
import math
import argparse
import warnings
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and preprocess dataset
    Configurable: input file path, missing value handling"""
    df = pd.read_csv(csv_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    # Forward/backward fill remaining missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


def select_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Select numerical features and target variable
    Configurable: feature selection criteria, target column name"""
    feature_df = df.select_dtypes(include=[np.number]).copy()
    if 'Target' not in feature_df.columns:
        raise ValueError("Target column 'Target' not found in dataset. Run built_dataset.py first.")
    y = feature_df['Target'].astype(int).values
    X = feature_df.drop(columns=['Target'])
    feature_names = list(X.columns)
    return X, y, feature_names


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


def time_series_cv_indices(n_rows: int,
                            n_splits: int,
                            val_window: int,
                            gap: int,
                            min_train_window: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start_val = max(min_train_window, 0)
    while start_val + gap + val_window <= n_rows and len(splits) < n_splits:
        train_end = start_val  # exclusive
        val_start = start_val + gap
        val_end = val_start + val_window
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        if len(train_idx) == 0 or len(val_idx) == 0:
            break
        splits.append((train_idx, val_idx))
        start_val = val_end
    if len(splits) == 0:
        raise ValueError("Unable to create any time-series CV splits. Consider reducing val_window or gap.")
    return splits


def cross_validate_linear(X: np.ndarray,
                            y: np.ndarray,
                            n_splits: int,
                            val_window: int,
                            gap: int,
                            min_train_window: int,
                            C_value: float,
                            solver: str,
                            verbose: bool = True,
                            threshold: float = 0.5) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, Any]]:
    """Perform time-series cross-validation for linear regression
    Configurable: regularization parameters, solvers, CV splits"""
    splits = time_series_cv_indices(n_rows=X.shape[0],
                                    n_splits=n_splits,
                                    val_window=val_window,
                                    gap=gap,
                                    min_train_window=min_train_window)

    fold_metrics: List[Dict[str, float]] = []
    fold_histories: List[Dict[str, List[float]]] = []
    # Out-of-fold predictions for overall CV metrics/plots
    oof_prob = np.full(X.shape[0], np.nan, dtype=np.float32)
    oof_true = np.full(X.shape[0], np.nan, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        if verbose:
            print(f"{'='*50}\nFOLD {fold}/{len(splits)}\n{'='*50}")
            print(f"  Train rows: [0, {train_idx[-1]}] ({len(train_idx)}) | Val rows: [{val_idx[0]}, {val_idx[-1]}] ({len(val_idx)}) | Gap: {gap}")

        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx]).astype(np.float32)
        X_val = scaler.transform(X[val_idx]).astype(np.float32)
        y_train, y_val = y[train_idx], y[val_idx]

        lr = LogisticRegression(C=C_value, solver=solver, class_weight='balanced', max_iter=2000, random_state=42)
        lr.fit(X_train, y_train)

        val_prob = lr.predict_proba(X_val)[:, 1]
        metrics = evaluate_binary(y_val, val_prob, threshold=threshold)
        fold_metrics.append(metrics)
        fold_histories.append({'val_auc': [metrics['auc']], 'val_f1': [metrics['f1']]})

        # store OOF
        oof_prob[val_idx] = val_prob.astype(np.float32)
        oof_true[val_idx] = y_val.astype(np.float32)

        if verbose:
            print(f"  Val AUC: {metrics['auc']:.4f} | Val F1@{threshold:.2f}: {metrics['f1']:.4f}")

    # Aggregate CV metrics
    agg = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}

    artifacts = {
        'fold_metrics': fold_metrics,
        'fold_histories': fold_histories,
        'oof_prob': [None if np.isnan(v) else float(v) for v in oof_prob.tolist()],
        'oof_true': [None if np.isnan(v) else int(v) for v in oof_true.tolist()],
    }
    return agg, fold_metrics, artifacts


def plot_training_history_linear(fold_metrics: List[Dict[str, float]], plots_dir: str, verbose: bool = True) -> None:
    if not verbose or not fold_metrics:
        return
    print("  📊 Generating training history plots...")
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Linear Model (LogReg) CV Performance', fontsize=16, fontweight='bold')

    # Plot 1: Average CV metrics
    metrics_names = list(fold_metrics[0].keys())
    avg_metrics = [np.mean([m[k] for m in fold_metrics]) for k in metrics_names]
    bars = axes[0].bar(metrics_names, avg_metrics, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0].set_title('Average CV Metrics')
    axes[0].set_ylabel('Score')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, avg_metrics):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Individual fold AUC and F1
    fold_aucs = [m['auc'] for m in fold_metrics]
    fold_f1s = [m['f1'] for m in fold_metrics]
    x = np.arange(len(fold_metrics))
    width = 0.35
    axes[1].bar(x - width/2, fold_aucs, width, label='AUC', alpha=0.7, color='lightcoral')
    axes[1].bar(x + width/2, fold_f1s, width, label='F1', alpha=0.7, color='lightgreen')
    axes[1].set_title('Per-Fold Performance')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(i+1) for i in range(len(fold_metrics))])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(plots_dir, 'linear_training_history.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  📈 Training history plot saved to: {out_path}")
    plt.show()


def plot_learning_curves_linear(Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray, C_value: float, solver: str, plots_dir: str, verbose: bool = True) -> None:
    if not verbose:
        return
    os.makedirs(plots_dir, exist_ok=True)
    sizes = np.linspace(0.2, 1.0, 6)
    train_auc_list, val_auc_list = [], []
    train_acc_list, val_acc_list = [], []
    for frac in sizes:
        n = max(20, int(len(Xtr) * float(frac)))
        X_sub, y_sub = Xtr[:n], ytr[:n]
        lr = LogisticRegression(C=C_value, solver=solver, class_weight='balanced', max_iter=2000, random_state=42)
        lr.fit(X_sub, y_sub)
        tr_probs = lr.predict_proba(X_sub)[:, 1]
        vl_probs = lr.predict_proba(Xval)[:, 1]
        tr_auc = roc_auc_score(y_sub, tr_probs) if len(np.unique(y_sub)) > 1 else 0.5
        vl_auc = roc_auc_score(yval, vl_probs) if len(np.unique(yval)) > 1 else 0.5
        tr_acc = accuracy_score(y_sub, (tr_probs >= 0.5).astype(int))
        vl_acc = accuracy_score(yval, (vl_probs >= 0.5).astype(int))
        train_auc_list.append(tr_auc)
        val_auc_list.append(vl_auc)
        train_acc_list.append(tr_acc)
        val_acc_list.append(vl_acc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Linear Model Learning Curves', fontsize=14, fontweight='bold')
    axes[0].plot(sizes, train_auc_list, label='Train AUC', marker='o')
    axes[0].plot(sizes, val_auc_list, label='Val AUC', marker='o')
    axes[0].set_xlabel('Training Fraction'); axes[0].set_ylabel('AUC'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(sizes, train_acc_list, label='Train Acc', marker='o')
    axes[1].plot(sizes, val_acc_list, label='Val Acc', marker='o')
    axes[1].set_xlabel('Training Fraction'); axes[1].set_ylabel('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    out_path = os.path.join(plots_dir, 'linear_learning_curves.png')
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  📈 Learning curves saved to: {out_path}")
    plt.show()


def plot_roc_pr_linear(yv: np.ndarray, pv: np.ndarray, yt: np.ndarray, pt: np.ndarray, plots_dir: str, verbose: bool = True) -> None:
    if not verbose:
        return
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Linear Model ROC and PR Curves', fontsize=14, fontweight='bold')
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
    out_path = os.path.join(plots_dir, 'linear_roc_pr.png')
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  📈 ROC/PR curves saved to: {out_path}")
    plt.show()


def plot_predictions_vs_price_linear(df: pd.DataFrame, dates: np.ndarray, prices: Optional[np.ndarray], y_true: np.ndarray, y_prob: np.ndarray, plots_dir: str, test_start_date: str, threshold: float = 0.5, verbose: bool = True) -> None:
    if not verbose:
        return
    print("  📊 Generating test set predictions vs price visualization...")
    os.makedirs(plots_dir, exist_ok=True)

    pred_binary = (y_prob >= threshold).astype(int)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Linear Model Predictions vs S&P 500 Price (Test Set: {test_start_date} onwards)', fontsize=16, fontweight='bold')

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
    ax2.set_title('Linear Model Prediction Probability (forward horizon)', fontweight='bold')
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
    out_path = os.path.join(plots_dir, 'linear_test_predictions_vs_price.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  📈 Test set predictions vs price plot saved to: {out_path}")
    plt.show()

    # Detailed analysis plot
    fig2, ((bx1, bx2), (bx3, bx4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f'Linear Model Test Set Analysis ({test_start_date} onwards)', fontsize=16, fontweight='bold')

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
    bx2.set_xlabel('Date'); bx2.set_ylabel('Confidence (0-1)'); bx2.grid(True, alpha=0.3)

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
        bx3.set_xlabel('Date'); bx3.set_ylabel('Accuracy'); bx3.legend(); bx3.grid(True, alpha=0.3)
    else:
        bx3.text(0.5, 0.5, 'Insufficient data for rolling accuracy', ha='center', va='center', transform=bx3.transAxes)
        bx3.set_title('Rolling Accuracy')

    bx4.scatter(y_true, y_prob, alpha=0.5, color='blue')
    bx4.set_title('Predictions vs Actual Values (Test Set)')
    bx4.set_xlabel('Actual (0=Down, 1=Up)'); bx4.set_ylabel('Predicted Probability'); bx4.grid(True, alpha=0.3)

    plt.tight_layout()
    detailed_path = os.path.join(plots_dir, 'linear_test_detailed_analysis.png')
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Test set detailed analysis plot saved to: {detailed_path}")
    plt.show()


def plot_training_vs_test_accuracy_linear(model: LogisticRegression,
                                            scaler: StandardScaler,
                                            data_csv: str,
                                            plots_dir: str,
                                            test_start_date: str,
                                            threshold: float = 0.5,
                                            window_size: int = 50,
                                            verbose: bool = True) -> None:
    if not verbose:
        return
    print("  📊 Generating training vs test accuracy comparison...")

    # Load and prepare data
    df = load_dataset(data_csv)
    X_df, y, _ = select_features_and_target(df)
    X = X_df.values.astype(np.float32)

    # Split by date
    test_start = pd.to_datetime(test_start_date)
    if 'Date' in df.columns:
        test_mask = df['Date'] >= test_start
        train_mask = ~test_mask
        train_dates = df['Date'][train_mask].values
        test_dates = df['Date'][test_mask].values
    else:
        split_idx = int(len(X) * 0.8)
        train_mask = np.arange(len(X)) < split_idx
        test_mask = ~train_mask
        train_dates = np.arange(train_mask.sum())
        test_dates = np.arange(test_mask.sum())

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    # Scale using provided scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    train_prob = model.predict_proba(X_train_scaled)[:, 1]
    test_prob = model.predict_proba(X_test_scaled)[:, 1]
    train_bin = (train_prob >= threshold).astype(int)
    test_bin = (test_prob >= threshold).astype(int)

    # Rolling accuracy
    train_roll, train_roll_dates = [], []
    test_roll, test_roll_dates = [], []
    if len(train_bin) >= window_size:
        for i in range(window_size, len(train_bin)):
            train_roll.append(np.mean(train_bin[i-window_size:i] == y_train[i-window_size:i]))
            train_roll_dates.append(train_dates[i])
    if len(test_bin) >= window_size:
        for i in range(window_size, len(test_bin)):
            test_roll.append(np.mean(test_bin[i-window_size:i] == y_test[i-window_size:i]))
            test_roll_dates.append(test_dates[i])

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Training vs Test Accuracy ({window_size}-day rolling)', fontsize=16, fontweight='bold')

    if train_roll:
        ax1.plot(train_roll_dates, train_roll, color='tab:blue', label='Train', linewidth=2)
    if test_roll:
        ax1.plot(test_roll_dates, test_roll, color='tab:red', label='Test', linewidth=2)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if 'Date' in df.columns:
        ax1.axvline(x=test_start, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Train/Test Split')
        ax1.legend()

    if train_roll:
        ax2.hist(train_roll, bins=20, alpha=0.6, density=True, label='Train', color='tab:blue')
    if test_roll:
        ax2.hist(test_roll, bins=20, alpha=0.6, density=True, label='Test', color='tab:red')
    ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Density')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if 'Date' in df.columns and (train_roll_dates or test_roll_dates):
        all_dates = []
        if train_roll_dates:
            all_dates.extend(train_roll_dates)
        if test_roll_dates:
            all_dates.extend(test_roll_dates)
        if all_dates:
            ax1.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
            ax1.xaxis.set_minor_locator(plt.matplotlib.dates.YearLocator(1))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            min_date, max_date = min(all_dates), max(all_dates)
            ax1.set_xlim(min_date, max_date)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, 'linear_training_vs_test_accuracy.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  📈 Training vs test accuracy plot saved to: {out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a linear (logistic regression) model for S&P 500 target prediction with time-series CV')
    parser.add_argument('--data-csv', type=str, default='data/final_dataset_for_modeling.csv')
    parser.add_argument('--models-dir', type=str, default='models/linear')
    parser.add_argument('--plots-dir', type=str, default='plots/linear')
    parser.add_argument('--test-start-date', type=str, default='2022-01-01')
    # CV configuration (align with LSTM defaults)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--val-window', type=int, default=180)
    parser.add_argument('--gap', type=int, default=14)
    parser.add_argument('--min-train-window', type=int, default=365)
    # Logistic Regression params
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--solver', type=str, default='lbfgs', choices=['lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'])
    parser.add_argument('--grid-C', type=str, default='')
    # Threshold handling
    parser.add_argument('--fixed-threshold', type=float, default=0.5, help='Fixed decision threshold; default 0.5')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Run a quick, lightweight training')
    args = parser.parse_args()

    set_seed(args.seed)

    if args.verbose:
        print(f"{'='*60}")
        print("LINEAR MODEL TRAINING (Logistic Regression)")
        print(f"{'='*60}")
        print(f"Data: {args.data_csv}")
        print(f"Models dir: {args.models_dir}")
        print(f"Plots dir: {args.plots_dir}")
        print(f"Test start date: {args.test_start_date}")
        print()

    df = load_dataset(args.data_csv)
    X_df, y, feature_names = select_features_and_target(df)
    X = X_df.values.astype(np.float32)

    if args.verbose:
        print(f"Rows: {X.shape[0]}, Features: {X.shape[1]}")

    # Smoke mode adjustments
    n_splits = args.n_splits if not args.smoke else 3
    val_window = args.val_window if not args.smoke else 120
    min_train_window = args.min_train_window if not args.smoke else 240
    gap = args.gap

    # Optional simple C grid over CV AUC
    try:
        c_grid = [float(x) for x in (args.grid_C.split(',') if args.grid_C else [])]
    except Exception:
        c_grid = []
    if not c_grid:
        c_grid = [args.C]

    best_cv_auc = -1.0
    best_C = c_grid[0]
    best_cv_metrics: Dict[str, float] = {}
    best_artifacts: Dict[str, Any] = {}
    for c_val in c_grid:
        cv_metrics, fold_metrics, artifacts = cross_validate_linear(
            X=X,
            y=y,
            n_splits=n_splits,
            val_window=val_window,
            gap=gap,
            min_train_window=min_train_window,
            C_value=c_val,
            solver=args.solver,
            verbose=args.verbose,
            threshold=float(args.fixed_threshold) if args.fixed_threshold is not None else 0.5,
        )
        if args.verbose:
            print(f"  CV with C={c_val}: AUC={cv_metrics['auc']:.4f}, F1={cv_metrics['f1']:.4f}")
        if cv_metrics['auc'] > best_cv_auc:
            best_cv_auc = cv_metrics['auc']
            best_C = c_val
            best_cv_metrics = cv_metrics
            best_artifacts = artifacts
            fold_metrics_best = fold_metrics

    # Plot CV history
    plot_training_history_linear(fold_metrics_best, args.plots_dir, args.verbose)

    # Determine holdout split (align with LSTM approach: 2022+ holdout)
    test_start_dt = pd.to_datetime(args.test_start_date)
    if 'Date' in df.columns:
        test_mask = df['Date'] >= test_start_dt
        pre_mask = ~test_mask
        dates_test_all = df['Date'][test_mask].values
        prices_test_all = df['Close'][test_mask].values if 'Close' in df.columns else None
    else:
        split_idx = int(len(X) * 0.85)
        pre_mask = np.zeros(len(X), dtype=bool)
        pre_mask[:split_idx] = True
        test_mask = ~pre_mask
        dates_test_all = np.arange(test_mask.sum())
        prices_test_all = None

    X_pre = X[pre_mask]
    y_pre = y[pre_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Within pre-holdout, create a tail validation window similar to LSTM holdout logic
    n_pre = X_pre.shape[0]
    val_window_holdout = min(val_window, max(1, n_pre // 5))
    val_start = max(0, n_pre - val_window_holdout)
    train_end = max(0, val_start - gap)
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(val_start, n_pre)

    # Fit scaler on training portion only, then fit LR on train and early-stop-like selection via val AUC
    if len(train_idx) == 0:
        train_idx = np.arange(0, max(1, n_pre - val_window_holdout))
        val_idx = np.arange(max(1, n_pre - val_window_holdout), n_pre)

    scaler = StandardScaler().fit(X_pre[train_idx])
    Xtr = scaler.transform(X_pre[train_idx]).astype(np.float32)
    Xval = scaler.transform(X_pre[val_idx]).astype(np.float32)
    ytr = y_pre[train_idx]
    yval = y_pre[val_idx]

    lr_final = LogisticRegression(C=best_C, solver=args.solver, class_weight='balanced', max_iter=2000, random_state=args.seed)
    lr_final.fit(Xtr, ytr)
    val_prob = lr_final.predict_proba(Xval)[:, 1]
    if args.verbose:
        print(f"Holdout pre-validation AUC: {roc_auc_score(yval, val_prob) if len(np.unique(yval))>1 else 0.5:.4f}")

    # Evaluate on holdout test using the fixed threshold
    Xte = scaler.transform(X_test).astype(np.float32)
    y_prob_test = lr_final.predict_proba(Xte)[:, 1]
    used_threshold = float(args.fixed_threshold) if args.fixed_threshold is not None else 0.5
    test_metrics = evaluate_binary(y_test, y_prob_test, threshold=used_threshold)

    # Build OOF vectors for CV ROC/PR (from best_artifacts)
    oof_prob_list = best_artifacts.get('oof_prob', [])
    oof_true_list = best_artifacts.get('oof_true', [])
    oof_mask = [p is not None and t is not None for p, t in zip(oof_prob_list, oof_true_list)]
    y_prob_val = np.array([oof_prob_list[i] for i in range(len(oof_prob_list)) if oof_mask[i]], dtype=np.float32)
    y_true_val = np.array([oof_true_list[i] for i in range(len(oof_true_list)) if oof_mask[i]], dtype=int)

    # Save artifacts
    os.makedirs(args.models_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_json_path = os.path.join(args.models_dir, f"linear_logreg_{timestamp}.json")
    model_artifact = {
        'type': 'logistic_regression',
        'feature_names': feature_names,
        'coef': lr_final.coef_.reshape(-1).tolist(),
        'intercept': float(lr_final.intercept_.reshape(-1)[0]) if hasattr(lr_final, 'intercept_') else 0.0,
        'solver': args.solver,
        'C': float(best_C),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'used_threshold': used_threshold,
        'test_start_date': args.test_start_date,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(model_json_path, 'w') as f:
        json.dump(model_artifact, f, indent=2)

    report = {
        'dataset_info': {
            'shape': X.shape,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'target_distribution': np.bincount(y).tolist(),
            'positive_class_ratio': float(y.mean())
        },
        'cv_metrics': best_cv_metrics,
        'cv_config': {
            'n_splits': n_splits,
            'val_window': val_window,
            'gap': gap,
            'min_train_window': min_train_window
        },
        'model': {
            'type': 'logistic_regression',
            'C': float(best_C),
            'solver': args.solver,
            'used_threshold': used_threshold,
            'model_path': model_json_path,
        },
        'test_metrics_since_' + args.test_start_date: test_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(args.models_dir, 'linear_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    if args.verbose:
        print(f"\n{'='*60}")
        print("LINEAR MODEL TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"💾 Model saved to: {model_json_path}")
        print(f"📊 Report saved to: {os.path.join(args.models_dir, 'linear_report.json')}")
        print(f"CV AUC: {best_cv_metrics.get('auc', float('nan')):.4f} | CV F1: {best_cv_metrics.get('f1', float('nan')):.4f}")

    # Plots
    os.makedirs(args.plots_dir, exist_ok=True)
    # Learning curves: use the pre-holdout train/val split
    plot_learning_curves_linear(Xtr, ytr, Xval, yval, C_value=best_C, solver=args.solver, plots_dir=args.plots_dir, verbose=args.verbose)
    # ROC/PR (CV OOF vs Test)
    if len(y_true_val) > 0:
        plot_roc_pr_linear(y_true_val, y_prob_val, y_test, y_prob_test, args.plots_dir, verbose=args.verbose)
    # Predictions vs price and detailed analysis
    plot_predictions_vs_price_linear(df, dates_test_all, prices_test_all, y_test, y_prob_test, args.plots_dir, args.test_start_date, threshold=used_threshold, verbose=args.verbose)
    # Rolling train vs test accuracy
    # Reconstruct model object from coefficients for plotting convenience (we already have lr_final)
    plot_training_vs_test_accuracy_linear(lr_final, scaler, args.data_csv, args.plots_dir, args.test_start_date, threshold=used_threshold, verbose=args.verbose)


if __name__ == '__main__':
    main()


