import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, classification_report,
)

from config import RESULT_DIR, FEATURES_CSV, DATE_COL


def compute_strategy_metrics(dates, probs, labels, price_returns):
    """
    Realistic strategy: go long if pred > 0.5, short otherwise.
    Uses RAW (actual market) next-day price returns, not denoised.
    """
    preds = (np.array(probs) > 0.5).astype(int)
    labels = np.array(labels).astype(int)
    returns = np.array(price_returns)

    position = np.where(preds == 1, 1.0, -1.0)
    strategy_returns = position * returns

    cumulative = np.cumsum(strategy_returns)
    n = len(strategy_returns)
    total_return = strategy_returns.sum()

    if n > 1:
        sr = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252)
    else:
        sr = 0.0

    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min()
    win_rate = (strategy_returns > 0).mean()
    correct = (preds == labels).astype(int)

    return {
        'total_trades': n,
        'total_return': float(total_return),
        'avg_return': float(np.mean(strategy_returns)),
        'sharpe_ratio': float(sr),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'direction_accuracy': float(correct.mean()),
        'equity_curve': cumulative.tolist(),
        'drawdown_curve': drawdown.tolist(),
    }


def main():
    print("=" * 60)
    print(" Step 5 — Evaluation")
    print("=" * 60)

    with open(RESULT_DIR / "fold_results.json") as f:
        results = json.load(f)

    df_features = pd.read_csv(FEATURES_CSV, parse_dates=[DATE_COL])
    df_features = df_features.sort_values(DATE_COL).reset_index(drop=True)

    def normalize_date(d):
        return pd.Timestamp(d).strftime('%Y-%m-%d')

    date_to_return = {}
    for i in range(len(df_features)):
        d = normalize_date(df_features.loc[i, DATE_COL])
        if 'forward_1d_return' in df_features.columns:
            raw_ret = df_features.loc[i, 'forward_1d_return']
        elif 'raw_daily_return' in df_features.columns:
            raw_ret = df_features.loc[i, 'raw_daily_return']
        elif 'daily_return' in df_features.columns:
            raw_ret = df_features.loc[i, 'daily_return']
        else:
            raw_ret = 0.0
        if pd.notna(raw_ret):
            date_to_return[d] = raw_ret

    all_probs, all_labels, all_dates, all_returns = [], [], [], []
    fold_accs = []
    seen_dates = set()

    for fold_name in sorted(results.keys()):
        fold_data = results[fold_name]
        fold_probs = fold_data['preds']
        fold_labels = fold_data['labels']
        fold_dates = [normalize_date(d) for d in fold_data['dates']]

        for i, d in enumerate(fold_dates):
            if d in seen_dates:
                continue
            seen_dates.add(d)
            all_probs.append(fold_probs[i])
            all_labels.append(fold_labels[i])
            all_dates.append(d)
            ret = date_to_return.get(d, 0.0)
            all_returns.append(ret)

        print(f"  {fold_name}:  test_acc={fold_data['test_acc']:.4f}  val_acc={fold_data['val_acc']:.4f}")
        fold_accs.append(fold_data['test_acc'])

    probs = np.array(all_probs)
    labels = np.array(all_labels).astype(int)
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    print(f"\n  Overall Accuracy:  {acc:.4f}")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"  F1 Score:          {f1:.4f}")
    print(f"  MCC:               {mcc:.4f}")
    print(f"  Avg fold acc:      {np.mean(fold_accs):.4f}")
    print(f"  Total unique predictions: {len(labels)}")

    print(f"\n  Classification Report:")
    print(classification_report(labels, preds, target_names=['Down (0)', 'Up (1)']))

    strategy = compute_strategy_metrics(all_dates, probs, labels, all_returns)
    print(f"\n  Strategy Metrics (real percentage returns):")
    print(f"    Total Trades:       {strategy['total_trades']}")
    print(f"    Total Return:        {strategy['total_return']:.4f}")
    print(f"    Avg Return/trade:    {strategy['avg_return']:.6f}")
    print(f"    Sharpe Ratio:        {strategy['sharpe_ratio']:.4f}")
    print(f"    Max Drawdown:        {strategy['max_drawdown']:.4f}")
    print(f"    Win Rate:            {strategy['win_rate']:.4f}")
    print(f"    Direction Accuracy:  {strategy['direction_accuracy']:.4f}")

    cm = confusion_matrix(labels, preds)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    im = axes[0, 0].imshow(cm, cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_xticklabels(['Down (0)', 'Up (1)'])
    axes[0, 0].set_yticklabels(['Down (0)', 'Up (1)'])

    axes[0, 1].plot(strategy['equity_curve'], linewidth=0.8)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title(f"Equity Curve  (Sharpe={strategy['sharpe_ratio']:.2f})")
    axes[0, 1].set_xlabel('Trade')
    axes[0, 1].set_ylabel('Cumulative Return')

    axes[1, 0].fill_between(range(len(strategy['drawdown_curve'])),
                            0, strategy['drawdown_curve'], alpha=0.3, color='red')
    axes[1, 0].plot(strategy['drawdown_curve'], color='red', linewidth=0.8)
    axes[1, 0].set_title(f"Drawdown  (Max DD={strategy['max_drawdown']:.4f})")
    axes[1, 0].set_xlabel('Trade')
    axes[1, 0].set_ylabel('Drawdown')

    fold_names = [f"fold_{y}" for y in sorted([int(k.split('_')[1]) for k in results.keys()])]
    fold_acc_vals = [results[k]['test_acc'] for k in fold_names]
    axes[1, 1].bar(fold_names, fold_acc_vals, color='steelblue')
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--')
    axes[1, 1].set_title('Accuracy by Test Year')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "evaluation_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'mcc': float(mcc),
        'avg_fold_acc': float(np.mean(fold_accs)),
        'fold_accs': {k: v for k, v in zip(fold_names, fold_acc_vals)},
        'strategy': strategy,
    }
    with open(RESULT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics and plots saved to {RESULT_DIR}/")

    print("Evaluation complete.\n")


if __name__ == "__main__":
    main()