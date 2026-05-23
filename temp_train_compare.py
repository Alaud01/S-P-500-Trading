import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report,
)
import importlib

from config import (
    FEATURES_CSV, DATE_COL, TARGET_COL, SEQUENCE_LENGTH, PREDICTION_HORIZON,
    TRAIN_START_YEAR, VAL_WINDOW_YEARS, TEST_START_YEAR,
    D_MODEL, N_BLOCKS, NUM_HEADS, EXPAND_FACTOR, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
    GRAD_CLIP, SEED,
)

xlstm = importlib.import_module('src.03_xlstm_model')
XLSTMTSModel = xlstm.XLSTMTSModel

torch.manual_seed(SEED)
np.random.seed(SEED)

PRUNED_FEATURES = [
    "ret_3d", "return_lag_1", "volume_lag_3_ratio", "dayofyear_sin",
    "price_MA60_ratio", "ret_1d", "daily_return", "log_return",
    "GDP_yoy_zscore", "volume_lag_1_ratio", "OBV_roc_10",
    "close_open_ratio", "Gold_return", "dayofyear_cos", "Sentiment_zscore",
    "price_MA14_ratio", "Unemployment_delta", "MACD_signal_pct",
    "stoch_d", "price_MA30_ratio", "volatility", "return_lag_7",
    "RSI", "Interest_rate_delta",
]


class EarlyStopping:
    def __init__(self, patience, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, metric, model):
        improved = (self.mode == 'max' and metric > self.best) or \
                   (self.mode == 'min' and metric < self.best)
        if improved:
            self.best = metric
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience


def build_sequences(X, y):
    X_seq, y_seq = [], []
    for i in range(len(X) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
        X_seq.append(X[i:i + SEQUENCE_LENGTH])
        y_seq.append(y[i + SEQUENCE_LENGTH + PREDICTION_HORIZON - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def get_fold_ranges(dates, test_year):
    train_mask = dates < pd.Timestamp(year=test_year - VAL_WINDOW_YEARS, month=1, day=1)
    val_mask = (dates >= pd.Timestamp(year=test_year - VAL_WINDOW_YEARS, month=1, day=1)) & \
               (dates < pd.Timestamp(year=test_year, month=1, day=1))
    test_mask = (dates >= pd.Timestamp(year=test_year, month=1, day=1)) & \
                (dates < pd.Timestamp(year=test_year + 1, month=1, day=1))
    return train_mask, val_mask, test_mask


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_loss += loss.item() * batch_x.size(0)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_x.size(0)
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total_loss += loss.item() * batch_x.size(0)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    return total_loss / total_samples, total_correct / total_samples, \
           np.array(all_probs), np.array(all_labels)


def compute_strategy_metrics(probs, labels, returns):
    preds = (np.array(probs) > 0.5).astype(int)
    labels = np.array(labels).astype(int)
    returns = np.array(returns)
    position = np.where(preds == 1, 1.0, -1.0)
    strategy_returns = position * returns
    cumulative = np.cumsum(strategy_returns)
    n = len(strategy_returns)
    total_return = strategy_returns.sum()
    sr = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252) if n > 1 else 0.0
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min()
    win_rate = (strategy_returns > 0).mean()
    correct = (preds == labels).astype(int)
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sr),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'direction_accuracy': float(correct.mean()),
    }


def run_experiment(feature_cols, label, df, device):
    print("=" * 70)
    print(f"  EXPERIMENT: {label}")
    print(f"  Features: {len(feature_cols)}")
    print("=" * 70)

    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df[TARGET_COL].values.astype(np.float32)
    dates = df[DATE_COL].values

    assert not np.isnan(X_all).any(), "NaN in features"
    assert not np.isinf(X_all).any(), "Inf in features"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    available_years = sorted(set(pd.DatetimeIndex(dates).year))
    test_years = [y for y in available_years if y >= TEST_START_YEAR]

    all_probs, all_labels, all_returns = [], [], []
    fold_results = {}
    seen_dates = set()

    for test_year in test_years:
        print(f"\n--- Fold: Test Year {test_year} ---")
        train_mask, val_mask, test_mask = get_fold_ranges(dates, test_year)

        X_train_raw = X_all[train_mask]
        y_train_raw = y_all[train_mask]
        X_val_raw = X_all[val_mask]
        y_val_raw = y_all[val_mask]
        X_test_raw = X_all[test_mask]
        y_test_raw = y_all[test_mask]
        dates_test = dates[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_raw)
        X_val_s = scaler.transform(X_val_raw)
        X_test_s = scaler.transform(X_test_raw)

        X_train_seq, y_train_seq = build_sequences(X_train_s, y_train_raw)
        X_val_seq, y_val_seq = build_sequences(X_val_s, y_val_raw)
        X_test_seq, y_test_seq = build_sequences(X_test_s, y_test_raw)

        if len(X_test_seq) < 10:
            print(f"  Skipping fold {test_year}")
            continue

        print(f"  Sequences: train={X_train_seq.shape} val={X_val_seq.shape} test={X_test_seq.shape}")

        n_features = len(feature_cols)
        model = XLSTMTSModel(n_features=n_features).to(device)

        n_pos = y_train_seq.sum()
        n_neg = len(y_train_seq) - n_pos
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

        train_ds = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
        val_ds = TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val_seq))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        stopper = EarlyStopping(patience=PATIENCE, mode='max')

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_acc)

            if epoch % 20 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d}  t_loss={train_loss:.4f}  v_loss={val_loss:.4f}  t_acc={train_acc:.3f}  v_acc={val_acc:.3f}")

            if stopper.step(val_acc, model):
                print(f"    Early stopping at epoch {epoch}  (best val_acc={stopper.best:.4f})")
                break

        model.load_state_dict(stopper.best_state)

        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test_seq), torch.from_numpy(y_test_seq)),
            batch_size=BATCH_SIZE, shuffle=False)
        _, test_acc, test_probs, test_labels = evaluate(model, test_loader, nn.BCEWithLogitsLoss(), device)

        print(f"  Test accuracy: {test_acc:.4f}")

        seq_offset = SEQUENCE_LENGTH + PREDICTION_HORIZON - 1
        pred_dates = dates_test[seq_offset:]

        fold_results[f"fold_{test_year}"] = {
            'test_acc': float(test_acc),
            'val_acc': float(stopper.best),
            'n_test': len(test_probs),
        }

        for i, d in enumerate(pred_dates):
            d_str = pd.Timestamp(d).strftime('%Y-%m-%d')
            if d_str in seen_dates:
                continue
            seen_dates.add(d_str)
            all_probs.append(test_probs[i])
            all_labels.append(test_labels[i])

            d_ts = pd.Timestamp(d)
            row_idx = df.index[df[DATE_COL] == d_ts]
            if len(row_idx) > 0 and 'raw_daily_return' in df.columns:
                all_returns.append(df.loc[row_idx[0], 'raw_daily_return'])
            else:
                all_returns.append(0.0)

    probs = np.array(all_probs)
    labels = np.array(all_labels).astype(int)
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    strategy = compute_strategy_metrics(probs, labels, all_returns)

    fold_accs = [fold_results[k]['test_acc'] for k in sorted(fold_results.keys())]

    result = {
        'label': label,
        'n_features': len(feature_cols),
        'features': feature_cols,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'mcc': float(mcc),
        'avg_fold_acc': float(np.mean(fold_accs)),
        'fold_accs': {k: v['test_acc'] for k, v in sorted(fold_results.items())},
        'strategy': strategy,
    }

    print(f"\n  === {label} Results ===")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1:         {f1:.4f}")
    print(f"  MCC:        {mcc:.4f}")
    print(f"  Avg Fold:   {np.mean(fold_accs):.4f}")
    print(f"  Sharpe:     {strategy['sharpe_ratio']:.4f}")
    print(f"  Max DD:     {strategy['max_drawdown']:.4f}")
    print(f"  Win Rate:    {strategy['win_rate']:.4f}")
    print(f"  Total Ret:  {strategy['total_return']:.4f}")
    for fold, fa in sorted(fold_results.items()):
        print(f"    {fold}: {fa['test_acc']:.4f}")

    return result


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    exclude = [DATE_COL, TARGET_COL, 'raw_close', 'raw_open', 'raw_high', 'raw_low',
               'raw_daily_return', 'raw_log_return']
    all_feature_cols = [c for c in df.columns if c not in exclude]

    pruned_feature_cols = [c for c in PRUNED_FEATURES if c in all_feature_cols]

    print(f"All features: {len(all_feature_cols)}")
    print(f"Pruned features: {len(pruned_feature_cols)}")
    print(f"Pruned list: {pruned_feature_cols}")

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    result_a = run_experiment(all_feature_cols, "ALL_FEATURES (58)", df, device)

    print("\n" + "#" * 70)
    print("  Switching to pruned feature set...")
    print("#" * 70 + "\n")

    result_b = run_experiment(pruned_feature_cols, "PRUNED_FEATURES (24)", df, device)

    print("\n\n" + "=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<20} {'ALL (58)':>12} {'PRUNED (24)':>14} {'Winner':>10}")
    print("-" * 60)

    metrics_to_compare = [
        ('accuracy', 'Accuracy', 'higher'),
        ('mcc', 'MCC', 'higher'),
        ('f1', 'F1', 'higher'),
        ('precision', 'Precision', 'higher'),
        ('recall', 'Recall', 'higher'),
        ('avg_fold_acc', 'Avg Fold Acc', 'higher'),
    ]

    for key, label, direction in metrics_to_compare:
        a_val = result_a[key]
        b_val = result_b[key]
        if direction == 'higher':
            winner = "ALL" if a_val > b_val else ("PRUNED" if b_val > a_val else "TIE")
        else:
            winner = "ALL" if a_val < b_val else ("PRUNED" if b_val < a_val else "TIE")
        print(f"{label:<20} {a_val:>12.4f} {b_val:>14.4f} {winner:>10}")

    strat_metrics = [
        ('sharpe_ratio', 'Sharpe', 'higher'),
        ('max_drawdown', 'Max DD', 'lower'),
        ('win_rate', 'Win Rate', 'higher'),
        ('total_return', 'Total Ret', 'higher'),
    ]

    for key, label, direction in strat_metrics:
        a_val = result_a['strategy'][key]
        b_val = result_b['strategy'][key]
        if direction == 'higher':
            winner = "ALL" if a_val > b_val else ("PRUNED" if b_val > a_val else "TIE")
        else:
            winner = "ALL" if a_val < b_val else ("PRUNED" if b_val < a_val else "TIE")
        print(f"{label:<20} {a_val:>12.4f} {b_val:>14.4f} {winner:>10}")

    print("\nPer-fold accuracy comparison:")
    all_folds = sorted(set(list(result_a['fold_accs'].keys()) + list(result_b['fold_accs'].keys())))
    print(f"{'Fold':<12} {'ALL':>10} {'PRUNED':>10} {'Winner':>10}")
    print("-" * 45)
    for fold in all_folds:
        a = result_a['fold_accs'].get(fold, float('nan'))
        b = result_b['fold_accs'].get(fold, float('nan'))
        winner = "ALL" if a > b else ("PRUNED" if b > a else "TIE")
        print(f"{fold:<12} {a:>10.4f} {b:>10.4f} {winner:>10}")

    comparison = {'all_features': result_a, 'pruned_features': result_b}
    with open("outputs/results/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved to outputs/results/comparison.json")


if __name__ == "__main__":
    main()