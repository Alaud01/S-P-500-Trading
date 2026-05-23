import sys
import importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import (
    FEATURES_CSV, MODEL_DIR, RESULT_DIR, DATE_COL, TARGET_COL,
    SEQUENCE_LENGTH, PREDICTION_HORIZON,
    TRAIN_START_YEAR, VAL_WINDOW_YEARS, TEST_START_YEAR, TEST_GAP_DAYS,
    D_MODEL, N_BLOCKS, NUM_HEADS, EXPAND_FACTOR, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
    GRAD_CLIP, LABEL_SMOOTHING, SEED,
)

xlstm = importlib.import_module('src.03_xlstm_model')
XLSTMTSModel = xlstm.XLSTMTSModel

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_and_prepare():
    df = pd.read_csv(FEATURES_CSV, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    exclude = [DATE_COL, TARGET_COL, 'raw_close', 'raw_open', 'raw_high', 'raw_low',
               'raw_daily_return', 'raw_log_return', 'forward_1d_return']
    feature_cols = [c for c in df.columns if c not in exclude]
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df[TARGET_COL].values.astype(np.float32)
    dates = df[DATE_COL].values

    assert not np.isnan(X_all).any(), "NaN values in features"
    assert not np.isinf(X_all).any(), "Inf values in features"

    return X_all, y_all, dates, feature_cols


def build_sequences(X, y):
    """
    (N, F) -> (N-S+1, S, F) input, (N-S+1,) target.
    Target is direction for next day after the sequence.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
        X_seq.append(X[i:i + SEQUENCE_LENGTH])
        y_seq.append(y[i + SEQUENCE_LENGTH + PREDICTION_HORIZON - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


class EarlyStopping:
    def __init__(self, patience, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_state = None
        self.best_acc = -float('inf')
        self.acc_counter = 0

    def step(self, metric, model, val_acc=None):
        if self.mode == 'val_loss_min_plus_acc':
            loss_improved = metric < self.best
            acc_improved = val_acc is not None and val_acc > self.best_acc
            if loss_improved:
                self.best = metric
                self.counter = 0
                self.best_state = copy.deepcopy(model.state_dict())
            else:
                self.counter += 1
            if acc_improved:
                self.best_acc = val_acc
                self.acc_counter = 0
                if not loss_improved:
                    self.best_state = copy.deepcopy(model.state_dict())
                    self.best = metric
                    self.counter = 0
            else:
                self.acc_counter += 1
            return self.counter >= self.patience and self.acc_counter >= self.patience
        else:
            improved = (self.mode == 'max' and metric > self.best) or \
                       (self.mode == 'min' and metric < self.best)
            if improved:
                self.best = metric
                self.counter = 0
                self.best_state = copy.deepcopy(model.state_dict())
            else:
                self.counter += 1
            return self.counter >= self.patience


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


def get_fold_ranges(dates, test_year):
    val_start = pd.Timestamp(year=test_year - VAL_WINDOW_YEARS, month=1, day=1)
    train_mask = dates < val_start
    val_mask = (dates >= val_start) & (dates < pd.Timestamp(year=test_year, month=1, day=1))
    test_start = pd.Timestamp(year=test_year, month=1, day=1) + pd.Timedelta(days=TEST_GAP_DAYS)
    test_mask = (dates >= test_start) & (dates < pd.Timestamp(year=test_year + 1, month=1, day=1))
    return train_mask, val_mask, test_mask


def train_fold(model, X_train, y_train, X_val, y_val, device, fold_label):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    ls = LABEL_SMOOTHING
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    stopper = EarlyStopping(patience=PATIENCE, mode='val_loss_min_plus_acc')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y_smooth = batch_y * (1 - ls) + 0.5 * ls
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y_smooth)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_loss += loss.item() * batch_x.size(0)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        model.eval()
        total_loss_v, total_correct_v, total_samples_v = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_y_smooth = batch_y * (1 - ls) + 0.5 * ls
                logits = model(batch_x)
                loss = criterion(logits, batch_y_smooth)
                probs = torch.sigmoid(logits)
                preds_v = (probs > 0.5).float()
                total_loss_v += loss.item() * batch_x.size(0)
                total_correct_v += (preds_v == batch_y).sum().item()
                total_samples_v += batch_x.size(0)
        val_loss = total_loss_v / total_samples_v
        val_acc = total_correct_v / total_samples_v

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}  |  t_loss={train_loss:.4f}  v_loss={val_loss:.4f}  "
                  f"t_acc={train_acc:.3f}  v_acc={val_acc:.3f}")

        if stopper.step(val_loss, model, val_acc=val_acc):
            print(f"    Early stopping at epoch {epoch}  (best val_loss={stopper.best:.4f})")
            break

    model.load_state_dict(stopper.best_state)
    return model, history, stopper.best_acc


def main():
    print("=" * 60)
    print(" Step 4 — Walk-Forward Training")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    X_all, y_all, dates, feature_cols = load_and_prepare()
    print(f"  Loaded features: {X_all.shape}  ({len(feature_cols)} features)")
    print(f"  Date range: {dates[0]} — {dates[-1]}")
    print(f"  Target distribution: 1={y_all.sum():.0f}  0={len(y_all)-y_all.sum():.0f}")
    print()

    available_years = sorted(set(pd.DatetimeIndex(dates).year))
    test_years = [y for y in available_years if y >= TEST_START_YEAR]

    all_fold_results = {}
    for test_year in test_years:
        print(f"--- Fold: Test Year {test_year} ---")

        train_mask, val_mask, test_mask = get_fold_ranges(dates, test_year)

        X_train_raw = X_all[train_mask]
        y_train_raw = y_all[train_mask]
        X_val_raw = X_all[val_mask]
        y_val_raw = y_all[val_mask]
        X_test_raw = X_all[test_mask]
        y_test_raw = y_all[test_mask]
        dates_test = dates[test_mask]

        print(f"  Sizes:  train={len(X_train_raw)}  val={len(X_val_raw)}  test={len(X_test_raw)}")

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_raw)
        X_val_s = scaler.transform(X_val_raw)
        X_test_s = scaler.transform(X_test_raw)

        X_train_seq, y_train_seq = build_sequences(X_train_s, y_train_raw)
        X_val_seq, y_val_seq = build_sequences(X_val_s, y_val_raw)
        X_test_seq, y_test_seq = build_sequences(X_test_s, y_test_raw)

        if len(X_test_seq) < 10:
            print(f"  Skipping fold {test_year} — insufficient test sequences ({len(X_test_seq)})")
            continue

        print(f"  Sequences: train={X_train_seq.shape}  val={X_val_seq.shape}  test={X_test_seq.shape}")

        n_features = X_all.shape[1]
        model = XLSTMTSModel(n_features=n_features).to(device)

        model, history, best_val = train_fold(
            model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, device, f"fold_{test_year}")

        _, test_acc, test_probs, test_labels = evaluate(
            model, DataLoader(
                TensorDataset(torch.from_numpy(X_test_seq), torch.from_numpy(y_test_seq)),
                batch_size=BATCH_SIZE, shuffle=False),
            nn.BCEWithLogitsLoss(), device)

        print(f"  Test accuracy: {test_acc:.4f}")

        fold_name = f"fold_{test_year}"
        torch.save({
            'model_state': model.state_dict(),
            'scaler': scaler,
            'feature_cols': feature_cols,
            'test_year': test_year,
            'val_acc': best_val,
        }, MODEL_DIR / f"xlstm_{fold_name}.pt")

        # We need dates aligned with test_seq predictions;
        # each prediction corresponds to the next day after sequence end
        seq_offset = SEQUENCE_LENGTH + PREDICTION_HORIZON - 1
        pred_dates = dates_test[seq_offset:]

        all_fold_results[fold_name] = {
            'dates': [str(d) for d in pred_dates],
            'preds': test_probs.tolist(),
            'labels': test_labels.tolist(),
            'test_acc': float(test_acc),
            'val_acc': float(best_val),
            'history': {k: [float(x) for x in v] for k, v in history.items()},
        }

        # Plot per-fold history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history['train_loss'], label='train')
        axes[0].plot(history['val_loss'], label='val')
        axes[0].set_title(f'Fold {test_year} — Loss')
        axes[0].legend()

        axes[1].plot(history['train_acc'], label='train')
        axes[1].plot(history['val_acc'], label='val')
        axes[1].axhline(y=0.5, color='gray', linestyle='--')
        axes[1].set_title(f'Fold {test_year} — Accuracy')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"training_history_{fold_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    with open(RESULT_DIR / "fold_results.json", "w") as f:
        json.dump(all_fold_results, f, indent=2)

    print(f"\nAll {len(test_years)} folds complete. Results saved to {RESULT_DIR}/fold_results.json")
    print("Training complete.\n")


if __name__ == "__main__":
    main()
