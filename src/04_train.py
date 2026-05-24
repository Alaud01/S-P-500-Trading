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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    FEATURES_CSV, MODEL_DIR, RESULT_DIR, DATE_COL, TARGET_COL,
    SEQUENCE_LENGTH, PREDICTION_HORIZON,
    TRAIN_START_YEAR, VAL_WINDOW_YEARS, TEST_START_YEAR,
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
    dates = pd.DatetimeIndex(df[DATE_COL])

    assert not np.isnan(X_all).any(), "NaN values in features"
    assert not np.isinf(X_all).any(), "Inf values in features"

    return X_all, y_all, dates, feature_cols


def get_label_known_dates(dates):
    """
    A row dated T has a 1-day-forward label that is only known after the
    next available trading close. Generalized to PREDICTION_HORIZON rows.
    """
    if PREDICTION_HORIZON != 1:
        raise ValueError("Walk-forward label availability currently expects PREDICTION_HORIZON=1")
    label_known = pd.Series(dates).shift(-PREDICTION_HORIZON)
    return pd.DatetimeIndex(label_known)


def build_sequences_for_indices(X, y, target_indices):
    """
    Build sequences whose prediction date is the target row itself.
    For target index i, features are X[i-SEQUENCE_LENGTH+1 : i+1] and label is y[i].
    """
    X_seq, y_seq = [], []
    used_indices = []
    for idx in target_indices:
        start = idx - SEQUENCE_LENGTH + 1
        if start < 0:
            continue
        X_seq.append(X[start:idx + 1])
        y_seq.append(y[idx])
        used_indices.append(idx)
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(used_indices, dtype=np.int64),
    )


def build_prediction_sequences(X, target_indices):
    X_seq, used_indices = [], []
    for idx in target_indices:
        start = idx - SEQUENCE_LENGTH + 1
        if start < 0:
            continue
        X_seq.append(X[start:idx + 1])
        used_indices.append(idx)
    return np.array(X_seq, dtype=np.float32), np.array(used_indices, dtype=np.int64)


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


def get_retrain_indices(dates):
    test_start = pd.Timestamp(year=TEST_START_YEAR, month=1, day=1)
    test_indices = np.flatnonzero(dates >= test_start)
    if len(test_indices) == 0:
        raise ValueError(f"No dates found at or after TEST_START_YEAR={TEST_START_YEAR}")

    test_frame = pd.DataFrame({
        'idx': test_indices,
        'quarter': dates[test_indices].to_period('Q'),
    })
    retrain_indices = test_frame.groupby('quarter', sort=True)['idx'].first().to_numpy()
    return test_indices, retrain_indices


def get_train_val_indices(dates, label_known_dates, retrain_date):
    eligible = (
        (dates >= pd.Timestamp(year=TRAIN_START_YEAR, month=1, day=1)) &
        pd.notna(label_known_dates) &
        (label_known_dates <= retrain_date)
    )
    val_start = retrain_date - pd.DateOffset(years=VAL_WINDOW_YEARS)
    train_indices = np.flatnonzero(eligible & (dates < val_start))
    val_indices = np.flatnonzero(eligible & (dates >= val_start) & (dates < retrain_date))

    if len(train_indices) == 0:
        raise ValueError(f"No training rows available for retrain date {retrain_date.date()}")
    if len(val_indices) == 0:
        raise ValueError(f"No validation rows available for retrain date {retrain_date.date()}")

    assert label_known_dates[train_indices].max() <= retrain_date
    assert label_known_dates[val_indices].max() <= retrain_date
    return train_indices, val_indices, val_start


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
    print(" Step 4 — Leakage-Free Daily Walk-Forward Training")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    X_all, y_all, dates, feature_cols = load_and_prepare()
    label_known_dates = get_label_known_dates(dates)
    print(f"  Loaded features: {X_all.shape}  ({len(feature_cols)} features)")
    print(f"  Date range: {dates[0]} — {dates[-1]}")
    print(f"  Target distribution: 1={y_all.sum():.0f}  0={len(y_all)-y_all.sum():.0f}")
    print()

    test_indices, retrain_indices = get_retrain_indices(dates)
    print(f"  Backtest dates: {dates[test_indices[0]].date()} to {dates[test_indices[-1]].date()}")
    print(f"  Daily predictions: {len(test_indices)}")
    print(f"  Quarterly retrains: {len(retrain_indices)}")

    all_fold_results = {}
    retrain_boundaries = list(retrain_indices) + [test_indices[-1] + 1]
    for fold_no, retrain_idx in enumerate(retrain_indices):
        retrain_date = dates[retrain_idx]
        next_boundary = retrain_boundaries[fold_no + 1]
        pred_indices = test_indices[(test_indices >= retrain_idx) & (test_indices < next_boundary)]
        if len(pred_indices) == 0:
            continue

        quarter = retrain_date.to_period('Q')
        fold_name = f"retrain_{quarter.year}_Q{quarter.quarter}"
        print(f"--- {fold_name}: retrain on {retrain_date.date()} ---")

        train_indices, val_indices, val_start = get_train_val_indices(
            dates, label_known_dates, retrain_date)

        print(
            f"  Split: train={len(train_indices)} "
            f"({dates[train_indices[0]].date()} to {dates[train_indices[-1]].date()}), "
            f"val={len(val_indices)} "
            f"({dates[val_indices[0]].date()} to {dates[val_indices[-1]].date()}), "
            f"predict={len(pred_indices)} "
            f"({dates[pred_indices[0]].date()} to {dates[pred_indices[-1]].date()})"
        )

        scaler = StandardScaler()
        scaler.fit(X_all[train_indices])
        assert scaler.n_samples_seen_ == len(train_indices), "Scaler fit row count mismatch"

        X_all_s = scaler.transform(X_all)
        X_train_seq, y_train_seq, train_seq_indices = build_sequences_for_indices(
            X_all_s, y_all, train_indices)
        X_val_seq, y_val_seq, val_seq_indices = build_sequences_for_indices(
            X_all_s, y_all, val_indices)
        X_test_seq, test_seq_indices = build_prediction_sequences(X_all_s, pred_indices)
        y_test_seq = y_all[test_seq_indices]

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            raise ValueError(f"Insufficient train/val sequences for {fold_name}")
        if len(X_test_seq) != len(pred_indices):
            missing = len(pred_indices) - len(X_test_seq)
            raise ValueError(f"{fold_name} missing {missing} prediction sequences")

        print(f"  Sequences: train={X_train_seq.shape}  val={X_val_seq.shape}  test={X_test_seq.shape}")
        assert dates[test_seq_indices].max() <= dates[pred_indices[-1]]
        for pred_idx in test_seq_indices:
            context_start = pred_idx - SEQUENCE_LENGTH + 1
            context_dates = dates[context_start:pred_idx + 1]
            assert context_dates.max() <= dates[pred_idx], "Prediction context uses future data"

        n_features = X_all.shape[1]
        model = XLSTMTSModel(n_features=n_features).to(device)

        model, history, best_val = train_fold(
            model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, device, fold_name)

        _, test_acc, test_probs, test_labels = evaluate(
            model, DataLoader(
                TensorDataset(torch.from_numpy(X_test_seq), torch.from_numpy(y_test_seq)),
                batch_size=BATCH_SIZE, shuffle=False),
            nn.BCEWithLogitsLoss(), device)

        print(f"  Window accuracy: {test_acc:.4f}")

        torch.save({
            'model_state': model.state_dict(),
            'scaler': scaler,
            'feature_cols': feature_cols,
            'retrain_date': str(retrain_date.date()),
            'prediction_start': str(dates[test_seq_indices[0]].date()),
            'prediction_end': str(dates[test_seq_indices[-1]].date()),
            'train_start': str(dates[train_indices[0]].date()),
            'train_end': str(dates[train_indices[-1]].date()),
            'val_start': str(dates[val_indices[0]].date()),
            'val_end': str(dates[val_indices[-1]].date()),
            'label_cutoff': str(retrain_date.date()),
            'scaler_fit_rows': int(len(train_indices)),
            'val_acc': best_val,
        }, MODEL_DIR / f"xlstm_{fold_name}.pt")

        pred_dates = dates[test_seq_indices]

        all_fold_results[fold_name] = {
            'dates': [str(d) for d in pred_dates],
            'preds': test_probs.tolist(),
            'labels': test_labels.tolist(),
            'test_acc': float(test_acc),
            'val_acc': float(best_val),
            'metadata': {
                'retrain_date': str(retrain_date.date()),
                'prediction_start': str(pred_dates[0].date()),
                'prediction_end': str(pred_dates[-1].date()),
                'train_start': str(dates[train_indices[0]].date()),
                'train_end': str(dates[train_indices[-1]].date()),
                'val_start': str(dates[val_indices[0]].date()),
                'val_end': str(dates[val_indices[-1]].date()),
                'label_cutoff': str(retrain_date.date()),
                'validation_policy': f'rolling_prior_{VAL_WINDOW_YEARS}y',
                'scaler_fit_rows': int(len(train_indices)),
                'train_sequence_rows': int(len(train_seq_indices)),
                'val_sequence_rows': int(len(val_seq_indices)),
            },
            'history': {k: [float(x) for x in v] for k, v in history.items()},
        }

        # Plot per-fold history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history['train_loss'], label='train')
        axes[0].plot(history['val_loss'], label='val')
        axes[0].set_title(f'{fold_name} — Loss')
        axes[0].legend()

        axes[1].plot(history['train_acc'], label='train')
        axes[1].plot(history['val_acc'], label='val')
        axes[1].axhline(y=0.5, color='gray', linestyle='--')
        axes[1].set_title(f'{fold_name} — Accuracy')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"training_history_{fold_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    with open(RESULT_DIR / "fold_results.json", "w") as f:
        json.dump(all_fold_results, f, indent=2)

    all_dates = [pd.Timestamp(d) for fold in all_fold_results.values() for d in fold['dates']]
    assert min(all_dates) == pd.Timestamp('2021-01-04'), "First prediction date is not 2021-01-04"
    assert max(all_dates) == dates[test_indices[-1]], "Last prediction date does not match feature end"
    assert len(set(all_dates)) == len(test_indices), "Prediction dates contain gaps or duplicates"

    print(f"\nAll {len(all_fold_results)} retrain windows complete. Results saved to {RESULT_DIR}/fold_results.json")
    print(f"  Prediction coverage: {min(all_dates).date()} to {max(all_dates).date()} ({len(all_dates)} trading days)")
    print("Training complete.\n")


if __name__ == "__main__":
    main()
