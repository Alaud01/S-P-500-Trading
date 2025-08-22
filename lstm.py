import os
import json
import math
import time
import random
import argparse
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    # Forward/backward fill remaining missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


def select_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    feature_df = df.select_dtypes(include=[np.number]).copy()
    if 'Target' not in feature_df.columns:
        raise ValueError("Target column 'Target' not found in dataset. Run built_dataset.py first.")
    y = feature_df['Target'].astype(int).values
    X = feature_df.drop(columns=['Target'])
    feature_names = list(X.columns)
    return X, y, feature_names


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    # Create sequences where each label corresponds to the last timestep in the window
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


def time_series_cv_indices(n_effective: int,
        n_splits: int,
        val_window: int,
        gap: int,
        min_train_window: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    start_val = max(min_train_window, 0)
    # Ensure room for gap and validation window
    while start_val + gap + val_window <= n_effective and len(splits) < n_splits:
        train_end = start_val  # exclusive for train effective index end
        train_idx = np.arange(0, train_end)
        val_start = start_val + gap
        val_end = val_start + val_window
        val_idx = np.arange(val_start, val_end)
        if len(train_idx) == 0 or len(val_idx) == 0:
            break
        splits.append((train_idx, val_idx))
        start_val = val_end  # rolling origin
    if len(splits) == 0:
        raise ValueError("Unable to create any time-series CV splits. Consider reducing val_window or gap.")
    return splits


class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray, indices: np.ndarray):
        self.X = X_seq[indices]
        self.y = y_seq[indices]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


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
        # x: (batch, seq, features)
        output, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            last_hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            last_hidden = hn[-1]
        logits = self.fc(self.dropout(last_hidden))
        return logits.squeeze(-1)


def compute_class_pos_weight(labels: np.ndarray) -> float:
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    if pos == 0:
        return 1.0
    return max(neg / max(pos, 1.0), 1.0)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    all_probs: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb.float())
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(yb.detach().cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
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


@torch.no_grad()
def collect_probs_and_labels(model: nn.Module,
                             loader: DataLoader,
                             device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list: List[float] = []
    labels_list: List[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb.float())
        probs = torch.sigmoid(logits)
        probs_list.extend(probs.detach().cpu().numpy().tolist())
        labels_list.extend(yb.detach().cpu().numpy().tolist())
    return np.array(probs_list), np.array(labels_list)


def find_best_threshold(probs: np.ndarray,
                        labels: np.ndarray,
                        metric: str = 'f1') -> Tuple[float, float]:
    best_thr = 0.5
    best_score = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        preds = (probs >= thr).astype(int)
        if metric == 'youden':
            # Youden's J = TPR - FPR
            tp = np.logical_and(preds == 1, labels == 1).sum()
            tn = np.logical_and(preds == 0, labels == 0).sum()
            fp = np.logical_and(preds == 1, labels == 0).sum()
            fn = np.logical_and(preds == 0, labels == 1).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            score = tpr - fpr
        else:
            _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
            score = float(f1)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def train_one_fold(X: np.ndarray,
                   y: np.ndarray,
                   feature_names: List[str],
                   seq_len: int,
                   train_idx_eff: np.ndarray,
                   val_idx_eff: np.ndarray,
                   params: Dict[str, Any],
                   device: torch.device) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[float]], float]:
    # Effective index t corresponds to sequence index s = t - (seq_len - 1)
    def eff_to_seq_idx(eff_idx: np.ndarray) -> np.ndarray:
        return eff_idx - (seq_len - 1)

    # Fit scaler on rows up to last training effective index (inclusive)
    last_train_eff = int(train_idx_eff.max())
    scaler_rows_end = last_train_eff + 1
    scaler = StandardScaler()
    scaler.fit(X[:scaler_rows_end])
    X_scaled = scaler.transform(X)

    # Build sequences from scaled features
    X_seq, y_seq = build_sequences(X_scaled, y, seq_len)

    train_seq_idx = eff_to_seq_idx(train_idx_eff)
    val_seq_idx = eff_to_seq_idx(val_idx_eff)

    train_ds = SequenceDataset(X_seq, y_seq, train_seq_idx)
    val_ds = SequenceDataset(X_seq, y_seq, val_seq_idx)

    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, drop_last=False)
    
    if params['verbose']:
        print(f"  Training samples: {len(train_ds)}")
        print(f"  Validation samples: {len(val_ds)}")
        print(f"  Input features: {X.shape[1]}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Batch size: {params['batch_size']}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")

    model = LSTMClassifier(
        input_size=X.shape[1],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        bidirectional=params.get('bidirectional', False)
    ).to(device)

    pos_weight_value = compute_class_pos_weight(y_seq[train_seq_idx])
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = None
    if params.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=params.get('scheduler_factor', 0.5),
            patience=params.get('scheduler_patience', 2),
            min_lr=params.get('min_lr', 1e-5)
        )

    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0
    
    # Track training history
    history = {
        'train_loss': [],
        'train_auc': [],
        'train_f1': [],
        'val_auc': [],
        'val_f1': [],
        'val_loss': [],
        'lr': []
    }

    if params['verbose']:
        print(f"  Starting training for {params['epochs']} epochs...")
        print(f"  Early stopping patience: {params['patience']} epochs")
        print(f"  Logging every {params['log_every']} epochs")
        print(f"  Positive class weight: {pos_weight_value:.3f}")
        print()
        
    best_threshold = 0.5
    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            logits = model(xb.float())
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            num_batches += 1

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_metrics = evaluate(model, train_loader, device, threshold=0.5)

        # Validation loss and metrics + threshold tuning
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.float().to(device)
                logits = model(xb.float())
                loss = criterion(logits, yb)
                val_epoch_loss += loss.item() * xb.size(0)
        val_avg_loss = val_epoch_loss / len(val_loader.dataset)

        # Collect validation probabilities to find best threshold
        val_probs, val_labels = collect_probs_and_labels(model, val_loader, device)
        tuned_thr, tuned_score = find_best_threshold(val_probs, val_labels, metric=params.get('threshold_metric', 'f1'))
        current_val_metrics = evaluate(model, val_loader, device, threshold=tuned_thr)
        val_metrics = current_val_metrics
        if val_metrics['f1'] >= tuned_score - 1e-9:
            best_threshold = tuned_thr
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_auc'].append(train_metrics['auc'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_loss'].append(val_avg_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state = {
                'model': model.state_dict(),
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'feature_names': feature_names,
                'seq_len': seq_len,
                'params': params,
            }
            epochs_no_improve = 0
            improvement = "★ NEW BEST"
        else:
            epochs_no_improve += 1
            improvement = ""

        if scheduler is not None:
            scheduler.step(val_metrics['auc'])

        if params['verbose'] and (epoch % max(1, params['log_every']) == 0 or epoch == params['epochs'] - 1):
            print(f"  Epoch {epoch+1:3d}/{params['epochs']} | "
                  f"TrainLoss: {avg_train_loss:.4f} | ValLoss: {val_avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Train AUC: {train_metrics['auc']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Train F1: {train_metrics['f1']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"No Improve: {epochs_no_improve}/{params['patience']} {improvement}")

        if epochs_no_improve >= params['patience']:
            if params['verbose']:
                print(f"  ⚠️  Early stopping at epoch {epoch+1}. Best Val AUC: {best_val_auc:.4f}")
            break

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state['model'])

    final_train_metrics = evaluate(model, train_loader, device, threshold=best_threshold)
    final_val_metrics = evaluate(model, val_loader, device, threshold=best_threshold)

    return final_train_metrics, final_val_metrics, history, best_threshold


def cross_validate(X: np.ndarray,
                   y: np.ndarray,
                   feature_names: List[str],
                   seq_len: int,
                   n_splits: int,
                   val_window: int,
                   gap: int,
                   min_train_window: int,
                   params: Dict[str, Any],
                   device: torch.device) -> Tuple[Dict[str, float], List[Dict[str, float]], List[Dict[str, List[float]]]]:
    n_effective = X.shape[0] - seq_len + 1
    splits = time_series_cv_indices(n_effective, n_splits, val_window, gap, min_train_window)
    fold_metrics: List[Dict[str, float]] = []
    fold_histories: List[Dict[str, List[float]]] = []

    if params['verbose']:
        print(f"Dataset info:")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Effective samples: {n_effective}")
        print(f"  CV splits: {len(splits)}")
        print()

    for fold, (train_eff, val_eff) in enumerate(splits, start=1):
        if params['verbose']:
            print(f"{'='*50}")
            print(f"FOLD {fold}/{len(splits)}")
            print(f"{'='*50}")
            print(f"  Training period: [0, {train_eff[-1]}] ({len(train_eff)} samples)")
            print(f"  Validation period: [{val_eff[0]}, {val_eff[-1]}] ({len(val_eff)} samples)")
            print(f"  Gap: {gap} samples")
            print()
            
        train_metrics, val_metrics, history, _ = train_one_fold(X, y, feature_names, seq_len, train_eff, val_eff, params, device)
        fold_metrics.append(val_metrics)
        fold_histories.append(history)
        
        if params['verbose']:
            print(f"  📊 Fold {fold} Results:")
            print(f"    Validation AUC: {val_metrics['auc']:.4f}")
            print(f"    Validation F1: {val_metrics['f1']:.4f}")
            print(f"    Validation Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"    Validation Precision: {val_metrics['precision']:.4f}")
            print(f"    Validation Recall: {val_metrics['recall']:.4f}")
            print()

    # Aggregate
    agg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
    
    if params['verbose']:
        print(f"{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print("Individual fold results:")
        for fold, metrics in enumerate(fold_metrics, 1):
            print(f"  Fold {fold}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
        print()
        print("Average metrics:")
        for metric, value in agg.items():
            print(f"  {metric.upper()}: {value:.4f}")
        print(f"{'='*60}")
        
    return agg, fold_metrics, fold_histories


def run_hyperparameter_search(X: np.ndarray,
                              y: np.ndarray,
                              feature_names: List[str],
                              base_params: Dict[str, Any],
                              search_space: Dict[str, List[Any]],
                              search_trials: int,
                              device: torch.device,
                              cv_cfg: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    def sample_params() -> Dict[str, Any]:
        sampled = base_params.copy()
        for k, v in search_space.items():
            sampled[k] = random.choice(v)
        return sampled

    best_score = -1.0
    best_params = None
    best_metrics = None
    trial_results = []

    if base_params['verbose']:
        print(f"Search space:")
        for param, values in search_space.items():
            print(f"  {param}: {values}")
        print()

    for t in range(search_trials):
        params = sample_params()
        if params['verbose']:
            print(f"\n{'='*50}")
            print(f"HYPERPARAMETER SEARCH TRIAL {t+1}/{search_trials}")
            print(f"{'='*50}")
            print("Parameters:")
            for param, value in params.items():
                if param in search_space:
                    print(f"  {param}: {value}")
            print()
            
        agg_metrics, _, _ = cross_validate(
            X=X,
            y=y,
            feature_names=feature_names,
            seq_len=params['seq_len'],
            n_splits=cv_cfg['n_splits'],
            val_window=cv_cfg['val_window'],
            gap=cv_cfg['gap'],
            min_train_window=cv_cfg['min_train_window'],
            params=params,
            device=device,
        )
        score = agg_metrics['auc']
        trial_results.append((score, params, agg_metrics))
        
        if params['verbose']:
            print(f"  🎯 Trial {t+1} Results:")
            print(f"    AUC: {score:.4f}")
            print(f"    F1: {agg_metrics['f1']:.4f}")
            print(f"    Accuracy: {agg_metrics['accuracy']:.4f}")
            
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = agg_metrics
            if params['verbose']:
                print(f"    🏆 NEW BEST SCORE!")
        print()

    if base_params['verbose']:
        print(f"{'='*60}")
        print("HYPERPARAMETER SEARCH SUMMARY")
        print(f"{'='*60}")
        print("All trial results (sorted by AUC):")
        trial_results.sort(key=lambda x: x[0], reverse=True)
        for i, (score, params, metrics) in enumerate(trial_results, 1):
            print(f"  {i}. AUC: {score:.4f} | F1: {metrics['f1']:.4f} | "
                  f"seq_len: {params['seq_len']} | hidden: {params['hidden_size']} | "
                  f"layers: {params['num_layers']} | lr: {params['lr']:.1e}")
        print()

    return best_params, best_metrics if best_metrics is not None else {}


def retrain_full_and_save(X: np.ndarray,
                          y: np.ndarray,
                          feature_names: List[str],
                          params: Dict[str, Any],
                          device: torch.device,
                          models_dir: str) -> str:
    if params['verbose']:
        print(f"\n{'='*60}")
        print("FINAL MODEL TRAINING ON FULL DATASET")
        print(f"{'='*60}")
        
    seq_len = params['seq_len']
    # Fit scaler on full data
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_seq, y_seq = build_sequences(X_scaled, y, seq_len)

    ds = SequenceDataset(X_seq, y_seq, np.arange(len(y_seq)))
    loader = DataLoader(ds, batch_size=params['batch_size'], shuffle=True)

    if params['verbose']:
        print(f"  Full dataset training:")
        print(f"    Total sequences: {len(ds)}")
        print(f"    Features: {X.shape[1]}")
        print(f"    Sequence length: {seq_len}")
        print(f"    Batch size: {params['batch_size']}")
        print(f"    Batches per epoch: {len(loader)}")
        print()

    model = LSTMClassifier(
        input_size=X.shape[1],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        bidirectional=params.get('bidirectional', False)
    ).to(device)

    pos_weight_value = compute_class_pos_weight(y_seq)
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    if params['verbose']:
        print(f"  Model architecture:")
        print(f"    Input size: {X.shape[1]}")
        print(f"    Hidden size: {params['hidden_size']}")
        print(f"    Number of layers: {params['num_layers']}")
        print(f"    Dropout: {params['dropout']}")
        print(f"    Positive class weight: {pos_weight_value:.3f}")
        print(f"    Learning rate: {params['lr']}")
        print(f"    Weight decay: {params['weight_decay']}")
        print()

    best_state = None
    best_loss = float('inf')
    epochs_no_improve = 0
    
    if params['verbose']:
        print(f"  Training for {params['epochs']} epochs with early stopping (patience: {params['patience']})...")
        
    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            logits = model(xb.float())
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        avg_loss = epoch_loss / len(ds)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
            improvement = "★ NEW BEST"
        else:
            epochs_no_improve += 1
            improvement = ""
            
        if params['verbose'] and (epoch % max(1, params['log_every']) == 0 or epoch == params['epochs'] - 1):
            print(f"    Epoch {epoch+1:3d}/{params['epochs']} | Loss: {avg_loss:.4f} | No Improve: {epochs_no_improve}/{params['patience']} {improvement}")
            
        if epochs_no_improve >= params['patience']:
            if params['verbose']:
                print(f"    ⚠️  Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(models_dir, exist_ok=True)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'feature_names': feature_names,
        'seq_len': seq_len,
        'params': params,
    }
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(models_dir, f"lstm_best_{timestamp}.pt")
    torch.save(ckpt, save_path)
    
    if params['verbose']:
        print(f"  💾 Model saved to: {save_path}")
        print(f"  📊 Final training loss: {best_loss:.4f}")
        print(f"{'='*60}")
        
    return save_path


def plot_training_history(fold_histories: List[Dict[str, List[float]]], 
                        fold_metrics: List[Dict[str, float]], 
                        models_dir: str,
                        plots_dir: str,
                        verbose: bool = True):
    """Create comprehensive training plots"""
    if not verbose:
        return
        
    print("  📊 Generating training plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LSTM Training History - Cross-Validation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: AUC over epochs for each fold
    for fold, history in enumerate(fold_histories, 1):
        epochs = range(1, len(history['train_auc']) + 1)
        axes[0, 0].plot(epochs, history['train_auc'], label=f'Train Fold {fold}', alpha=0.7, linewidth=2)
        axes[0, 0].plot(epochs, history['val_auc'], label=f'Val Fold {fold}', alpha=0.7, linewidth=2, linestyle='--')
    
    axes[0, 0].set_title('AUC Score Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('AUC Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: F1 Score over epochs for each fold
    for fold, history in enumerate(fold_histories, 1):
        epochs = range(1, len(history['train_f1']) + 1)
        axes[0, 1].plot(epochs, history['train_f1'], label=f'Train Fold {fold}', alpha=0.7, linewidth=2)
        axes[0, 1].plot(epochs, history['val_f1'], label=f'Val Fold {fold}', alpha=0.7, linewidth=2, linestyle='--')
    
    axes[0, 1].set_title('F1 Score Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss and LR over epochs for each fold
    for fold, history in enumerate(fold_histories, 1):
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0, 2].plot(epochs, history['train_loss'], label=f'Train Fold {fold}', alpha=0.7, linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) == len(history['train_loss']):
            axes[0, 2].plot(epochs, history['val_loss'], label=f'Val Fold {fold}', alpha=0.7, linewidth=2, linestyle='--')
        if 'lr' in history and len(history['lr']) == len(history['train_loss']):
            ax_lr = axes[0, 2].twinx()
            ax_lr.plot(epochs, history['lr'], label=f'LR Fold {fold}', alpha=0.3, color='purple')
            ax_lr.set_ylabel('Learning Rate')
    
    axes[0, 2].set_title('Loss (Train/Val) and LR Over Epochs')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Final CV metrics comparison
    metrics_names = ['auc', 'f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall']
    fold_avg_metrics = []
    for metric in metrics_names:
        avg_val = np.mean([fold_metric[metric] for fold_metric in fold_metrics])
        fold_avg_metrics.append(avg_val)
    
    bars = axes[1, 0].bar(metrics_names, fold_avg_metrics, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[1, 0].set_title('Average CV Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, fold_avg_metrics):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Individual fold performance
    fold_numbers = list(range(1, len(fold_metrics) + 1))
    fold_aucs = [fold_metric['auc'] for fold_metric in fold_metrics]
    fold_f1s = [fold_metric['f1'] for fold_metric in fold_metrics]
    
    x = np.arange(len(fold_numbers))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, fold_aucs, width, label='AUC', alpha=0.7, color='lightcoral')
    bars2 = axes[1, 1].bar(x + width/2, fold_f1s, width, label='F1', alpha=0.7, color='lightgreen')
    
    axes[1, 1].set_title('Individual Fold Performance')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(fold_numbers)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Learning curves summary
    # Calculate average learning curves across folds
    max_epochs = max(len(history['train_auc']) for history in fold_histories)
    avg_train_auc = np.zeros(max_epochs)
    avg_val_auc = np.zeros(max_epochs)
    avg_train_f1 = np.zeros(max_epochs)
    avg_val_f1 = np.zeros(max_epochs)
    
    for history in fold_histories:
        for i in range(len(history['train_auc'])):
            avg_train_auc[i] += history['train_auc'][i]
            avg_val_auc[i] += history['val_auc'][i]
            avg_train_f1[i] += history['train_f1'][i]
            avg_val_f1[i] += history['val_f1'][i]
    
    n_folds = len(fold_histories)
    avg_train_auc /= n_folds
    avg_val_auc /= n_folds
    avg_train_f1 /= n_folds
    avg_val_f1 /= n_folds
    
    epochs = range(1, max_epochs + 1)
    axes[1, 2].plot(epochs, avg_train_auc, label='Avg Train AUC', linewidth=3, color='blue')
    axes[1, 2].plot(epochs, avg_val_auc, label='Avg Val AUC', linewidth=3, color='red', linestyle='--')
    axes[1, 2].plot(epochs, avg_train_f1, label='Avg Train F1', linewidth=3, color='green', alpha=0.7)
    axes[1, 2].plot(epochs, avg_val_f1, label='Avg Val F1', linewidth=3, color='orange', linestyle='--', alpha=0.7)
    
    axes[1, 2].set_title('Average Learning Curves')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'lstm_training_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  📈 Training plots saved to: {plot_path}")
    plt.show()


def plot_test_predictions_vs_price(model_path: str, 
                                 data_csv: str,
                                 models_dir: str,
                                 plots_dir: str,
                                 test_start_date: str,
                                 device: torch.device,
                                 verbose: bool = True):
    """
    Create a plot showing model predictions vs S&P 500 price for the test set (2022 onwards).
    """
    if not verbose:
        return
        
    print("  📊 Generating test set predictions vs price visualization...")
    
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    model = LSTMClassifier(
        input_size=len(checkpoint['feature_names']),
        hidden_size=checkpoint['params']['hidden_size'],
        num_layers=checkpoint['params']['num_layers'],
        dropout=checkpoint['params']['dropout'],
        bidirectional=checkpoint['params'].get('bidirectional', False)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and prepare data
    df = load_dataset(data_csv)
    X_df, y, feature_names = select_features_and_target(df)
    X = X_df.values.astype(np.float32)
    
    # Split data into train and test based on date
    test_start = pd.to_datetime(test_start_date)
    if 'Date' in df.columns:
        test_mask = df['Date'] >= test_start
        train_mask = ~test_mask
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        test_dates = df['Date'][test_mask].values
        test_prices = df['Close'][test_mask].values if 'Close' in df.columns else None
    else:
        # Fallback if no date column
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        test_dates = np.arange(len(X_test))
        test_prices = None
    
    # Scale data using only training data (no data leakage)
    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build sequences for test set
    seq_len = checkpoint['seq_len']
    X_test_seq, y_test_seq = build_sequences(X_test_scaled, y_test, seq_len)
    
    # Get predictions for test set
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_test_seq), 128):  # Process in batches
            batch = X_test_seq[i:i+128]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Adjust dates and prices for sequence offset
    if test_prices is not None:
        test_prices = test_prices[seq_len-1:]
    test_dates = test_dates[seq_len-1:]
    
    # Create the visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'LSTM Model Predictions vs S&P 500 Price (Test Set: {test_start_date} onwards)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: S&P 500 Price
    if test_prices is not None:
        ax1.plot(test_dates, test_prices, color='blue', linewidth=1.5, alpha=0.8)
        ax1.set_title('S&P 500 Closing Price (Test Set)', fontweight='bold')
        ax1.set_ylabel('Price ($)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add price trend
        if len(test_prices) > 30:
            ma_30 = np.convolve(test_prices, np.ones(30)/30, mode='valid')
            ma_dates = test_dates[29:]
            ax1.plot(ma_dates, ma_30, color='red', linewidth=2, alpha=0.7, label='30-day MA')
            ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Price data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('S&P 500 Closing Price (Test Set)', fontweight='bold')
    
    # Plot 2: Model Predictions (Probability)
    ax2.plot(test_dates, predictions, color='green', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold (0.5)')
    ax2.set_title('LSTM Prediction Probability (14-day forward)', fontweight='bold')
    ax2.set_ylabel('Probability', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Predictions vs Actual (Binary)
    # Convert predictions to binary using 0.5 threshold
    pred_binary = (predictions >= 0.5).astype(int)
    actual_binary = y_test_seq
    
    # Create scatter plot
    correct_up = (pred_binary == 1) & (actual_binary == 1)
    correct_down = (pred_binary == 0) & (actual_binary == 0)
    wrong_up = (pred_binary == 1) & (actual_binary == 0)
    wrong_down = (pred_binary == 0) & (actual_binary == 1)
    
    if test_prices is not None:
        ax3.scatter(test_dates[correct_up], test_prices[correct_up], color='green', alpha=0.6, s=20, label='Correct Up Prediction')
        ax3.scatter(test_dates[correct_down], test_prices[correct_down], color='blue', alpha=0.6, s=20, label='Correct Down Prediction')
        ax3.scatter(test_dates[wrong_up], test_prices[wrong_up], color='red', alpha=0.6, s=20, label='Wrong Up Prediction')
        ax3.scatter(test_dates[wrong_down], test_prices[wrong_down], color='orange', alpha=0.6, s=20, label='Wrong Down Prediction')
    else:
        ax3.scatter(test_dates[correct_up], np.zeros_like(test_dates[correct_up]), color='green', alpha=0.6, s=20, label='Correct Up Prediction')
        ax3.scatter(test_dates[correct_down], np.zeros_like(test_dates[correct_down]), color='blue', alpha=0.6, s=20, label='Correct Down Prediction')
        ax3.scatter(test_dates[wrong_up], np.zeros_like(test_dates[wrong_up]), color='red', alpha=0.6, s=20, label='Wrong Up Prediction')
        ax3.scatter(test_dates[wrong_down], np.zeros_like(test_dates[wrong_down]), color='orange', alpha=0.6, s=20, label='Wrong Down Prediction')
    
    ax3.set_title('Prediction Accuracy vs Price (Test Set)', fontweight='bold')
    ax3.set_ylabel('Price ($)' if test_prices is not None else 'Index', fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Calculate and display accuracy metrics
    accuracy = np.mean(pred_binary == actual_binary)
    precision = np.sum((pred_binary == 1) & (actual_binary == 1)) / max(np.sum(pred_binary == 1), 1)
    recall = np.sum((pred_binary == 1) & (actual_binary == 1)) / max(np.sum(actual_binary == 1), 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
    
    # Add text box with metrics
    metrics_text = f'Test Set Metrics:\nAccuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
    ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Format x-axis dates
    if len(test_dates) > 0 and isinstance(test_dates[0], pd.Timestamp):
        ax3.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
        ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'lstm_test_predictions_vs_price.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  📈 Test set predictions vs price plot saved to: {plot_path}")
    plt.show()
    
    # Create additional detailed analysis plot for test set
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f'LSTM Model Test Set Analysis ({test_start_date} onwards)', fontsize=16, fontweight='bold')
    
    # Plot 1: Prediction distribution
    ax1.hist(predictions, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax1.set_title('Distribution of Prediction Probabilities (Test Set)')
    ax1.set_xlabel('Prediction Probability')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction confidence over time
    confidence = np.abs(predictions - 0.5) * 2  # Convert to 0-1 confidence scale
    ax2.plot(test_dates, confidence, color='purple', alpha=0.7)
    ax2.set_title('Model Confidence Over Time (Test Set)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Confidence (0-1)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling accuracy
    window_size = min(50, len(pred_binary) // 4)  # Adaptive window size
    if window_size > 10:
        rolling_accuracy = []
        rolling_dates = []
        for i in range(window_size, len(pred_binary)):
            window_acc = np.mean(pred_binary[i-window_size:i] == actual_binary[i-window_size:i])
            rolling_accuracy.append(window_acc)
            rolling_dates.append(test_dates[i])
        
        ax3.plot(rolling_dates, rolling_accuracy, color='green', alpha=0.7)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')
        ax3.set_title(f'Rolling Accuracy ({window_size}-day window)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for rolling accuracy', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Rolling Accuracy')
    
    # Plot 4: Prediction vs actual correlation
    ax4.scatter(actual_binary, predictions, alpha=0.5, color='blue')
    ax4.set_title('Predictions vs Actual Values (Test Set)')
    ax4.set_xlabel('Actual (0=Down, 1=Up)')
    ax4.set_ylabel('Predicted Probability')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save detailed analysis plot
    detailed_plot_path = os.path.join(plots_dir, 'lstm_test_detailed_analysis.png')
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Test set detailed analysis plot saved to: {detailed_plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train LSTM with time-series CV for S&P 500 target prediction')
    parser.add_argument('--data-csv', type=str, default='data/final_dataset_for_modeling.csv')
    parser.add_argument('--models-dir', type=str, default='models/lstm')
    parser.add_argument('--plots-dir', type=str, default='plots/lstm')
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--val-window', type=int, default=180)
    parser.add_argument('--gap', type=int, default=14)
    parser.add_argument('--min-train-window', type=int, default=365)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--use-scheduler', action='store_true')
    parser.add_argument('--scheduler-factor', type=float, default=0.7)
    parser.add_argument('--scheduler-patience', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--threshold-metric', type=str, default='f1', choices=['f1', 'youden'])
    parser.add_argument('--test-start-date', type=str, default='2022-01-01')
    parser.add_argument('--search-trials', type=int, default=0, help='Number of random trials. 0 disables search')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--smoke', action='store_true', help='Run a very small training for quick validation')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    
    if args.verbose:
        print(f"{'='*60}")
        print("LSTM MODEL TRAINING FOR S&P 500 PREDICTION")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Random seed: {args.seed}")
        print(f"Data file: {args.data_csv}")
        print(f"Models directory: {args.models_dir}")
        print(f"Plots directory: {args.plots_dir}")
        print(f"Verbose mode: {args.verbose}")
        print(f"Smoke test: {args.smoke}")
        print()

    df = load_dataset(args.data_csv)
    X_df, y, feature_names = select_features_and_target(df)
    X = X_df.values.astype(np.float32)
    
    if args.verbose:
        print(f"Dataset loaded successfully:")
        print(f"  Shape: {X.shape}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Target distribution: {np.bincount(y)}")
        print(f"  Positive class ratio: {y.mean():.3f}")
        print()

    # Adjust for smoke test
    base_params = {
        'seq_len': args.seq_len if not args.smoke else 30,
        'hidden_size': args.hidden_size if not args.smoke else 32,
        'num_layers': args.num_layers if not args.smoke else 1,
        'dropout': args.dropout if not args.smoke else 0.1,
        'batch_size': args.batch_size if not args.smoke else 64,
        'epochs': args.epochs if not args.smoke else 8,
        'patience': args.patience if not args.smoke else 3,
        'lr': args.lr if not args.smoke else 1e-3,
        'weight_decay': args.weight_decay,
        'verbose': args.verbose,
        'log_every': args.log_every,
        'bidirectional': args.bidirectional if not args.smoke else False,
        'use_scheduler': args.use_scheduler if not args.smoke else False,
        'scheduler_factor': args.scheduler_factor,
        'scheduler_patience': args.scheduler_patience,
        'min_lr': args.min_lr,
        'threshold_metric': args.threshold_metric,
    }

    cv_cfg = {
        'n_splits': args.n_splits if not args.smoke else 3,
        'val_window': args.val_window if not args.smoke else 120,
        'gap': args.gap,
        'min_train_window': args.min_train_window if not args.smoke else 240,
    }

    if args.search_trials and args.search_trials > 0:
        if args.verbose:
            print(f"\n{'='*60}")
            print("STARTING HYPERPARAMETER SEARCH")
            print(f"{'='*60}")
            print(f"Search trials: {args.search_trials}")
            print(f"CV splits: {cv_cfg['n_splits']}")
            print(f"Validation window: {cv_cfg['val_window']} days")
            print(f"Gap: {cv_cfg['gap']} days")
            print(f"Min train window: {cv_cfg['min_train_window']} days")
            print(f"{'='*60}\n")
            
        search_space = {
            'seq_len': [30, 45, 60, 90],
            'hidden_size': [32, 64, 96],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'batch_size': [64, 96, 128],
            'lr': [1e-3, 7e-4, 5e-4, 3e-4],
            'bidirectional': [False, True],
            'use_scheduler': [False, True],
        }
        trials = args.search_trials if not args.smoke else min(args.search_trials, 3)
        best_params, best_cv_metrics = run_hyperparameter_search(
            X=X,
            y=y,
            feature_names=feature_names,
            base_params=base_params,
            search_space=search_space,
            search_trials=trials,
            device=device,
            cv_cfg=cv_cfg,
        )
        if args.verbose:
            print(f"\n{'='*60}")
            print("HYPERPARAMETER SEARCH RESULTS")
            print(f"{'='*60}")
            print("Best CV metrics:")
            for metric, value in best_cv_metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
            print("\nBest parameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"{'='*60}\n")
        final_params = best_params
    else:
        # Single CV with base params
        if args.verbose:
            print(f"\n{'='*60}")
            print("STARTING CROSS-VALIDATION TRAINING")
            print(f"{'='*60}")
            print(f"Sequence length: {base_params['seq_len']}")
            print(f"Hidden size: {base_params['hidden_size']}")
            print(f"Number of layers: {base_params['num_layers']}")
            print(f"Dropout: {base_params['dropout']}")
            print(f"Batch size: {base_params['batch_size']}")
            print(f"Learning rate: {base_params['lr']}")
            print(f"Epochs: {base_params['epochs']}")
            print(f"CV splits: {cv_cfg['n_splits']}")
            print(f"Validation window: {cv_cfg['val_window']} days")
            print(f"Gap: {cv_cfg['gap']} days")
            print(f"Min train window: {cv_cfg['min_train_window']} days")
            print(f"{'='*60}\n")
            
        agg_metrics, fold_metrics, fold_histories = cross_validate(
            X=X,
            y=y,
            feature_names=feature_names,
            seq_len=base_params['seq_len'],
            n_splits=cv_cfg['n_splits'],
            val_window=cv_cfg['val_window'],
            gap=cv_cfg['gap'],
            min_train_window=cv_cfg['min_train_window'],
            params=base_params,
            device=device,
        )
        if args.verbose:
            print(f"\n{'='*60}")
            print("CROSS-VALIDATION RESULTS")
            print(f"{'='*60}")
            print("Average CV metrics:")
            for metric, value in agg_metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
            print(f"{'='*60}\n")
            
            # Plot training history
            plot_training_history(fold_histories, fold_metrics, args.models_dir, args.plots_dir, args.verbose)
        final_params = base_params

    # Holdout evaluation from a specific date (default: 2022-01-01)
    test_start_date = pd.to_datetime(args.test_start_date)
    if 'Date' in df.columns:
        test_mask = df['Date'] >= test_start_date
        if test_mask.any() and (~test_mask).any():
            X_train_df = X_df[~test_mask]
            y_train = y[~test_mask]
            X_test_df = X_df[test_mask]
            y_test = y[test_mask]

            if args.verbose:
                print(f"{'='*60}")
                print("HOLDOUT EVALUATION")
                print(f"{'='*60}")
                print(f"Train rows: {X_train_df.shape[0]} | Test rows (since {args.test_start_date}): {X_test_df.shape[0]}")
                print()

            # Scale on train only
            scaler_holdout = StandardScaler().fit(X_train_df.values.astype(np.float32))
            X_train_scaled = scaler_holdout.transform(X_train_df.values.astype(np.float32))
            X_test_scaled = scaler_holdout.transform(X_test_df.values.astype(np.float32))

            seq_len = final_params['seq_len']
            X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train, seq_len)
            X_test_seq, y_test_seq = build_sequences(X_test_scaled, y_test, seq_len)

            train_ds_holdout = SequenceDataset(X_train_seq, y_train_seq, np.arange(len(y_train_seq)))
            test_ds_holdout = SequenceDataset(X_test_seq, y_test_seq, np.arange(len(y_test_seq)))
            train_loader_holdout = DataLoader(train_ds_holdout, batch_size=final_params['batch_size'], shuffle=True)
            test_loader_holdout = DataLoader(test_ds_holdout, batch_size=final_params['batch_size'], shuffle=False)

            # Train fresh model on pre-test data
            holdout_model = LSTMClassifier(
                input_size=X_train_df.shape[1],
                hidden_size=final_params['hidden_size'],
                num_layers=final_params['num_layers'],
                dropout=final_params['dropout'],
                bidirectional=final_params.get('bidirectional', False)
            ).to(device)

            pos_weight_value = compute_class_pos_weight(y_train_seq)
            criterion_holdout = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
            optimizer_holdout = torch.optim.Adam(holdout_model.parameters(), lr=final_params['lr'], weight_decay=final_params['weight_decay'])
            scheduler_holdout = None
            if final_params.get('use_scheduler', False):
                scheduler_holdout = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_holdout, mode='max', factor=final_params.get('scheduler_factor', 0.5),
                    patience=final_params.get('scheduler_patience', 2), min_lr=final_params.get('min_lr', 1e-5)
                )

            best_auc = -1.0
            best_state = None
            best_thr_holdout = 0.5
            epochs_no_improve = 0
            for epoch in range(final_params['epochs']):
                holdout_model.train()
                total_loss = 0.0
                for xb, yb in train_loader_holdout:
                    xb = xb.to(device)
                    yb = yb.float().to(device)
                    optimizer_holdout.zero_grad()
                    logits = holdout_model(xb.float())
                    loss = criterion_holdout(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(holdout_model.parameters(), max_norm=1.0)
                    optimizer_holdout.step()
                    total_loss += loss.item() * xb.size(0)

                avg_loss = total_loss / len(train_loader_holdout.dataset)
                # Tune threshold on train
                train_probs, train_labels = collect_probs_and_labels(holdout_model, train_loader_holdout, device)
                tuned_thr, _ = find_best_threshold(train_probs, train_labels, metric=final_params.get('threshold_metric', 'f1'))
                val_metrics = evaluate(holdout_model, test_loader_holdout, device, threshold=tuned_thr)
                if scheduler_holdout is not None:
                    scheduler_holdout.step(val_metrics['auc'])
                is_improve = val_metrics['auc'] > best_auc
                if is_improve:
                    best_auc = val_metrics['auc']
                    best_state = holdout_model.state_dict()
                    best_thr_holdout = tuned_thr
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if args.verbose and (epoch % max(1, args.log_every) == 0 or epoch == final_params['epochs'] - 1):
                    print(f"    [Holdout] Epoch {epoch+1:3d}/{final_params['epochs']} | Loss: {avg_loss:.4f} | Val AUC: {val_metrics['auc']:.4f} | Thr: {tuned_thr:.2f} | No Improve: {epochs_no_improve}/{final_params['patience']}")
                if epochs_no_improve >= final_params['patience']:
                    if args.verbose:
                        print("    [Holdout] Early stopping")
                    break

            if best_state is not None:
                holdout_model.load_state_dict(best_state)
            holdout_metrics = evaluate(holdout_model, test_loader_holdout, device, threshold=best_thr_holdout)
        else:
            holdout_metrics = {}
    else:
        holdout_metrics = {}

    # Retrain on full data with final params and save
    model_path = retrain_full_and_save(X, y, feature_names, final_params, device, args.models_dir)

    # Save report
    report = {
        'device': str(device),
        'final_params': final_params,
        'models_dir': args.models_dir,
        'model_path': model_path,
        'dataset_info': {
            'shape': X.shape,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'target_distribution': np.bincount(y).tolist(),
            'positive_class_ratio': float(y.mean())
        },
        'training_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seed': args.seed,
            'smoke_test': args.smoke
        }
    }
    
    if 'agg_metrics' in locals():
        report['cv_metrics'] = agg_metrics
    if 'best_cv_metrics' in locals():
        report['best_cv_metrics'] = best_cv_metrics
    if holdout_metrics:
        report['holdout_metrics_since_' + args.test_start_date] = holdout_metrics
        
    os.makedirs(args.models_dir, exist_ok=True)
    report_path = os.path.join(args.models_dir, 'lstm_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    if args.verbose:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"💾 Model saved to: {model_path}")
        print(f"📊 Report saved to: {report_path}")
        if 'agg_metrics' in locals():
            print(f"📈 CV Performance:")
            for metric, value in agg_metrics.items():
                print(f"    {metric.upper()}: {value:.4f}")
        print(f"{'='*60}")
    
    # Generate test set predictions vs price visualization
    plot_test_predictions_vs_price(model_path, args.data_csv, args.models_dir, args.plots_dir, args.test_start_date, device, args.verbose)


if __name__ == '__main__':
    main()


